import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd


class BatteryDispatchModel:
    """A class to model and optimize battery dispatch given a set of hourly and half hourly prices.

    To simplify the modelling we assume that the core time resolution is half-hourly,
    and the hourly market variables are constrained to be the same for both half-hourly
    periods within the hour.

    Becuase of this, the maximum charge and discharge rates are halved to reflect
    the fact that the battery can only charge or discharge at that rate for half an hour.

    Attributes:
        battery_capacity (float): Maximum capacity of the battery (MWh).
        initial_soc (float): Initial state of charge of the battery (MWh).
        max_charge_rate (float): Maximum charge rate of the battery (MW).
        max_discharge_rate (float): Maximum discharge rate of the battery (MW).
        charge_efficiency (float): Efficiency of charging the battery (0 < efficiency <= 1).
        discharge_efficiency (float): Efficiency of discharging the battery (0 < efficiency <=
            1).
        time_horizon (int): Number of time periods to model (e.g., 24 for one day).
        maximum_cycles (int): Maximum number of charge/discharge cycles.
        storage_degradation (float): Degradation factor per cycle (0 < degradation < 1).
        capex (float): Capital expenditure per kWh of battery capacity ($/kWh).
        opex (float): Operational expenditure per kWh of energy throughput ($/kWh).
        market_data_path (str): Path to the Excel file containing market data.
        model (pyo.ConcreteModel): Pyomo model instance.

    Methods:
    __init__: Initializes the battery dispatch model with given parameters.
    _build_model: Constructs the Pyomo model with variables, constraints, and objective.
    _load_data: Loads time series data for load and renewable generation.
    solve: Solves the optimization problem using a specified solver.
    save_results: Prints and saves the results of the optimization.
    """

    def __init__(
        self,
        battery_capacity: float = 4.0,
        initial_soc: float = 2.0,
        max_charge_rate: float = 2.0,
        max_discharge_rate: float = 2.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        time_horizon: int = 8760 * 2 + 8784,  # three years of hourly data
        maximum_cycles: int = 5000,
        maximum_age: int = 10,
        storage_degradation: float = 0.0001,
        capex: float = 500000,
        opex: float = 5000,
        market_data_path: str = "./data/assignment_data_mini.xlsx",
    ):
        """Initializes the battery dispatch model with given parameters.

        Args:

            battery_capacity (float): Maximum capacity of the battery (MWh).
            initial_soc (float): Initial state of charge of the battery (MWh).
            max_charge_rate (float): Maximum charge rate of the battery (MW).
            max_discharge_rate (float): Maximum discharge rate of the battery (MW).
            charge_efficiency (float): Efficiency of charging the battery (0 < efficiency <= 1).
            discharge_efficiency (float): Efficiency of discharging the battery (0 < efficiency <=
                1).
            time_horizon (int): Number of time periods to model (e.g., 24 for one day).
            maximum_cycles (int): Maximum number of charge/discharge cycles.
            maximum_age (int): Maximum age of the battery in years.
            storage_degradation (float): Degradation factor per cycle (0 < degradation < 1).
            capex (float): Capital expenditure per kWh of battery capacity ($/kWh).
            opex (float): Operational expenditure per kWh of energy throughput ($/kWh).
            market_data_path (str): Path to the Excel file containing market data.
        """
        self.battery_capacity = battery_capacity
        self.initial_soc = initial_soc
        self.max_charge_rate = max_charge_rate / 2  # because half-hourly
        self.max_discharge_rate = max_discharge_rate / 2  # because half-hourly
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.time_horizon = time_horizon
        self.maximum_cycles = maximum_cycles
        self.maximum_age = maximum_age
        self.battery_degradation = storage_degradation
        self.capex = capex
        self.opex = opex

        self.model = pyo.ConcreteModel()
        self._load_data(market_data_path)
        self._build_model()

    def _build_model(self):
        """Constructs the Pyomo model with variables, constraints, and objective.

        This method sets up the optimization model including:
            - Sets for time periods
            - Parameters for prices of energy markets
            - Variables for battery charge, discharge, and state of charge
            - Constraints for battery operation and market rules
            - Objective function for profit maximization

        Note:
            This is an internal method called by __init__.
        """
        m = self.model

        # Sets
        m.T_halfhour = pyo.RangeSet(0, self.time_horizon * 2 - 1)

        # Variables
        m.charge_hour = pyo.Var(
            m.T_halfhour, within=pyo.NonNegativeReals, bounds=(0, self.max_charge_rate)
        )
        m.discharge_hour = pyo.Var(
            m.T_halfhour,
            within=pyo.NonNegativeReals,
            bounds=(0, self.max_discharge_rate),
        )
        m.is_charging_hour = pyo.Var(m.T_halfhour, within=pyo.Binary)

        m.charge_halfhour = pyo.Var(
            m.T_halfhour, within=pyo.NonNegativeReals, bounds=(0, self.max_charge_rate)
        )
        m.discharge_halfhour = pyo.Var(
            m.T_halfhour,
            within=pyo.NonNegativeReals,
            bounds=(0, self.max_discharge_rate),
        )
        m.is_charging_halfhour = pyo.Var(m.T_halfhour, within=pyo.Binary)

        m.current_degradation = pyo.Var(m.T_halfhour, within=pyo.NonNegativeReals)

        m.total_cycles = pyo.Var(m.T_halfhour, within=pyo.NonNegativeReals)

        m.soc = pyo.Var(
            m.T_halfhour,
            within=pyo.NonNegativeReals,
            bounds=(0, self.battery_capacity),
        )

        # Parameters

        m.initial_soc = pyo.Param(initialize=self.initial_soc)

        # Initialize price parameters
        m.hourly_price = pyo.Param(
            m.T_halfhour,
            within=pyo.Reals,
            initialize=self.hourly_data["Market 2 Price [£/MWh]"].to_dict(),
        )
        m.halfhourly_price = pyo.Param(
            m.T_halfhour,
            within=pyo.Reals,
            initialize=self.halfhourly_data["Market 1 Price [£/MWh]"].to_dict(),
        )

        # Constraints

        # Constraint: Battery cannot charge and discharge simultaneously.
        def charge_discharge_hour_rule_1(m, t):
            M = self.max_charge_rate
            return m.charge_hour[t] <= M * m.is_charging_hour[t]

        def charge_discharge_hour_rule_2(m, t):
            M = self.max_charge_rate
            return m.discharge_hour[t] <= M * (1 - m.is_charging_hour[t])

        def charge_discharge_halfhour_rule_1(m, t):
            M = self.max_charge_rate
            return m.charge_halfhour[t] <= M * m.is_charging_halfhour[t]

        def charge_discharge_halfhour_rule_2(m, t):
            M = self.max_charge_rate
            return m.discharge_halfhour[t] <= M * (1 - m.is_charging_halfhour[t])

        # Linking constraint between hourly and half-hourly market
        def charge_discharge_hour_halfhour_link_rule_1(m, t):
            M = self.max_charge_rate
            return m.charge_halfhour[t] <= M * m.is_charging_hour[t]

        def charge_discharge_hour_halfhour_link_rule_2(m, t):
            M = self.max_charge_rate
            return m.discharge_halfhour[t] <= M * (1 - m.is_charging_hour[t])

        m.charge_discharge_hour_constraint_1 = pyo.Constraint(
            m.T_halfhour, rule=charge_discharge_hour_rule_1
        )
        m.charge_discharge_hour_constraint_2 = pyo.Constraint(
            m.T_halfhour, rule=charge_discharge_hour_rule_2
        )
        m.charge_discharge_halfhour_constraint_1 = pyo.Constraint(
            m.T_halfhour, rule=charge_discharge_halfhour_rule_1
        )
        m.charge_discharge_halfhour_constraint_2 = pyo.Constraint(
            m.T_halfhour, rule=charge_discharge_halfhour_rule_2
        )
        m.charge_discharge_hour_halfhour_link_constraint_1 = pyo.Constraint(
            m.T_halfhour, rule=charge_discharge_hour_halfhour_link_rule_1
        )
        m.charge_discharge_hour_halfhour_link_constraint_2 = pyo.Constraint(
            m.T_halfhour, rule=charge_discharge_hour_halfhour_link_rule_2
        )

        # Constraint: the hourly behaviour should be the same for the full hour
        def charge_hourly_consistency_rule(m, t):
            if t % 2 == 0:
                return m.charge_hour[t] == m.charge_hour[t + 1]
            else:
                return pyo.Constraint.Skip

        def discharge_hourly_consistency_rule(m, t):
            if t % 2 == 0:
                return m.discharge_hour[t] == m.discharge_hour[t + 1]
            else:
                return pyo.Constraint.Skip

        m.charge_hourly_consistency_constraint = pyo.Constraint(
            m.T_halfhour, rule=charge_hourly_consistency_rule
        )
        m.discharge_hourly_consistency_constraint = pyo.Constraint(
            m.T_halfhour, rule=discharge_hourly_consistency_rule
        )

        # Constraint: State of Charge (SoC) dynamics.

        def soc_dynamics_rule(m, t):
            if t == 0:
                expr = (
                    m.initial_soc
                    + m.charge_hour[t] * self.charge_efficiency
                    - m.discharge_hour[t]
                    + m.charge_halfhour[t] * self.charge_efficiency
                    - m.discharge_halfhour[t]
                )
            else:
                expr = (
                    m.soc[t - 1]
                    + m.charge_hour[t] * self.charge_efficiency
                    - m.discharge_hour[t]
                    + m.charge_halfhour[t] * self.charge_efficiency
                    - m.discharge_halfhour[t]
                )
            return m.soc[t] == expr

        m.soc_constraint = pyo.Constraint(m.T_halfhour, rule=soc_dynamics_rule)

        # Constraint: make sure battery doesn't discharge more than its current SoC
        def discharge_limit_rule(m, t):
            return m.discharge_hour[t] + m.discharge_halfhour[t] <= m.soc[t]

        m.discharge_limit_constraint = pyo.Constraint(
            m.T_halfhour, rule=discharge_limit_rule
        )

        # Constraint: make sure battery doesn't charge more than its remaining capacity including degradation and charge efficiency
        def charge_limit_rule(m, t):
            return (
                (m.charge_hour[t] + m.charge_halfhour[t]) * self.charge_efficiency
                <= self.battery_capacity - m.current_degradation[t] - m.soc[t]
            )

        m.charge_limit_constraint = pyo.Constraint(m.T_halfhour, rule=charge_limit_rule)

        # Constraint on total charge max rate
        def total_charge_rate_rule(m, t):
            return m.charge_hour[t] + m.charge_halfhour[t] <= self.max_charge_rate

        m.total_charge_rate_constraint = pyo.Constraint(
            m.T_halfhour, rule=total_charge_rate_rule
        )

        # Constraint on total discharge max rate
        def total_discharge_rate_rule(m, t):
            return (
                m.discharge_hour[t] + m.discharge_halfhour[t] <= self.max_discharge_rate
            )

        m.total_discharge_rate_constraint = pyo.Constraint(
            m.T_halfhour, rule=total_discharge_rate_rule
        )

        # Constraint: Limit on total cycles
        # SIMPLIFICATION - this does not include degradation effects on capacity as that makes the problem non-linear
        def total_cycles_rule(m, t):
            if t == 0:
                total_cycles = (m.charge_hour[t] + m.discharge_hour[t]) / (
                    self.battery_capacity
                ) / 2 + (m.charge_halfhour[t] + m.discharge_halfhour[t]) / (
                    self.battery_capacity
                ) / 2
            else:
                total_cycles = (
                    m.total_cycles[t - 1]
                    + (m.charge_hour[t] + m.discharge_hour[t])
                    / (self.battery_capacity)
                    / 2
                    + (m.charge_halfhour[t] + m.discharge_halfhour[t])
                    / (self.battery_capacity)
                    / 2
                )

            return m.total_cycles[t] == total_cycles

        m.total_cycles_constraint = pyo.Constraint(m.T_halfhour, rule=total_cycles_rule)

        def max_cycles_rule(m, t):
            return m.total_cycles[t] <= self.maximum_cycles

        m.max_cycles_constraint = pyo.Constraint(m.T_halfhour, rule=max_cycles_rule)

        # Constraint: battery degradation

        def battery_degradation_rule(m, t):
            # Division by 4 because h full cycle is charge + discharge

            current_degradation = m.total_cycles[t] * self.battery_degradation
            return m.current_degradation[t] == current_degradation

        m.battery_degradation_constraint = pyo.Constraint(
            m.T_halfhour, rule=battery_degradation_rule
        )

        # Constraint: limit the maximum lifespan of the battery
        def max_age_charge_hour_rule(m, t):
            max_periods = (
                self.maximum_age * 365 * 48
            )  # half-hourly periods in maximum_age years
            if self.time_horizon > max_periods and t >= max_periods:
                return m.charge_hour[t] == 0
            return pyo.Constraint.Skip

        def max_age_discharge_hour_rule(m, t):
            max_periods = self.maximum_age * 365 * 48
            if self.time_horizon > max_periods and t >= max_periods:
                return m.discharge_hour[t] == 0
            return pyo.Constraint.Skip

        def max_age_charge_halfhour_rule(m, t):
            max_periods = self.maximum_age * 365 * 48
            if self.time_horizon > max_periods and t >= max_periods:
                return m.charge_halfhour[t] == 0
            return pyo.Constraint.Skip

        def max_age_discharge_halfhour_rule(m, t):
            max_periods = self.maximum_age * 365 * 48
            if self.time_horizon > max_periods and t >= max_periods:
                return m.discharge_halfhour[t] == 0
            return pyo.Constraint.Skip

        m.max_age_charge_hour_constraint = pyo.Constraint(
            m.T_halfhour, rule=max_age_charge_hour_rule
        )
        m.max_age_discharge_hour_constraint = pyo.Constraint(
            m.T_halfhour, rule=max_age_discharge_hour_rule
        )
        m.max_age_charge_halfhour_constraint = pyo.Constraint(
            m.T_halfhour, rule=max_age_charge_halfhour_rule
        )
        m.max_age_discharge_halfhour_constraint = pyo.Constraint(
            m.T_halfhour, rule=max_age_discharge_halfhour_rule
        )

        # Objective: maximize profit (maximise profit)
        def objective_rule(m):
            energy_cost = sum(
                (
                    m.charge_hour[t] * m.hourly_price[t]
                    + m.charge_halfhour[t] * m.halfhourly_price[t]
                )
                for t in m.T_halfhour
            )

            revenue = sum(
                (
                    m.discharge_hour[t] * self.discharge_efficiency * m.hourly_price[t]
                    + m.discharge_halfhour[t]
                    * self.discharge_efficiency
                    * m.halfhourly_price[t]
                )
                for t in m.T_halfhour
            )

            operational_cost = self.opex * min(
                self.time_horizon / 8766, self.maximum_age
            )

            capital_cost = self.capex

            total_profit = revenue - energy_cost - operational_cost - capital_cost
            return total_profit

        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def _load_data(self, filepath: str):
        """Loads time series price data for hourly and half hourly markets.

        Args:
            filepath (str): Path to the CSV file containing hourly and half hourly price data.

        Raises:
            ValueError: If the length of profiles doesn't match the model's time horizon.
        """
        # Load data from Excel file
        xls = pd.ExcelFile(filepath)
        hourly_data = pd.read_excel(xls, "Hourly data")
        halfhourly_data = pd.read_excel(xls, "Half-hourly data")

        # Adding rows for the hourly data to match the half-hourly periods
        hourly_data = pd.DataFrame(
            np.repeat(hourly_data.values, 2, axis=0),
            columns=hourly_data.columns,
        ).reset_index(drop=True)

        self.hourly_data = hourly_data
        self.halfhourly_data = halfhourly_data

    def solve(self, solver: str = "highs", mip_gap: float = 0.09):
        """Solves the optimization problem using the specified solver.

        Args:
            solver (str, optional): Name of the solver to use. Defaults to "highs".
            mip_gap (float, optional): The relative MIP gap tolerance. Defaults to 0.09 (9%).

        Returns:
            pyomo.opt.results.SolverResults: Results from the optimization solver.
        """
        opt = SolverFactory(solver)

        # Set solver options including MIP gap
        if solver.lower() == "highs":
            opt.options["mip_rel_gap"] = mip_gap

        results = opt.solve(self.model, tee=True)
        self.model.solutions.load_from(results)
        return results

    def save_results(self, filepath: str):
        """Saves the results of the optimization to a CSV file.

        Args:
            filepath (str): Path to the output CSV file.
        """
        m = self.model
        results = {
            "Time": list(m.T_halfhour),
            "State of Charge (MWh)": [pyo.value(m.soc[t]) for t in m.T_halfhour],
            "Charge Take from Hourly Market (MWh)": [
                pyo.value(m.charge_hour[t]) for t in m.T_halfhour
            ],
            "Discharge to Hourly market (MWh)": [
                pyo.value(m.discharge_hour[t]) for t in m.T_halfhour
            ],
            "Is Charging in Hourly Market": [
                pyo.value(m.is_charging_hour[t]) for t in m.T_halfhour
            ],
            "Charge taken from Half Hourly Market (MWh)": [
                pyo.value(m.charge_halfhour[t]) for t in m.T_halfhour
            ],
            "Discharge to Half Hourly Market (MWh)": [
                pyo.value(m.discharge_halfhour[t]) for t in m.T_halfhour
            ],
            "Is Charging in Half Hourly Market": [
                pyo.value(m.is_charging_halfhour[t]) for t in m.T_halfhour
            ],
            "Current Degradation (MWh)": [
                pyo.value(m.current_degradation[t]) for t in m.T_halfhour
            ],
            "Total Cycles": [pyo.value(m.total_cycles[t]) for t in m.T_halfhour],
            "Hourly Price (£/MWh)": [
                pyo.value(m.hourly_price[t]) for t in m.T_halfhour
            ],
            "Half Hourly Price (£/MWh)": [
                pyo.value(m.halfhourly_price[t]) for t in m.T_halfhour
            ],
            "Objective Value (£)": pyo.value(m.objective),
        }
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
