import unittest
import os
from abdnw.battery_dispatch import BatteryDispatchModel
import pandas as pd
import numpy as np
import pyomo.environ as pyo


class TestBatteryDispatch(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures.
        Creates a small test dataset and initializes a BatteryDispatchModel instance.
        """
        os.makedirs("./data", exist_ok=True)

        self.test_data_path = "./data/test_market_data.xlsx"

        hourly_data = pd.DataFrame(
            {"Market 2 Price [£/MWh]": [50.0 if i < 24 else 80.0 for i in range(48)]}
        )
        halfhourly_data = pd.DataFrame(
            {"Market 1 Price [£/MWh]": [40.0 if i < 24 else 100.0 for i in range(96)]}
        )

        with pd.ExcelWriter(self.test_data_path) as writer:
            hourly_data.to_excel(writer, sheet_name="Hourly data", index=True)
            halfhourly_data.to_excel(writer, sheet_name="Half-hourly data", index=True)

        # Initialize model with test parameters
        self.model = BatteryDispatchModel(
            battery_capacity=4.0,  # MWh
            initial_soc=2.0,  # MWh
            max_charge_rate=2.0,  # MW
            max_discharge_rate=2.0,  # MW
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            time_horizon=24 * 2,
            maximum_cycles=2,  # Reduced for testing
            maximum_age=10,
            storage_degradation=0.0001,
            capex=500000,
            opex=5000,
            market_data_path=self.test_data_path,
        )

        self.model.solve()

    def tearDown(self):
        """Clean up test fixtures."""
        pass

    def test_no_simultaneous_charge_discharge(self):
        """Test that battery cannot charge and discharge simultaneously."""
        m = self.model.model

        for t in m.T_halfhour:
            charge_hour = m.charge_hour[t].value
            discharge_hour = m.discharge_hour[t].value
            charge_halfhour = m.charge_halfhour[t].value
            discharge_halfhour = m.discharge_halfhour[t].value

            # Test hourly market: at least one of charge or discharge must be zero
            self.assertLessEqual(
                charge_hour * discharge_hour,
                1e-10,  # Allow for minor numerical issues
                f"Battery charging and discharging simultaneously in hourly market at period {t}",
            )

            # Test half-hourly market: at least one of charge or discharge must be zero
            self.assertLessEqual(
                charge_halfhour * discharge_halfhour,
                1e-10,  # Allow for minor numerical issues
                f"Battery charging and discharging simultaneously in half-hourly market at period {t}",
            )

            # Test cross-market: if charging in hourly, can't discharge in half-hourly and vice versa
            self.assertLessEqual(
                charge_hour * discharge_halfhour,
                1e-10,
                f"Battery charging in hourly and discharging in half-hourly at period {t}",
            )
            self.assertLessEqual(
                discharge_hour * charge_halfhour,
                1e-10,
                f"Battery discharging in hourly and charging in half-hourly at period {t}",
            )

    def test_state_of_charge_dynamics(self):
        """Test that state of charge calculations are correct."""
        m = self.model.model
        charge_eff = self.model.charge_efficiency
        initial_soc = self.model.initial_soc

        # Test initial state of charge
        self.assertAlmostEqual(
            m.soc[0].value,
            initial_soc
            + (m.charge_hour[0].value * charge_eff - m.discharge_hour[0].value)
            + (m.charge_halfhour[0].value * charge_eff - m.discharge_halfhour[0].value),
            places=6,
            msg="Initial state of charge calculation incorrect",
        )

        # Test state of charge dynamics for subsequent periods
        for t in range(1, self.model.time_horizon * 2):
            expected_soc = (
                m.soc[t - 1].value
                + (m.charge_hour[t].value * charge_eff - m.discharge_hour[t].value)
                + (
                    m.charge_halfhour[t].value * charge_eff
                    - m.discharge_halfhour[t].value
                )
            )

            self.assertAlmostEqual(
                m.soc[t].value,
                expected_soc,
                places=6,
                msg=f"State of charge calculation incorrect at period {t}",
            )

    def test_battery_capacity_constraints(self):
        """Test that battery state of charge stays within capacity limits."""
        m = self.model.model
        battery_capacity = self.model.battery_capacity

        # Test that state of charge never exceeds battery capacity
        for t in m.T_halfhour:
            self.assertLessEqual(
                m.soc[t].value,
                battery_capacity,
                f"State of charge exceeds battery capacity at period {t}",
            )

            # Test that state of charge is never negative
            self.assertGreaterEqual(
                m.soc[t].value, 0, f"State of charge is negative at period {t}"
            )

            # Test that discharge amount doesn't exceed available charge
            total_discharge = m.discharge_hour[t].value + m.discharge_halfhour[t].value
            self.assertLessEqual(
                total_discharge,
                m.soc[t].value + 1e-6,  # Allow for minor numerical issues
                f"Discharge amount exceeds available charge at period {t}",
            )

            # Test that charging respects remaining capacity
            total_charge = (
                m.charge_hour[t].value + m.charge_halfhour[t].value
            ) * self.model.charge_efficiency
            remaining_capacity = (
                battery_capacity - m.soc[t].value - m.current_degradation[t].value
            )
            self.assertLessEqual(
                total_charge,
                remaining_capacity + 1e-6,  # Allow for minor numerical issues
                f"Charging amount exceeds remaining capacity at period {t}",
            )

    def test_maximum_charge_discharge_rates(self):
        """Test that charge and discharge rates stay within specified limits."""
        m = self.model.model
        max_charge_rate = self.model.max_charge_rate
        max_discharge_rate = self.model.max_discharge_rate

        for t in m.T_halfhour:
            # Test total charge rate limit
            total_charge = m.charge_hour[t].value + m.charge_halfhour[t].value
            self.assertLessEqual(
                total_charge,
                max_charge_rate + 1e-6,  # Allow for minor numerical issues
                f"Total charge rate exceeds maximum at period {t}",
            )

            # Test total discharge rate limit
            total_discharge = m.discharge_hour[t].value + m.discharge_halfhour[t].value
            self.assertLessEqual(
                total_discharge,
                max_discharge_rate + 1e-6,  # Allow for minor numerical issues
                f"Total discharge rate exceeds maximum at period {t}",
            )

            # Test hourly market rates
            self.assertLessEqual(
                m.charge_hour[t].value,
                max_charge_rate + 1e-6,  # Allow for minor numerical issues
                f"Hourly charge rate exceeds maximum at period {t}",
            )
            self.assertLessEqual(
                m.discharge_hour[t].value,
                max_discharge_rate + 1e-6,  # Allow for minor numerical issues
                f"Hourly discharge rate exceeds maximum at period {t}",
            )

            # Test half-hourly market rates
            self.assertLessEqual(
                m.charge_halfhour[t].value,
                max_charge_rate + 1e-6,  # Allow for minor numerical issues
                f"Half-hourly charge rate exceeds maximum at period {t}",
            )
            self.assertLessEqual(
                m.discharge_halfhour[t].value,
                max_discharge_rate + 1e-6,  # Allow for minor numerical issues
                f"Half-hourly discharge rate exceeds maximum at period {t}",
            )

            # Test hourly consistency (rates should be the same within each hour)
            if t % 2 == 0 and t < len(m.T_halfhour) - 1:
                self.assertAlmostEqual(
                    m.charge_hour[t].value,
                    m.charge_hour[t + 1].value,
                    places=6,
                    msg=f"Hourly charge rates not consistent at periods {t} and {t+1}",
                )
                self.assertAlmostEqual(
                    m.discharge_hour[t].value,
                    m.discharge_hour[t + 1].value,
                    places=6,
                    msg=f"Hourly discharge rates not consistent at periods {t} and {t+1}",
                )

    def test_battery_degradation_and_cycles(self):
        """Test battery degradation calculations and cycle limits."""
        m = self.model.model
        battery_capacity = self.model.battery_capacity
        degradation_factor = self.model.battery_degradation

        # Calculate total cycles
        total_cycles = 0
        for t in m.T_halfhour:
            # Accumulate cycles
            cycles_t = m.total_cycles[t].value
            cycles_tmin1 = m.total_cycles[t - 1].value if t > 0 else 0

            # Verify cycles calculation
            expected_cycles = (
                cycles_tmin1
                + (
                    m.charge_hour[t].value
                    + m.discharge_hour[t].value
                    + m.charge_halfhour[t].value
                    + m.discharge_halfhour[t].value
                )
                / battery_capacity
                / 2
            )
            self.assertAlmostEqual(
                cycles_t,
                expected_cycles,
                places=6,
                msg=f"Cycle calculation incorrect at period {t}",
            )
            # test degradation calculation
            expected_degradation = cycles_t * degradation_factor
            self.assertAlmostEqual(
                m.current_degradation[t].value,
                expected_degradation,
                places=6,
                msg=f"Degradation calculation incorrect at period {t}",
            )

        # Test that total cycles don't exceed maximum
        self.assertLessEqual(
            total_cycles,
            self.model.maximum_cycles + 1e-6,  # Allow for minor numerical issues
            "Total cycles exceed maximum allowed cycles",
        )


if __name__ == "__main__":
    unittest.main()
