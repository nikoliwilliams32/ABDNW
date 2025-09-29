import abdnw.battery_dispatch as bd
import pyomo.environ as pyo

if __name__ == "__main__":
    # Uncomment lines to run the mini model
    bd_model = bd.BatteryDispatchModel(
        # time_horizon=24 * 4,
        # market_data_path="./data/assignment_data_mini.xlsx",
        market_data_path="./data/assignment_data.xlsx",
        # maximum_cycles=30,
    )
    bd_model.solve()
    # Print main objective value
    print(f"Optimal profit: £{pyo.value(bd_model.model.objective):.2f}")
    bd_model.save_results("./data/battery_dispatch_results.csv")
