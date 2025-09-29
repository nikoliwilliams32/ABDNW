# Battery Dispatch Model

This repo is the code written by Nik Williams to prepare for the Aurora stage 2 interview.

This battery dispatch model parameterises the problem as an MIP trying to maximise profit, ensuring all key constraints such as that of the two markets, charging and discharging rates, degredation and lifespan etc. are met. The two markets are modelled as distinct markets with the hourly market also operating on a half-hour basis and the aditional constraint that the behaviour in the hourly market must be the same for the full hour.

The MIP model is structured as a model class with a selection of methods to create, solve and export results for further analysis.

I have included the data and results file in the data directory for simplicity. I know it is not best practice and can cause repo bloat but in this case it seemed the easiest option.

The MIP has been solved with a MIP gap of 9% as getting to 7.28% was already taking 1.5 hours using HIGHS.

## How to run the model

This repo uses poetry to manage the packages and dependencies.

Ensure you have a suitable python version installed and then run the command `poetry install` when cd'd into the cloned repo directory to create the environment.

To run the model you can run the example script main.py by running the command `poetry run python main.py`. This will generate a results file in the data directory. Please see the example results from my run.

## Testing
A set of unit tests have been written to check that the required properties of the market and battery behaviour are observed.

This project uses the inbuilt unittest as a framework for unit tests. To run them yourself use the command `poetry run python -m unittest tests/battery_dispatch_tests.py`.

## Simplifications

- This model doesn't discount any cashflows when making NPV calculations in the objective. This could be postprocessed later but in view of time that wasn't implemented.
- The battery degradation will not factor into the calculation for maximum cycles as this causes a solver issue (suspected non-linearity).
- The battery cannot chose to not participate in the market for a year or end operation early. The battery operates for its entire lifespan.
- The OPEX is pro-rated hourly.

## Ideas for further development

- This model solves for a given potential future with perfect hindsight. Interacting this with multiple price forecasts could lead to a more robust predicition of battery performace.
- Generalisation to incorporate other markets with different time resolutions. There are a few hard-coded numbers to ensure the half hourly conversion and making that robust would be worth-while.
- Build in automatic units using a package like `Pint`.
- Test a more efficient MIP formulation/ better solvers to deal with the time to solve.
