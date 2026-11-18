# Bandwidth Minimization Problem — Exact Solvers (CP & ILP)

Benchmark suite for the Bandwidth Minimization Problem using Constraint Programming (CPLEX CP, OR-Tools CP-SAT) and Integer Linear Programming (CPLEX MIP, Gurobi MIP).

## Project Structure

- `CP, ILP run/main.py` — Run solvers on a single sparse-matrix instance.
- `CP, ILP run/benchmark_all.py` — Batch benchmark across instances in `input/` and export results to `output/`.
- `utils/` — Matrix parser (`hb_parser.py`) and lower-bound / solution verification helpers (`lb_loader.py`).
- `solver/` — Solver implementations (`CP/CplexCP.py`, `CP/CPSat.py`, `ILP/CplexMIP.py`, `ILP/gurobi.py`).
- `input/` — Instance files (`small_instances/`, `large_instances/`) and precomputed lower bounds (`lb.csv`).
- `output/` — Exported per-solver CSV results grouped by instance category.

## Running `main.py` (Single Instance)

Run from the project root using your Python environment (e.g. `.venv/bin/python` or `python`):

```bash
# Run all four solvers on the default instance (bcspwr01.mtx.rnd) with a 120s time limit
python "CP, ILP run/main.py"

# Run on a specific input file
python "CP, ILP run/main.py" --file input/small_instances/arc130.mtx.rnd --time-limit 120

# Run only specific solvers (choices: cplex_cp, cplex_mip, ortools, gurobi)
python "CP, ILP run/main.py" --file input/small_instances/arc130.mtx.rnd --solvers ortools cplex_cp

# Print results without appending to output CSV files
python "CP, ILP run/main.py" --file input/small_instances/arc130.mtx.rnd --no-export

# Use a custom output directory or lower-bounds CSV
python "CP, ILP run/main.py" --file input/small_instances/arc130.mtx.rnd --output-dir output --lb-csv input/lb.csv
```

## Running `benchmark_all.py` (Batch Benchmark)

Run from the project root:

```bash
# Run all solvers on all instances (small and large) with default 120s time limit
python "CP, ILP run/benchmark_all.py"

# Run only small_instances or large_instances
python "CP, ILP run/benchmark_all.py" --category small_instances
python "CP, ILP run/benchmark_all.py" --category large_instances

# Run specific solvers with a custom time limit (in seconds)
python "CP, ILP run/benchmark_all.py" --solvers ortools gurobi --time-limit 60

# Run only specific instances by name
python "CP, ILP run/benchmark_all.py" --instances arc130 bcspwr01

# Specify custom input/output directories and lower-bounds CSV
python "CP, ILP run/benchmark_all.py" --input-dir input --output-dir output --lb-csv input/lb.csv
```