# Bandwidth Minimization Problem — Exact Solvers (SAT, CP & ILP)

Benchmark suite for the Bandwidth Minimization Problem using:
- **Propositional Satisfiability (SAT):** Order Encoding and Pairwise Encoding with Incremental SAT & Multiple SAT (CaDiCaL / Glucose via PySAT).
- **Constraint Programming (CP):** CPLEX CP Optimizer, Google OR-Tools CP-SAT.
- **Integer Linear Programming (ILP):** CPLEX MIP, Gurobi MIP.

## Project Structure

- `CP, ILP run/main.py` — Run solvers on a single sparse-matrix instance.
- `CP, ILP run/benchmark_all.py` — Batch benchmark across instances in `input/` and export results to `output/`.
- `Order run/benchmark_runner.py` — Isolated subprocess batch benchmark runner for Order Encoding SAT solvers.
- `Pairwise run/benchmark_runner.py` — Isolated subprocess batch benchmark runner for Pairwise Encoding SAT solvers.
- `docs/PAIRWISE_SAT.md` — Technical documentation for Pairwise Encoding, optimizations, and benchmark usage.
- `encoder/order.py` — Dedicated Order & Hybrid Encoding CNF (`OrderEncoder`: Sequential Counter AMO, 2-hop bounds, assumption-guarded thresholds, symmetry breaking & degree pruning).
- `encoder/pairwise.py` — Dedicated Pairwise Encoding CNF (`PairwiseEncoder`: direct $X_{v,a}$ encoding, multi-hop BFS domain pruning, arc consistency, symmetry breaking, support clauses & incremental delta encoding).
- `solver/SAT/` — Strictly separated SAT solvers:
  - Order Encoding: `orderIncrementalSAT.py`, `orderMultipleSAT.py`.
  - Pairwise Encoding: `pairwiseIncrementalSAT.py`, `pairwiseMultipleSAT.py`.
  - Unified routers: `incrementalSAT.py`, `multipleSAT.py`.
- `solver/CP/` — CP solvers (`CplexCP.py`, `CPSat.py`).
- `solver/ILP/` — ILP solvers (`CplexMIP.py`, `gurobi.py`).
- `utils/` — Matrix parser (`hb_parser.py`) and lower-bound / solution verification helpers (`lb_loader.py`).
- `input/` — Instance files (`small_instances/`, `large_instances/`) and precomputed lower bounds (`lb.csv`).
- `output/` — Exported per-solver CSV results grouped by instance category.

## Running `main.py` (Single Instance)

Run from the project root using your Python environment:

```bash
# Run all default solvers on default instance (bcspwr01.mtx.rnd) with 120s limit
python "CP, ILP run/main.py"

# Run SAT solvers on a specific input file
python "CP, ILP run/main.py" --file input/small_instances/bcspwr01.mtx.rnd --solvers order_incremental_sat order_multiple_sat

# Run specific solvers (choices: cplex_cp, cplex_mip, ortools, gurobi, order_multiple_sat, order_incremental_sat, pairwise_multiple_sat, pairwise_incremental_sat)
python "CP, ILP run/main.py" --file input/small_instances/bcspwr01.mtx.rnd --solvers ortools gurobi order_incremental_sat

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