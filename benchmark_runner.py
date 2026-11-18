"""
Benchmark Runner for Order Encoding SAT Solvers
Runs MultipleSAT and IncrementalSAT across small instances,
tracks performance, and outputs CSV reports according to requirements.md.
"""

import os
import glob
import time
from utils import load_lower_bounds, write_result_csv
from solver.SAT.multipleSAT import solve_file as solve_multiple
from solver.SAT.incrementalSAT import solve_file as solve_incremental

def run_benchmarks(instances, timeout=60.0):
    lb_map = load_lower_bounds("input/lb.csv")
    
    multiple_results = []
    incremental_results = []
    
    print(f"Running benchmarks on {len(instances)} instances (timeout={timeout}s)...")
    print("-" * 80)
    print(f"{'Instance':<12} | {'n':<4} | {'m':<5} | {'LB':<4} | {'Mult BW':<8} {'Time(s)':<8} | {'Incr BW':<8} {'Time(s)':<8}")
    print("-" * 80)
    
    for f in instances:
        inst_name = os.path.basename(f).replace(".mtx.rnd", "").replace(".mtx", "")
        
        # 1. Run MultipleSAT
        res_m = solve_multiple(f, lb_map, timeout=timeout)
        multiple_results.append(res_m)
        t_m = res_m["encoding_time"] + res_m["solve_stime"]
        
        # 2. Run IncrementalSAT
        res_i = solve_incremental(f, lb_map, timeout=timeout)
        incremental_results.append(res_i)
        t_i = res_i["encoding_time"] + res_i["solve_stime"]
        
        print(f"{inst_name:<12} | {res_m['n']:<4} | {res_m['edges']:<5} | {res_m['lb']:<4} | "
              f"{res_m['bandwidth']:<8} {t_m:<8.2f} | {res_i['bandwidth']:<8} {t_i:<8.2f}")

    # Write output CSVs
    write_result_csv("results_multiple_sat.csv", multiple_results)
    write_result_csv("results_incremental_sat.csv", incremental_results)
    print("-" * 80)
    print("Results saved to results_multiple_sat.csv and results_incremental_sat.csv")
    return multiple_results, incremental_results

if __name__ == "__main__":
    # Select small instances sorted by vertex count n
    import re
    all_small = glob.glob("input/small_instances/*.mtx.rnd")
    
    # Sort by n
    def get_n(f):
        with open(f, 'r') as fp:
            for line in fp:
                p = line.split()
                if len(p) == 3 and p[0].isdigit():
                    return int(p[0])
        return 9999
        
    all_small.sort(key=get_n)
    
    # Run first 10 small instances
    run_benchmarks(all_small[:10], timeout=60.0)
