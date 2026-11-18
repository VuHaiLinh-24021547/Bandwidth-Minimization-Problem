use time.perf_counter() to track encoding and solving time.
interrupt process if either encoding time or solving time exceeds time limit
output csv files consist of instance(input file name withou .mtx.rnd), n(number of vertices), edges, lb(stored in /input/lb.csv), encoding_time, solve_stime 