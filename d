warning: in the working copy of 'encoder/order.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'solver/SAT/incrementalSAT.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'solver/SAT/multipleSAT.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/encoder/order.py b/encoder/order.py[m
[1mindex be9d6d2..61f051f 100644[m
[1m--- a/encoder/order.py[m
[1m+++ b/encoder/order.py[m
[36m@@ -14,6 +14,7 @@[m [mclass OrderEncoder:[m
         use_strengthening: bool = True,[m
         use_symmetry_breaking: bool = True,[m
         use_degree_bounds: bool = True,[m
[32m+[m[32m        use_2hop: bool = True,[m
         amo_encoding: str = "seqcounter",[m
     ):[m
         """[m
[36m@@ -24,12 +25,14 @@[m [mclass OrderEncoder:[m
         :param use_strengthening: Whether to add resolvents between exact and order variables.[m
         :param use_symmetry_breaking: Whether to add reflection symmetry breaking.[m
         :param use_degree_bounds: Whether to prune boundaries for high-degree vertices.[m
[32m+[m[32m        :param use_2hop: Whether to add 2-hop distance bound constraints.[m
         :param amo_encoding: AMO constraint encoding ('seqcounter' or 'pairwise').[m
         """[m
         self.n = n[m
         self.use_strengthening = use_strengthening[m
         self.use_symmetry_breaking = use_symmetry_breaking[m
         self.use_degree_bounds = use_degree_bounds[m
[32m+[m[32m        self.use_2hop = use_2hop[m
         self.amo_encoding = amo_encoding.lower()[m
 [m
         # Normalize 0-indexed edges to 1-indexed (1..n) internally[m
[36m@@ -55,6 +58,22 @@[m [mclass OrderEncoder:[m
 [m
         self.edges: List[Tuple[int, int]] = sorted(list(norm_edges))[m
 [m
[32m+[m[32m        # Adjacency and 2-hop neighbor pairs (d_G(u, v) == 2)[m
[32m+[m[32m        adj = [set() for _ in range(n + 1)][m
[32m+[m[32m        for u, v in self.edges:[m
[32m+[m[32m            adj[u].add(v)[m
[32m+[m[32m            adj[v].add(u)[m
[32m+[m[32m        self.adj = adj[m
[32m+[m
[32m+[m[32m        two_hop_set = set()[m
[32m+[m[32m        if self.use_2hop:[m
[32m+[m[32m            for u in range(1, n + 1):[m
[32m+[m[32m                for w in adj[u]:[m
[32m+[m[32m                    for v in adj[w]:[m
[32m+[m[32m                        if v > u and v not in adj[u]:[m
[32m+[m[32m                            two_hop_set.add((u, v))[m
[32m+[m[32m        self.two_hop_edges: List[Tuple[int, int]] = sorted(list(two_hop_set))[m
[32m+[m
         # Select anchor vertex with maximum degree for symmetry breaking[m
         self.anchor_vertex = max(range(1, n + 1), key=lambda v: (self.degrees[v], -v))[m
 [m
[36m@@ -180,6 +199,14 @@[m [mclass OrderEncoder:[m
                         clauses.append([-self.var_X(u, a), -self.var_O(v, a + k + 1)])[m
                         clauses.append([-self.var_X(v, a), -self.var_O(u, a + k + 1)])[m
 [m
[32m+[m[32m        # 2-hop distance bounds for pairs with d_G(u, v) == 2[m
[32m+[m[32m        if self.use_2hop and 2 * k < n - 1:[m
[32m+[m[32m            two_k = 2 * k[m
[32m+[m[32m            for u, v in self.two_hop_edges:[m
[32m+[m[32m                for t in range(two_k + 2, n + 1):[m
[32m+[m[32m                    clauses.append([-self.var_O(u, t), self.var_O(v, t - two_k)])[m
[32m+[m[32m                    clauses.append([-self.var_O(v, t), self.var_O(u, t - two_k)])[m
[32m+[m
         return clauses[m
 [m
     def encode(self, k: int) -> List[List[int]]:[m
[1mdiff --git a/output/small_instances/Order_IncrementalSAT.csv b/output/small_instances/Order_IncrementalSAT.csv[m
[1mindex 2f40d8a..e29400f 100644[m
[1m--- a/output/small_instances/Order_IncrementalSAT.csv[m
[1m+++ b/output/small_instances/Order_IncrementalSAT.csv[m
[36m@@ -1,11 +1,11 @@[m
 instance,n,edges,lb,solver,bandwidth,encoding_time,solve_time[m
[31m-pores_1,30,103,7,Order IncrementalSAT,7,0.0354,0.0078[m
[31m-ibm32,32,90,11,Order IncrementalSAT,11,0.0238,0.4750[m
[31m-bcspwr01,39,46,5,Order IncrementalSAT,5,0.0282,0.0280[m
[31m-bcsstk01,48,176,16,Order IncrementalSAT,16,0.2300,6.1019[m
[31m-bcspwr02,49,59,7,Order IncrementalSAT,7,0.0658,0.0991[m
[31m-curtis54,54,124,10,Order IncrementalSAT,10,0.0890,0.8590[m
[31m-will57,57,127,6,Order IncrementalSAT,6,0.2123,10.7999[m
[31m-impcol_b,59,281,19,Order IncrementalSAT,22,0.6374,15.3655[m
[31m-steam3,80,424,7,Order IncrementalSAT,7,0.0004,0.0000[m
[31m-ash85,85,219,9,Order IncrementalSAT,15,0.1743,15.3799[m
[32m+[m[32mpores_1,30,103,7,Order IncrementalSAT,7,0.0327,0.0087[m
[32m+[m[32mibm32,32,90,11,Order IncrementalSAT,11,0.0412,0.0236[m
[32m+[m[32mbcspwr01,39,46,5,Order IncrementalSAT,5,0.0401,0.0241[m
[32m+[m[32mbcsstk01,48,176,16,Order IncrementalSAT,16,0.2682,0.6454[m
[32m+[m[32mbcspwr02,49,59,7,Order IncrementalSAT,7,0.2716,0.6844[m
[32m+[m[32mcurtis54,54,124,10,Order IncrementalSAT,10,0.4272,0.9738[m
[32m+[m[32mwill57,57,127,6,Order IncrementalSAT,9,0.7802,14.3203[m
[32m+[m[32mimpcol_b,59,281,19,Order IncrementalSAT,22,0.7719,14.2094[m
[32m+[m[32msteam3,80,424,7,Order IncrementalSAT,7,0.0017,0.0000[m
[32m+[m[32mash85,85,219,9,Order IncrementalSAT,14,1.3743,13.6078[m
[1mdiff --git a/output/small_instances/Order_MultipleSAT.csv b/output/small_instances/Order_MultipleSAT.csv[m
[1mindex e3655c5..21d647e 100644[m
[1m--- a/output/small_instances/Order_MultipleSAT.csv[m
[1m+++ b/output/small_instances/Order_MultipleSAT.csv[m
[36m@@ -1,11 +1,11 @@[m
 instance,n,edges,lb,solver,bandwidth,encoding_time,solve_time[m
[31m-pores_1,30,103,7,Order MultipleSAT,7,0.0062,0.0068[m
[31m-ibm32,32,90,11,Order MultipleSAT,11,0.0426,0.2841[m
[31m-bcspwr01,39,46,5,Order MultipleSAT,5,0.0179,0.0468[m
[31m-bcsstk01,48,176,16,Order MultipleSAT,16,0.0528,2.6741[m
[31m-bcspwr02,49,59,7,Order MultipleSAT,7,0.0256,0.0287[m
[31m-curtis54,54,124,10,Order MultipleSAT,10,0.0386,0.4263[m
[31m-will57,57,127,6,Order MultipleSAT,6,0.0731,2.0853[m
[31m-impcol_b,59,281,19,Order MultipleSAT,23,0.0973,14.9351[m
[31m-steam3,80,424,7,Order MultipleSAT,7,0.0004,0.0000[m
[31m-ash85,85,219,9,Order MultipleSAT,12,0.1193,15.6058[m
[32m+[m[32mpores_1,30,103,7,Order MultipleSAT,7,0.0092,0.0100[m
[32m+[m[32mibm32,32,90,11,Order MultipleSAT,11,0.0114,0.1150[m
[32m+[m[32mbcspwr01,39,46,5,Order MultipleSAT,5,0.0132,0.0751[m
[32m+[m[32mbcsstk01,48,176,16,Order MultipleSAT,16,0.0792,3.9741[m
[32m+[m[32mbcspwr02,49,59,7,Order MultipleSAT,7,0.0958,0.3392[m
[32m+[m[32mcurtis54,54,124,10,Order MultipleSAT,10,0.1559,4.0135[m
[32m+[m[32mwill57,57,127,6,Order MultipleSAT,6,0.0971,3.4922[m
[32m+[m[32mimpcol_b,59,281,19,Order MultipleSAT,23,0.5226,21.3491[m
[32m+[m[32msteam3,80,424,7,Order MultipleSAT,7,0.0023,0.0000[m
[32m+[m[32mash85,85,219,9,Order MultipleSAT,9,0.8414,11.6253[m
[1mdiff --git a/solver/SAT/incrementalSAT.py b/solver/SAT/incrementalSAT.py[m
[1mindex 4e08911..4282a57 100644[m
[1m--- a/solver/SAT/incrementalSAT.py[m
[1m+++ b/solver/SAT/incrementalSAT.py[m
[36m@@ -47,6 +47,7 @@[m [mdef solve([m
     time_limit: int = 60,[m
     lower_bound: Optional[int] = None,[m
     solver_name: str = "glucose4",[m
[32m+[m[32m    use_2hop: bool = True,[m
 ) -> Dict:[m
     """Incremental SAT solver using Order Encoding with descending threshold search."""[m
     total_encoding_time = 0.0[m
[36m@@ -77,7 +78,13 @@[m [mdef solve([m
 [m
     # Base encoding[m
     t_enc0 = time.perf_counter()[m
[31m-    encoder = OrderEncoder(n, edges, use_strengthening=True, use_symmetry_breaking=True)[m
[32m+[m[32m    encoder = OrderEncoder([m
[32m+[m[32m        n,[m
[32m+[m[32m        edges,[m
[32m+[m[32m        use_strengthening=True,[m
[32m+[m[32m        use_symmetry_breaking=True,[m
[32m+[m[32m        use_2hop=use_2hop,[m
[32m+[m[32m    )[m
     base_clauses = encoder.get_base_clauses()[m
     total_encoding_time += (time.perf_counter() - t_enc0)[m
 [m
[1mdiff --git a/solver/SAT/multipleSAT.py b/solver/SAT/multipleSAT.py[m
[1mindex 700bbee..b82ded1 100644[m
[1m--- a/solver/SAT/multipleSAT.py[m
[1m+++ b/solver/SAT/multipleSAT.py[m
[36m@@ -47,6 +47,7 @@[m [mdef solve([m
     time_limit: int = 60,[m
     lower_bound: Optional[int] = None,[m
     solver_name: str = "glucose4",[m
[32m+[m[32m    use_2hop: bool = True,[m
 ) -> Dict:[m
     """Solve BMP using binary search over bandwidth with independent SAT instances."""[m
     total_encoding_time = 0.0[m
[36m@@ -75,7 +76,13 @@[m [mdef solve([m
             "solve_time": 0.0,[m
         }[m
 [m
[31m-    encoder = OrderEncoder(n, edges, use_strengthening=True, use_symmetry_breaking=True)[m
[32m+[m[32m    encoder = OrderEncoder([m
[32m+[m[32m        n,[m
[32m+[m[32m        edges,[m
[32m+[m[32m        use_strengthening=True,[m
[32m+[m[32m        use_symmetry_breaking=True,[m
[32m+[m[32m        use_2hop=use_2hop,[m
[32m+[m[32m    )[m
 [m
     low = lb[m
     high = best_bw - 1[m
