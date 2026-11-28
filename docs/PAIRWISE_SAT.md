# Pairwise SAT Encoding & Solvers — Documentation

Tài liệu mô tả chi tiết thiết kế, mô hình hóa SAT, các kỹ thuật tối ưu hóa, cơ chế kiểm soát thời gian/bộ nhớ và hướng dẫn sử dụng bộ giải **Pairwise Encoding** cho bài toán **Bandwidth Minimization Problem (BMP)** trên nhánh `pairwise`.

---

## 1. Cấu trúc mã nguồn

Bộ giải Pairwise SAT được tách biệt hoàn toàn với Order Encoding theo kiến trúc chuẩn của dự án:

| Tệp | Vai trò |
|---|---|
| `encoder/pairwise.py` | Lớp `PairwiseEncoder` sinh công thức CNF theo mã hóa trực tiếp (Direct / Pairwise Encoding), tích hợp cắt tỉa miền, phá đối xứng, mệnh đề hỗ trợ và sinh mệnh đề delta cho Incremental SAT. |
| `solver/SAT/pairwiseIncrementalSAT.py` | Bộ giải **Pairwise IncrementalSAT** dùng tìm kiếm nhị phân với biến giả định (*assumption-guarded delta binary search*) và hỗ trợ chế độ tuyến tính (`search_strategy="linear"`). |
| `solver/SAT/pairwiseMultipleSAT.py` | Bộ giải **Pairwise MultipleSAT** dùng tìm kiếm nhị phân trên $[LB, UB - 1]$, khởi tạo SAT solver độc lập với miền cắt tỉa chặt cho từng giá trị $k$. |
| `solver/SAT/incrementalSAT.py` | Router thống nhất điều phối `encoding="order"` và `encoding="pairwise"`. |
| `solver/SAT/multipleSAT.py` | Router thống nhất điều phối `encoding="order"` và `encoding="pairwise"`. |
| `Pairwise run/benchmark_runner.py` | Trình chạy benchmark cô lập tiến trình con (subprocess worker + checkpointing + song song `--workers`), xuất CSV theo chuẩn `output/<category>/Pairwise_*.csv`. |
| `CP, ILP run/main.py` & `CP, ILP run/benchmark_all.py` | Tích hợp sẵn 2 khóa solver `pairwise_incremental_sat` và `pairwise_multiple_sat`. |
| `utils/hb_parser.py` | Bổ sung fallback đọc file `.mtx.rnd` / `.mtx` thuần Python khi môi trường không cài `scipy`. |

---

## 2. Mô hình hóa SAT (Pairwise / Direct Encoding)

Cho đồ thị vô hướng đơn $G = (V, E)$ với $|V| = n$ và ngưỡng băng thông mục tiêu $k \in \{1, \dots, n - 1\}$.

### 2.1. Biến mệnh đề
Sử dụng $n^2$ biến nhị phân trực tiếp:
$$X_{v, a} \in \{\text{True}, \text{False}\}, \quad v \in \{1, \dots, n\},\; a \in \{1, \dots, n\}$$
với ý nghĩa $X_{v, a} = \text{True} \iff f(v) = a$ (đỉnh $v$ được gán vị trí $a$). Chỉ số biến DIMACS là $\text{id}(v, a) = (v - 1)\cdot n + a$.

### 2.2. Ràng buộc song ánh (Bijection Constraints)
1. **Mỗi đỉnh có ít nhất một vị trí (ALO theo đỉnh):**
   $$\bigvee_{a=1}^{n} X_{v, a} \quad (\forall v \in V)$$
2. **Mỗi vị trí có ít nhất một đỉnh (ALO theo vị trí):**
   $$\bigvee_{v=1}^{n} X_{v, a} \quad (\forall a \in \{1, \dots, n\})$$
3. **Mỗi đỉnh có nhiều nhất một vị trí (Pairwise AMO theo đỉnh):**
   $$\neg X_{v, a} \lor \neg X_{v, b} \quad (\forall v \in V,\; 1 \le a < b \le n)$$
4. **Mỗi vị trí chứa nhiều nhất một đỉnh (Pairwise AMO theo vị trí):**
   $$\neg X_{u, a} \lor \neg X_{v, a} \quad (\forall a \in \{1, \dots, n\},\; 1 \le u < v \le n)$$

### 2.3. Ràng buộc băng thông $|f(u) - f(v)| \le k$
- **Mệnh đề xung đột trực tiếp (Direct Conflict Clauses):** Với mọi cạnh $(u, v) \in E$ và cặp vị trí $(a, b)$ thỏa $|a - b| > k$:
  $$\neg X_{u, a} \lor \neg X_{v, b}$$
- **Độ phức tạp:**
  - Số biến cơ sở: $O(n^2)$
  - Số mệnh đề cơ sở (ALO + Pairwise AMO): $2n + 2n \binom{n}{2} = \Theta(n^3)$
  - Số mệnh đề cạnh cho một giá trị $k$: $O(|E| \cdot (n - k)^2)$

---

## 3. Các kỹ thuật tối ưu hóa

1. **Chặn trên chặt bằng Multi-start RCM (`get_rcm_upper_bound`)**
   - Chạy đa khởi điểm Reverse Cuthill-McKee thuần Python (ưu tiên các đỉnh bậc nhỏ và đỉnh biên khoảng cách BFS lớn) kết hợp `scipy.sparse.csgraph.reverse_cuthill_mckee` nếu có.
   - Thu hẹp đáng kể khoảng tìm kiếm $[LB, UB - 1]$; nếu $UB \le LB$ thì kết luận tối ưu ngay với `solve_time = 0.0`.

2. **Phá đối xứng (`use_symmetry_breaking=True`)**
   - Do phép đảo ngược thứ tự $a \mapsto n + 1 - a$ bảo toàn băng thông, đỉnh neo $v^*$ có bậc lớn nhất (`anchor_vertex`) được cố định vào nửa đầu:
     $$f(v^*) \le \left\lfloor \frac{n + 1}{2} \right\rfloor \implies \neg X_{v^*, a} \quad \forall a > \left\lfloor \frac{n + 1}{2} \right\rfloor$$

3. **Cắt tỉa miền đa bước & Lan truyền nhất quán cung (`compute_domain_bounds(k)`)**
   - Tiền xử lý BFS từ mọi đỉnh $u$ để tính khoảng cách `dist[u][w]` và kích thước bóng lân cận $h$-bước $B_h(u) = |\{w : 1 \le d_G(u, w) \le h\}|$.
   - **Cận bóng lân cận:** Với ngưỡng $k$, toàn bộ $B_h(u)$ phải nằm trong đoạn $[f(u) - hk, f(u) + hk]$. Nếu $B_h(u) > hk$ thì thu hẹp ngay:
     $$f(u) \ge B_h(u) - hk + 1, \quad f(u) \le n - B_h(u) + hk$$
   - **Cận theo đỉnh neo:** Với mọi đỉnh $u$ cách đỉnh neo $v^*$ khoảng $d = d_G(v^*, u)$, ta có $f(u) \le \max\_p[v^*] + d \cdot k$.
   - **Lan truyền cố định điểm (Arc-Consistency Propagation):** Lặp trên mọi cạnh $(u, v) \in E$ để siết chặt $min\_p[u] \ge min\_p[v] - k$ và $max\_p[u] \le max\_p[v] + k$ đến hội tụ. Nếu tồn tại $u$ sao cho $min\_p[u] > max\_p[u]$, công thức lập tức vô nghiệm (UNSAT) mà không cần gọi SAT solver.

4. **Mệnh đề hỗ trợ (`use_support=True`)**
   - Với mọi cạnh $(u, v) \in E$ và vị trí hợp lệ $a \in [min\_p[u], max\_p[u]]$, thêm mệnh đề:
     $$\neg X_{u, a} \lor \bigvee_{b = \max(min\_p[v], a - k)}^{\min(max\_p[v], a + k)} X_{v, b}$$
   - Tăng cường đáng kể khả năng lan truyền đơn vị (Unit Propagation) trong bộ giải CDCL (Glucose4).

5. **Tìm kiếm nhị phân gia tăng có biến giả định (`get_bandwidth_clauses_delta`)**
   - Trong `pairwiseIncrementalSAT.py`, bộ giải khởi tạo `Solver` một lần duy nhất với các mệnh đề cơ sở (bijection + symmetry breaking).
   - Ở mỗi bước nhị phân thử $k = \text{mid}$, tạo một biến giả định mới `act_var`, chỉ sinh các mệnh đề chênh lệch giữa ngưỡng đã khóa gần nhất `locked_k` và `mid` được bảo vệ bởi literal `-act_var`, rồi gọi `solve_limited(assumptions=[act_var])`:
     - Nếu **SAT**: khóa vĩnh viễn `[act_var]`, cập nhật `locked_k = mid` và thu hẹp `high = min(mid - 1, actual_bw - 1)`.
     - Nếu **UNSAT**: vô hiệu hóa vĩnh viễn `[-act_var]` (giữ nguyên các mệnh đề đã học hợp lệ) và nâng `low = mid + 1`.

---

## 4. Kiểm soát thời gian & An toàn thực thi (`requirements.md`)

- **Đo thời gian chính xác bằng `time.perf_counter()`:**
  - `encoding_time`: gồm thời gian chạy RCM upper bound + khởi tạo `PairwiseEncoder` (BFS) + sinh CNF + nạp clauses vào PySAT (`Solver(..., bootstrap_with=...)` và `append_formula(...)`).
  - `solve_time`: chỉ tính thời gian chạy `solver.solve_limited(...)`.
- **Ngắt đúng giới hạn thời gian (`time_limit = 120s` mặc định):**
  - Kiểm tra `deadline` liên tục trong quá trình BFS, sinh mệnh đề (ném `EncodingTimeoutError`), trước/sau khi nạp vào solver và dùng `threading.Timer(..., solver.interrupt)` khi giải.
  - Bắt `MemoryError` và trả về nghiệm tốt nhất đã tìm được (tối thiểu là nghiệm RCM hợp lệ).
- **Cô lập tiến trình con trong `Pairwise run/benchmark_runner.py`:**
  - Mỗi cặp `(instance, solver)` chạy trong một tiến trình Python riêng có `hard_timeout = time_limit + 30s` và lưu checkpoint liên tục qua `progress_callback`.
  - Đảm bảo khi gặp đồ thị lớn ($n \ge 300$, nơi Pairwise AMO $\Theta(n^3)$ có thể tốn nhiều bộ nhớ/thời gian), tiến trình con bị ngắt sạch sẽ và không làm gián đoạn toàn bộ phiên benchmark.

---

## 5. Hướng dẫn sử dụng

Chạy tất cả các lệnh từ thư mục gốc của dự án:

### 5.1. Chạy trên một instance đơn lẻ (`CP, ILP run/main.py`)
```bash
# Chạy cả 2 bộ giải Pairwise SAT với giới hạn 120s (không ghi vào output CSV)
python "CP, ILP run/main.py" --file input/small_instances/bcspwr01.mtx.rnd --solvers pairwise_incremental_sat pairwise_multiple_sat --time-limit 120 --no-export

# So sánh trực tiếp Pairwise SAT với Order SAT và CP-SAT
python "CP, ILP run/main.py" --file input/small_instances/ibm32.mtx.rnd --solvers pairwise_incremental_sat pairwise_multiple_sat order_incremental_sat ortools --no-export
```

### 5.2. Chạy qua SAT router (`solver/SAT/`)
```python
from solver.SAT import incrementalSAT, multipleSAT
from utils.hb_parser import load_hb_file

n, edges = load_hb_file("input/small_instances/bcspwr01.mtx.rnd")
res_inc = incrementalSAT.solve(n, edges, time_limit=120, lower_bound=5, encoding="pairwise")
res_mul = multipleSAT.solve(n, edges, time_limit=120, lower_bound=5, encoding="pairwise")
```

### 5.3. Chạy Batch Benchmark bằng `Pairwise run/benchmark_runner.py`
```bash
# Chạy toàn bộ small_instances (33 đồ thị) với 4 workers song song, time limit 120s
python "Pairwise run/benchmark_runner.py" --category small_instances --time-limit 120 --workers 4

# Chạy toàn bộ large_instances (40 đồ thị) với 2 workers song song, time limit 120s
python "Pairwise run/benchmark_runner.py" --category large_instances --time-limit 120 --workers 2

# Chạy cả 73 đồ thị (small + large) và xuất thẳng vào output/
python "Pairwise run/benchmark_runner.py" --category all --time-limit 120 --workers 4

# Chạy thử trên danh sách instance cụ thể
python "Pairwise run/benchmark_runner.py" --category small_instances --instances bcspwr01 bcspwr02 ibm32 --time-limit 30
```

Kết quả xuất ra 4 file CSV theo đúng chuẩn cột `instance,n,edges,lb,solver,bandwidth,encoding_time,solve_time`:
- `output/small_instances/Pairwise_IncrementalSAT.csv`
- `output/small_instances/Pairwise_MultipleSAT.csv`
- `output/large_instances/Pairwise_IncrementalSAT.csv`
- `output/large_instances/Pairwise_MultipleSAT.csv`

---

## 6. Kết quả kiểm thử xác minh (Smoke Test)

Kiểm thử xác minh độ chính xác nghiệm (`verify_solution`) và cơ chế ngắt timeout trên máy cục bộ (Python 3.13 + Glucose4):

| Instance | $n$ | $|E|$ | LB | Solver | Bandwidth | Enc. Time (s) | Solve Time (s) | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `bcspwr01` | 39 | 46 | 5 | Pairwise IncrementalSAT | **5** | 0.0525 | 0.0145 | Optimal |
| `bcspwr01` | 39 | 46 | 5 | Pairwise MultipleSAT | **5** | 0.0403 | 0.0086 | Optimal |
| `bcspwr02` | 49 | 59 | 7 | Pairwise IncrementalSAT | **7** | 0.1636 | 0.2632 | Optimal |
| `bcspwr02` | 49 | 59 | 7 | Pairwise MultipleSAT | **7** | 0.2223 | 0.0138 | Optimal |
| `ibm32` | 32 | 90 | 11 | Pairwise IncrementalSAT | **11** | 0.0545 | 0.0431 | Optimal |
| `ibm32` | 32 | 90 | 11 | Pairwise MultipleSAT | **11** | 0.0704 | 0.1229 | Optimal |
| `ash292` *(timeout=2s)* | 292 | 958 | 16 | Pairwise IncrementalSAT | 26 | 2.0812 | 0.0000 | TimeLimitReached |
| `ash292` *(timeout=2s)* | 292 | 958 | 16 | Pairwise MultipleSAT | 26 | 2.0349 | 0.0000 | TimeLimitReached |

---

## 7. Hạn chế đặc thù của Pairwise Encoding & Bước tiếp theo

- **Đặc thù độ phức tạp $\Theta(n^3)$:** Do dùng mã hóa Pairwise AMO ($O(n^3)$ mệnh đề cơ sở), Pairwise Encoding rất hiệu quả về lan truyền trên các đồ thị nhỏ ($n \lesssim 150$), nhưng trên `large_instances` ($n \in [200, 1600]$), thời gian sinh mệnh đề và bộ nhớ sẽ tăng nhanh theo $n^3$ và chạm ngưỡng `120s` sớm (khi đó solver trả về nghiệm hợp lệ tốt nhất từ RCM).
- **Bước tiếp theo:** Chạy full benchmark 73 instances bằng `Pairwise run/benchmark_runner.py --category all --time-limit 120 --workers 4` để tạo đủ 4 file CSV trong `output/` rồi tạo Pull Request merge vào `main`.
