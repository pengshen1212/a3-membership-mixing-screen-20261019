# B2 code and budget notes (Beijing data)

## Required code change

`a3_coeff_override.load_partition` currently defaults `n_expected=11095` (Porto).
Beijing is 40,306. Before B2:

- pass `n_expected=40306` in the override branch of `train_graph.py`, or
- generalize `load_partition` to infer `len(seg2region)`.

Otherwise `train_graph.py --variant induced_coeff_override` stops at
`load_partition` before building the operator. The data itself was validated
with explicit `n_expected=40306`:

- clique operator shape 40,306 x 40,306, nnz 480,176,828
- row-sum max abs error 4.4e-16
- 8,758 hyperedges (8,656 regions + 14 types + 88 boundary pairs)

The operator is >20k-node scale; use the existing row-chunked path.

## Budget

- Archived Beijing 40k run: ~2.85 h per 5000-episode run on the 4090D.
- B2 25 formal + 1 repro: ~74 h.
- The B2 document's 10 h cap is Porto-scale and is infeasible for full Beijing.
  Choose before starting:
  1. full 5-seed B2, new cap ~80 h;
  2. 3-seed reduced design (15+1 runs, ~46 h; wider paired CI);
  3. 2-seed reduced design (~31 h; weaker CI);
  4. a smaller second city/subnetwork, requiring new data qualification.

## Threshold

Beijing has 32,261 labeled roads and speed 0.085-34.99 (mean 8.19).
The Porto +/-0.02/0.01 m/s thresholds must be re-reviewed and pre-registered
for Beijing; do not reuse them without justification.
