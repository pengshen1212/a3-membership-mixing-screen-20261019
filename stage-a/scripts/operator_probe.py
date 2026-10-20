"""Stage A operator probe: rebuild every training-path operator with the pinned
modules, validate the exact-coefficient identities, cross-check nnz/row-sums
against `train_graph.py --build-only`, and record byte hashes for the manifest.

Glue replicates train_graph.py exactly (hg construction for exact, H@H^T with
zero-row self-loop for clique_self); no pinned file is modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
sys.path.insert(0, str(A3))
import a3_coeff_override as coeff_override  # noqa: E402
import a3_exact as a3_exact_mod  # noqa: E402

from dhg import Hypergraph  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
SHARED = ["road_feature.csv", "roadmap.rel", "edge_attr.csv",
          "w_by_hop_sparse.npz", "win_speed.csv", "roadmap.geo"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def op_hash(M: sp.csr_matrix) -> str:
    M = M.tocsr()
    M.sort_indices()
    h = hashlib.sha256()
    h.update(np.int32(M.indptr).tobytes())
    h.update(np.int32(M.indices).tobytes())
    h.update(M.data.astype(np.float32).tobytes())
    return h.hexdigest()


def build_hg(seg: pd.DataFrame, rf: pd.DataFrame, rel_path: Path, n: int) -> Hypergraph:
    """Replicates train_graph.py hg_type=4 construction (types 1..3)."""
    hg = Hypergraph(n, None)
    for _, group in seg.groupby("region_id"):
        hg.add_hyperedges(e_list=group["geo_id"].to_list(), merge_op="sum")
    hi = rf["highway_info"]
    for i in range(int(hi.max()) + 1):
        hg.add_hyperedges(np.where(hi == i)[0].tolist(), merge_op="sum")
    rel = pd.read_csv(rel_path)
    origin = rel["origin_id"].to_list()
    destination = rel["destination_id"].to_list()
    dest_set = set(destination)
    for o, d in zip(origin, destination):
        if o not in dest_set:
            hg.add_hyperedges((o, d))
    return hg


def clique_matrix(seg: pd.DataFrame, rf: pd.DataFrame, rel_path: Path, n: int) -> sp.csr_matrix:
    """Replicates train_graph.py induced_clique_self: H H^T, keep diagonal,
    zero rows get a self-loop, row-normalize (float32, same dtypes)."""
    e_lists = [g["geo_id"].to_list() for _, g in seg.groupby("region_id")]
    hi = rf["highway_info"]
    e_lists += [np.where(hi == i)[0].tolist() for i in range(int(hi.max()) + 1)]
    rel = pd.read_csv(rel_path)
    dest_ids = set(rel["destination_id"].tolist())
    e_lists += [(int(o), int(d)) for o, d in zip(rel["origin_id"], rel["destination_id"])
                if o not in dest_ids]
    rows = np.concatenate([np.repeat(i, len(e)) for i, e in enumerate(e_lists)])
    cols = np.concatenate([np.asarray(e, dtype=np.int64) for e in e_lists])
    H = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (cols, rows)),
                      shape=(n, len(e_lists))).tocsr()
    A = (H @ H.T).tocsr()
    A.eliminate_zeros()
    # EXACT copy of train_graph._row_normalize (float64 1.0/deg scaling) —
    # a float32 scale changes last-ULP values and the byte hash.
    deg = np.asarray(A.sum(1)).flatten()
    zr = np.where(deg == 0)[0]
    if len(zr) > 0:
        A = (A + sp.coo_matrix((np.ones(len(zr), dtype=np.float32), (zr, zr)), shape=A.shape)).tocsr()
        deg = np.asarray(A.sum(1)).flatten()
    return sp.diags(1.0 / deg).dot(A).tocsr()


def row_sum_err(M: sp.csr_matrix) -> float:
    # canonicalize first: float32 summation is storage-order dependent and the
    # training entry's An is sorted via .tocoo().tocsr() (verified byte-identical
    # against the in-process operator, see validation JSON note)
    M = M.tocsr()
    M.sort_indices()
    M.eliminate_zeros()
    return float(np.max(np.abs(np.asarray(M.sum(1)).ravel() - 1.0)))


def main() -> None:
    rf_path = A3 / "data" / "het_g8p0" / "road_feature.csv"
    rel_path = A3 / "data" / "het_g8p0" / "roadmap.rel"
    rf = pd.read_csv(rf_path).sort_values("road_id").reset_index(drop=True)
    n = len(rf)
    real_seg = pd.read_csv(A3 / "data" / "het_g8p0" / "seg2region.csv").sort_values("geo_id").reset_index(drop=True)
    p3_seg = pd.read_csv(A3 / "data" / "het_g8p0_perm3" / "seg2region.csv").sort_values("geo_id").reset_index(drop=True)

    P_real = coeff_override.load_partition(A3 / "data" / "het_g8p0" / "seg2region.csv", rf_path, rel_path)
    P_p3 = coeff_override.load_partition(A3 / "data" / "het_g8p0_perm3" / "seg2region.csv", rf_path, rel_path)

    rep: dict = {"status": "evaluated", "paths": {}, "identities": {}, "cross_check": {}}

    # --- training-path operators (exactly what each arm builds) ---
    print("building RR exact (dhg hg, real)...", flush=True)
    rr_exact = a3_exact_mod.exact_mean_matrix(build_hg(real_seg, rf, rel_path, n)).tocsr()
    print("building PP exact (dhg hg, perm3)...", flush=True)
    pp_exact = a3_exact_mod.exact_mean_matrix(build_hg(p3_seg, rf, rel_path, n)).tocsr()
    print("building RR clique...", flush=True)
    rr_clique = clique_matrix(real_seg, rf, rel_path, n)
    print("building PP clique...", flush=True)
    pp_clique = clique_matrix(p3_seg, rf, rel_path, n)
    print("building PR clique (coeff override)...", flush=True)
    pr_clique = coeff_override.build_operator(P_p3, P_real, "clique").tocsr()

    rep["paths"] = {
        "RR_exact": {"nnz": int(rr_exact.nnz), "row_sum_err": row_sum_err(rr_exact),
                     "min": float(rr_exact.data.min()), "max": float(rr_exact.data.max()),
                     "sha256": op_hash(rr_exact)},
        "RR_clique": {"nnz": int(rr_clique.nnz), "row_sum_err": row_sum_err(rr_clique),
                      "min": float(rr_clique.data.min()), "max": float(rr_clique.data.max()),
                      "sha256": op_hash(rr_clique)},
        "PP_exact": {"nnz": int(pp_exact.nnz), "row_sum_err": row_sum_err(pp_exact),
                     "min": float(pp_exact.data.min()), "max": float(pp_exact.data.max()),
                     "sha256": op_hash(pp_exact)},
        "PP_clique": {"nnz": int(pp_clique.nnz), "row_sum_err": row_sum_err(pp_clique),
                      "min": float(pp_clique.data.min()), "max": float(pp_clique.data.max()),
                      "sha256": op_hash(pp_clique)},
        "PR_clique": {"nnz": int(pr_clique.nnz), "row_sum_err": row_sum_err(pr_clique),
                      "min": float(pr_clique.data.min()), "max": float(pr_clique.data.max()),
                      "sha256": op_hash(pr_clique)},
    }

    # --- identities (float64 coeff_override path, must be exactly 0) ---
    pr_exact_f64 = coeff_override.build_operator(P_p3, P_real, "exact")
    pp_exact_f64 = coeff_override.build_operator(P_p3, P_p3, "exact")
    rp_exact_f64 = coeff_override.build_operator(P_real, P_p3, "exact")
    rr_exact_f64 = coeff_override.build_operator(P_real, P_real, "exact")
    rep["identities"] = {
        "PR_exact_vs_PP_exact": coeff_override.max_abs_diff(pr_exact_f64, pp_exact_f64),
        "RP_exact_vs_RR_exact": coeff_override.max_abs_diff(rp_exact_f64, rr_exact_f64),
        "PR_clique_vs_PP_clique_nonzero": coeff_override.max_abs_diff(
            coeff_override.build_operator(P_p3, P_real, "clique"),
            coeff_override.build_operator(P_p3, P_p3, "clique")),
        "PP_exact_dhg_vs_independent_ref": coeff_override.max_abs_diff(
            pp_exact.astype(np.float64), coeff_override.incidence_reference(P_p3, "exact")),
        "PP_clique_vs_independent_ref": coeff_override.max_abs_diff(
            pp_clique.astype(np.float64), coeff_override.incidence_reference(P_p3, "clique")),
        "RR_exact_dhg_vs_independent_ref": coeff_override.max_abs_diff(
            rr_exact.astype(np.float64), coeff_override.incidence_reference(P_real, "exact")),
    }

    # --- cross-check vs train_graph --build-only (the real entry glue) ---
    build_only_specs = [
        ("RR_exact", "het_g8p0", "induced_exact", None),
        ("RR_clique", "het_g8p0", "induced_clique_self", None),
        ("PP_exact", "het_g8p0_perm3", "induced_exact", None),
        ("PP_clique", "het_g8p0_perm3", "induced_clique_self", None),
        ("PR_clique", "het_g8p0_perm3", "induced_coeff_override", "data/het_g8p0/seg2region.csv"),
    ]
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", DST_DATASET="het_g8p0")
    for name, dataset, variant, coeff_seg in build_only_specs:
        cmd = [sys.executable, "train_graph.py", "--dataset", dataset, "--ver", f"probe_{name}",
               "--variant", variant, "--seed", "0", "--device", "cuda", "--build-only"]
        if coeff_seg:
            cmd += ["--coeff-seg2region", coeff_seg, "--override-weighting", "clique"]
        out = subprocess.run(cmd, cwd=A3, env=env, capture_output=True, text=True, timeout=1200)
        try:
            printed = json.loads(out.stdout[out.stdout.index("{"):])
        except Exception:
            rep["cross_check"][name] = {"status": "build_only_unparseable",
                                        "stdout_tail": out.stdout[-400:], "stderr_tail": out.stderr[-400:]}
            continue
        want = rep["paths"][name]
        rep["cross_check"][name] = {
            "nnz_match": printed["nnz"] == want["nnz"],
            "row_sum_err_match": abs(printed["row_sum_max_abs_error_to_1"] - want["row_sum_err"]) <= 1e-9,
            "build_only_nnz": printed["nnz"], "probe_nnz": want["nnz"],
        }

    rep["data_pins"] = {
        f"data/het_g8p0/{f}": sha256_file(A3 / "data" / "het_g8p0" / f) for f in SHARED
    } | {
        f"data/het_g8p0_perm3/{f}": sha256_file(A3 / "data" / "het_g8p0_perm3" / f) for f in SHARED
    } | {
        "data/het_g8p0/seg2region.csv": sha256_file(A3 / "data" / "het_g8p0" / "seg2region.csv"),
        "data/het_g8p0_perm3/seg2region.csv": sha256_file(A3 / "data" / "het_g8p0_perm3" / "seg2region.csv"),
    }

    gate = {
        "exact_identity_PR_PP_zero": rep["identities"]["PR_exact_vs_PP_exact"] == 0.0,
        "exact_identity_RP_RR_zero": rep["identities"]["RP_exact_vs_RR_exact"] == 0.0,
        "clique_content_diff_nonzero": rep["identities"]["PR_clique_vs_PP_clique_nonzero"] > 0,
        "refs_match": all(rep["identities"][k] <= 1e-5 for k in
                          ("PP_exact_dhg_vs_independent_ref", "PP_clique_vs_independent_ref",
                           "RR_exact_dhg_vs_independent_ref")),
        "row_sums": all(v["row_sum_err"] <= (1e-5 if "exact" in k else 1e-5)
                        for k, v in rep["paths"].items()),
        "cross_check_all_match": all(v.get("nnz_match") and v.get("row_sum_err_match")
                                     for v in rep["cross_check"].values()),
    }
    rep["gate"] = gate
    rep["gate_passed"] = all(gate.values())
    rep["note"] = (
        "op_hash is computed on sorted-canonical CSR; a one-time instrumented run of a "
        "diagnostic COPY of the pinned entry (pinned file untouched, copy deleted after) "
        "verified the probe's RR_clique operator is BYTE-IDENTICAL to the entry's "
        "in-process operator (sha256 cae2cec95680eb7647cb817f41e1a8e753a89778e42daa76ed4e45fdc39c196d). "
        "Row-sum errors are computed on sorted storage because float32 accumulation is "
        "storage-order dependent; the unsorted reconstruction differed by last-ULP only."
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "operator_validation_stage_a.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"gate": gate, "gate_passed": rep["gate_passed"],
                      "identities": rep["identities"]}, indent=1))
    if not rep["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
