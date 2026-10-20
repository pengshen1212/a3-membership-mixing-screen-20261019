"""Determinism assessment for Stage A (numerical-equivalence revision).

User approved option 2 on 2026-10-20 ('选2吧'): the reproduction criterion is
revised from bitwise identity to NUMERICAL EQUIVALENCE. Uses the EXISTING
pair of artifacts (original RR_clique s0 run vs the rerun from the Stage A
runner's reproduction check) — no new GPU work.

PRE-REGISTERED GATE (fixed before computing, this file):
  |MAE(rerun emb) - MAE(original emb)| <= 0.005 m/s on all 9,384 labeled roads
  with the frozen probe (Ridge + QuantileTransformer + KFold(5, rs=0)).
  0.005 = 25% of the +0.02 meaningful threshold. Embedding-level statistics
  (max-abs, RMS, relative Frobenius, row cosines) are REPORTED, not gated:
  run-level chaos with estimand-level stability is precisely what the
  revision is designed to tolerate.

Writes outputs/stage_a_determinism_assessment.json:
  verdict benign_nondeterminism  -> B1 trigger satisfied (revised criterion)
  verdict not_benign             -> B1 stays halted; report to user
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

SA = Path(__file__).resolve().parents[1]
A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
ORIG = SA / "outputs" / "stage_a" / "RR_clique" / "seed0" / "emb_induced_clique_self_s0.pt"
RERUN = SA / "outputs" / "stage_a" / "repro_check_RR_clique_s0" / "emb_induced_clique_self_s0.pt"
GATE_DELTA_MAE = 0.005


def per_road_errors(X, y):
    err = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        qt = QuantileTransformer(n_quantiles=min(1000, len(tr)),
                                 output_distribution="normal", random_state=0)
        yt = qt.fit_transform(y[tr].reshape(-1, 1)).ravel()
        pred = qt.inverse_transform(
            Ridge().fit(X[tr], yt).predict(X[te]).reshape(-1, 1)).ravel()
        err[te] = np.abs(y[te] - pred)
    return err


def main() -> None:
    for p in (ORIG, RERUN):
        if not p.exists():
            print(json.dumps({"status": "missing_artifact", "path": str(p)}))
            raise SystemExit(2)
    a = torch.load(ORIG, map_location="cpu", weights_only=False)["hg"].numpy()
    b = torch.load(RERUN, map_location="cpu", weights_only=False)["hg"].numpy()

    d = b - a
    emb_stats = {
        "max_abs_diff": float(np.abs(d).max()),
        "rms_diff": float(np.sqrt((d ** 2).mean())),
        "relative_frobenius_error": float(np.linalg.norm(d) / np.linalg.norm(a)),
        "mean_row_cosine": float(np.mean(
            (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))),
        "min_row_cosine": float(np.min(
            (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))),
    }

    y_all = pd.read_csv(A3 / "data" / "het_g8p0" / "win_speed.csv")["speed"].to_numpy(float)
    valid = ~np.isnan(y_all)
    mae_a = float(np.mean(per_road_errors(a[valid], y_all[valid])))
    mae_b = float(np.mean(per_road_errors(b[valid], y_all[valid])))
    delta = abs(mae_b - mae_a)

    res = json.loads((SA / "outputs" / "stage_a_result.json").read_text(encoding="utf-8"))
    per_seed = res["subsets"]["all_roads"]["delta_content"]["per_seed"]
    seed_std = float(np.std(per_seed, ddof=1))
    seed_range = float(max(per_seed) - min(per_seed))

    gate_passed = delta <= GATE_DELTA_MAE
    report = {
        "status": "evaluated",
        "revision": "user-approved option 2 (2026-10-20 '选2吧'): bitwise -> numerical equivalence",
        "gate": {"criterion": "|MAE_rerun - MAE_orig| <= 0.005 m/s (25% of the +0.02 threshold)",
                 "delta_mae": delta, "threshold": GATE_DELTA_MAE, "passed": gate_passed},
        "mae_original": mae_a, "mae_rerun": mae_b,
        "embedding_stats_reported_not_gated": emb_stats,
        "context": {"seed_std_delta_content": seed_std,
                    "seed_range_delta_content": seed_range,
                    "delta_over_seed_std": delta / seed_std if seed_std else None,
                    "bitwise_identical": False,
                    "interpretation_note": "estimand-level stability under run-level "
                                           "re-initialization/atomic-order chaos is the "
                                           "scientifically relevant reproducibility property"},
        "verdict": "benign_nondeterminism" if gate_passed else "not_benign",
    }
    out = SA / "outputs" / "stage_a_determinism_assessment.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("verdict", "gate", "mae_original", "mae_rerun",
                       "embedding_stats_reported_not_gated", "context")}, indent=1))
    raise SystemExit(0 if gate_passed else 2)


if __name__ == "__main__":
    main()
