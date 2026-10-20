"""Stage A evaluator (no training, no API).

Reads only Stage A outputs (all arms retrained on this host — zero reuse).
Same frozen probe as the v2 perm0 screen: Ridge(alpha=1.0) on
QuantileTransformer(normal), KFold(5, shuffle, random_state=0), out-of-fold
absolute errors, MAE per arm/seed; verdict via the pre-registered
screen_rules.classify_content_effect. Exits 2 if any embedding is missing.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "a3-membership-mixing" / "src"))
from screen_rules import classify_content_effect  # noqa: E402

A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
CFG = json.loads((ROOT / "configs" / "stage_a_freeze.json").read_text(encoding="utf-8"))
OUT = ROOT / "outputs" / "stage_a_result.json"
SEEDS = range(5)
SA = ROOT / "outputs" / "stage_a"


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


def t_ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    m, s = x.mean(), x.std(ddof=1)
    half = stats.t.ppf(0.975, len(x) - 1) * s / np.sqrt(len(x))
    return float(m), float(m - half), float(m + half)


def load_emb(rel_dir: str, variant: str, seed: int):
    f = SA / rel_dir / f"seed{seed}" / f"emb_{variant}_s{seed}.pt"
    obj = torch.load(f, map_location="cpu", weights_only=False)
    if int(obj.get("seed", -1)) != seed or obj.get("variant") != variant:
        raise RuntimeError(f"artifact metadata mismatch: {f}")
    return obj["hg"].numpy()


def main() -> None:
    needed = [("RR_exact", "RR_exact", "induced_exact"), ("RR_clique", "RR_clique", "induced_clique_self"),
              ("PP_exact", "PP_exact", "induced_exact"), ("PP_clique", "PP_clique", "induced_clique_self"),
              ("PR_clique", "PR_clique", "induced_coeff_override")]
    missing = [str(SA / d / f"seed{s}" / f"emb_{v}_s{s}.pt")
               for d, _, v in needed for s in SEEDS
               if not (SA / d / f"seed{s}" / f"emb_{v}_s{s}.pt").exists()]
    if missing:
        print(json.dumps({"status": "embeddings_missing", "n_missing": len(missing),
                          "first_missing": missing[:2]}, indent=1))
        raise SystemExit(2)

    # perm3 movable subset for the secondary continuity report
    seg = pd.read_csv(A3 / "data" / "het_g8p0_perm3" / "seg2region.csv")
    sizes = seg.groupby("region_id").size()
    cls = sizes.map(sizes.value_counts())
    movable = (seg["region_id"].map(sizes) > 1) & (seg["region_id"].map(cls) > 1)
    movable_idx = seg["geo_id"].to_numpy()[movable.to_numpy()]

    y_all = pd.read_csv(A3 / "data" / "het_g8p0" / "win_speed.csv")["speed"].to_numpy(float)
    valid = ~np.isnan(y_all)
    n = len(seg)

    emb = {(d, s): load_emb(d, v, s) for d, _, v in needed for s in SEEDS}

    out = {"status": "evaluated", "design": CFG["design"], "instance": CFG["instance"],
           "reuse_level": CFG["reuse"]["level"], "certified": CFG["certified"],
           "thresholds": CFG["thresholds"], "subsets": {}}
    for subset_name, mask in (("all_roads", np.ones(n, dtype=bool)),
                              ("perm3_movable", movable_idx)):
        m = np.zeros(n, dtype=bool)
        m[mask] = True
        per_road = {}
        for d, _, v in needed:
            per_road[d] = {}
            for s in SEEDS:
                e = np.full(n, np.nan)
                e[valid] = per_road_errors(emb[(d, s)][valid], y_all[valid])
                per_road[d][s] = e
        mae = {}
        for d, _, v in needed:
            kind = "exact" if v.endswith("exact") else "clique"
            mae.setdefault(d, {})[kind] = [float(np.nanmean(per_road[d][s][valid & m])) for s in SEEDS]
        E = {
            "RR": [mae["RR_clique"]["clique"][s] - mae["RR_exact"]["exact"][s] for s in SEEDS],
            "PP": [mae["PP_clique"]["clique"][s] - mae["PP_exact"]["exact"][s] for s in SEEDS],
            "PR": [mae["PR_clique"]["clique"][s] - mae["PP_exact"]["exact"][s] for s in SEEDS],
        }
        content = [E["RR"][s] - E["PR"][s] for s in SEEDS]
        coeff = [E["PR"][s] - E["PP"][s] for s in SEEDS]
        total = [E["RR"][s] - E["PP"][s] for s in SEEDS]
        cm, clo, chi = t_ci(content)
        km, klo, khi = t_ci(coeff)
        tm, tlo, thi = t_ci(total)
        out["subsets"][subset_name] = {
            "n_labeled": int((valid & m).sum()),
            "mae_per_arm_seed": mae,
            "contrast_E_per_arm_seed": E,
            "delta_content": {"per_seed": content, "mean": cm, "ci95": [clo, chi]},
            "delta_coefficient": {"per_seed": coeff, "mean": km, "ci95": [klo, khi]},
            "delta_total": {"per_seed": total, "mean": tm, "ci95": [tlo, thi]},
            "additivity_max_abs_error": float(np.max(np.abs(
                np.array(content) + np.array(coeff) - np.array(total)))),
        }

    primary = out["subsets"]["all_roads"]["delta_content"]
    verdict = classify_content_effect(primary["mean"], primary["ci95"][0], primary["ci95"][1],
                                      CFG["thresholds"]["meaningful_positive_m_s"],
                                      -CFG["thresholds"]["negligible_band_m_s"][0])
    out["verdict"] = verdict
    out["stop_rule"] = CFG["stop_rules"].get(verdict, "stop")
    repro = json.loads((ROOT / "outputs" / "stage_a_manifest.json").read_text(encoding="utf-8"))
    out["reproducibility_check"] = repro.get("reproducibility_check")
    # determinism handling per the user-approved option-2 revision (2026-10-20):
    # bitwise identity is RECORDED but the criterion is numerical equivalence
    # (stage_a_determinism_assessment.json); no determinism_suspect flag when benign
    det_path = ROOT / "outputs" / "stage_a_determinism_assessment.json"
    det = json.loads(det_path.read_text(encoding="utf-8")) if det_path.exists() else None
    out["determinism"] = {
        "bitwise_identical": (out["reproducibility_check"] or {}).get("bitwise_identical"),
        "assessment_verdict": det.get("verdict") if det else None,
        "mae_drift": (det or {}).get("gate", {}).get("delta_mae"),
        "criterion": "numerical equivalence: |MAE_rerun - MAE_orig| <= 0.005 m/s "
                     "(revision 2026-10-20, user-approved option 2); bitwise identity "
                     "recorded but not the criterion",
        "assessment_file": "outputs/stage_a_determinism_assessment.json",
    }
    if not (det and det.get("verdict") == "benign_nondeterminism"):
        out["determinism_flag"] = "determinism_suspect"
    out["certified_definition"] = (
        "certified here means manifest-complete + numerical-equivalence reproducibility: "
        "per-run code/data/environment/operator/init/output hashes are recorded in "
        "stage_a_manifest.json, and the rerun check satisfies the pre-registered "
        "numerical-equivalence gate; bitwise identity is NOT claimed (run-level GPU "
        "atomic-order chaos); see CERTIFICATION.md at the repository root")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "primary_delta_content": primary,
        "delta_coefficient": out["subsets"]["all_roads"]["delta_coefficient"],
        "delta_total": out["subsets"]["all_roads"]["delta_total"],
        "n_labeled_all": out["subsets"]["all_roads"]["n_labeled"],
        "secondary_movable_delta_content": out["subsets"]["perm3_movable"]["delta_content"],
        "reproducibility_bitwise_identical": (out["reproducibility_check"] or {}).get("bitwise_identical"),
    }, indent=1))


if __name__ == "__main__":
    main()
