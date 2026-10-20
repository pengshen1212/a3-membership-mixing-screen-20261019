"""Stage B1 evaluator (no training, no API).

RR read from Stage A outputs (hash re-verified against the Stage A manifest);
PP/PR from Stage B1 outputs. Same frozen probe/judgment as Stage A. Also
reports the cross-instance comparison (perm3 vs perm4). Exits 2 if any
embedding is missing.
"""
from __future__ import annotations

import hashlib
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

B1 = Path(__file__).resolve().parents[1]
SA = B1.parent / "a3-membership-mixing-stage-a"
sys.path.insert(0, str(B1.parent / "a3-membership-mixing" / "src"))
from screen_rules import classify_content_effect  # noqa: E402

A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
CFG = json.loads((B1 / "configs" / "stage_b1_freeze.json").read_text(encoding="utf-8"))
OUT = B1 / "outputs" / "stage_b1_result.json"
SEEDS = range(5)
SB = B1 / "outputs" / "stage_b1"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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


def load_local(rel_dir: str, variant: str, seed: int):
    f = SB / rel_dir / f"seed{seed}" / f"emb_{variant}_s{seed}.pt"
    obj = torch.load(f, map_location="cpu", weights_only=False)
    if int(obj.get("seed", -1)) != seed or obj.get("variant") != variant:
        raise RuntimeError(f"artifact metadata mismatch: {f}")
    return obj["hg"].numpy()


def load_rr_reused(manifest_runs: dict, variant: str, seed: int):
    arm = "RR_exact" if variant == "induced_exact" else "RR_clique"
    f = SA / "outputs" / "stage_a" / arm / f"seed{seed}" / f"emb_{variant}_s{seed}.pt"
    rec = manifest_runs.get((arm, seed))
    if not f.exists() or not rec or rec.get("embedding_sha256") != sha256_file(f):
        raise RuntimeError(f"certified RR reuse hash mismatch: {f}")
    obj = torch.load(f, map_location="cpu", weights_only=False)
    if int(obj.get("seed", -1)) != seed or obj.get("variant") != variant:
        raise RuntimeError(f"RR artifact metadata mismatch: {f}")
    return obj["hg"].numpy()


def main() -> None:
    sa_man = json.loads((SA / "outputs" / "stage_a_manifest.json").read_text(encoding="utf-8"))
    man_runs = {}
    for r in sa_man.get("runs", []):
        man_runs[(r.get("arm"), r.get("seed"))] = r

    needed = [("PP_exact", "PP_exact", "induced_exact"),
              ("PP_clique", "PP_clique", "induced_clique_self"),
              ("PR_clique", "PR_clique", "induced_coeff_override")]
    missing = [str(SB / d / f"seed{s}" / f"emb_{v}_s{s}.pt")
               for d, _, v in needed for s in SEEDS
               if not (SB / d / f"seed{s}" / f"emb_{v}_s{s}.pt").exists()]
    if missing:
        print(json.dumps({"status": "embeddings_missing", "n_missing": len(missing)}, indent=1))
        raise SystemExit(2)

    seg = pd.read_csv(A3 / "data" / "het_g8p0_perm4" / "seg2region.csv")
    sizes = seg.groupby("region_id").size()
    cls = sizes.map(sizes.value_counts())
    movable = (seg["region_id"].map(sizes) > 1) & (seg["region_id"].map(cls) > 1)
    movable_idx = seg["geo_id"].to_numpy()[movable.to_numpy()]

    y_all = pd.read_csv(A3 / "data" / "het_g8p0" / "win_speed.csv")["speed"].to_numpy(float)
    valid = ~np.isnan(y_all)
    n = len(seg)

    emb = {("RR_exact", s): load_rr_reused(man_runs, "induced_exact", s) for s in SEEDS}
    emb.update({("RR_clique", s): load_rr_reused(man_runs, "induced_clique_self", s) for s in SEEDS})
    emb.update({(d, s): load_local(d, v, s) for d, _, v in needed for s in SEEDS})

    out = {"status": "evaluated", "design": CFG["design"], "instance": CFG["instance"],
           "reuse": CFG["reuse"], "certified": CFG["certified"],
           "rr_reuse_hash_verified": True,
           "thresholds": CFG["thresholds"], "subsets": {}}
    for subset_name, mask in (("all_roads", np.ones(n, dtype=bool)),
                              ("perm4_movable", movable_idx)):
        m = np.zeros(n, dtype=bool)
        m[mask] = True
        per_road = {}
        for d in ("RR_exact", "RR_clique") + tuple(x[0] for x in needed):
            per_road[d] = {}
            for s in SEEDS:
                e = np.full(n, np.nan)
                e[valid] = per_road_errors(emb[(d, s)][valid], y_all[valid])
                per_road[d][s] = e
        mae = {}
        for d, kind in (("RR_exact", "exact"), ("RR_clique", "clique"),
                        ("PP_exact", "exact"), ("PP_clique", "clique"),
                        ("PR_clique", "clique")):
            mae[d] = [float(np.nanmean(per_road[d][s][valid & m])) for s in SEEDS]
        E = {
            "RR": [mae["RR_clique"][s] - mae["RR_exact"][s] for s in SEEDS],
            "PP": [mae["PP_clique"][s] - mae["PP_exact"][s] for s in SEEDS],
            "PR": [mae["PR_clique"][s] - mae["PP_exact"][s] for s in SEEDS],
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
    b1_man = B1 / "outputs" / "stage_b1_manifest.json"
    if b1_man.exists():
        out["reproducibility_check"] = json.loads(
            b1_man.read_text(encoding="utf-8")).get("reproducibility_check")
        # numerical-equivalence criterion (revision 2026-10-20, user-approved
        # option 2): MAE drift of the rerun vs original PP_clique s0 <= 0.005;
        # bitwise identity is reported but NOT the gating condition
        repro_path = SB / "repro_check_PP_clique_s0" / "emb_induced_clique_self_s0.pt"
        rc = out["reproducibility_check"] or {}
        if repro_path.exists() and rc.get("returncode") == 0:
            robj = torch.load(repro_path, map_location="cpu", weights_only=False)
            re_ = np.full(n, np.nan)
            re_[valid] = per_road_errors(robj["hg"].numpy()[valid], y_all[valid])
            mae_repro = float(np.nanmean(re_[valid]))
            mae_orig = out["subsets"]["all_roads"]["mae_per_arm_seed"]["PP_clique"][0]
            drift = abs(mae_repro - mae_orig)
            benign = drift <= 0.005
            out["repro_numerical_equivalence"] = {
                "mae_original": mae_orig, "mae_rerun": mae_repro,
                "abs_drift": drift, "threshold": 0.005, "benign": benign,
            }
            if not benign:
                out["determinism_flag"] = "determinism_suspect"
        else:
            out["determinism_flag"] = "determinism_suspect"

    sa_res_path = SA / "outputs" / "stage_a_result.json"
    if sa_res_path.exists():
        sa = json.loads(sa_res_path.read_text(encoding="utf-8"))
        out["cross_instance"] = {
            "perm3_stage_a": {"verdict": sa.get("verdict"),
                              "delta_content": sa["subsets"]["all_roads"]["delta_content"],
                              "delta_coefficient": sa["subsets"]["all_roads"]["delta_coefficient"],
                              "delta_total": sa["subsets"]["all_roads"]["delta_total"]},
            "perm4_stage_b1": {"verdict": verdict,
                               "delta_content": primary,
                               "delta_coefficient": out["subsets"]["all_roads"]["delta_coefficient"],
                               "delta_total": out["subsets"]["all_roads"]["delta_total"]},
            "note": "same-city second instance; city-level external validity requires B2 (new city data)",
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "primary_delta_content": primary,
        "delta_coefficient": out["subsets"]["all_roads"]["delta_coefficient"],
        "delta_total": out["subsets"]["all_roads"]["delta_total"],
        "n_labeled_all": out["subsets"]["all_roads"]["n_labeled"],
        "rr_reuse_hash_verified": True,
        "reproducibility_bitwise_identical": (out.get("reproducibility_check") or {}).get("bitwise_identical"),
        "stage_a_verdict_for_reference": (out.get("cross_instance") or {}).get("perm3_stage_a", {}).get("verdict"),
    }, indent=1))


if __name__ == "__main__":
    main()
