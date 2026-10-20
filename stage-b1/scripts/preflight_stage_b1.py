"""Stage B1 preflight: verify the trigger, bind the certified RR reuse, run
perm4 operator checks. Writes outputs/stage_b1_preflight.json and exits 0 only
if every gate passes. Run AFTER Stage A's result exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

B1 = Path(__file__).resolve().parents[1]
SA = B1.parent / "a3-membership-mixing-stage-a"
A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
sys.path.insert(0, str(A3))
OUT = B1 / "outputs"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    report: dict = {"gates": {}}
    ok = True

    # 1. trigger: stage A verdict + benign nondeterminism (numerical-equivalence
    #    revision, user-approved option 2 on 2026-10-20; supersedes bitwise)
    res_path = SA / "outputs" / "stage_a_result.json"
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        verdict_pass = res.get("verdict") == "meaningful_positive"
    else:
        verdict_pass = False
    det_path = SA / "outputs" / "stage_a_determinism_assessment.json"
    if det_path.exists():
        det = json.loads(det_path.read_text(encoding="utf-8"))
        det_ok = det.get("verdict") == "benign_nondeterminism"
    else:
        det_ok = False
    report["gates"]["stage_a_meaningful_positive"] = verdict_pass
    report["gates"]["stage_a_determinism_benign"] = det_ok
    ok &= verdict_pass and det_ok

    # 2. RR certified reuse: hash-bind all 10 RR embeddings against the
    # Stage A manifest (dedupe manifest runs by (arm, seed), keep last)
    man = json.loads((SA / "outputs" / "stage_a_manifest.json").read_text(encoding="utf-8"))
    runs = {}
    for r in man.get("runs", []):
        runs[(r.get("arm"), r.get("seed"))] = r
    rr_bind = {}
    rr_ok = True
    for arm, variant in (("RR_exact", "induced_exact"), ("RR_clique", "induced_clique_self")):
        for s in range(5):
            f = SA / "outputs" / "stage_a" / arm / f"seed{s}" / f"emb_{variant}_s{s}.pt"
            rec = runs.get((arm, s))
            good = f.exists() and rec and rec.get("returncode") == 0 and \
                rec.get("embedding_sha256") == sha256_file(f)
            rr_bind[f"{arm}_s{s}"] = {"exists": f.exists(), "hash_matches_manifest": bool(good)}
            rr_ok &= good
    report["gates"]["rr_certified_reuse_bound"] = rr_ok
    report["rr_binding"] = rr_bind
    ok &= rr_ok

    # 3. perm4 instance generated and validated
    gen_path = OUT / "perm4_generation.json"
    if gen_path.exists():
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        seg = A3 / "data" / "het_g8p0_perm4" / "seg2region.csv"
        gen_ok = (gen.get("status") == "generated"
                  and all(v is True for v in gen["checks"].values() if isinstance(v, bool))
                  and seg.exists() and sha256_file(seg) == gen["sha256"])
    else:
        gen_ok = False
    report["gates"]["perm4_instance_valid"] = gen_ok
    ok &= gen_ok

    # 4. perm4 operator identities (exact must be exactly 0; clique must differ)
    if gen_ok:
        import a3_coeff_override as co
        rf = A3 / "data" / "het_g8p0" / "road_feature.csv"
        rel = A3 / "data" / "het_g8p0" / "roadmap.rel"
        P_r = co.load_partition(A3 / "data" / "het_g8p0" / "seg2region.csv", rf, rel)
        P_4 = co.load_partition(A3 / "data" / "het_g8p0_perm4" / "seg2region.csv", rf, rel)
        pr_exact = co.build_operator(P_4, P_r, "exact")
        pp_exact = co.build_operator(P_4, P_4, "exact")
        ident = co.max_abs_diff(pr_exact, pp_exact)
        pr_cl = co.build_operator(P_4, P_r, "clique")
        pp_cl = co.build_operator(P_4, P_4, "clique")
        diff_cl = co.max_abs_diff(pr_cl, pp_cl)
        report["operator_identity"] = {
            "PR_exact_vs_PP_exact": ident,
            "PR_clique_vs_PP_clique": diff_cl,
            "row_sum_err_PR_clique": float(abs(
                __import__("numpy").asarray(pr_cl.sum(1)).ravel() - 1.0).max()),
        }
        ident_ok = ident == 0.0 and diff_cl > 0
        report["gates"]["perm4_operator_identity"] = ident_ok
        ok &= ident_ok

    # 5. shared inputs of perm4 dir are byte-identical to het_g8p0
    shared_ok = True
    for f in ("road_feature.csv", "roadmap.rel", "edge_attr.csv",
              "w_by_hop_sparse.npz", "win_speed.csv"):
        a, b = A3 / "data" / "het_g8p0" / f, A3 / "data" / "het_g8p0_perm4" / f
        same = a.exists() and b.exists() and sha256_file(a) == sha256_file(b)
        if not same:
            shared_ok = False
    report["gates"]["perm4_shared_inputs_identical"] = shared_ok
    ok &= shared_ok

    # 6. init reference from Stage A applies (construction is instance-independent)
    init_path = SA / "outputs" / "init_reference.json"
    init_ok = init_path.exists() and json.loads(
        init_path.read_text(encoding="utf-8")).get("all_deterministic") is True
    report["gates"]["init_reference_available"] = init_ok
    ok &= init_ok

    report["status"] = "pass" if ok else "fail"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_b1_preflight.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=1))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
