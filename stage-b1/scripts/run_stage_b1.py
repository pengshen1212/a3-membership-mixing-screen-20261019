"""Stage B1 runner: 15 trainings (PP exact/clique, PR clique) + 1 bitwise
reproduction check (PP_clique s0). RR is certified-reused from Stage A
(hash-bound; verified by preflight_stage_b1.py). Gates: conditional freeze
authorized, preflight pass, code/data pins, no existing outputs, exclusive
ledger, 5h wall-clock cap, no retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

B1 = Path(__file__).resolve().parents[1]
SA = B1.parent / "a3-membership-mixing-stage-a"
sys.path.insert(0, str(B1.parent / "a3-membership-mixing" / "src"))
from screen_rules import budget_allows  # noqa: E402

A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
CFG_PATH = B1 / "configs" / "stage_b1_freeze.json"
LEDGER = B1 / "outputs" / "stage_b1_runs.json"
MANIFEST = B1 / "outputs" / "stage_b1_manifest.json"
LOGDIR = B1 / "outputs" / "stage_b1_logs"

CODE_FILES = ["train_graph.py", "a3_coeff_override.py", "a3_output_paths.py", "model.py",
              "config.py", "preprocess.py", "utils.py", "a3_exact.py", "dataset.py"]
RUN_INPUTS = {
    "PP": ["data/het_g8p0_perm4/seg2region.csv", "data/het_g8p0_perm4/road_feature.csv",
           "data/het_g8p0_perm4/roadmap.rel", "data/het_g8p0_perm4/edge_attr.csv",
           "data/het_g8p0_perm4/w_by_hop_sparse.npz", "data/het_g8p0_perm4/roadmap.geo"],
    "PR": ["data/het_g8p0_perm4/seg2region.csv", "data/het_g8p0_perm4/road_feature.csv",
           "data/het_g8p0_perm4/roadmap.rel", "data/het_g8p0_perm4/edge_attr.csv",
           "data/het_g8p0_perm4/w_by_hop_sparse.npz", "data/het_g8p0_perm4/roadmap.geo",
           "data/het_g8p0/seg2region.csv"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arms() -> list[dict]:
    runs = []
    for seed in range(5):
        runs.append({"arm": "PP_exact", "dataset": "het_g8p0_perm4", "variant": "induced_exact",
                     "coeff_seg": None, "seed": seed})
    for seed in range(5):
        runs.append({"arm": "PP_clique", "dataset": "het_g8p0_perm4", "variant": "induced_clique_self",
                     "coeff_seg": None, "seed": seed})
    for seed in range(5):
        runs.append({"arm": "PR_clique", "dataset": "het_g8p0_perm4",
                     "variant": "induced_coeff_override",
                     "coeff_seg": "data/het_g8p0/seg2region.csv", "seed": seed})
    return runs


def command(r: dict, out_dir: Path, python_exe: str) -> list[str]:
    cmd = [python_exe, "train_graph.py", "--dataset", r["dataset"],
           "--ver", f"stage_b1_{r['arm']}_s{r['seed']}", "--variant", r["variant"],
           "--seed", str(r["seed"]), "--device", "cuda", "--output-dir", str(out_dir)]
    if r["coeff_seg"]:
        cmd += ["--coeff-seg2region", r["coeff_seg"], "--override-weighting", "clique"]
    return cmd


def environment() -> dict:
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.platform()}


def preflight(cfg: dict, dry_run: bool) -> list[str]:
    problems = []
    if not cfg.get("authorized"):
        problems.append("config authorized=false")
    pre = B1 / "outputs" / "stage_b1_preflight.json"
    if not pre.exists() or json.loads(pre.read_text(encoding="utf-8")).get("status") != "pass":
        problems.append("stage_b1 preflight not passed (trigger / RR binding / perm4 gates)")
    for rel, want in cfg.get("code_pins", {}).items():
        p = A3 / rel
        if not p.exists() or sha256(p) != want:
            problems.append(f"code pin mismatch: {rel}")
    for rel, want in cfg.get("data_pins", {}).items():
        p = A3 / rel
        if not p.exists() or sha256(p) != want:
            problems.append(f"data pin mismatch: {rel}")
    seg4 = A3 / "data" / "het_g8p0_perm4" / "seg2region.csv"
    gen = B1 / "outputs" / "perm4_generation.json"
    if not (seg4.exists() and gen.exists()
            and sha256(seg4) == json.loads(gen.read_text(encoding="utf-8"))["sha256"]):
        problems.append("perm4 seg2region not bound to its generation report")
    if not dry_run:
        existing = [str(p) for p in (B1 / "outputs" / "stage_b1").rglob("*") if p.is_file()]
        if existing:
            problems.append(f"stage_b1 outputs already exist: {existing[:3]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=os.environ.get("A3_PYTHON"))
    args = ap.parse_args()
    python_exe = args.python or sys.executable

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    problems = preflight(cfg, args.dry_run)
    specs = arms()
    all_cmds = []
    for r in specs:
        out_dir = B1 / "outputs" / "stage_b1" / r["arm"] / f"seed{r['seed']}"
        all_cmds.append({"spec": r, "out_dir": out_dir, "cmd": command(r, out_dir, python_exe)})
    repro_dir = B1 / "outputs" / "stage_b1" / "repro_check_PP_clique_s0"
    repro_cmd = command({"arm": "PP_clique", "dataset": "het_g8p0_perm4",
                         "variant": "induced_clique_self", "coeff_seg": None, "seed": 0},
                        repro_dir, python_exe)

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "n_runs": len(all_cmds),
                          "cap_hours": cfg["cost"]["cap_gpu_hours"],
                          "repro_check": " ".join(repro_cmd),
                          "refused": problems,
                          "commands": [" ".join(c["cmd"]) for c in all_cmds]}, indent=1))
        return
    if problems:
        print(json.dumps({"status": "refused", "problems": problems}, indent=1))
        raise SystemExit(3)

    pre = json.loads((B1 / "outputs" / "stage_b1_preflight.json").read_text(encoding="utf-8"))
    code_hashes = {rel: sha256(A3 / rel) for rel in CODE_FILES}
    init_ref = json.loads((SA / "outputs" / "init_reference.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_sha256": sha256(CFG_PATH),
        "freeze_doc_sha256": sha256(B1 / "FREEZE_STAGE_B1.md"),
        "preflight_sha256": sha256(B1 / "outputs" / "stage_b1_preflight.json"),
        "perm4_generation_sha256": sha256(B1 / "outputs" / "perm4_generation.json"),
        "environment": environment(), "code_hashes": code_hashes,
        "rr_certified_reuse": pre["rr_binding"],
        "rr_source": {"experiment": "stage_a", "manifest": str(SA / "outputs" / "stage_a_manifest.json")},
        "init_reference": init_ref["seeds"],
        "runs": [], "reproducibility_check": None,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LEDGER.open("x", encoding="utf-8") as f:
            json.dump({"started_at": manifest["started_at"], "cap_hours": cfg["cost"]["cap_gpu_hours"],
                       "runs": []}, f, indent=1)
    except FileExistsError:
        print(json.dumps({"status": "refused_existing_ledger", "ledger": str(LEDGER)}))
        raise SystemExit(3)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    def save_all(rec=None):
        # upsert by (arm, seed): one entry per run across the two writes
        if rec is not None:
            key = (rec["arm"], rec["seed"])
            manifest["runs"] = [r for r in manifest["runs"]
                                if (r.get("arm"), r.get("seed")) != key] + [rec]
        MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        led = json.loads(LEDGER.read_text(encoding="utf-8"))
        if rec is not None:
            led["runs"] = [r for r in led["runs"]
                           if (r.get("arm"), r.get("seed")) != (rec["arm"], rec["seed"])] + [
                {k: rec.get(k) for k in ("arm", "seed", "status", "returncode", "hours",
                                         "cumulative_hours", "embedding_sha256")}]
        led["stopped"] = manifest.get("stopped")
        LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")

    elapsed = 0.0
    for item in all_cmds:
        r, out_dir, cmd = item["spec"], item["out_dir"], item["cmd"]
        if not budget_allows(elapsed, cfg["cost"]["estimated_hours_per_run"],
                             cfg["cost"]["cap_gpu_hours"]):
            manifest["stopped"] = "budget_would_exceed_cap"
            break
        emb_path = out_dir / f"emb_{r['variant']}_s{r['seed']}.pt"
        rec = {"arm": r["arm"], "seed": r["seed"], "dataset": r["dataset"], "variant": r["variant"],
               "cmd": " ".join(cmd), "status": "running",
               "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "code_hashes": code_hashes,
               "input_data_hashes": {rel: sha256(A3 / rel)
                                     for rel in RUN_INPUTS[r["arm"].split("_")[0]]},
               "init_reference_sha256": init_ref["seeds"][str(r["seed"])]["state_dict_sha256"]}
        save_all(rec)
        log_path = LOGDIR / f"{r['arm']}_s{r['seed']}.log"
        t0 = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, cwd=A3,
                                      env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
                                               DST_DATASET="het_g8p0"),
                                      stdout=logf, stderr=subprocess.STDOUT,
                                      timeout=max(1.0, (cfg["cost"]["cap_gpu_hours"] - elapsed) * 3600.0))
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = -999
            manifest["stopped"] = "cap_exceeded_mid_run"
        except OSError as exc:
            returncode = -998
            rec["error"] = str(exc)
            manifest["stopped"] = "launch_failed"
        hours = (time.perf_counter() - t0) / 3600.0
        elapsed += hours
        emb_ok = emb_path.exists()
        rec.update({"status": "finished", "returncode": returncode,
                    "hours": round(hours, 4), "cumulative_hours": round(elapsed, 4),
                    "embedding_exists": emb_ok,
                    "embedding_sha256": sha256(emb_path) if emb_ok else None,
                    "graph_model_sha256": sha256(out_dir / "graph_model.pt")
                    if (out_dir / "graph_model.pt").exists() else None,
                    "log_sha256": sha256(log_path)})
        if returncode == 0 and not emb_ok:
            manifest["stopped"] = "output_missing"
        save_all(rec)
        print(json.dumps({k: rec.get(k) for k in ("arm", "seed", "returncode", "hours",
                                                  "cumulative_hours")}), flush=True)
        if manifest.get("stopped") or returncode != 0:
            manifest.setdefault("stopped", "run_failed" if returncode not in (0, -999)
                                else "cap_exceeded_mid_run")
            break

    if not manifest.get("stopped"):
        t0 = time.perf_counter()
        log_path = LOGDIR / "repro_check_PP_clique_s0.log"
        try:
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(repro_cmd, cwd=A3,
                                      env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
                                               DST_DATASET="het_g8p0"),
                                      stdout=logf, stderr=subprocess.STDOUT, timeout=3600.0)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = -999
        except OSError:
            returncode = -998
        hours = (time.perf_counter() - t0) / 3600.0
        elapsed += hours
        orig = all_cmds[5]  # PP_clique seed0
        orig_emb = orig["out_dir"] / "emb_induced_clique_self_s0.pt"
        repro_emb = repro_dir / "emb_induced_clique_self_s0.pt"
        same = (orig_emb.exists() and repro_emb.exists()
                and sha256(orig_emb) == sha256(repro_emb))
        num_stats = None
        if orig_emb.exists() and repro_emb.exists():
            import numpy as _np
            import torch as _torch
            _a = _torch.load(orig_emb, map_location="cpu", weights_only=False)["hg"].numpy()
            _b = _torch.load(repro_emb, map_location="cpu", weights_only=False)["hg"].numpy()
            _d = _b - _a
            num_stats = {
                "max_abs_diff": float(_np.abs(_d).max()),
                "rms_diff": float(_np.sqrt((_d ** 2).mean())),
                "relative_frobenius_error": float(_np.linalg.norm(_d) / _np.linalg.norm(_a)),
            }
        manifest["reproducibility_check"] = {
            "cmd": " ".join(repro_cmd), "returncode": returncode, "hours": round(hours, 4),
            "cumulative_hours": round(elapsed, 4),
            "original_sha256": sha256(orig_emb) if orig_emb.exists() else None,
            "repro_sha256": sha256(repro_emb) if repro_emb.exists() else None,
            "bitwise_identical": bool(same),
            "embedding_stats": num_stats,
            "criterion": "numerical equivalence: MAE drift <= 0.005 m/s, evaluated by "
                         "eval_stage_b1.py (revision 2026-10-20, user-approved option 2); "
                         "bitwise is reported only, not gating",
        }
    if not manifest.get("stopped"):
        manifest["stopped"] = None
    manifest["status"] = "completed" if not manifest.get("stopped") else f"stopped_{manifest['stopped']}"
    manifest["total_hours"] = round(elapsed, 4)
    save_all()
    print(json.dumps({"status": manifest["status"], "total_hours": manifest["total_hours"],
                      "reproducibility": manifest["reproducibility_check"]}, indent=1))


if __name__ == "__main__":
    main()
