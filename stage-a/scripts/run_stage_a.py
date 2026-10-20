"""Certified Stage A runner: 25 trainings + 1 bitwise reproduction check.

Gate: frozen config authorized+certified, code/data pins, no existing outputs,
exclusive ledger, wall-clock cap, no retry. Per-run certified manifest:
code hashes, input data hashes, environment, operator probe reference,
init reference, output hashes, log hash. --dry-run prints everything and
trains nothing.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "a3-membership-mixing" / "src"))
from screen_rules import budget_allows  # noqa: E402

A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
CFG_PATH = ROOT / "configs" / "stage_a_freeze.json"
LEDGER = ROOT / "outputs" / "stage_a_runs.json"
MANIFEST = ROOT / "outputs" / "stage_a_manifest.json"
LOGDIR = ROOT / "outputs" / "stage_a_logs"

CODE_FILES = ["train_graph.py", "a3_coeff_override.py", "a3_output_paths.py", "model.py",
              "config.py", "preprocess.py", "utils.py", "a3_exact.py", "dataset.py"]
RUN_INPUTS = {
    "RR": ["data/het_g8p0/seg2region.csv", "data/het_g8p0/road_feature.csv",
           "data/het_g8p0/roadmap.rel", "data/het_g8p0/edge_attr.csv",
           "data/het_g8p0/w_by_hop_sparse.npz", "data/het_g8p0/roadmap.geo"],
    "PP": ["data/het_g8p0_perm3/seg2region.csv", "data/het_g8p0_perm3/road_feature.csv",
           "data/het_g8p0_perm3/roadmap.rel", "data/het_g8p0_perm3/edge_attr.csv",
           "data/het_g8p0_perm3/w_by_hop_sparse.npz", "data/het_g8p0_perm3/roadmap.geo"],
    "PR": ["data/het_g8p0_perm3/seg2region.csv", "data/het_g8p0_perm3/road_feature.csv",
           "data/het_g8p0_perm3/roadmap.rel", "data/het_g8p0_perm3/edge_attr.csv",
           "data/het_g8p0_perm3/w_by_hop_sparse.npz", "data/het_g8p0_perm3/roadmap.geo",
           "data/het_g8p0/seg2region.csv"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arms() -> list[dict]:
    runs = []
    for seed in range(5):
        runs.append({"arm": "RR_exact", "dataset": "het_g8p0", "variant": "induced_exact",
                     "coeff_seg": None, "seed": seed})
    for seed in range(5):
        runs.append({"arm": "RR_clique", "dataset": "het_g8p0", "variant": "induced_clique_self",
                     "coeff_seg": None, "seed": seed})
    for seed in range(5):
        runs.append({"arm": "PP_exact", "dataset": "het_g8p0_perm3", "variant": "induced_exact",
                     "coeff_seg": None, "seed": seed})
    for seed in range(5):
        runs.append({"arm": "PP_clique", "dataset": "het_g8p0_perm3", "variant": "induced_clique_self",
                     "coeff_seg": None, "seed": seed})
    for seed in range(5):
        runs.append({"arm": "PR_clique", "dataset": "het_g8p0_perm3",
                     "variant": "induced_coeff_override",
                     "coeff_seg": "data/het_g8p0/seg2region.csv", "seed": seed})
    return runs


def command(r: dict, out_dir: Path, python_exe: str) -> list[str]:
    cmd = [python_exe, "train_graph.py", "--dataset", r["dataset"],
           "--ver", f"stage_a_{r['arm']}_s{r['seed']}", "--variant", r["variant"],
           "--seed", str(r["seed"]), "--device", "cuda", "--output-dir", str(out_dir)]
    if r["coeff_seg"]:
        cmd += ["--coeff-seg2region", r["coeff_seg"], "--override-weighting", "clique"]
    return cmd


def environment() -> dict:
    import torch
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.platform(),
    }


def preflight(cfg: dict, dry_run: bool) -> list[str]:
    problems = []
    if not cfg.get("authorized"):
        problems.append("config authorized=false")
    if not cfg.get("certified"):
        problems.append("config certified=false (user authorized certified level)")
    for rel, want in cfg.get("code_pins", {}).items():
        p = A3 / rel
        if not p.exists() or sha256(p) != want:
            problems.append(f"code pin mismatch: {rel}")
    for rel, want in cfg.get("data_pins", {}).items():
        p = A3 / rel
        if not p.exists() or sha256(p) != want:
            problems.append(f"data pin mismatch: {rel}")
    val = ROOT / "outputs" / "operator_validation_stage_a.json"
    if not val.exists() or not json.loads(val.read_text(encoding="utf-8")).get("gate_passed"):
        problems.append("operator validation gate not passed")
    init_ref = ROOT / "outputs" / "init_reference.json"
    if not init_ref.exists() or not json.loads(init_ref.read_text(encoding="utf-8")).get("all_deterministic"):
        problems.append("init reference missing or non-deterministic")
    if not dry_run:
        existing = [str(p) for p in (ROOT / "outputs" / "stage_a").rglob("*") if p.is_file()]
        if existing:
            problems.append(f"stage_a outputs already exist: {existing[:3]}")
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
    repro_dir = ROOT / "outputs" / "stage_a" / "repro_check_RR_clique_s0"
    all_cmds = []
    for r in specs:
        out_dir = ROOT / "outputs" / "stage_a" / r["arm"] / f"seed{r['seed']}"
        all_cmds.append({"spec": r, "out_dir": out_dir, "cmd": command(r, out_dir, python_exe)})
    repro_cmd = command({"arm": "RR_clique", "dataset": "het_g8p0",
                         "variant": "induced_clique_self", "coeff_seg": None, "seed": 0},
                        repro_dir, python_exe)

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "n_runs": len(all_cmds),
                          "cap_hours": cfg["cost"]["cap_gpu_hours"],
                          "estimated_hours_per_run": cfg["cost"]["estimated_hours_per_run"],
                          "repro_check": " ".join(repro_cmd),
                          "commands": [" ".join(c["cmd"]) for c in all_cmds]}, indent=1))
        return
    if problems:
        print(json.dumps({"status": "refused", "problems": problems}, indent=1))
        raise SystemExit(3)

    code_hashes = {rel: sha256(A3 / rel) for rel in CODE_FILES}
    env_info = environment()
    val = json.loads((ROOT / "outputs" / "operator_validation_stage_a.json").read_text(encoding="utf-8"))
    init_ref = json.loads((ROOT / "outputs" / "init_reference.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_sha256": sha256(CFG_PATH),
        "freeze_doc_sha256": sha256(ROOT / "FREEZE_STAGE_A.md"),
        "instance_validation_sha256": sha256(ROOT / "outputs" / "perm3_generation.json"),
        "operator_validation_sha256": sha256(ROOT / "outputs" / "operator_validation_stage_a.json"),
        "environment": env_info, "code_hashes": code_hashes,
        "operator_probe": val["paths"], "init_reference": init_ref["seeds"],
        "runs": [], "reproducibility_check": None,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LEDGER.open("x", encoding="utf-8") as f:
            json.dump({"started_at": manifest["started_at"], "cap_hours": cfg["cost"]["cap_gpu_hours"],
                       "runs": []}, f, indent=1)
    except FileExistsError:
        print(json.dumps({"status": "refused_existing_ledger", "ledger": str(LEDGER),
                          "note": "previous attempts retain their budget; automatic restart is disabled"}))
        raise SystemExit(3)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    def save_all(rec=None):
        # upsert by (arm, seed): save_all is called twice per run (start with
        # status=running, finish with results) and must keep ONE entry per run,
        # not append. (Fixed 2026-10-19 after review; the already-running
        # process holds the pre-fix code, so its ledger gets duplicate entries
        # with identical final values — the status script dedupes on read.)
        if rec is not None:
            key = (rec["arm"], rec["seed"])
            manifest["runs"] = [r for r in manifest["runs"]
                                if (r.get("arm"), r.get("seed")) != key] + [rec]
        MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        led = json.loads(LEDGER.read_text(encoding="utf-8"))
        if rec is not None:
            led["runs"] = [r for r in led["runs"]
                           if (r.get("arm"), r.get("seed")) != (rec["arm"], rec["seed"])] + [
                {k: rec.get(k) for k in
                 ("arm", "seed", "status", "returncode", "hours",
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
               "input_data_hashes": {rel: sha256(A3 / rel) for rel in RUN_INPUTS[r["arm"].split("_")[0]]},
               "environment": env_info,
               "operator_probe": val["paths"][r["arm"]],
               "init_reference_sha256": init_ref["seeds"][str(r["seed"])]["state_dict_sha256"]}
        save_all(rec)
        log_path = LOGDIR / f"{r['arm']}_s{r['seed']}.log"
        t0 = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, cwd=A3, env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
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
                    "embedding_size": emb_path.stat().st_size if emb_ok else None,
                    "graph_model_sha256": sha256(out_dir / "graph_model.pt")
                    if (out_dir / "graph_model.pt").exists() else None,
                    "log_sha256": sha256(log_path)})
        if returncode == 0 and not emb_ok:
            manifest["stopped"] = "output_missing"
        save_all(rec)
        print(json.dumps({k: rec.get(k) for k in ("arm", "seed", "returncode", "hours",
                                                  "cumulative_hours", "embedding_sha256")}), flush=True)
        if manifest.get("stopped"):
            break
        if returncode != 0:
            manifest.setdefault("stopped", "run_failed" if returncode != -999 else "cap_exceeded_mid_run")
            break

    if not manifest.get("stopped"):
        # end-to-end bitwise reproduction check (freeze §5)
        t0 = time.perf_counter()
        log_path = LOGDIR / "repro_check_RR_clique_s0.log"
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
        orig = all_cmds[5]  # RR_clique seed0
        orig_emb = orig["out_dir"] / "emb_induced_clique_self_s0.pt"
        repro_emb = repro_dir / "emb_induced_clique_self_s0.pt"
        same = (orig_emb.exists() and repro_emb.exists()
                and sha256(orig_emb) == sha256(repro_emb))
        manifest["reproducibility_check"] = {
            "cmd": " ".join(repro_cmd), "returncode": returncode, "hours": round(hours, 4),
            "cumulative_hours": round(elapsed, 4),
            "original_sha256": sha256(orig_emb) if orig_emb.exists() else None,
            "repro_sha256": sha256(repro_emb) if repro_emb.exists() else None,
            "bitwise_identical": bool(same),
            "note": "embedding must match bitwise; mismatch flags all Stage A results determinism_suspect",
        }
        if elapsed > cfg["cost"]["cap_gpu_hours"]:
            manifest["stopped"] = manifest.get("stopped", "cap_exceeded_after_repro")
    if not manifest.get("stopped"):
        manifest["stopped"] = None
    manifest["status"] = "completed" if not manifest.get("stopped") else f"stopped_{manifest['stopped']}"
    manifest["total_hours"] = round(elapsed, 4)
    save_all()
    print(json.dumps({"status": manifest["status"], "total_hours": manifest["total_hours"],
                      "reproducibility": manifest["reproducibility_check"]}, indent=1))


if __name__ == "__main__":
    main()
