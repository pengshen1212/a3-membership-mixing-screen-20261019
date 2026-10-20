"""Generate the Stage B1 membership instance perm4 (pre-registered algorithm).

Identical family/algorithm to perm3 (make_perm3.py) with generation seed
20261020. Must differ from real/perm0/perm3 assignments. Refuses to
overwrite an existing perm4 directory. Writes outputs/perm4_generation.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
OUT = Path(__file__).resolve().parents[1] / "outputs"
GEN_SEED = 20261020
N_EXPECTED = 11095


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    target_dir = A3 / "data" / "het_g8p0_perm4"
    if target_dir.exists():
        print(json.dumps({"status": "refused_existing_instance", "dir": str(target_dir)}))
        raise SystemExit(3)

    real = pd.read_csv(A3 / "data" / "het_g8p0" / "seg2region.csv")
    n = len(real)
    assert n == N_EXPECTED and (real["geo_id"].to_numpy() == np.arange(n)).all()

    sizes = real.groupby("region_id").size().sort_index()
    region_ids = sizes.index.to_numpy()
    rng = np.random.RandomState(GEN_SEED)
    labels = np.concatenate([np.full(int(s), r) for r, s in zip(region_ids, sizes.values)])
    order = rng.permutation(n)
    new_region = np.empty(n, dtype=np.int64)
    new_region[order] = labels

    checks = {}
    perm = pd.DataFrame({"geo_id": real["geo_id"].to_numpy(), "region_id": new_region})
    sz_new = perm.groupby("region_id").size().sort_index()
    checks["n_rows"] = bool(len(perm) == n)
    checks["geo_order"] = bool((perm["geo_id"].to_numpy() == np.arange(n)).all())
    checks["region_count"] = bool(len(sz_new) == 2160)
    checks["ids_contiguous"] = bool(sz_new.index.to_numpy().tolist() == list(range(2160)))
    checks["size_multiset"] = bool(np.array_equal(np.sort(sizes.values), np.sort(sz_new.values)))
    checks["no_empty_region"] = bool((sz_new.values > 0).all())
    checks["differs_from_real"] = bool((new_region != real["region_id"].to_numpy()).sum() > 0)
    checks["changed_vs_real"] = int((new_region != real["region_id"].to_numpy()).sum())
    for name in ("het_g8p0_perm0", "het_g8p0_perm3"):
        p = A3 / "data" / name / "seg2region.csv"
        if p.exists():
            other = pd.read_csv(p)["region_id"].to_numpy()
            checks[f"differs_from_{name.split('_')[-1]}"] = bool((new_region != other).sum() > 0)
            checks[f"changed_vs_{name.split('_')[-1]}"] = int((new_region != other).sum())

    if not all(v is True for k, v in checks.items() if isinstance(v, bool)):
        print(json.dumps({"status": "validation_failed", "checks": checks}, indent=1))
        raise SystemExit(2)

    sys.path.insert(0, str(A3))
    import a3_coeff_override as coeff_override
    rf = A3 / "data" / "het_g8p0" / "road_feature.csv"
    rel = A3 / "data" / "het_g8p0" / "roadmap.rel"
    target_dir.mkdir(parents=True)
    csv_path = target_dir / "seg2region.csv"
    perm.to_csv(csv_path, index=None)
    part = coeff_override.load_partition(csv_path, rf, rel)
    checks["load_partition_ok"] = bool(part.n == N_EXPECTED and part.n_edges == 2199)

    report = {
        "status": "generated",
        "instance": "perm4",
        "generation_seed": GEN_SEED,
        "algorithm": "identical family/algorithm to perm3; global node permutation consuming the real region-size histogram in ascending region-id order",
        "sha256": sha256_file(csv_path),
        "checks": checks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "perm4_generation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if not checks["load_partition_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())
