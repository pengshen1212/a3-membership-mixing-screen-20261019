"""Generate the Stage A membership instance perm3 (pre-registered algorithm).

Family: global region reshuffle preserving the region-size multiset — the same
family as the archived perm0/1/2 (verified against perm0). New RNG seed.
Writes data/het_g8p0_perm3/seg2region.csv into A3_ROOT and a validation JSON.

Refuses to overwrite an existing perm3 directory.
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
GEN_SEED = 20261019
N_EXPECTED = 11095


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    target_dir = A3 / "data" / "het_g8p0_perm3"
    if target_dir.exists():
        print(json.dumps({"status": "refused_existing_instance", "dir": str(target_dir)}))
        raise SystemExit(3)

    real = pd.read_csv(A3 / "data" / "het_g8p0" / "seg2region.csv")
    n = len(real)
    assert n == N_EXPECTED and (real["geo_id"].to_numpy() == np.arange(n)).all()

    sizes = real.groupby("region_id").size().sort_index()
    region_ids = sizes.index.to_numpy()
    rng = np.random.RandomState(GEN_SEED)

    # pre-registered algorithm: shuffle node order once, consume the real
    # region-size histogram in ascending region-id order
    labels = np.concatenate([np.full(int(s), r) for r, s in zip(region_ids, sizes.values)])
    order = rng.permutation(n)
    new_region = np.empty(n, dtype=np.int64)
    new_region[order] = labels

    checks = {}
    perm = pd.DataFrame({"geo_id": real["geo_id"].to_numpy(), "region_id": new_region})
    sz_new = perm.groupby("region_id").size().sort_index()
    checks["n_rows"] = bool(len(perm) == n)
    checks["geo_order"] = bool((perm["geo_id"].to_numpy() == np.arange(n)).all())
    checks["region_count"] = bool(len(sz_new) == len(sizes) == 2160)
    checks["ids_contiguous"] = bool(sz_new.index.to_numpy().tolist() == list(range(2160)))
    checks["size_multiset"] = bool(np.array_equal(np.sort(sizes.values), np.sort(sz_new.values)))
    checks["no_empty_region"] = bool((sz_new.values > 0).all())
    checks["differs_from_real"] = bool((new_region != real["region_id"].to_numpy()).sum() > 0)
    p0_path = A3 / "data" / "het_g8p0_perm0" / "seg2region.csv"
    if p0_path.exists():
        p0 = pd.read_csv(p0_path)
        checks["differs_from_perm0"] = bool((new_region != p0["region_id"].to_numpy()).sum() > 0)
        checks["changed_vs_real"] = int((new_region != real["region_id"].to_numpy()).sum())
        checks["changed_vs_perm0"] = int((new_region != p0["region_id"].to_numpy()).sum())

    if not all(v is True for k, v in checks.items() if isinstance(v, bool)):
        print(json.dumps({"status": "validation_failed", "checks": checks}, indent=1))
        raise SystemExit(2)

    # loadable by the pinned loader (coeff partition semantics identical)
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
        "instance": "perm3",
        "generation_seed": GEN_SEED,
        "algorithm": "single uniform node permutation consuming the real region-size histogram in ascending region-id order (same family as archived perm0/1/2)",
        "sha256": sha256_file(csv_path),
        "checks": checks,
        "note": "per-node region size is NOT preserved (matches archived perm0 behaviour); per-node incidence d_i=2+b_i is preserved by construction (roadmap.rel unchanged)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "perm3_generation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
