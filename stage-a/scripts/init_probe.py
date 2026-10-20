"""Init reference probe (certified manifest field: initialization).

Replicates train_graph.py's exact construction sequence with the pinned
modules only: setup_seed -> hop-weight tuple -> model_config -> RoadModel -> RNG
snapshot -> PairHGNN swap -> RNG restore -> weight_init, then hashes the
initial state_dict.

Runs every seed twice to demonstrate determinism. The construction consumed by
the RNG stream is identical across arms (same config dims, same sparse hop
prior, same PairHGNN swap), so one reference per seed covers all arms; the
runner asserts this and the end-to-end bitwise reproduction check (freeze §5)
validates it against a real training run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

A3 = Path(os.environ.get("A3_ROOT", r"C:\Users\SHEN.PENG23\.zcode\workspace\default\A3_ROOT_DST"))
sys.path.insert(0, str(A3))

OUT = Path(__file__).resolve().parents[1] / "outputs"


def state_dict_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k]
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_once(seed: int, device: str) -> str:
    import torch
    from utils import setup_seed, weight_init
    from config import model_config
    from model import RoadModel
    import model

    setup_seed(seed=seed)

    # hop weight tuple, replicating train_graph's sparse path exactly
    W_sp = sp.load_npz(A3 / "data" / "het_g8p0" / "w_by_hop_sparse.npz").astype(np.float32).tocsr()
    row_sum = np.asarray(W_sp.sum(axis=1)).ravel()
    zr = np.where(row_sum == 0)[0]
    if len(zr) > 0:
        W_sp = (W_sp + sp.coo_matrix((np.ones(len(zr), dtype=np.float32), (zr, zr)),
                                     shape=W_sp.shape)).tocsr()
        row_sum = np.asarray(W_sp.sum(axis=1)).ravel()
    W_sp = sp.diags(1.0 / row_sum).dot(W_sp).tocsr()
    coo = W_sp.tocoo()
    hop_weight = (torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int32)),
                  torch.FloatTensor(coo.data.astype(np.float32)), coo.shape)
    num_nodes = W_sp.shape[0]

    # model_config exactly as train_graph derives it (road_feature identical
    # across het_g8p0 / het_g8p0_perm3, hash-pinned)
    rf = pd.read_csv(A3 / "data" / "het_g8p0" / "road_feature.csv")
    model_config["road_id_in"] = rf["road_id"].max() + 1
    model_config["lon_in"] = rf["lon_grid"].max() + 1
    model_config["lat_in"] = rf["lat_grid"].max() + 1
    model_config["highway_in"] = rf["highway_info"].max() + 1
    model_config["lanes_in"] = rf["lanes_info"].max() + 1
    model_config["length_in"] = rf["length_info"].max() + 1
    model_config["maxspeed_in"] = rf["maxspeed_info"].max() + 1

    road_model = RoadModel(model_config=model_config, w=hop_weight, n=num_nodes).to(device)
    _rng_cpu = torch.get_rng_state()
    _rng_cuda = torch.cuda.get_rng_state() if device == "cuda" else None
    in_dim = model_config["emb_out"]["id"] + model_config["emb_out"]["dis"] * 9
    road_model.hgnn = model.PairHGNN(in_dim, model_config["hid_size"], model_config["hid_size"],
                                     drop_rate=model_config["drop_rate"]).to(device)
    torch.set_rng_state(_rng_cpu)
    if _rng_cuda is not None:
        torch.cuda.set_rng_state(_rng_cuda)
    road_model.apply(weight_init)
    return state_dict_hash(road_model.state_dict())


def main() -> None:
    import torch
    device = "cuda"
    ref = {}
    for seed in range(5):
        h1 = build_once(seed, device)
        h2 = build_once(seed, device)
        ref[str(seed)] = {"state_dict_sha256": h1, "deterministic": h1 == h2}
        print(f"seed {seed}: {h1[:16]}... deterministic={h1 == h2}", flush=True)
    report = {
        "status": "evaluated",
        "device": device,
        "torch": torch.__version__,
        "note": "one reference per seed covers all arms: construction sequence consumed by the RNG stream is identical across arms (same config dims from hash-identical road_feature.csv, same hop prior, same PairHGNN swap); end-to-end bitwise reproduction check in the runner validates this against a real training run",
        "seeds": ref,
    }
    report["all_deterministic"] = all(v["deterministic"] for v in ref.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "init_reference.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"all_deterministic": report["all_deterministic"], "n_seeds": len(ref)}))
    if not report["all_deterministic"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
