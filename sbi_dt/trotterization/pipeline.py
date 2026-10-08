"""Steps shared by the two Trotter pipelines: run setup, config snapshot, cached training data."""

import hashlib
import json
import os
import random
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import torch


def setup_run(cfg: dict) -> dict:
    """Seed RNGs, create the tagged output/cache directories and write config_snapshot.json.

    Outputs and cache go to ``<output_dir|data_cache_dir>/<tag>`` where tag is the directory that
    holds ``sbi_params_file`` (e.g. ``12q_npe_brickwall``). Updates ``cfg`` in place.
    """
    seed = cfg["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parts = str(cfg["sbi_params_file"]).replace("\\", "/").split("/")
    tag = parts[-2] if len(parts) >= 2 else "unknown_sbi"
    cfg["output_dir"] = os.path.join(cfg["output_dir"], tag)
    cfg["data_cache_dir"] = os.path.join(cfg["data_cache_dir"], tag)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    if cfg.get("cache_data"):
        os.makedirs(cfg["data_cache_dir"], exist_ok=True)

    def _jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, tuple):
            return list(v)
        return v

    path = os.path.join(cfg["output_dir"], "config_snapshot.json")
    with open(path, "w") as f:
        json.dump({**{k: _jsonable(v) for k, v in cfg.items()}, "timestamp": datetime.now().isoformat()}, f, indent=2)
    print(f"  Config snapshot: {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return {"device": device, "output_dir": cfg["output_dir"]}


def load_noise_inputs(cfg: dict):
    """Return (SBI posterior samples, true noise parameters) read from the config's csv files."""
    print("\n=== Loading SBI posterior samples ===")
    sbi_samples = np.loadtxt(cfg["sbi_params_file"], delimiter=",")
    print(f"  {len(sbi_samples)} samples × {sbi_samples.shape[1]} params")
    true_noise_params = np.loadtxt(cfg["true_noise_params_file"], delimiter=",")
    if true_noise_params.ndim == 2:
        true_noise_params = true_noise_params[0]
    print(f"  True noise: {cfg['true_noise_params_file']}")
    return sbi_samples, true_noise_params


def _cache_base(cfg: dict, n_qubits: int, noise_dim: int) -> str:
    """Cache file stem; encodes every setting that changes the data (names match older caches)."""
    j_eval = cfg.get("J_eval", [])
    j_tag = hashlib.sha256(np.array(sorted(j_eval)).tobytes()).hexdigest()[:12] if j_eval else "none"
    inc = "inc" if cfg.get("initial_non_clifford", False) else "i0"
    steps_tag = hashlib.sha256(np.array(cfg["n_steps_train"]).tobytes()).hexdigest()[:10]
    sbi_tag = hashlib.sha256(cfg.get("sbi_params_file", "").encode()).hexdigest()[:10]
    return os.path.join(
        cfg["data_cache_dir"],
        f"train_clifford_nq{n_qubits}_nt{len(cfg['n_steps_train'])}_st{steps_tag}"
        f"_ncps1_cpc{cfg['circuits_per_combo']}_seed{cfg['random_seed']}_nd{noise_dim}_nsmean_{inc}"
        f"_sbi{sbi_tag}_jex{j_tag}",
    )


def _load_cache(base: str, cfg: dict) -> Optional[Dict[str, np.ndarray]]:
    meta_path = base + "_meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    data = {}
    for key in ("noisy", "ideal"):  # older caches also hold noise_params, which is not needed
        csv_path = base + "_" + key + ".csv"
        if key not in meta["keys"] or not os.path.exists(csv_path):
            return None
        data[key] = np.loadtxt(csv_path, delimiter=",").reshape(tuple(meta["shapes"][key]))
    # Depth labels are not stored: the data is ordered depth-major with circuits_per_combo each.
    data["n_steps"] = np.repeat(np.array(cfg["n_steps_train"], dtype=np.float32), cfg["circuits_per_combo"])
    return data


def _save_cache(base: str, data: Dict[str, np.ndarray]) -> None:
    saved = ["noisy", "ideal"]
    with open(base + "_meta.json", "w") as f:
        json.dump({"keys": saved, "shapes": {k: list(data[k].shape) for k in saved},
                   "timestamp": datetime.now().isoformat()}, f, indent=2)
    for k in saved:
        np.savetxt(base + "_" + k + ".csv", data[k].reshape(len(data[k]), -1), delimiter=",")


def load_or_generate_train_data(
    cfg: dict,
    n_qubits: int,
    sbi_samples: np.ndarray,
    generate: Callable[[], Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Return the training data from the csv cache if present, else ``generate()`` and cache it."""
    base = _cache_base(cfg, n_qubits, sbi_samples.shape[1])
    if cfg.get("cache_data") and os.path.exists(base + "_meta.json"):
        print(f"  Loading cached train data: {base}_*.csv")
        cached = _load_cache(base, cfg)
        if cached is not None:
            return cached
        print("  Cache incomplete: regenerating train data.")
    data = generate()
    if cfg.get("cache_data"):
        _save_cache(base, data)
        print(f"  Cached train data: {base}_*.csv")
    return data
