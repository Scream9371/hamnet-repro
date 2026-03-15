"""Utility helpers used by the minimal HAM-Net reproducibility path."""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SEEDS: list[int] = [0, 7, 42, 123, 2025, 3407, 4096, 8192, 16384, 32768]


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


class JSONLWriter:


    def __init__(self, path: Path) -> None:

        self.path = path
        ensure_dir(path.parent)
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, record: dict) -> None:

        self._fh.write(json.dumps(record, ensure_ascii=False) + os.linesep)
        self._fh.flush()

    def close(self) -> None:

        self._fh.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        self.close()


def aggregate_mean_std(
    records: list[dict[str, float]], *, digits: int = 3
) -> dict[str, float | str]:

    if not records:
        return {}
    keys = [k for k in records[0].keys()]
    out: dict[str, float | str] = {}
    n = 0
    for k in keys:
        vals = [
            float(v[k])
            for v in records
            if k in v and isinstance(v[k], (int, float)) and np.isfinite(float(v[k]))
        ]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=0))
        out[k] = mean_val
        out[f"{k}_std"] = std_val
        out[f"{k}_mean_std"] = f"{mean_val:.{digits}f}±{std_val:.{digits}f}"
        n = len(vals)
    if n:
        out["runs"] = float(n)
    return out


def format_mean_std(mean_val: float, std_val: float, digits: int = 3) -> str:

    return f"{mean_val:.{digits}f}±{std_val:.{digits}f}"


def resolve_seeds(seeds: list[int] | None) -> list[int]:

    return list(seeds) if seeds is not None else DEFAULT_SEEDS.copy()
