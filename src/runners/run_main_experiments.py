"""Minimal batch runner for the HAM-Net main experiment path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hamnet.utils import aggregate_mean_std, ensure_dir, save_json


DEFAULT_DATASETS = [
    Path("datasets/promise_java.jsonl"),
    Path("datasets/defactors_python.jsonl"),
    Path("datasets/bigvul_c.jsonl"),
    Path("datasets/bugsinpy_python.jsonl"),
    Path("datasets/devign_c.jsonl"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the minimal batch runner."""
    parser = argparse.ArgumentParser(
        description="Run the fixed HAM-Net main experiment configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/main.json"),
        help="Path to the main experiment config file.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=("samplewise", "cpdp"),
        required=True,
        help="Evaluation protocol to run.",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="*",
        default=None,
        help="Optional subset of dataset JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/main"),
        help="Root directory for generated outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional override for the seed list.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def resolve_datasets(cli_datasets: list[Path] | None) -> list[Path]:
    """Resolve the dataset list used by the runner."""
    datasets = cli_datasets if cli_datasets else DEFAULT_DATASETS
    return [path if path.is_absolute() else (REPO_ROOT / path) for path in datasets]


def load_config(path: Path) -> dict[str, Any]:
    """Load the main experiment configuration JSON."""
    cfg_path = path if path.is_absolute() else (REPO_ROOT / path)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_run_command(
    *,
    dataset_path: Path,
    split_file: Path,
    output_dir: Path,
    seed: int,
    cfg_common: dict[str, Any],
    batch_size: int,
) -> list[str]:
    """Build a single-run command for the HAM-Net main path."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src/runners/run_hamnet.py"),
        "--data-path",
        str(dataset_path),
        "--split-file",
        str(split_file),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--encoder-name",
        str(cfg_common.get("encoder_name", "microsoft/codebert-base")),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(cfg_common["epochs"]),
        "--learning-rate",
        str(cfg_common["learning_rate"]),
        "--weight-decay",
        str(cfg_common["weight_decay"]),
        "--warmup-ratio",
        str(cfg_common["warmup_ratio"]),
        "--grad-accum",
        str(cfg_common["grad_accum"]),
        "--num-workers",
        str(cfg_common["num_workers"]),
        "--max-grad-norm",
        "1.0",
        "--max-length",
        str(cfg_common["max_length"]),
        "--segment-len",
        str(cfg_common["segment_len"]),
        "--graph-hidden",
        str(cfg_common["graph_hidden"]),
        "--unfreeze-last-n",
        str(cfg_common["unfreeze_last_n"]),
        "--pos-weight-strategy",
        str(cfg_common["pos_weight_strategy"]),
        "--early-stop-patience",
        str(cfg_common["early_stop_patience"]),
    ]
    if cfg_common.get("amp", False):
        cmd.append("--amp")
    if cfg_common.get("freeze_encoder", False):
        cmd.append("--freeze-encoder")
    caps_file = REPO_ROOT / "caps" / f"{dataset_path.stem}_train_caps.json"
    if caps_file.exists():
        cmd.extend(
            [
                "--caps-file",
                str(caps_file),
                "--caps-apply-splits",
                str(cfg_common.get("caps_apply_splits", "train,val")),
            ]
        )
    encoder_local_path = cfg_common.get("encoder_local_path")
    if encoder_local_path not in (None, "", "None"):
        local_path = Path(str(encoder_local_path))
        if not local_path.is_absolute():
            local_path = (REPO_ROOT / local_path).resolve()
        if local_path.exists():
            cmd.extend(["--encoder-local-path", str(local_path)])
        else:
            print(
                "[WARN] encoder_local_path is set but does not exist: "
                f"{local_path}. Falling back to encoder-name."
            )
    return cmd


def read_test_metrics(run_dir: Path) -> dict[str, float] | None:
    """Read test metrics from one finished run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_metrics = data.get("best_test_at_threshold") or data.get("test_metrics")
    return test_metrics if isinstance(test_metrics, dict) else None


def main() -> None:
    """Main entry."""
    args = parse_args()
    cfg = load_config(args.config)
    common = dict(cfg["common"])
    protocol_cfg = dict(cfg["protocols"][args.protocol])
    datasets = resolve_datasets(args.datasets)
    split_seed = int(protocol_cfg["split_seed"])
    seeds = list(args.seeds) if args.seeds else [int(x) for x in protocol_cfg["seeds"]]
    batch_size = int(protocol_cfg["batch_size"])

    summary_all: dict[str, dict[str, float | str]] = {}
    protocol_root = args.output_root if args.output_root.is_absolute() else (REPO_ROOT / args.output_root)
    protocol_root = protocol_root / args.protocol
    ensure_dir(protocol_root)

    for dataset_path in datasets:
        dataset_name = dataset_path.stem
        split_dir = protocol_cfg["splits_dir"]
        split_file = REPO_ROOT / split_dir / f"{dataset_name}_seed{split_seed}.json"
        dataset_root = protocol_root / dataset_name
        ensure_dir(dataset_root)

        run_metrics: list[dict[str, float]] = []
        for run_idx, seed in enumerate(seeds, start=1):
            run_dir = dataset_root / f"run{run_idx}"
            ensure_dir(run_dir)
            cmd = build_run_command(
                dataset_path=dataset_path,
                split_file=split_file,
                output_dir=run_dir,
                seed=seed,
                cfg_common=common,
                batch_size=batch_size,
            )
            print("[INFO] Running:", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True, cwd=REPO_ROOT)
                one = read_test_metrics(run_dir)
                if one:
                    run_metrics.append(one)

        summary = aggregate_mean_std(run_metrics) if run_metrics else {}
        save_json(summary, dataset_root / "summary.json")
        summary_all[dataset_name] = summary

    save_json(summary_all, protocol_root / "summary_all.json")


if __name__ == "__main__":
    main()
