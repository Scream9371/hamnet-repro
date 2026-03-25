"""Minimal HAM-Net training and evaluation entry for the paper path."""

import argparse
import copy
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from hamnet.data import (
    ClassMilDataset,
    DatasetBundle,
    FunctionGraphDataset,
    apply_caps_to_records,
    build_ast_vocab,
    collate_class_batch,
    collate_graph_batch,
    infer_dataset_name,
    load_caps_map,
    load_records,
)
from hamnet.early_stop import EarlyStopper, pick_monitor_key
from hamnet.metrics import compute_classification_metrics
from hamnet.model import HAMNetModel
from hamnet.thresholds import search_best_threshold
from hamnet.utils import JSONLWriter, ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """Parse the minimal CLI required by the paper-aligned HAM-Net path."""
    parser = argparse.ArgumentParser(
        description="Run a single HAM-Net training/evaluation job for a fixed split."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to a dataset JSONL file.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Path to a fixed split JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="microsoft/codebert-base",
        help=(
            "Hugging Face model name or local model directory. "
            "If a remote model name is used, Transformers will download it once "
            "and reuse the local cache on later runs."
        ),
    )
    parser.add_argument(
        "--encoder-local-path",
        type=Path,
        default=None,
        help=(
            "Optional local encoder directory used instead of encoder-name. "
            "This is recommended for offline or review environments."
        ),
    )
    parser.add_argument(
        "--caps-file",
        type=Path,
        default=None,
        help="Optional caps file applied to bag-level data.",
    )
    parser.add_argument(
        "--caps-dataset-name",
        type=str,
        default=None,
        help="Optional dataset name override used when computing bag ids for caps.",
    )
    parser.add_argument(
        "--caps-apply-splits",
        type=str,
        default="train,val",
        help="Comma-separated split names where caps are applied.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum code token length.",
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=32,
        help="Segment length used by hierarchical attention.",
    )
    parser.add_argument(
        "--graph-hidden",
        type=int,
        default=128,
        help="Hidden size of the graph branch.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the encoder and only fine-tune the upper layers/modules.",
    )
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=2,
        help="Number of top transformer layers to unfreeze when encoder freezing is enabled.",
    )
    parser.add_argument(
        "--pos-weight-strategy",
        type=str,
        default="one",
        choices=("ratio", "sqrt", "half", "one"),
        help="Positive class weight strategy.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=6,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable the graph branch.",
    )
    parser.add_argument(
        "--no-hier-attn",
        action="store_true",
        help="Disable hierarchical attention and fall back to CLS pooling.",
    )

    args = parser.parse_args()

    # Fixed internal defaults for the paper-focused path.
    args.threshold = 0.5
    args.max_samples = None
    return args


def prepare_dataloaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int,
    collate_fn,
) -> Dict[str, Optional[DataLoader]]:
    """Build DataLoaders for train/validation/test splits."""
    pin_memory = torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loaders: Dict[str, Optional[DataLoader]] = {
        "train": DataLoader(
            bundle.train,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    }
    eval_batch = batch_size * 2
    loaders["val"] = (
        DataLoader(bundle.val, batch_size=eval_batch, shuffle=False, **loader_kwargs)
        if bundle.val
        else None
    )
    loaders["test"] = (
        DataLoader(bundle.test, batch_size=eval_batch, shuffle=False, **loader_kwargs)
        if bundle.test
        else None
    )
    return loaders


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a batch to the target device."""
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif key == "graphs":
            moved[key] = [
                {
                    "node_ids": graph["node_ids"].to(device, non_blocking=True),
                    "edge_index": graph["edge_index"].to(device, non_blocking=True),
                }
                for graph in value
            ]
        else:
            moved[key] = value
    return moved


def run_eval(
    model: HAMNetModel,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> Optional[Dict[str, float]]:
    """Compute evaluation metrics on a validation or test split."""
    if not loader:
        return None
    model.eval()
    all_labels: List[float] = []
    all_probs: List[float] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            with autocast(
                device_type="cuda", enabled=use_amp and torch.cuda.is_available()
            ):
                logits, _ = model(
                    input_ids=batch_on_device["input_ids"],
                    attention_mask=batch_on_device["attention_mask"],
                    graphs=batch_on_device["graphs"],
                    bag_idx=batch_on_device.get("bag_idx"),
                )
            probs = torch.sigmoid(logits)
            loss = criterion(logits, batch_on_device["labels"])
            losses.append(loss.item())
            all_labels.extend(batch_on_device["labels"].tolist())
            all_probs.extend(probs.tolist())
    metrics = compute_classification_metrics(all_labels, all_probs, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def collect_probs(
    model: HAMNetModel,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
):
    """Collect labels, logits, probabilities, and mean loss from one split."""
    if loader is None:
        return [], [], [], 0.0
    labels: List[float] = []
    probs: List[float] = []
    logits_list: List[float] = []
    losses: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            with autocast(
                device_type="cuda", enabled=use_amp and torch.cuda.is_available()
            ):
                logits, _ = model(
                    input_ids=batch_on_device["input_ids"],
                    attention_mask=batch_on_device["attention_mask"],
                    graphs=batch_on_device["graphs"],
                    bag_idx=batch_on_device.get("bag_idx"),
                )
            prob = torch.sigmoid(logits)
            loss = criterion(logits, batch_on_device["labels"])
            losses.append(loss.item())
            labels.extend(batch_on_device["labels"].tolist())
            probs.extend(prob.tolist())
            logits_list.extend(logits.tolist())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return labels, probs, logits_list, mean_loss


def summarize_distribution(
    values: Sequence[float], threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Summarize a distribution of logits or probabilities."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    summary = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "quantiles": {str(q): float(np.quantile(arr, q)) for q in quantiles},
    }
    if threshold is not None:
        summary["threshold_quantile"] = float(np.mean(arr <= threshold))
    return summary


def count_predicted_positive(
    probs: Sequence[float], threshold: float
) -> Dict[str, float]:
    """Count predicted positives at a given threshold."""
    if not probs:
        return {"count": 0, "ratio": 0.0}
    total = len(probs)
    pos = sum(p >= threshold for p in probs)
    return {"count": int(pos), "ratio": float(pos / total)}


def build_bundle_from_split(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    tokenizer,
    max_length: int,
) -> DatasetBundle:
    """Build a DatasetBundle from fixed split record lists."""
    is_bag_level = bool(train_records and "functions" in train_records[0])
    vocab = build_ast_vocab(train_records)
    dataset_cls = ClassMilDataset if is_bag_level else FunctionGraphDataset
    dataset_kwargs = {}

    train_ds = dataset_cls(
        train_records, tokenizer, max_length, vocab, split_name="train", **dataset_kwargs
    )
    val_ds = (
        dataset_cls(val_records, tokenizer, max_length, vocab, split_name="val", **dataset_kwargs)
        if val_records
        else None
    )
    test_ds = (
        dataset_cls(test_records, tokenizer, max_length, vocab, split_name="test", **dataset_kwargs)
        if test_records
        else None
    )

    def _count(items: Sequence[dict]) -> Dict[str, int]:
        from collections import Counter

        counter = Counter(rec["label"] for rec in items)
        return {str(k): int(v) for k, v in counter.items()}

    label_stats = {
        "train": _count(train_records),
        "val": _count(val_records),
        "test": _count(test_records),
    }
    return DatasetBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        node_vocab=vocab,
        label_stats=label_stats,
        bag_level=is_bag_level,
    )


def auto_thresholds(labels, probs):
    """Run the shared threshold search policy."""
    return search_best_threshold(labels, probs)


def main() -> None:
    """Main entry."""
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    records = load_records(args.data_path, max_samples=args.max_samples, seed=args.seed)
    print(f"[INFO] Loaded {len(records)} records from {args.data_path}")
    caps_map = None
    caps_apply_splits = {
        s.strip() for s in str(args.caps_apply_splits).split(",") if s.strip()
    }
    caps_dataset_name = args.caps_dataset_name or infer_dataset_name(args.data_path)
    if args.caps_file:
        caps_map = load_caps_map(args.caps_file)
        print(
            f"[INFO] Loaded caps: {args.caps_file} "
            f"(bags={len(caps_map)}, dataset={caps_dataset_name}, apply={sorted(caps_apply_splits)})"
        )

    split_path = Path(args.split_file)
    data = json.loads(split_path.read_text(encoding="utf-8"))
    splits = data.get("splits") or {}
    train_idx = splits.get("train", {}).get("indices", [])
    val_idx = splits.get("val", {}).get("indices", [])
    test_idx = splits.get("test", {}).get("indices", [])
    train_records = [records[i] for i in train_idx if i < len(records)]
    val_records = [records[i] for i in val_idx if i < len(records)]
    test_records = [records[i] for i in test_idx if i < len(records)]
    if caps_map and train_records and "functions" in train_records[0]:
        if "train" in caps_apply_splits:
            train_records, train_caps_stats = apply_caps_to_records(
                train_records,
                dataset_name=caps_dataset_name,
                caps_map=caps_map,
            )
            print("[INFO] Caps train:", train_caps_stats)
        if "val" in caps_apply_splits:
            val_records, val_caps_stats = apply_caps_to_records(
                val_records,
                dataset_name=caps_dataset_name,
                caps_map=caps_map,
            )
            print("[INFO] Caps val:", val_caps_stats)
        if "test" in caps_apply_splits:
            test_records, test_caps_stats = apply_caps_to_records(
                test_records,
                dataset_name=caps_dataset_name,
                caps_map=caps_map,
            )
            print("[INFO] Caps test:", test_caps_stats)

    encoder_source = args.encoder_name
    if args.encoder_local_path:
        local_path = Path(args.encoder_local_path)
        if local_path.exists():
            encoder_source = str(local_path)
        else:
            print(
                f"[WARN] Local encoder path does not exist: {local_path}. Falling back to {args.encoder_name}."
            )
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)
    bundle = build_bundle_from_split(
        train_records,
        val_records,
        test_records,
        tokenizer,
        args.max_length,
    )
    if bundle.bag_level:
        collate_fn = partial(
            collate_class_batch, tokenizer=tokenizer, max_length=args.max_length
        )
    else:
        collate_fn = collate_graph_batch
    loaders = prepare_dataloaders(
        bundle, args.batch_size, args.num_workers, collate_fn
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HAMNetModel(
        encoder_name=encoder_source,
        node_vocab_size=len(bundle.node_vocab),
        segment_len=args.segment_len,
        graph_hidden=args.graph_hidden,
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_n=args.unfreeze_last_n,
        use_graph=not args.no_graph,
        use_hier_attn=not args.no_hier_attn,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    try:
        scaler = GradScaler(
            device_type="cuda", enabled=args.amp and torch.cuda.is_available()
        )
    except TypeError:
        scaler = GradScaler(enabled=args.amp and torch.cuda.is_available())

    train_counts = bundle.label_stats["train"]
    pos_count = train_counts.get("1") or train_counts.get(1) or 1
    neg_count = train_counts.get("0") or train_counts.get(0) or 1
    ratio_weight = max(neg_count / max(pos_count, 1), 1e-3)
    if args.pos_weight_strategy == "ratio":
        pos_weight_value = ratio_weight
    elif args.pos_weight_strategy == "sqrt":
        pos_weight_value = max(math.sqrt(ratio_weight), 1e-3)
    elif args.pos_weight_strategy == "half":
        pos_weight_value = max(ratio_weight / 2.0, 1e-3)
    elif args.pos_weight_strategy == "one":
        pos_weight_value = 1.0
    else:
        raise ValueError(f"Unknown pos_weight strategy: {args.pos_weight_strategy}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )

    total_steps = (
        (len(loaders["train"]) + args.grad_accum - 1) // args.grad_accum * args.epochs
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    config_path = args.output_dir / "config.json"
    config_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    save_json(config_dict, config_path)
    save_json(bundle.label_stats, args.output_dir / "label_stats.json")

    best_metric = -float("inf")
    early_stopper = EarlyStopper(patience=args.early_stop_patience, mode="max")
    best_state = None
    best_epoch = 0

    log_writer = JSONLWriter(args.output_dir / "training_log.jsonl")

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_losses: List[float] = []
            optimizer.zero_grad()
            start_time = time.time()

            for step, batch in enumerate(loaders["train"], 1):
                batch_on_device = move_batch_to_device(batch, device)
                with autocast(
                    device_type="cuda", enabled=args.amp and torch.cuda.is_available()
                ):
                    logits, _ = model(
                        input_ids=batch_on_device["input_ids"],
                        attention_mask=batch_on_device["attention_mask"],
                        graphs=batch_on_device["graphs"],
                        bag_idx=batch_on_device.get("bag_idx"),
                    )
                    labels = batch_on_device["labels"]
                    loss = criterion(logits, labels) / args.grad_accum
                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                epoch_losses.append(loss.item() * args.grad_accum)

                if step % args.grad_accum == 0:
                    if args.amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if args.amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            train_loss = float(np.mean(epoch_losses))
            val_metrics = run_eval(
                model, loaders["val"], criterion, device, args.threshold, args.amp
            )
            monitor_key = pick_monitor_key(val_metrics)
            eval_target = val_metrics.get(monitor_key) if val_metrics else -train_loss
            if eval_target > best_metric:
                best_metric = eval_target
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            current_score = (val_metrics or {}).get(monitor_key)
            should_stop = early_stopper.step(current_score)

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "duration_sec": time.time() - start_time,
            }
            log_writer.write(epoch_record)
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                f"val_{monitor_key}={(val_metrics or {}).get(monitor_key)}"
            )
            if (
                args.early_stop_patience > 0
                and current_score is not None
                and should_stop
            ):
                print(
                    "[INFO] Early stopping triggered: "
                    f"val {monitor_key} not improved for {early_stopper.bad_epochs} epochs."
                )
                break
    finally:
        log_writer.close()

    if best_state:
        torch.save(best_state, args.output_dir / "best_model.pt")
        model.load_state_dict(best_state)
    print(f"[INFO] Best epoch: {best_epoch} (metric={best_metric:.4f})")

    test_metrics = run_eval(
        model, loaders["test"], criterion, device, args.threshold, args.amp
    )
    val_metrics = run_eval(
        model, loaders["val"], criterion, device, args.threshold, args.amp
    )

    val_labels, val_probs, val_logits, val_loss = collect_probs(
        model, loaders["val"], criterion, device, args.amp
    )
    thresholds, val_records, best_entry = auto_thresholds(val_labels, val_probs)
    best_th = best_entry["threshold"]

    test_best_metrics = None
    test_best_loss = None
    if loaders.get("test") is not None:
        test_labels, test_probs, test_logits, test_best_loss = collect_probs(
            model, loaders["test"], criterion, device, args.amp
        )
        test_best_metrics = compute_classification_metrics(
            test_labels, test_probs, threshold=best_th
        )
        _, test_records, test_best_entry = auto_thresholds(test_labels, test_probs)
        test_best_f1_entry = (
            max(test_records, key=lambda x: x["f1"]) if test_records else None
        )
    else:
        test_labels, test_probs, test_logits = [], [], []
        test_records, test_best_entry, test_best_f1_entry = [], None, None

    logit_stats = {
        "val": {
            "prob": summarize_distribution(val_probs, threshold=best_th),
            "logit": summarize_distribution(val_logits),
        },
        "test": {
            "prob": summarize_distribution(test_probs, threshold=best_th)
            if test_probs
            else None,
            "logit": summarize_distribution(test_logits) if test_logits else None,
        },
    }

    predicted_positive = {
        "val": count_predicted_positive(val_probs, best_th),
        "test": count_predicted_positive(test_probs, best_th) if test_probs else None,
    }

    threshold_selection = {
        "best_threshold": best_th,
        "val_mcc": best_entry["mcc"],
        "val_f1": best_entry["f1"],
        "val_metrics": best_entry,
        "val_loss": val_loss,
    }

    test_threshold_rescan = (
        {
            "best_threshold": test_best_entry["threshold"],
            "best_metrics": test_best_entry,
            "records": test_records,
        }
        if test_best_entry
        else None
    )
    test_best_f1 = (
        {"threshold": test_best_f1_entry["threshold"], "metrics": test_best_f1_entry}
        if test_best_f1_entry
        else None
    )

    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "label_stats": bundle.label_stats,
        "best_threshold": best_th,
        "best_val_at_threshold": best_entry,
        "best_val_loss": val_loss,
        "best_test_at_threshold": test_best_metrics,
        "best_test_loss": test_best_loss,
        "label_distribution": bundle.label_stats,
        "threshold_selection": threshold_selection,
        "logit_stats": logit_stats,
        "predicted_positive": predicted_positive,
        "test_threshold_rescan": test_threshold_rescan,
        "test_best_f1_upper_bound": test_best_f1,
    }
    save_json(summary, args.output_dir / "metrics.json")

    threshold_eval = {
        "threshold_records": val_records,
        "best_threshold": best_th,
        "best_val": best_entry,
        "val_loss": val_loss,
        "test_metrics": test_best_metrics,
        "test_loss": test_best_loss,
        "pos_weight": pos_weight_value,
        "logit_stats": logit_stats,
        "predicted_positive": predicted_positive,
        "test_threshold_rescan": test_threshold_rescan,
        "test_best_f1_upper_bound": test_best_f1,
    }
    save_json(threshold_eval, args.output_dir / "threshold_eval.json")

    print(
        f"[INFO] Experiment complete. Results stored in {args.output_dir}, "
        f"best_threshold={best_th:.4f}, best_val_mcc={best_entry['mcc']:.4f}"
    )


if __name__ == "__main__":
    main()
