"""BiLSTM baseline for fixed-split experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hamnet.data import (
    apply_caps_to_records,
    infer_dataset_name,
    load_caps_map,
    load_records,
)
from hamnet.early_stop import EarlyStopper, pick_monitor_key
from hamnet.metrics import compute_classification_metrics
from hamnet.thresholds import search_best_threshold
from hamnet.utils import ensure_dir, save_json, set_seed


class BertSeqDataset(Dataset):
    """Function-level dataset backed by raw code strings."""

    def __init__(
        self,
        records: Sequence[dict],
    ) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        return {"code": str(rec.get("code", "")), "label": int(rec.get("label", 0))}


class ClassMilSeqDataset(Dataset):
    """Bag-level dataset that defers flattening to the collate function."""

    def __init__(self, records: Sequence[dict]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.records[idx]


def build_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    """Create a collate function for function-level inputs."""

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts = [str(item["code"]) for item in batch]
        labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.float)
        enc = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate


def build_mil_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    """Create a collate function that expands bag-level samples into instances."""

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        all_codes: List[str] = []
        bag_indices: List[int] = []
        labels = torch.tensor([int(item.get("label", 0)) for item in batch], dtype=torch.float)

        for bag_id, item in enumerate(batch):
            functions = item.get("functions")
            if not isinstance(functions, list) or not functions:
                all_codes.append(str(item.get("code", "")))
                bag_indices.append(bag_id)
                continue
            for func in functions:
                all_codes.append(str(func.get("code", "")))
                bag_indices.append(bag_id)

        enc = tokenizer(
            all_codes,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "bag_idx": torch.tensor(bag_indices, dtype=torch.long),
        }

    return collate


class Attention(nn.Module):
    """Attention pooling over BiLSTM outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_hidden = torch.tanh(self.proj(hidden_states))
        scores = self.score(attn_hidden).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class BiLSTMAttnModel(nn.Module):
    """CodeBERT encoder followed by BiLSTM, pooling, and a linear head."""

    def __init__(
        self,
        encoder_name: str,
        lstm_hidden: int = 64,
        freeze_encoder: bool = False,
        unfreeze_last_n: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n = max(int(unfreeze_last_n), 0)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if self.unfreeze_last_n > 0:
                encoder_layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
                if encoder_layers is not None:
                    for layer in encoder_layers[-self.unfreeze_last_n :]:
                        for param in layer.parameters():
                            param.requires_grad = True

        enc_dim = int(self.encoder.config.hidden_size)
        self.bilstm = nn.LSTM(
            input_size=enc_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = Attention(hidden_dim=2 * lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(3 * 2 * lstm_hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder and self.unfreeze_last_n <= 0:
            with torch.no_grad():
                enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden = enc_out.last_hidden_state
        hidden, _ = self.bilstm(hidden)
        mask = attention_mask

        masked_hidden = hidden.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        avg_pool = masked_hidden.sum(dim=1) / lengths
        max_pool, _ = hidden.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(dim=1)
        att_vec = self.attn(hidden, mask)

        features = torch.cat([att_vec, max_pool, avg_pool], dim=-1)
        features = self.dropout(features)
        return self.classifier(features).squeeze(-1)


def reduce_bag_logits(instance_logits: torch.Tensor, bag_idx: torch.Tensor, num_bags: int) -> torch.Tensor:
    """Aggregate instance logits into bag logits via max pooling."""
    bag_logits = instance_logits.new_full((num_bags,), float("-inf"))
    for bag_id in range(num_bags):
        mask = bag_idx == bag_id
        if mask.any():
            bag_logits[bag_id] = instance_logits[mask].max()
    return bag_logits


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum: int,
) -> float:
    """Run one training epoch."""
    model.train()
    losses: List[float] = []
    grad_accum = max(int(grad_accum), 1)
    optimizer.zero_grad()
    step = 0
    for step, batch in enumerate(tqdm(loader, desc="train", unit="batch"), start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if "bag_idx" in batch:
            bag_idx = batch["bag_idx"].to(device)
            bag_labels = batch["labels"].to(device)
            instance_logits = model(input_ids, attention_mask)
            bag_logits = reduce_bag_logits(instance_logits, bag_idx, bag_labels.numel())
            loss = criterion(bag_logits, bag_labels)
        else:
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        loss = loss / grad_accum
        loss.backward()
        if step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        losses.append(float(loss.item() * grad_accum))

    if step and step % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()
    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate one split and compute classification metrics."""
    model.eval()
    losses: List[float] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", unit="batch", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if "bag_idx" in batch:
                bag_idx = batch["bag_idx"].to(device)
                bag_labels = batch["labels"].to(device)
                instance_logits = model(input_ids, attention_mask)
                bag_logits = reduce_bag_logits(instance_logits, bag_idx, bag_labels.numel())
                loss = criterion(bag_logits, bag_labels)
                probs = torch.sigmoid(bag_logits)
                losses.append(float(loss.item()))
                all_labels.extend(int(x) for x in bag_labels.tolist())
                all_probs.extend(float(x) for x in probs.tolist())
            else:
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                losses.append(float(loss.item()))
                all_labels.extend(int(x) for x in labels.tolist())
                all_probs.extend(float(x) for x in probs.tolist())

    metrics = compute_classification_metrics(all_labels, all_probs, threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[List[int], List[float], float]:
    """Collect labels and probabilities for threshold search."""
    model.eval()
    losses: List[float] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="threshold_eval", unit="batch", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if "bag_idx" in batch:
                bag_idx = batch["bag_idx"].to(device)
                bag_labels = batch["labels"].to(device)
                instance_logits = model(input_ids, attention_mask)
                bag_logits = reduce_bag_logits(instance_logits, bag_idx, bag_labels.numel())
                loss = criterion(bag_logits, bag_labels)
                probs = torch.sigmoid(bag_logits)
                losses.append(float(loss.item()))
                all_labels.extend(int(x) for x in bag_labels.tolist())
                all_probs.extend(float(x) for x in probs.tolist())
            else:
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                losses.append(float(loss.item()))
                all_labels.extend(int(x) for x in labels.tolist())
                all_probs.extend(float(x) for x in probs.tolist())

    return all_labels, all_probs, float(np.mean(losses)) if losses else 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the BiLSTM baseline on a fixed split.")
    parser.add_argument("--data-path", type=Path, required=True, help="Input dataset JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="microsoft/codebert-base",
        help="Pretrained encoder name or local path.",
    )
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=6,
        help="Patience used by early stopping.",
    )
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--lstm-hidden", type=int, default=64, help="BiLSTM hidden size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze the encoder weights.")
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=0,
        help="If the encoder is frozen, unfreeze the top N transformer layers.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Validation-time threshold.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional debug subset size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Fixed split JSON file with train/val/test indices.",
    )
    parser.add_argument("--caps-file", type=Path, default=None, help="Optional caps JSON path.")
    parser.add_argument(
        "--caps-dataset-name",
        type=str,
        default=None,
        help="Optional dataset name used when computing bag ids for caps.",
    )
    parser.add_argument(
        "--caps-apply-splits",
        type=str,
        default="train,val",
        help="Comma-separated split names where caps should be applied.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training and evaluation for the BiLSTM baseline."""
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(
        {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        args.output_dir / "config.json",
    )

    records = load_records(args.data_path, max_samples=args.max_samples, seed=args.seed)
    is_mil = bool(records and isinstance(records[0], dict) and "functions" in records[0])

    split_data = json.loads(args.split_file.read_text(encoding="utf-8"))
    splits = split_data.get("splits") or {}
    train_idx = splits.get("train", {}).get("indices", [])
    val_idx = splits.get("val", {}).get("indices", [])
    test_idx = splits.get("test", {}).get("indices", [])
    train_records = [records[i] for i in train_idx if i < len(records)]
    val_records = [records[i] for i in val_idx if i < len(records)]
    test_records = [records[i] for i in test_idx if i < len(records)]

    caps_map = None
    caps_apply_splits = {
        split.strip() for split in str(args.caps_apply_splits).split(",") if split.strip()
    }
    caps_dataset_name = args.caps_dataset_name or infer_dataset_name(args.data_path)
    if args.caps_file:
        caps_map = load_caps_map(args.caps_file)
        print(
            f"[INFO] Loaded caps: {args.caps_file} "
            f"(bags={len(caps_map)}, dataset={caps_dataset_name}, apply={sorted(caps_apply_splits)})"
        )
    if is_mil and caps_map:
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

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, use_fast=True)
    if is_mil:
        collate_fn = build_mil_collate_fn(tokenizer, args.max_length)
        train_ds = ClassMilSeqDataset(train_records)
        val_ds = ClassMilSeqDataset(val_records)
        test_ds = ClassMilSeqDataset(test_records)
    else:
        collate_fn = build_collate_fn(tokenizer, args.max_length)
        train_ds = BertSeqDataset(train_records)
        val_ds = BertSeqDataset(val_records)
        test_ds = BertSeqDataset(test_records)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttnModel(
        encoder_name=args.encoder_name,
        lstm_hidden=args.lstm_hidden,
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_n=args.unfreeze_last_n,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_metric = -1.0
    best_state = None
    early_stopper = EarlyStopper(patience=args.early_stop_patience, mode="max")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_accum=args.grad_accum,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )
        monitor_key = pick_monitor_key(val_metrics)
        monitor_value = val_metrics.get(monitor_key)
        print(
            f"[INFO] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
        )
        if monitor_value is not None and monitor_value > best_metric:
            best_metric = float(monitor_value)
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
            }
        if monitor_value is not None and early_stopper.step(monitor_value):
            print(
                "[INFO] Early stopping triggered: "
                f"val {monitor_key} did not improve for {early_stopper.bad_epochs} epochs."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        torch.save(best_state, args.output_dir / "bilstm_best.pt")

    val_labels, val_probs, val_loss = collect_outputs(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )
    if val_labels:
        _, _, best_val = search_best_threshold(val_labels, val_probs)
    else:
        best_val = {"threshold": float(args.threshold), "mcc": 0.0, "f1": 0.0}

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=float(best_val["threshold"]),
    )
    test_metrics["threshold"] = float(best_val["threshold"])
    test_metrics["val_mcc"] = float(best_val.get("mcc", 0.0))
    test_metrics["val_f1"] = float(best_val.get("f1", 0.0))
    test_metrics["val_loss"] = float(val_loss)

    save_json(
        {"val_best": best_val, "test_metrics": test_metrics},
        args.output_dir / "metrics.json",
    )
    print("[INFO] Baseline evaluation finished. Results written to", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
