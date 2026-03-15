"""DeepJIT baseline for fixed-split experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

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


class DeepJITModel(nn.Module):
    """CNN-based DeepJIT model."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        vocab_msg = int(args.vocab_msg)
        vocab_code = int(args.vocab_code)
        emb_dim = int(args.embedding_dim)
        num_filters = int(args.num_filters)
        filter_sizes = list(args.filter_sizes)

        self.args = args
        self.embed_msg = nn.Embedding(vocab_msg, emb_dim)
        self.embed_code = nn.Embedding(vocab_code, emb_dim)
        self.convs_msg = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (kernel, emb_dim)) for kernel in filter_sizes]
        )
        self.convs_code_line = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (kernel, emb_dim)) for kernel in filter_sizes]
        )
        self.convs_code_file = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, (kernel, num_filters * len(filter_sizes)))
                for kernel in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(float(args.dropout_keep_prob))
        self.fc1 = nn.Linear(2 * len(filter_sizes) * num_filters, int(args.hidden_units))
        self.fc2 = nn.Linear(int(args.hidden_units), 1)

    def forward_msg(self, tensor: torch.Tensor, convs: nn.ModuleList) -> torch.Tensor:
        tensor = tensor.unsqueeze(1)
        tensor = [F.relu(conv(tensor)).squeeze(3) for conv in convs]
        tensor = [F.max_pool1d(part, part.size(2)).squeeze(2) for part in tensor]
        return torch.cat(tensor, 1)

    def forward_code(
        self,
        tensor: torch.Tensor,
        convs_line: nn.ModuleList,
        convs_file: nn.ModuleList,
    ) -> torch.Tensor:
        batch_size, file_count = tensor.shape[0], tensor.shape[1]
        tensor = tensor.reshape(batch_size * file_count, tensor.shape[2], tensor.shape[3])
        tensor = self.forward_msg(tensor, convs_line)
        tensor = tensor.reshape(
            batch_size,
            file_count,
            self.args.num_filters * len(self.args.filter_sizes),
        )
        return self.forward_msg(tensor, convs_file)

    def forward(self, msg: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        msg_features = self.embed_msg(msg)
        msg_features = self.forward_msg(msg_features, self.convs_msg)

        code_features = self.embed_code(code)
        code_features = self.forward_code(
            code_features,
            self.convs_code_line,
            self.convs_code_file,
        )

        features = torch.cat((msg_features, code_features), 1)
        features = self.dropout(features)
        features = F.relu(self.fc1(features))
        return torch.sigmoid(self.fc2(features)).squeeze(1)


def build_vocab_from_sequences(seqs: Iterable[Iterable[str]]) -> Dict[str, int]:
    """Build a token vocabulary that always includes <NULL>."""
    counter: Counter[str] = Counter()
    for seq in seqs:
        counter.update(seq)

    vocab: Dict[str, int] = {"<NULL>": 0}
    for token, _ in counter.most_common():
        token = token.lower()
        if token == "<null>" or token in vocab:
            continue
        vocab[token] = len(vocab)
    return vocab


def padding_length(text: str, max_length: int) -> str:
    """Pad or truncate one whitespace-tokenized sequence."""
    tokens = str(text).split()
    if len(tokens) < max_length:
        tokens = tokens + ["<NULL>"] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
    return " ".join(tokens)


def padding_message(data: Sequence[str], max_length: int) -> List[str]:
    """Pad or truncate all message strings."""
    return [padding_length(item, max_length=max_length) for item in data]


def mapping_dict_msg(pad_msg: Sequence[str], dict_msg: Dict[str, int]) -> np.ndarray:
    """Map padded messages to integer ids."""
    null_id = dict_msg["<NULL>"]
    return np.asarray(
        [
            np.asarray([dict_msg.get(token.lower(), null_id) for token in line.split()], dtype=np.int64)
            for line in pad_msg
        ],
        dtype=np.int64,
    )


def padding_data(
    data: Sequence[str],
    dictionary: Dict[str, int],
    msg_length: int,
) -> np.ndarray:
    """Pad message strings and map them to integer ids."""
    return mapping_dict_msg(padding_message(data, max_length=msg_length), dictionary)


def sample_indices(pool: Sequence[int], sample_size: int) -> List[int]:
    """Sample indices with replacement if the pool is too small."""
    if sample_size <= 0:
        return []
    if len(pool) >= sample_size:
        return random.sample(list(pool), sample_size)
    return random.choices(list(pool), k=sample_size)


def mini_batches_test(
    x_msg: np.ndarray,
    x_code: np.ndarray,
    y: np.ndarray,
    mini_batch_size: int = 64,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create ordered mini-batches for evaluation."""
    total = x_msg.shape[0]
    mini_batches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    num_complete = int(math.floor(total / float(mini_batch_size)))

    for batch_id in range(num_complete):
        start = batch_id * mini_batch_size
        end = start + mini_batch_size
        mini_batches.append((x_msg[start:end], x_code[start:end], y[start:end]))

    if total % mini_batch_size != 0:
        start = num_complete * mini_batch_size
        mini_batches.append((x_msg[start:total], x_code[start:total], y[start:total]))
    return mini_batches


def mini_batches_train(
    x_msg: np.ndarray,
    x_code: np.ndarray,
    y: np.ndarray,
    mini_batch_size: int = 64,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create class-balanced mini-batches for training."""
    total = x_msg.shape[0]
    labels = y.tolist()
    pos_idx = [idx for idx, label in enumerate(labels) if int(label) == 1]
    neg_idx = [idx for idx, label in enumerate(labels) if int(label) == 0]
    if not pos_idx or not neg_idx:
        return mini_batches_test(x_msg, x_code, y, mini_batch_size=mini_batch_size)

    mini_batches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    num_batches = int(math.floor(total / float(mini_batch_size))) + 1
    half = max(int(mini_batch_size / 2), 1)
    for _ in range(num_batches):
        indices = sorted(sample_indices(pos_idx, half) + sample_indices(neg_idx, half))
        mini_batches.append((x_msg[indices], x_code[indices], y[indices]))
    return mini_batches


def records_to_msg_code(
    records: Sequence[dict],
    msg_source: str,
) -> Tuple[List[str], List[List[str]], np.ndarray]:
    """Convert HAM-Net records into DeepJIT-style messages and code hunks."""
    msgs: List[str] = []
    codes: List[List[str]] = []
    labels: List[int] = []

    for rec in records:
        labels.append(int(rec.get("label", 0)))

        if msg_source == "empty":
            msg = "<EMPTY>"
        else:
            project = str(rec.get("project", "")).strip()
            class_name = str(rec.get("class_name", "") or rec.get("file_path", "")).strip()
            msg = f"{project} {class_name}".strip() or "<EMPTY>"
        msgs.append(msg)

        lines: List[str] = []
        functions = rec.get("functions")
        if isinstance(functions, list) and functions:
            for func in functions:
                for line in str(func.get("code", "")).splitlines():
                    line = line.strip()
                    if line:
                        lines.append(line)
        else:
            for line in str(rec.get("code", "")).splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
        if not lines:
            lines = [""]
        codes.append(lines)

    return msgs, codes, np.asarray(labels, dtype=np.int64)


def build_padded_code_ids(
    codes: Sequence[Sequence[str]],
    dict_code: Dict[str, int],
    code_line: int,
    code_length: int,
) -> np.ndarray:
    """Encode code hunks as a fixed-shape integer tensor."""
    total = len(codes)
    null_id = dict_code.get("<NULL>", 0)
    pad_code = np.full((total, code_line, code_length), null_id, dtype=np.int64)

    for i, lines in enumerate(codes):
        for j, line in enumerate(lines[:code_line]):
            tokens = str(line).split()
            for k, token in enumerate(tokens[:code_length]):
                pad_code[i, j, k] = dict_code.get(token.lower(), null_id)
    return pad_code


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the DeepJIT baseline on a fixed split.")
    parser.add_argument("--data-path", type=Path, required=True, help="Input dataset JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--msg-length", type=int, default=256, help="Maximum message length.")
    parser.add_argument(
        "--msg-source",
        type=str,
        choices=["metadata", "empty"],
        default="metadata",
        help="Pseudo-message source.",
    )
    parser.add_argument("--code-line", type=int, default=10, help="Maximum number of code lines.")
    parser.add_argument("--code-length", type=int, default=256, help="Maximum tokens per code line.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding size.")
    parser.add_argument(
        "--filter-sizes",
        type=str,
        default="1,2,3",
        help="Comma-separated convolution kernel sizes.",
    )
    parser.add_argument("--num-filters", type=int, default=64, help="Filters per kernel size.")
    parser.add_argument("--hidden-units", type=int, default=512, help="Hidden layer size.")
    parser.add_argument(
        "--dropout-keep-prob",
        type=float,
        default=0.5,
        help="Dropout probability passed to nn.Dropout.",
    )
    parser.add_argument("--l2-reg-lambda", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=6,
        help="Patience used by early stopping.",
    )
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=int, default=-1, help="Reserved DeepJIT compatibility flag.")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA.")
    return parser.parse_args()


def main() -> None:
    """Run training and evaluation for the DeepJIT baseline."""
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

    records = load_records(args.data_path, max_samples=None, seed=args.seed)
    if not records:
        raise RuntimeError(f"No samples were loaded from {args.data_path}.")

    split_data = json.loads(args.split_file.read_text(encoding="utf-8"))
    splits = split_data.get("splits") or {}
    train_idx = [int(i) for i in splits.get("train", {}).get("indices", [])]
    val_idx = [int(i) for i in splits.get("val", {}).get("indices", [])]
    test_idx = [int(i) for i in splits.get("test", {}).get("indices", [])]

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

    all_records = train_records + val_records + test_records
    msgs, codes, labels = records_to_msg_code(all_records, msg_source=args.msg_source)

    msg_token_seqs = (msg.split() for msg in msgs)
    code_token_seqs = (line.split() for commit in codes for line in commit)
    dict_msg = build_vocab_from_sequences(msg_token_seqs)
    dict_code = build_vocab_from_sequences(code_token_seqs)

    pad_msg = padding_data(msgs, dict_msg, msg_length=args.msg_length)
    pad_code = build_padded_code_ids(
        codes=codes,
        dict_code=dict_code,
        code_line=args.code_line,
        code_length=args.code_length,
    )

    n_train = len(train_records)
    n_val = len(val_records)
    train_msg, val_msg, test_msg = (
        pad_msg[:n_train],
        pad_msg[n_train : n_train + n_val],
        pad_msg[n_train + n_val :],
    )
    train_code, val_code, test_code = (
        pad_code[:n_train],
        pad_code[n_train : n_train + n_val],
        pad_code[n_train + n_val :],
    )
    train_y, val_y, test_y = (
        labels[:n_train],
        labels[n_train : n_train + n_val],
        labels[n_train + n_val :],
    )

    class ModelArgs:
        def __init__(self, ns: argparse.Namespace) -> None:
            self.msg_length = ns.msg_length
            self.code_line = ns.code_line
            self.code_length = ns.code_length
            self.embedding_dim = ns.embedding_dim
            self.filter_sizes = [int(value) for value in ns.filter_sizes.split(",") if value.strip()]
            self.num_filters = ns.num_filters
            self.hidden_units = ns.hidden_units
            self.dropout_keep_prob = ns.dropout_keep_prob
            self.l2_reg_lambda = ns.l2_reg_lambda
            self.learning_rate = ns.learning_rate
            self.batch_size = ns.batch_size
            self.num_epochs = ns.epochs
            self.save_dir = str(ns.output_dir)
            self.device = ns.device
            self.no_cuda = ns.no_cuda
            self.vocab_msg = len(dict_msg)
            self.vocab_code = len(dict_code)
            self.class_num = 1
            self.cuda = (not self.no_cuda) and torch.cuda.is_available()

    model_args = ModelArgs(args)
    device = torch.device("cuda" if model_args.cuda else "cpu")
    model = DeepJITModel(model_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_args.learning_rate,
        weight_decay=model_args.l2_reg_lambda,
    )
    criterion = nn.BCELoss()

    def eval_split(
        split_msg: np.ndarray,
        split_code: np.ndarray,
        split_y: np.ndarray,
        desc: str,
    ) -> Tuple[List[int], List[float]]:
        model.eval()
        labels_list: List[int] = []
        probs_list: List[float] = []
        batches = mini_batches_test(
            x_msg=split_msg,
            x_code=split_code,
            y=split_y,
            mini_batch_size=model_args.batch_size,
        )
        with torch.no_grad():
            for msg_batch, code_batch, label_batch in tqdm(batches, desc=desc, unit="batch"):
                msg_t = torch.tensor(msg_batch, dtype=torch.long, device=device)
                code_t = torch.tensor(code_batch, dtype=torch.long, device=device)
                pred = model(msg_t, code_t)
                probs_list.extend(float(x) for x in pred.detach().cpu().numpy().tolist())
                labels_list.extend(int(x) for x in np.asarray(label_batch).tolist())
        return labels_list, probs_list

    early_stopper = EarlyStopper(patience=args.early_stop_patience, mode="max")
    best_state = None
    best_metric = -1.0

    for epoch in range(1, model_args.num_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        batches = mini_batches_train(
            x_msg=train_msg,
            x_code=train_code,
            y=train_y,
            mini_batch_size=model_args.batch_size,
        )
        for msg_batch, code_batch, label_batch in tqdm(
            batches,
            desc=f"epoch_{epoch}/{model_args.num_epochs}",
            unit="batch",
        ):
            msg_t = torch.tensor(msg_batch, dtype=torch.long, device=device)
            code_t = torch.tensor(code_batch, dtype=torch.long, device=device)
            label_t = torch.tensor(label_batch, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            pred = model(msg_t, code_t)
            loss = criterion(pred, label_t)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        print(f"[INFO] Epoch {epoch}/{model_args.num_epochs} train_loss={np.mean(epoch_losses):.4f}")
        val_labels_ep, val_probs_ep = eval_split(
            val_msg,
            val_code,
            val_y,
            desc="val_early_stop",
        )
        val_metrics_ep = compute_classification_metrics(val_labels_ep, val_probs_ep, threshold=0.5)
        monitor_key = pick_monitor_key(val_metrics_ep)
        monitor_value = val_metrics_ep.get(monitor_key)
        if monitor_value is not None and monitor_value > best_metric:
            best_metric = float(monitor_value)
            best_state = copy.deepcopy(model.state_dict())
        if monitor_value is not None and early_stopper.step(monitor_value):
            print(
                "[INFO] Early stopping triggered: "
                f"val {monitor_key} did not improve for {early_stopper.bad_epochs} epochs."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs = eval_split(val_msg, val_code, val_y, desc="val_eval")
    test_labels, test_probs = eval_split(test_msg, test_code, test_y, desc="test_eval")

    if val_labels:
        _, _, best_val = search_best_threshold(val_labels, val_probs)
    else:
        best_val = {"threshold": 0.5, "mcc": 0.0, "f1": 0.0}

    test_metrics = compute_classification_metrics(
        test_labels,
        test_probs,
        threshold=float(best_val["threshold"]),
    )
    test_metrics["threshold"] = float(best_val["threshold"])
    try:
        test_metrics["roc_auc"] = float(roc_auc_score(test_labels, test_probs))
    except Exception:
        pass

    save_json(
        {"val_best": best_val, "test_metrics": test_metrics},
        args.output_dir / "metrics.json",
    )
    torch.save(model.state_dict(), args.output_dir / "deepjit_best.pt")
    print("[INFO] Baseline evaluation finished. Results written to", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
