"""Interpretability evaluation entry."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
RUNNER_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))

from hamnet.data import (
    collate_class_batch,
    load_records,
)
from hamnet.model import HAMNetModel
from hamnet.utils import save_json, set_seed
from run_hamnet import build_bundle_from_split


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run the interpretability evaluation for a trained HAM-Net model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Training output directory containing best_model.pt and config.json.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional dataset path that overrides config.json.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--max-bags",
        type=int,
        default=0,
        help="Maximum number of bags to evaluate. Use 0 for all bags.",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.1,0.2,0.3,0.5",
        help="Comma-separated deletion/retention ratios.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=20,
        help="Number of random-control trials.",
    )
    parser.add_argument(
        "--peer-model-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional peer model directories used for cross-run stability evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to model-dir/interpretability_eval.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def parse_ratios(text: str) -> List[float]:

    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        val = float(token)
        if val <= 0.0 or val >= 1.0:
            continue
        values.append(val)
    values = sorted(set(values))
    if not values:
        values = [0.1, 0.2, 0.3, 0.5]
    return values


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif key == "graphs":
            moved[key] = [
                {
                    "node_ids": graph["node_ids"].to(device),
                    "edge_index": graph["edge_index"].to(device),
                }
                for graph in value
            ]
        else:
            moved[key] = value
    return moved


def build_bundle_from_config(cfg: Dict[str, Any], tokenizer, data_path: Path):
    def _resolve_repo_relative_path(raw: str) -> Path:
        p = Path(str(raw))
        if p.exists() or not p.is_absolute():
            return p
        repo_root = REPO_ROOT
        parts = p.parts

        for marker in ("splits_cpdp", "splits_samplewise", "splits_sample", "splits"):
            if marker in parts:
                idx = parts.index(marker)
                candidate = (repo_root / Path(*parts[idx:])).resolve()
                if candidate.exists():
                    return candidate

        for base in ("splits_cpdp", "splits_samplewise", "splits_sample", "splits"):
                candidate = (repo_root / base / p.name).resolve()
                if candidate.exists():
                    return candidate
        return p
    records = load_records(
        data_path, max_samples=cfg.get("max_samples"), seed=int(cfg.get("seed", 42))
    )
    split_file = cfg.get("split_file")
    if split_file in (None, "", "None"):
        raise RuntimeError(
            "Interpretability evaluation in the reproducibility repository "
            "requires a fixed split_file in config.json."
        )
    split_path = _resolve_repo_relative_path(str(split_file))
    if not split_path.is_absolute():
        repo_root = REPO_ROOT
        candidate = (repo_root / split_path).resolve()
        if candidate.exists():
            split_path = candidate
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    splits = split_data.get("splits") or {}
    train_idx = splits.get("train", {}).get("indices", [])
    val_idx = splits.get("val", {}).get("indices", [])
    test_idx = splits.get("test", {}).get("indices", [])
    train_records = [records[i] for i in train_idx if i < len(records)]
    val_records = [records[i] for i in val_idx if i < len(records)]
    test_records = [records[i] for i in test_idx if i < len(records)]
    return build_bundle_from_split(
        train_records,
        val_records,
        test_records,
        tokenizer,
        int(cfg.get("max_length", 256)),
    )


def load_model_from_dir(
    model_dir: Path,
    cfg: Dict[str, Any],
    node_vocab_size: int,
    device: torch.device,
) -> HAMNetModel:

    encoder_source = (
        str(cfg.get("encoder_local_path"))
        if cfg.get("encoder_local_path") not in (None, "", "None")
        and Path(str(cfg.get("encoder_local_path"))).exists()
        else cfg.get("encoder_name", "microsoft/codebert-base")
    )
    model = HAMNetModel(
        encoder_name=encoder_source,
        node_vocab_size=node_vocab_size,
        segment_len=int(cfg.get("segment_len", 32)),
        graph_hidden=int(cfg.get("graph_hidden", 128)),
        freeze_encoder=bool(cfg.get("freeze_encoder", False)),
        unfreeze_last_n=int(cfg.get("unfreeze_last_n", 0)),
        use_graph=not bool(cfg.get("no_graph", False)),
        use_hier_attn=not bool(cfg.get("no_hier_attn", False)),
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def select_dataset_split(bundle, split: str):

    if split == "train":
        return bundle.train
    if split == "val":
        if bundle.val is None:
            raise RuntimeError("The current configuration does not provide a validation split.")
        return bundle.val
    if bundle.test is None:
        raise RuntimeError("The current configuration does not provide a test split.")
    return bundle.test


def forward_single_bag(
    model: HAMNetModel,
    sample: Dict[str, Any],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> Dict[str, Any]:

    batch_cpu = collate_class_batch([sample], tokenizer=tokenizer, max_length=max_length)
    batch = move_batch_to_device(batch_cpu, device)
    with torch.no_grad():
        logits, attn = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            graphs=batch["graphs"],
            bag_idx=batch.get("bag_idx"),
        )
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return {"prob": float(probs[0]), "attn": attn, "batch": batch_cpu}


def _top_k_count(total: int, ratio: float) -> int:

    return max(1, int(math.ceil(total * ratio)))


def _mask_function_subset(
    batch_cpu: Dict[str, Any], keep_indices: Sequence[int]
) -> Dict[str, Any]:

    keep = sorted(set(int(i) for i in keep_indices))

    new_batch = dict(batch_cpu)
    new_batch["input_ids"] = batch_cpu["input_ids"][keep].clone()
    new_batch["attention_mask"] = batch_cpu["attention_mask"][keep].clone()
    new_batch["bag_idx"] = torch.zeros(len(keep), dtype=torch.long)
    new_batch["graphs"] = [batch_cpu["graphs"][i] for i in keep]

    function_names = batch_cpu.get("function_names")
    if function_names is not None:
        new_batch["function_names"] = [function_names[i] for i in keep]
    function_labels = batch_cpu.get("function_labels")
    if function_labels is not None:
        new_batch["function_labels"] = [function_labels[i] for i in keep]
    return new_batch


def _build_batched_function_subset_batch(
    batch_cpu: Dict[str, Any],
    keep_sets: Sequence[Sequence[int]],
) -> Dict[str, Any]:

    input_chunks = []
    mask_chunks = []
    bag_idx_all: List[int] = []
    graphs_all: List[Dict[str, torch.Tensor]] = []

    for bag_id, keep_indices in enumerate(keep_sets):
        keep = sorted(set(int(i) for i in keep_indices))
        if not keep:
            continue
        input_chunks.append(batch_cpu["input_ids"][keep])
        mask_chunks.append(batch_cpu["attention_mask"][keep])
        graphs_all.extend(batch_cpu["graphs"][i] for i in keep)
        bag_idx_all.extend([bag_id] * len(keep))

    if not input_chunks:
        raise ValueError("keep_sets is empty, so no batched subset can be constructed.")

    out: Dict[str, Any] = {
        "input_ids": torch.cat(input_chunks, dim=0).clone(),
        "attention_mask": torch.cat(mask_chunks, dim=0).clone(),
        "graphs": graphs_all,
        "bag_idx": torch.tensor(bag_idx_all, dtype=torch.long),
    }
    return out


def _run_batched_probs(
    model: HAMNetModel, batch_cpu: Dict[str, Any], device: torch.device, num_bags: int
) -> List[float]:

    batch = move_batch_to_device(batch_cpu, device)
    with torch.inference_mode():
        logits, _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            graphs=batch["graphs"],
            bag_idx=batch.get("bag_idx"),
        )
    probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()

    if len(probs) < num_bags:
        probs.extend([float("nan")] * (num_bags - len(probs)))
    return [float(p) for p in probs[:num_bags]]


def _run_batch_prob(model: HAMNetModel, batch_cpu: Dict[str, Any], device: torch.device) -> float:

    batch = move_batch_to_device(batch_cpu, device)
    with torch.inference_mode():
        logits, _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            graphs=batch["graphs"],
            bag_idx=batch.get("bag_idx"),
        )
    return float(torch.sigmoid(logits)[0].item())


def _extract_token_importance(
    token_weights: torch.Tensor, attention_mask: torch.Tensor, input_ids: torch.Tensor, special_ids: set[int]
) -> Tuple[np.ndarray, np.ndarray]:

    w = token_weights.detach().cpu().reshape(-1).numpy()
    seq_len = int(attention_mask.numel())
    if w.size < seq_len:
        pad = np.zeros(seq_len - w.size, dtype=float)
        w = np.concatenate([w, pad], axis=0)
    w = w[:seq_len]
    mask_np = attention_mask.detach().cpu().numpy().astype(bool)
    ids_np = input_ids.detach().cpu().numpy()
    valid = []
    for i in range(seq_len):
        if not mask_np[i]:
            continue
        if int(ids_np[i]) in special_ids:
            continue
        valid.append(i)
    return w, np.asarray(valid, dtype=int)


def _extract_node_importance(
    node_weights: Optional[torch.Tensor], node_ids: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:

    n = int(node_ids.numel())
    if node_weights is None:
        w = np.zeros(n, dtype=float)
    else:
        w = node_weights.detach().cpu().reshape(-1).numpy()
        if w.size < n:
            w = np.concatenate([w, np.zeros(n - w.size, dtype=float)], axis=0)
        w = w[:n]
    ids_np = node_ids.detach().cpu().numpy()
    valid = np.where(ids_np != 0)[0]
    return w, valid.astype(int)

def evaluate_localization(bag_scores: List[Dict[str, Any]]) -> Dict[str, Any]:

    hits1, hits3, mrrs, aps, aucs = [], [], [], [], []
    used = 0
    for item in bag_scores:
        labels = item["func_labels"]
        scores = item["func_scores"]
        if labels is None:
            continue
        y = np.asarray(labels, dtype=float)
        s = np.asarray(scores, dtype=float)
        if y.size == 0 or np.unique(y).size < 2 or y.sum() <= 0:
            continue
        used += 1
        order = np.argsort(-s)
        pos_ranks = np.where(y[order] > 0.5)[0]
        top1 = int(order[0] in np.where(y > 0.5)[0])
        k3 = min(3, len(order))
        top3 = int(any(idx in np.where(y > 0.5)[0] for idx in order[:k3]))
        first_rank = int(pos_ranks[0]) + 1
        hits1.append(top1)
        hits3.append(top3)
        mrrs.append(1.0 / first_rank)
        aps.append(float(average_precision_score(y, s)))
        try:
            aucs.append(float(roc_auc_score(y, s)))
        except ValueError:
            pass
    return {
        "bags_with_func_label": used,
        "hit_at_1": float(np.mean(hits1)) if hits1 else None,
        "hit_at_3": float(np.mean(hits3)) if hits3 else None,
        "mrr": float(np.mean(mrrs)) if mrrs else None,
        "ap": float(np.mean(aps)) if aps else None,
        "auc": float(np.mean(aucs)) if aucs else None,
    }


def _append_metric(store: Dict[float, List[float]], ratio: float, value: float) -> None:

    store.setdefault(ratio, []).append(float(value))


def summarize_ratio_metrics(values: Dict[float, List[float]]) -> Dict[str, Dict[str, float]]:

    out: Dict[str, Dict[str, float]] = {}
    for ratio in sorted(values.keys()):
        arr = np.asarray(values[ratio], dtype=float)
        out[f"{ratio:.3f}"] = {
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "std": float(np.std(arr)) if arr.size else float("nan"),
            "n": int(arr.size),
        }
    return out


def _is_file_fallback_bag(functions: Sequence[Dict[str, Any]]) -> bool:

    if not functions:
        return False
    names = [str(fn.get("function_name", "")).strip() for fn in functions]
    return bool(names) and all(name == "<file>" for name in names)


def evaluate_faithfulness(
    model: HAMNetModel,
    dataset,
    tokenizer,
    max_length: int,
    ratios: Sequence[float],
    random_trials: int,
    max_bags: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:

    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    pad_token_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    func_del_top: Dict[float, List[float]] = {}
    func_del_rand: Dict[float, List[float]] = {}
    func_keep_top: Dict[float, List[float]] = {}
    func_keep_rand: Dict[float, List[float]] = {}
    tok_del_top: Dict[float, List[float]] = {}
    tok_del_rand: Dict[float, List[float]] = {}
    node_del_top: Dict[float, List[float]] = {}
    node_del_rand: Dict[float, List[float]] = {}

    bag_scores: List[Dict[str, Any]] = []
    total = len(dataset)
    seen_bags = 0
    used_bags = 0
    skipped_empty_functions = 0
    skipped_file_fallback = 0

    for idx in tqdm(range(total), desc="interpretability", unit="bag"):
        if max_bags > 0 and used_bags >= max_bags:
            break
        sample = dataset[idx]
        seen_bags += 1
        funcs = sample.get("functions", [])
        if not funcs:
            skipped_empty_functions += 1
            continue
        if _is_file_fallback_bag(funcs):
            skipped_file_fallback += 1
            continue
        used_bags += 1
        out = forward_single_bag(model, sample, tokenizer, max_length, device)
        base_prob = float(out["prob"])
        attn = out["attn"]
        batch_cpu = out["batch"]

        bag_attn = attn.get("bag_attn", [])
        if not bag_attn:
            continue
        func_scores = bag_attn[0].detach().cpu().numpy()
        n_funcs = int(len(func_scores))
        order = np.argsort(-func_scores)
        top_func_idx = int(order[0])

        raw_func_labels = batch_cpu.get("function_labels")
        func_labels = None
        if raw_func_labels is not None and any(v is not None for v in raw_func_labels):
            mapped = []
            for v in raw_func_labels:
                if v is None:
                    mapped.append(0.0)
                else:
                    mapped.append(float(v))
            func_labels = mapped

        bag_scores.append(
            {
                "bag_index": idx,
                "project": batch_cpu["projects"][0] if batch_cpu.get("projects") else None,
                "class_name": batch_cpu["class_names"][0] if batch_cpu.get("class_names") else None,
                "file_path": batch_cpu["file_paths"][0] if batch_cpu.get("file_paths") else None,
                "func_scores": func_scores.tolist(),
                "func_labels": func_labels,
            }
        )

        if n_funcs >= 2:
            all_idx = list(range(n_funcs))
            for ratio in ratios:
                k = _top_k_count(n_funcs, ratio)
                k = min(k, n_funcs - 1)
                topk = order[:k].tolist()
                keep_after_delete = [i for i in all_idx if i not in set(topk)]
                keep_top = topk


                keep_sets: List[Sequence[int]] = []

                keep_sets.append(keep_after_delete)
                keep_sets.append(keep_top)

                rand_ks = []
                for _ in range(random_trials):
                    rand_pick = random.sample(all_idx, k)
                    rand_keep_after_delete = [i for i in all_idx if i not in set(rand_pick)]
                    rand_ks.append((rand_keep_after_delete, rand_pick))
                    keep_sets.append(rand_keep_after_delete)
                    keep_sets.append(rand_pick)

                batched = _build_batched_function_subset_batch(batch_cpu, keep_sets)
                probs = _run_batched_probs(model, batched, device, num_bags=len(keep_sets))

                prob_del_top = probs[0]
                prob_keep_top = probs[1]
                _append_metric(func_del_top, ratio, base_prob - prob_del_top)
                _append_metric(func_keep_top, ratio, base_prob - prob_keep_top)

                rand_del_drops = [base_prob - probs[2 + 2 * i] for i in range(random_trials)]
                rand_keep_drops = [base_prob - probs[2 + 2 * i + 1] for i in range(random_trials)]
                _append_metric(func_del_rand, ratio, float(np.mean(rand_del_drops)))
                _append_metric(func_keep_rand, ratio, float(np.mean(rand_keep_drops)))

        token_weights = attn.get("token_weights")
        graph_node_weights = attn.get("graph_node_weights")
        if token_weights is not None and token_weights.size(0) > top_func_idx:
            w_token, valid_tok_pos = _extract_token_importance(
                token_weights[top_func_idx],
                batch_cpu["attention_mask"][top_func_idx],
                batch_cpu["input_ids"][top_func_idx],
                special_ids,
            )
        else:
            w_token, valid_tok_pos = np.zeros(0, dtype=float), np.zeros(0, dtype=int)

        node_weights_i = None
        if graph_node_weights is not None and len(graph_node_weights) > top_func_idx:
            node_weights_i = graph_node_weights[top_func_idx]
        w_node, valid_node_pos = _extract_node_importance(
            node_weights_i,
            batch_cpu["graphs"][top_func_idx]["node_ids"],
        )

        for ratio in ratios:
            if valid_tok_pos.size > 0:
                k_tok = min(_top_k_count(valid_tok_pos.size, ratio), valid_tok_pos.size)
                tok_order = valid_tok_pos[np.argsort(-w_token[valid_tok_pos])]
                tok_topk = tok_order[:k_tok]

                top_batch = dict(batch_cpu)
                top_batch["input_ids"] = batch_cpu["input_ids"].clone()
                top_batch["attention_mask"] = batch_cpu["attention_mask"].clone()
                top_batch["input_ids"][top_func_idx, tok_topk] = pad_token_id
                top_batch["attention_mask"][top_func_idx, tok_topk] = 0
                prob_tok_top = _run_batch_prob(model, top_batch, device)
                _append_metric(tok_del_top, ratio, base_prob - prob_tok_top)

                rand_vals = []
                for _ in range(random_trials):
                    rand_pos = np.asarray(random.sample(valid_tok_pos.tolist(), k_tok), dtype=int)
                    rb = dict(batch_cpu)
                    rb["input_ids"] = batch_cpu["input_ids"].clone()
                    rb["attention_mask"] = batch_cpu["attention_mask"].clone()
                    rb["input_ids"][top_func_idx, rand_pos] = pad_token_id
                    rb["attention_mask"][top_func_idx, rand_pos] = 0
                    rand_vals.append(base_prob - _run_batch_prob(model, rb, device))
                _append_metric(tok_del_rand, ratio, float(np.mean(rand_vals)))

            if valid_node_pos.size > 0:
                k_node = min(_top_k_count(valid_node_pos.size, ratio), valid_node_pos.size)
                node_order = valid_node_pos[np.argsort(-w_node[valid_node_pos])]
                node_topk = node_order[:k_node]

                nb = dict(batch_cpu)
                nb["graphs"] = list(batch_cpu["graphs"])
                nb_g = dict(batch_cpu["graphs"][top_func_idx])
                nb_g["node_ids"] = batch_cpu["graphs"][top_func_idx]["node_ids"].clone()
                nb_g["node_ids"][node_topk] = 0
                nb["graphs"][top_func_idx] = nb_g
                prob_node_top = _run_batch_prob(model, nb, device)
                _append_metric(node_del_top, ratio, base_prob - prob_node_top)

                rand_vals = []
                for _ in range(random_trials):
                    rand_pos = np.asarray(random.sample(valid_node_pos.tolist(), k_node), dtype=int)
                    rb = dict(batch_cpu)
                    rb["graphs"] = list(batch_cpu["graphs"])
                    rb_g = dict(batch_cpu["graphs"][top_func_idx])
                    rb_g["node_ids"] = batch_cpu["graphs"][top_func_idx]["node_ids"].clone()
                    rb_g["node_ids"][rand_pos] = 0
                    rb["graphs"][top_func_idx] = rb_g
                    rand_vals.append(base_prob - _run_batch_prob(model, rb, device))
                _append_metric(node_del_rand, ratio, float(np.mean(rand_vals)))

    summary = {
        "function_deletion_top": summarize_ratio_metrics(func_del_top),
        "function_deletion_random": summarize_ratio_metrics(func_del_rand),
        "function_keep_top": summarize_ratio_metrics(func_keep_top),
        "function_keep_random": summarize_ratio_metrics(func_keep_rand),
        "token_deletion_top": summarize_ratio_metrics(tok_del_top),
        "token_deletion_random": summarize_ratio_metrics(tok_del_rand),
        "node_deletion_top": summarize_ratio_metrics(node_del_top),
        "node_deletion_random": summarize_ratio_metrics(node_del_rand),
    }
    filter_stats = {
        "dataset_bags": int(total),
        "seen_bags": int(seen_bags),
        "evaluated_bags": int(used_bags),
        "excluded_empty_functions_bags": int(skipped_empty_functions),
        "excluded_file_fallback_bags": int(skipped_file_fallback),
        "excluded_file_fallback_ratio_over_seen": (
            float(skipped_file_fallback / seen_bags) if seen_bags > 0 else 0.0
        ),
    }
    return summary, bag_scores, filter_stats


def _bag_identity(entry: Dict[str, Any]) -> str:

    return "|".join(
        [
            str(entry.get("project")),
            str(entry.get("file_path")),
            str(entry.get("class_name")),
            str(entry.get("bag_index")),
        ]
    )


def _topk_jaccard(a_scores: Sequence[float], b_scores: Sequence[float], k: int = 3) -> float:

    a = np.asarray(a_scores, dtype=float)
    b = np.asarray(b_scores, dtype=float)
    ka = min(k, a.size)
    kb = min(k, b.size)
    sa = set(np.argsort(-a)[:ka].tolist())
    sb = set(np.argsort(-b)[:kb].tolist())
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb) / len(union))


def evaluate_stability_with_peers(
    base_scores: List[Dict[str, Any]],
    peer_score_list: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:

    if not peer_score_list:
        return {}
    base_map = {_bag_identity(x): x for x in base_scores}
    top1_agreements, jaccards = [], []
    compared_bags = 0
    for peer_scores in peer_score_list:
        peer_map = {_bag_identity(x): x for x in peer_scores}
        common = set(base_map.keys()) & set(peer_map.keys())
        for key in common:
            a = np.asarray(base_map[key]["func_scores"], dtype=float)
            b = np.asarray(peer_map[key]["func_scores"], dtype=float)
            if a.size == 0 or b.size == 0:
                continue
            compared_bags += 1
            top1_agreements.append(float(np.argmax(a) == np.argmax(b)))
            jaccards.append(_topk_jaccard(a, b, k=3))
    return {
        "compared_bags": compared_bags,
        "top1_agreement": float(np.mean(top1_agreements)) if top1_agreements else None,
        "top3_jaccard": float(np.mean(jaccards)) if jaccards else None,
    }


def collect_only_bag_scores(
    model: HAMNetModel,
    dataset,
    tokenizer,
    max_length: int,
    max_bags: int,
    device: torch.device,
) -> List[Dict[str, Any]]:

    scores: List[Dict[str, Any]] = []
    total = len(dataset)
    limit = total if max_bags <= 0 else min(total, max_bags)
    for idx in tqdm(range(limit), desc="peer_attention", unit="bag", leave=False):
        sample = dataset[idx]
        out = forward_single_bag(model, sample, tokenizer, max_length, device)
        attn = out["attn"]
        bag_attn = attn.get("bag_attn", [])
        if not bag_attn:
            continue
        func_scores = bag_attn[0].detach().cpu().numpy().tolist()
        batch_cpu = out["batch"]
        scores.append(
            {
                "bag_index": idx,
                "project": batch_cpu["projects"][0] if batch_cpu.get("projects") else None,
                "class_name": batch_cpu["class_names"][0] if batch_cpu.get("class_names") else None,
                "file_path": batch_cpu["file_paths"][0] if batch_cpu.get("file_paths") else None,
                "func_scores": func_scores,
            }
        )
    return scores


def main() -> None:

    args = parse_args()
    model_dir = args.model_dir.resolve()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    set_seed(int(cfg.get("seed", 42)))
    ratios = parse_ratios(args.ratios)
    device = torch.device(args.device)

    data_path = args.data_path or Path(str(cfg.get("data_path")))
    encoder_source = (
        str(cfg.get("encoder_local_path"))
        if cfg.get("encoder_local_path") not in (None, "", "None")
        and Path(str(cfg.get("encoder_local_path"))).exists()
        else cfg.get("encoder_name", "microsoft/codebert-base")
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)

    bundle = build_bundle_from_config(cfg, tokenizer, data_path)
    if not bundle.bag_level:
        raise RuntimeError(
            "The current model was trained on flat function-level samples, so MIL "
            "interpretability evaluation is unavailable."
        )
    dataset = select_dataset_split(bundle, args.split)
    max_length = int(cfg.get("max_length", 256))

    model = load_model_from_dir(model_dir, cfg, len(bundle.node_vocab), device)

    faithfulness, bag_scores, filter_stats = evaluate_faithfulness(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        ratios=ratios,
        random_trials=int(args.random_trials),
        max_bags=int(args.max_bags),
        device=device,
    )
    localization = evaluate_localization(bag_scores)

    stability = {}
    peer_dirs = args.peer_model_dirs or []
    if peer_dirs:
        peer_scores_all: List[List[Dict[str, Any]]] = []
        for peer_dir in peer_dirs:
            peer_dir = peer_dir.resolve()
            if not (peer_dir / "best_model.pt").exists():
                continue
            peer_cfg_path = peer_dir / "config.json"
            if not peer_cfg_path.exists():
                continue
            peer_cfg = json.loads(peer_cfg_path.read_text(encoding="utf-8"))
            peer_model = load_model_from_dir(peer_dir, peer_cfg, len(bundle.node_vocab), device)
            peer_scores = collect_only_bag_scores(
                model=peer_model,
                dataset=dataset,
                tokenizer=tokenizer,
                max_length=max_length,
                max_bags=int(args.max_bags),
                device=device,
            )
            peer_scores_all.append(peer_scores)
        stability = evaluate_stability_with_peers(bag_scores, peer_scores_all)

    result = {
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "split": args.split,
        "ratios": [float(r) for r in ratios],
        "random_trials": int(args.random_trials),
        "max_bags": int(args.max_bags),
        "filtering": filter_stats,
        "faithfulness": faithfulness,
        "localization": localization,
        "stability": stability,
    }

    output_json = args.output_json or (model_dir / "interpretability_eval.json")
    save_json(result, Path(output_json))
    print(f"[INFO] Interpretability evaluation finished. Results written to: {output_json}")


if __name__ == "__main__":
    main()
