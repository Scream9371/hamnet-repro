"""Dataset loading and tensor packaging utilities for HAM-Net."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _encode_code_tokens(
    tokenizer,
    code: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    encoded = tokenizer(
        code,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_attention_mask=True,
    )
    return (
        torch.tensor(encoded["input_ids"], dtype=torch.long),
        torch.tensor(encoded["attention_mask"], dtype=torch.long),
    )


def infer_dataset_name(data_path: Path | str) -> str:

    return Path(data_path).stem


def stable_bag_key(dataset: str, bag: Dict[str, Any]) -> str:

    if bag.get("sample_id") is not None:
        return f"{dataset}|sample_id={bag.get('sample_id')}"
    project = bag.get("project") or bag.get("repo") or "UNK"
    file_path = bag.get("file_path") or bag.get("path") or "UNK"
    extra = (
        bag.get("bug_id")
        or bag.get("snapshot_commit")
        or bag.get("commit")
        or bag.get("version")
        or ""
    )
    return f"{dataset}|{project}|{extra}|{file_path}"


def stable_bag_id(dataset: str, bag: Dict[str, Any]) -> str:

    key = stable_bag_key(dataset, bag)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_caps_map(caps_file: Path) -> Dict[str, List[int]]:

    data = json.loads(Path(caps_file).read_text(encoding="utf-8"))
    out: Dict[str, List[int]] = {}
    for bag_id, info in data.items():
        if isinstance(info, dict):
            kept = info.get("kept_func_indices", [])
        else:
            kept = info
        if not isinstance(kept, list):
            continue
        clean = [int(i) for i in kept if isinstance(i, int) or str(i).isdigit()]
        out[str(bag_id)] = clean
    return out


def apply_caps_to_records(
    records: Sequence[dict],
    *,
    dataset_name: str,
    caps_map: Dict[str, List[int]],
) -> tuple[List[dict], Dict[str, int]]:

    new_records: List[dict] = []
    stats = {
        "total_bags": 0,
        "caps_hit": 0,
        "caps_missing": 0,
        "caps_empty_after_filter": 0,
    }
    for rec in records:
        funcs = rec.get("functions")
        if not isinstance(funcs, list):
            new_records.append(rec)
            continue
        stats["total_bags"] += 1
        bag_id = stable_bag_id(dataset_name, rec)
        kept = caps_map.get(bag_id)
        if kept is None:
            stats["caps_missing"] += 1
            new_records.append(rec)
            continue
        stats["caps_hit"] += 1
        chosen = [funcs[i] for i in kept if 0 <= i < len(funcs)]
        if not chosen:
            stats["caps_empty_after_filter"] += 1
            new_records.append(rec)
            continue
        rec_copy = dict(rec)
        rec_copy["functions"] = chosen
        new_records.append(rec_copy)
    return new_records, stats


def load_records(
    data_path: Path, *, max_samples: Optional[int] = None, seed: int = 42
) -> List[dict]:
    path = Path(data_path)
    raw_records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Loading records", unit="record"):
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))

    if raw_records and "functions" in raw_records[0]:
        records: List[dict] = []
        for raw in raw_records:
            functions = []
            for func in raw.get("functions", []):
                functions.append(
                    {
                        "code": func.get("code", ""),
                        "ast_nodes": func.get("ast_nodes", []),
                        "ast_edges": func.get("ast_edges", []),
                        "function_name": func.get("function_name"),
                        "func_label": func.get(
                            "func_label", func.get("function_label")
                        ),
                    }
                )
            records.append(
                {
                    "functions": functions,
                    "label": int(raw.get("label", 0)),
                    "project": raw.get("project") or raw.get("repo") or "unknown",
                    "class_name": raw.get("class_name"),
                    "file_path": raw.get("file_path"),
                    "sample_id": raw.get("sample_id"),
                    "bug_id": raw.get("bug_id"),
                    "snapshot_commit": raw.get("snapshot_commit"),
                    "commit": raw.get("commit"),
                    "version": raw.get("version"),
                    "repo": raw.get("repo"),
                    "path": raw.get("path"),
                }
            )
        if max_samples is not None and max_samples < len(records):
            rng = random.Random(seed)
            records = rng.sample(records, max_samples)
        return records

    records: List[dict] = []
    for raw in raw_records:
        records.append(
            {
                "code": raw["code"],
                "label": int(raw["label"]),
                "ast_nodes": raw["ast_nodes"],
                "ast_edges": raw["ast_edges"],
                "project": raw.get("project") or raw.get("repo") or "unknown",
                "bug_id": raw.get("bug_id"),
                "file_path": raw.get("file_path"),
            }
        )

    if max_samples is not None and max_samples < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)
    return records


def build_ast_vocab(records: Sequence[dict]) -> Dict[str, int]:

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for rec in tqdm(records, desc="Building AST vocabulary", unit="sample"):
        if "functions" in rec:
            node_iter = (
                node_type
                for func in rec.get("functions", [])
                for node_type in func.get("ast_nodes", [])
            )
        else:
            node_iter = rec.get("ast_nodes", [])
        for node_type in node_iter:
            if node_type not in vocab:
                vocab[node_type] = len(vocab)
    return vocab


def _edge_tensor(edges: Sequence[Sequence[int]], node_count: int) -> torch.Tensor:

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long)
        if edge_index.dim() == 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    if edge_index.numel() > 0:
        rev_edge = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, rev_edge], dim=1)
    self_loops = torch.arange(node_count, dtype=torch.long)
    loop_index = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


class FunctionGraphDataset(Dataset):


    def __init__(
        self,
        records: Sequence[dict],
        tokenizer,
        max_length: int,
        vocab: Dict[str, int],
        split_name: str,
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = vocab
        self._unk = vocab.get(UNK_TOKEN, 1)
        self.pad_token_id = (
            tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
        )
        self.samples: List[Dict[str, Any]] = []
        iterator = records
        desc = f"Building {split_name} graphs" if split_name else "Building graphs"
        for rec in tqdm(iterator, desc=desc, unit="sample"):
            input_ids, attention_mask = _encode_code_tokens(
                tokenizer=tokenizer,
                code=rec["code"],
                max_length=max_length,
            )
            node_ids = [
                vocab.get(node_type, self._unk) for node_type in rec["ast_nodes"]
            ]
            if not node_ids:
                node_ids = [self._unk]
            node_tensor = torch.tensor(node_ids, dtype=torch.long)
            edge_tensor = _edge_tensor(rec["ast_edges"], node_tensor.size(0))
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pad_token_id": self.pad_token_id,
                    "label": float(rec["label"]),
                    "project": rec.get("project", "unknown"),
                    "node_ids": node_tensor,
                    "edge_index": edge_tensor,
                }
            )

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": torch.tensor(sample["label"], dtype=torch.float32),
            "node_ids": sample["node_ids"],
            "edge_index": sample["edge_index"],
            "project": sample["project"],
        }


class ClassMilDataset(Dataset):


    def __init__(
        self,
        records: Sequence[dict],
        tokenizer,
        max_length: int,
        vocab: Dict[str, int],
        split_name: str,
        max_funcs: Optional[int] = None,
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = vocab
        self.max_funcs = max_funcs
        self._unk = vocab.get(UNK_TOKEN, 1)
        self.pad_token_id = (
            tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
        )
        self.samples: List[Dict[str, Any]] = []
        desc = f"Building {split_name} bag samples" if split_name else "Building bag samples"
        for rec in tqdm(records, desc=desc, unit="bag"):
            func_items = []
            for func in rec.get("functions", []):
                code = func.get("code", "")
                input_ids, attention_mask = _encode_code_tokens(
                    tokenizer=tokenizer,
                    code=code,
                    max_length=max_length,
                )
                node_ids = [
                    vocab.get(node_type, self._unk)
                    for node_type in func.get("ast_nodes", [])
                ]
                if not node_ids:
                    node_ids = [self._unk]
                node_tensor = torch.tensor(node_ids, dtype=torch.long)
                edge_tensor = _edge_tensor(
                    func.get("ast_edges", []), node_tensor.size(0)
                )
                func_items.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pad_token_id": self.pad_token_id,
                        "node_ids": node_tensor,
                        "edge_index": edge_tensor,
                        "function_name": func.get("function_name"),
                        "function_label": func.get(
                            "func_label", func.get("function_label")
                        ),
                    }
                )
            if not func_items:
                node_tensor = torch.tensor([self._unk], dtype=torch.long)
                input_ids, attention_mask = _encode_code_tokens(
                    tokenizer=tokenizer,
                    code="",
                    max_length=max_length,
                )
                func_items.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pad_token_id": self.pad_token_id,
                        "node_ids": node_tensor,
                        "edge_index": _edge_tensor([], node_tensor.size(0)),
                        "function_name": None,
                        "function_label": None,
                    }
                )
            self.samples.append(
                {
                    "functions": func_items,
                    "label": float(rec["label"]),
                    "project": rec.get("project", "unknown"),
                    "class_name": rec.get("class_name"),
                    "file_path": rec.get("file_path"),
                }
            )

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        sample = self.samples[idx]
        functions = sample["functions"]
        if self.max_funcs is not None and len(functions) > self.max_funcs:
            chosen = random.sample(range(len(functions)), self.max_funcs)
            functions = [functions[i] for i in chosen]
        return {
            "functions": functions,
            "labels": torch.tensor(sample["label"], dtype=torch.float32),
            "project": sample["project"],
            "class_name": sample.get("class_name"),
            "file_path": sample.get("file_path"),
        }


@dataclass
class DatasetBundle:
    train: Dataset
    val: Optional[Dataset]
    test: Optional[Dataset]
    node_vocab: Dict[str, int]
    label_stats: Dict[str, Dict[str, int]]
    bag_level: bool


def collate_graph_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pad_ids = [int(sample.get("pad_token_id", 0)) for sample in samples]
    pad_id = max(set(pad_ids), key=pad_ids.count) if pad_ids else 0
    input_ids = pad_sequence(
        [sample["input_ids"] for sample in samples],
        batch_first=True,
        padding_value=pad_id,
    )
    attention_mask = pad_sequence(
        [sample["attention_mask"] for sample in samples],
        batch_first=True,
        padding_value=0,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack([sample["labels"] for sample in samples], dim=0),
        "projects": [sample["project"] for sample in samples],
        "graphs": [
            {
                "node_ids": sample["node_ids"],
                "edge_index": sample["edge_index"],
            }
            for sample in samples
        ],
    }
    return batch


def collate_class_batch(
    samples: Sequence[Dict[str, Any]], tokenizer, max_length: int
) -> Dict[str, Any]:

    token_ids_list: List[torch.Tensor] = []
    token_mask_list: List[torch.Tensor] = []
    graphs: List[Dict[str, torch.Tensor]] = []
    bag_idx: List[int] = []
    labels: List[torch.Tensor] = []
    projects: List[str] = []
    class_names: List[str | None] = []
    file_paths: List[str | None] = []
    function_names: List[str | None] = []
    function_labels: List[float | None] = []
    for b_idx, sample in enumerate(samples):
        labels.append(sample["labels"])
        projects.append(sample.get("project", "unknown"))
        class_names.append(sample.get("class_name"))
        file_paths.append(sample.get("file_path"))
        for func in sample["functions"]:
            token_ids_list.append(func["input_ids"])
            token_mask_list.append(func["attention_mask"])
            graphs.append(
                {
                    "node_ids": func["node_ids"],
                    "edge_index": func["edge_index"],
                }
            )
            function_names.append(func.get("function_name"))
            raw_func_label = func.get("function_label")
            function_labels.append(
                float(raw_func_label) if raw_func_label is not None else None
            )
            bag_idx.append(b_idx)
    if token_ids_list:
        pad_id = (
            tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else 0
        )

        input_ids = pad_sequence(
            token_ids_list,
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = pad_sequence(
            token_mask_list,
            batch_first=True,
            padding_value=0,
        )
    else:
        input_ids = torch.empty((0, max_length), dtype=torch.long)
        attention_mask = torch.empty((0, max_length), dtype=torch.long)

    labels_tensor = (
        torch.stack(labels)
        if labels and isinstance(labels[0], torch.Tensor)
        else torch.tensor(labels, dtype=torch.float32)
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
        "bag_idx": torch.tensor(bag_idx, dtype=torch.long),
        "projects": projects,
        "class_names": class_names,
        "file_paths": file_paths,
        "function_names": function_names,
        "function_labels": function_labels,
        "graphs": graphs,
    }
    return batch
