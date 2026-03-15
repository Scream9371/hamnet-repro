"""Dataset loading and tensor packaging utilities for HAM-Net."""

import difflib
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
AST_SEQ_LEN: int | None = None


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


    if raw_records and _looks_like_patch_dataset(raw_records[0]):
        raw_records = _apply_diff_window(raw_records)


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


def _looks_like_patch_dataset(sample: Dict[str, Any]) -> bool:

    return all(
        key in sample for key in ("bug_id", "file_path", "function_name", "version")
    )


def _compute_changed_lines(old_src: str, new_src: str) -> Tuple[int, int]:

    old_lines = old_src.splitlines()
    new_lines = new_src.splitlines()
    changed: set[int] = set()

    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile="a", tofile="b", lineterm=""
    )

    old_line_no = 0
    new_line_no = 0

    for line in diff:
        if line.startswith("@@"):
            try:
                header = line.split(" ")
                old_range = header[1]
                old_start = int(old_range.split(",")[0][1:])
                old_line_no = old_start - 1
                new_range = header[2]
                new_start = int(new_range.split(",")[0][1:])
                new_line_no = new_start - 1
            except Exception:
                continue
        elif line.startswith("-"):
            old_line_no += 1
            changed.add(old_line_no)
        elif line.startswith("+"):
            new_line_no += 1
        else:
            old_line_no += 1
            new_line_no += 1

    if not changed:

        return 1, len(old_lines) if old_lines else 1
    return min(changed), max(changed)


def _apply_diff_window(records: Sequence[Dict[str, Any]], window_radius: int = 20) -> List[Dict[str, Any]]:

    from collections import defaultdict


    grouped: Dict[Tuple[str, str, str], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        bug_id = rec.get("bug_id")
        file_path = rec.get("file_path")
        func_name = rec.get("function_name")
        label_raw = rec.get("label", None)
        if bug_id is None or file_path is None or func_name is None:
            continue
        try:
            label = int(label_raw)
        except Exception:
            continue
        key = (str(bug_id), str(file_path), str(func_name))

        if label not in grouped[key]:
            grouped[key][label] = rec


    windowed_code: Dict[Tuple[str, str, str], Dict[int, str]] = {}
    iter_pairs = [
        (key, pair) for key, pair in grouped.items() if 0 in pair and 1 in pair
    ]
    if not iter_pairs:
        return list(records)

    for key, pair in tqdm(iter_pairs, desc="Building diff windows", unit="function"):
        buggy_rec = pair[1]
        fixed_rec = pair[0]
        buggy_src = buggy_rec.get("code", "")
        fixed_src = fixed_rec.get("code", "")


        buggy_start, buggy_end = _compute_changed_lines(buggy_src, fixed_src)
        buggy_lines = buggy_src.splitlines()
        if not buggy_lines:
            buggy_window = buggy_src
        else:
            start = max(buggy_start - window_radius, 1)
            end = min(buggy_end + window_radius, len(buggy_lines))
            buggy_window = "\n".join(buggy_lines[start - 1 : end])


        fixed_start, fixed_end = _compute_changed_lines(fixed_src, buggy_src)
        fixed_lines = fixed_src.splitlines()
        if not fixed_lines:
            fixed_window = fixed_src
        else:
            start_f = max(fixed_start - window_radius, 1)
            end_f = min(fixed_end + window_radius, len(fixed_lines))
            fixed_window = "\n".join(fixed_lines[start_f - 1 : end_f])

        windowed_code[key] = {1: buggy_window, 0: fixed_window}


    new_records: List[Dict[str, Any]] = []
    for rec in records:
        bug_id = rec.get("bug_id")
        file_path = rec.get("file_path")
        func_name = rec.get("function_name")
        if bug_id is None or file_path is None or func_name is None:
            new_records.append(rec)
            continue
        key = (str(bug_id), str(file_path), str(func_name))
        code_map = windowed_code.get(key)
        if not code_map:
            new_records.append(rec)
            continue
        try:
            label = int(rec.get("label"))
        except Exception:
            new_records.append(rec)
            continue
        if label not in code_map:
            new_records.append(rec)
            continue
        rec_copy = dict(rec)
        rec_copy["code"] = code_map[label]
        new_records.append(rec_copy)

    return new_records


def _split_records(
    records: Sequence[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> (List[dict], List[dict], List[dict]):

    total = train_ratio + val_ratio + test_ratio
    assert total > 0, "The sum of split ratios must be greater than 0."

    train_r = train_ratio / total
    val_r = val_ratio / total
    test_r = test_ratio / total

    labels = [rec["label"] for rec in records]
    train_records, temp_records = train_test_split(
        records,
        test_size=val_r + test_r,
        stratify=labels,
        random_state=seed,
    )
    if val_r == 0 and test_r == 0:
        return list(train_records), [], []

    if val_r == 0:
        return list(train_records), [], list(temp_records)
    if test_r == 0:
        return list(train_records), list(temp_records), []

    temp_labels = [rec["label"] for rec in temp_records]
    val_size = val_r / (val_r + test_r)
    val_records, test_records = train_test_split(
        temp_records,
        test_size=1 - val_size,
        stratify=temp_labels,
        random_state=seed,
    )
    return list(train_records), list(val_records), list(test_records)


def _group_split_records(
    records: Sequence[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_key: str,
) -> (List[dict], List[dict], List[dict]):

    total = train_ratio + val_ratio + test_ratio
    assert total > 0, "The sum of split ratios must be greater than 0."
    train_r = train_ratio / total
    val_r = val_ratio / total
    test_r = test_ratio / total
    from collections import defaultdict

    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        key = str(rec.get(group_key, "unknown"))
        groups[key].append(rec)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    train_cut = int(n * train_r)
    val_cut = train_cut + int(n * val_r)


    train_ids = group_ids[:train_cut]
    val_ids = group_ids[train_cut:val_cut]
    if abs(test_ratio) < 1e-6:
        train_ids = train_ids + group_ids[val_cut:]
        test_ids = []
    else:
        test_ids = group_ids[val_cut:]

    def _collect(ids: Sequence[str]) -> List[dict]:
        out: List[dict] = []
        for gid in ids:
            out.extend(groups[gid])
        return out

    return _collect(train_ids), _collect(val_ids), _collect(test_ids)


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


def build_datasets(
    records: List[dict],
    tokenizer,
    *,
    max_length: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_key: Optional[str] = None,
    max_funcs_per_class: Optional[int] = None,
    caps_map: Optional[Dict[str, List[int]]] = None,
    caps_dataset_name: Optional[str] = None,
    caps_apply_splits: Optional[Sequence[str]] = None,
) -> DatasetBundle:

    if group_key:
        train_records, val_records, test_records = _group_split_records(
            records, train_ratio, val_ratio, test_ratio, seed, group_key
        )
    else:
        train_records, val_records, test_records = _split_records(
            records, train_ratio, val_ratio, test_ratio, seed
        )
    is_bag_level = bool(train_records and "functions" in train_records[0])
    if is_bag_level and caps_map:
        apply_set = set(caps_apply_splits or ("train", "val"))
        dataset_name = caps_dataset_name or "dataset"
        if "train" in apply_set:
            train_records, _ = apply_caps_to_records(
                train_records, dataset_name=dataset_name, caps_map=caps_map
            )
        if "val" in apply_set:
            val_records, _ = apply_caps_to_records(
                val_records, dataset_name=dataset_name, caps_map=caps_map
            )
        if "test" in apply_set:
            test_records, _ = apply_caps_to_records(
                test_records, dataset_name=dataset_name, caps_map=caps_map
            )
    vocab = build_ast_vocab(train_records)

    if is_bag_level:
        dataset_cls = ClassMilDataset
        dataset_kwargs = {"max_funcs": max_funcs_per_class}
    else:
        dataset_cls = FunctionGraphDataset
        dataset_kwargs = {}

    train_dataset = dataset_cls(
        train_records,
        tokenizer,
        max_length,
        vocab,
        split_name="train",
        **dataset_kwargs,
    )
    val_dataset = (
        dataset_cls(
            val_records,
            tokenizer,
            max_length,
            vocab,
            split_name="val",
            **dataset_kwargs,
        )
        if val_records
        else None
    )
    test_dataset = (
        dataset_cls(
            test_records,
            tokenizer,
            max_length,
            vocab,
            split_name="test",
            **dataset_kwargs,
        )
        if test_records
        else None
    )

    def _count(items: Sequence[dict]) -> Dict[str, int]:
        from collections import Counter

        counter = Counter(rec["label"] for rec in items)
        return {str(k): int(v) for k, v in counter.items()}

    stats = {
        "train": _count(train_records),
        "val": _count(val_records) if val_records else {},
        "test": _count(test_records) if test_records else {},
    }

    return DatasetBundle(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        node_vocab=vocab,
        label_stats=stats,
        bag_level=is_bag_level,
    )


def collate_graph_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    global AST_SEQ_LEN

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
    if AST_SEQ_LEN:
        seqs = []
        masks = []
        for sample in samples:
            ids = sample["node_ids"]
            trimmed = ids[:AST_SEQ_LEN]
            pad_len = AST_SEQ_LEN - trimmed.size(0)
            if pad_len > 0:
                trimmed = torch.cat(
                    [trimmed, torch.zeros(pad_len, dtype=ids.dtype)], dim=0
                )
            seqs.append(trimmed)
            mask = torch.zeros(AST_SEQ_LEN, dtype=torch.long)
            mask[: min(len(ids), AST_SEQ_LEN)] = 1
            masks.append(mask)
        batch["ast_seq"] = torch.stack(seqs, dim=0)
        batch["ast_mask"] = torch.stack(masks, dim=0)
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
    global AST_SEQ_LEN
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
    if AST_SEQ_LEN:
        seqs = []
        masks = []
        for graph in graphs:
            ids = graph["node_ids"]
            trimmed = ids[:AST_SEQ_LEN]
            pad_len = AST_SEQ_LEN - trimmed.size(0)
            if pad_len > 0:
                trimmed = torch.cat(
                    [trimmed, torch.zeros(pad_len, dtype=ids.dtype)], dim=0
                )
            seqs.append(trimmed)
            mask = torch.zeros(AST_SEQ_LEN, dtype=torch.long)
            mask[: min(len(ids), AST_SEQ_LEN)] = 1
            masks.append(mask)
        batch["ast_seq"] = (
            torch.stack(seqs, dim=0)
            if seqs
            else torch.empty((0, AST_SEQ_LEN), dtype=torch.long)
        )
        batch["ast_mask"] = (
            torch.stack(masks, dim=0)
            if masks
            else torch.empty((0, AST_SEQ_LEN), dtype=torch.long)
        )

    return batch


def set_ast_seq_len(length: Optional[int]) -> None:

    global AST_SEQ_LEN
    AST_SEQ_LEN = length if length and length > 0 else None
