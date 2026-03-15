"""TF-IDF + logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hamnet.utils import ensure_dir, save_json, set_seed
from hamnet.metrics import compute_classification_metrics
from hamnet.data import (
    apply_caps_to_records,
    infer_dataset_name,
    load_caps_map,
    load_records,
)
from hamnet.thresholds import search_best_threshold


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run the TF-IDF + logistic-regression baseline."
    )
    parser.add_argument(
        "--data-path", type=Path, required=True, help="Input dataset JSONL path."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory where metrics.json will be written.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram length for TF-IDF.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Pooling method used when a record contains multiple functions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Fixed split JSON file containing train/val/test indices.",
    )
    parser.add_argument(
        "--caps-file",
        type=Path,
        default=None,
        help="Optional caps JSON path keyed by bag_id.",
    )
    parser.add_argument(
        "--caps-dataset-name",
        type=str,
        default=None,
        help="Optional dataset name override used when computing bag ids.",
    )
    parser.add_argument(
        "--caps-apply-splits",
        type=str,
        default="train,val",
        help="Comma-separated split names where caps should be applied.",
    )
    return parser.parse_args()


def load_texts_labels(records: List[dict], pooling: str) -> Tuple[List[str], List[int]]:

    texts: List[str] = []
    labels: List[int] = []
    for rec in tqdm(records, desc="load_records", unit="record"):
        label = int(rec.get("label", 0))
        functions = rec.get("functions")
        if isinstance(functions, list) and functions:

            code = "\n".join(str(f.get("code", "")) for f in functions)
        else:
            code = str(rec.get("code", ""))
        texts.append(code)
        labels.append(label)
    return texts, labels


def load_instance_texts_labels(records: List[dict]) -> Tuple[List[List[str]], List[int]]:

    func_texts: List[List[str]] = []
    labels: List[int] = []
    for rec in tqdm(records, desc="load_instances", unit="record"):
        label = int(rec.get("label", 0))
        functions = rec.get("functions")
        if isinstance(functions, list) and functions:
            codes = [str(f.get("code", "")) for f in functions]
        else:
            codes = [str(rec.get("code", ""))]
        func_texts.append(codes)
        labels.append(label)
    return func_texts, labels


def pool_bag_vectors(vectors: List[np.ndarray], pooling: str) -> np.ndarray:

    if not vectors:
        raise ValueError("Cannot pool an empty bag.")
    stacked = np.vstack([v.toarray() for v in vectors])
    if pooling == "mean":
        return stacked.mean(axis=0)
    return stacked.max(axis=0)


def main() -> None:

    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    records = load_records(args.data_path, max_samples=None, seed=args.seed)
    is_mil = bool(records and isinstance(records[0], dict) and "functions" in records[0])


    data = json.loads(args.split_file.read_text(encoding="utf-8"))
    splits = data.get("splits") or {}
    train_idx = splits.get("train", {}).get("indices", [])
    val_idx = splits.get("val", {}).get("indices", [])
    test_idx = splits.get("test", {}).get("indices", [])
    train_records = [records[i] for i in train_idx if i < len(records)]
    val_records = [records[i] for i in val_idx if i < len(records)]
    test_records = [records[i] for i in test_idx if i < len(records)]
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
    if is_mil and caps_map:
        if "train" in caps_apply_splits:
            train_records, train_caps_stats = apply_caps_to_records(
                train_records, dataset_name=caps_dataset_name, caps_map=caps_map
            )
            print("[INFO] Caps train:", train_caps_stats)
        if "val" in caps_apply_splits:
            val_records, val_caps_stats = apply_caps_to_records(
                val_records, dataset_name=caps_dataset_name, caps_map=caps_map
            )
            print("[INFO] Caps val:", val_caps_stats)
        if "test" in caps_apply_splits:
            test_records, test_caps_stats = apply_caps_to_records(
                test_records, dataset_name=caps_dataset_name, caps_map=caps_map
            )
            print("[INFO] Caps test:", test_caps_stats)
    X_train, y_train = load_texts_labels(train_records, args.pooling)
    X_val, y_val = load_texts_labels(val_records, args.pooling)
    X_test, y_test = load_texts_labels(test_records, args.pooling)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_test = np.asarray(y_test, dtype=int)


    def build_vectorizer(min_df: int, use_default_token: bool = False) -> TfidfVectorizer:
        kw = dict(
            ngram_range=(1, args.ngram_max),
            max_features=args.max_features,
            min_df=min_df,
        )
        if not use_default_token:
            kw["token_pattern"] = r"(?u)[A-Za-z_]\w+"
        return TfidfVectorizer(**kw)

    try:
        vectorizer = build_vectorizer(args.min_df)
        X_train_tf = vectorizer.fit_transform(X_train)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            try:
                vectorizer = build_vectorizer(1)
                X_train_tf = vectorizer.fit_transform(X_train)
            except ValueError:

                vectorizer = build_vectorizer(1, use_default_token=True)
                X_train_tf = vectorizer.fit_transform(X_train)
        else:
            raise
    X_val_tf = vectorizer.transform(X_val)
    X_test_tf = vectorizer.transform(X_test)


    ft_train, _ = load_instance_texts_labels(train_records)
    ft_val, _ = load_instance_texts_labels(val_records)
    ft_test, _ = load_instance_texts_labels(test_records)

    if any(len(f) > 1 for f in ft_train):

        def pool_split(func_lists, tf_mat):
            pooled = []
            cursor = 0
            for funcs in func_lists:
                vecs = []
                for code in funcs:
                    v = vectorizer.transform([code])
                    vecs.append(v)
                pooled.append(pool_bag_vectors(vecs, args.pooling))
                cursor += 1
            return np.vstack(pooled)

        X_train_tf = pool_split(ft_train, X_train_tf)
        X_val_tf = pool_split(ft_val, X_val_tf)
        X_test_tf = pool_split(ft_test, X_test_tf)
    else:

        X_train_tf = X_train_tf.toarray()
        X_val_tf = X_val_tf.toarray()
        X_test_tf = X_test_tf.toarray()


    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )
    clf.fit(X_train_tf, y_train)


    val_probs = clf.predict_proba(X_val_tf)[:, 1]
    _, _, val_best = search_best_threshold(y_val, val_probs)
    best_thr = float(val_best["threshold"])


    test_probs = clf.predict_proba(X_test_tf)[:, 1]
    test_metrics = compute_classification_metrics(y_test, test_probs, threshold=best_thr)
    test_metrics["threshold"] = float(best_thr)

    save_json(
        {"val_best": {**val_best, "threshold": best_thr}, "test_metrics": test_metrics},
        args.output_dir / "metrics.json",
    )
    print("[INFO] Baseline evaluation finished. Results written to", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
