"""Official XES3G5M question-level protocol and response-free cold-item splits.

Importing this module never downloads data or opens the test CSV. Downloads are
an explicit CLI operation; ``prepare`` reads train/validation and embeddings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

BAIM_COMMIT = "a9501b5e44dafc0e81d2f17e717766a4106d0122"
EMBEDDING_URL = (f"https://raw.githubusercontent.com/sjin4861/BAIM/{BAIM_COMMIT}/"
                 "embedding/xes3g5m/polya_tensor_all_layers_mean_pca768.pt")
ARCHIVE_URL = ("https://drive.usercontent.google.com/download?"
               "id=1eFiIYyh5O2V90RA0brammGH6EpHvPDQe&export=download&confirm=t")
EXPECTED_ARCHIVE_BYTES = 373235093
EXPECTED_EMBEDDING_BYTES = 94029649
SEQUENCE_LENGTH = 200
MAX_CONCEPTS = 7
NUM_CONCEPTS = 865


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, destination: Path, expected_bytes: int) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != expected_bytes:
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "semantic-evidence-kt/0.1"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as out:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                out.write(block)
        if temporary.stat().st_size != expected_bytes:
            raise ValueError(f"Unexpected download length: {temporary}; retained for inspection")
        temporary.replace(destination)
    return {"url": url, "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def _question_directory(root: Path) -> Path:
    for candidate in [root / "XES3G5M" / "question_level", root / "question_level"]:
        if (candidate / "train_valid_sequences_quelevel.csv").exists():
            return candidate
    matches = list(root.glob("**/question_level/train_valid_sequences_quelevel.csv"))
    if len(matches) == 1:
        return matches[0].parent
    raise FileNotFoundError("Expected official XES3G5M/question_level data under " + str(root))


def download(root: str | Path) -> dict:
    """Explicit, idempotent download. Never invoked by training or ``prepare``."""
    root = Path(root).resolve()
    archive = root / "downloads" / "XES3G5M.tar.gz"
    manifest = {"baim_commit": BAIM_COMMIT,
                "dataset": _download(ARCHIVE_URL, archive, EXPECTED_ARCHIVE_BYTES),
                "embeddings": _download(EMBEDDING_URL, root / "baim_embeddings.pt",
                                        EXPECTED_EMBEDDING_BYTES)}
    try:
        _question_directory(root)
    except FileNotFoundError:
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream.getmembers():
                target = (root / member.name).resolve()
                if not target.is_relative_to(root) or member.issym() or member.islnk():
                    raise ValueError("Unsafe archive member: " + member.name)
            stream.extractall(root, filter="data")
        _question_directory(root)
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",")]


def _read_sequences(path: Path, n_items: int) -> dict[str, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            q = _ints(row["questions"])
            r = _ints(row["responses"])
            c = row["concepts"].split(",")
            s = _ints(row.get("selectmasks", ",".join("1" for _ in q)))
            t = _ints(row["timestamps"]) if row.get("timestamps") else [0] * len(q)
            if not (len(q) == len(r) == len(c) == len(s) == len(t)):
                raise ValueError(f"Misaligned sequence columns at row {len(rows)} in {path}")
            if len(q) > SEQUENCE_LENGTH:
                raise ValueError("Use the official length-200 segmented files; no implicit truncation")
            length = len(q)
            padding = SEQUENCE_LENGTH - length
            q = q + [-1] * padding
            r = r + [-1] * padding
            c = c + ["-1"] * padding
            s = s + [-1] * padding
            t = t + [0] * padding
            valid = np.asarray(r) != -1
            if np.any(valid[1:] & ~valid[:-1]):
                raise ValueError("Only right padding is supported")
            if any((qid < 0 or qid >= n_items) for qid, ok in zip(q, valid) if ok):
                raise ValueError("Question IDs do not match the embedding row indices")
            if any(response not in (0, 1) for response, ok in zip(r, valid) if ok):
                raise ValueError("Responses must be binary")
            concepts = []
            for entry, ok in zip(c, valid):
                kc = [int(x) for x in entry.split("_")] if ok else []
                kc = [x for x in kc if x >= 0]
                if len(kc) > MAX_CONCEPTS or any(x >= NUM_CONCEPTS for x in kc):
                    raise ValueError("Unexpected concept IDs or count")
                concepts.append(kc + [-1] * (MAX_CONCEPTS - len(kc)))
            scoremask = (np.asarray(s) != -1) & valid
            scoremask[0] = False  # Official KTQueDataset evaluates shifted targets.
            rows.append((q, r, concepts, t, valid, scoremask,
                         int(row.get("fold", -1)), str(row.get("uid", len(rows)))))
    if not rows:
        raise ValueError("Empty dataset " + str(path))
    fields = ["qids", "responses", "concepts", "timestamps", "valid", "scoremask", "folds", "uids"]
    types = [np.int32, np.int8, np.int16, np.int64, bool, bool, np.int8, str]
    result = {name: np.asarray([row[i] for row in rows], dtype=dtype)
              for i, (name, dtype) in enumerate(zip(fields, types))}
    # Pad IDs are safe for direct embedding lookup and cannot be scored.
    result["qids"][~result["valid"]] = 0
    result["responses"][~result["valid"]] = 0
    return result


class SequenceDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], row_indices: np.ndarray | None = None):
        self.arrays = arrays
        self.row_indices = np.arange(len(arrays["qids"])) if row_indices is None else row_indices

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.row_indices[index])
        result = {key: torch.as_tensor(self.arrays[key][row], dtype=(torch.bool if key in
                  {"valid", "scoremask"} else torch.long)) for key in
                  ["qids", "responses", "concepts", "timestamps", "valid", "scoremask"]}
        result["sequence_row"] = torch.tensor(row, dtype=torch.long)
        return result


def _summary(dataset: SequenceDataset) -> dict:
    a, index = dataset.arrays, dataset.row_indices
    return {"sequences": len(dataset), "students": len(np.unique(a["uids"][index])),
            "valid_events": int(a["valid"][index].sum()),
            "scored_events": int(a["scoremask"][index].sum())}


def _counts(dataset: SequenceDataset, n_items: int) -> torch.Tensor:
    q = dataset.arrays["qids"][dataset.row_indices]
    valid = dataset.arrays["valid"][dataset.row_indices]
    return torch.from_numpy(np.bincount(q[valid], minlength=n_items).astype(np.float32))


def _cold_partition(n_items: int, seed: int) -> dict:
    """Approved catalog-wide 70/15/15 split; independent of all responses."""
    rng = np.random.default_rng(seed)
    items = np.arange(n_items)
    rng.shuffle(items)
    n_train, n_val = int(0.70 * n_items), int(0.15 * n_items)
    warm = sorted(items[:n_train].tolist())
    val = sorted(items[n_train:n_train + n_val].tolist())
    test = sorted(items[n_train + n_val:].tolist())
    payload = {"seed": seed, "algorithm": "all-catalog-default_rng-shuffle-70-15-15-v1",
               "cold_val_items": val, "cold_test_items": test, "warm_items": warm,
               "item_counts": {"warm": len(warm), "cold_val": len(val), "cold_test": len(test)}}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def _warm_only(arrays: dict, rows: np.ndarray, warm_items: list[int]) -> SequenceDataset:
    warm = np.isin(arrays["qids"], warm_items) & arrays["valid"]
    keep_rows = rows[warm[rows].sum(-1) >= 2]
    out = {key: value[keep_rows].copy() for key, value in arrays.items()}
    for new_row, old_row in enumerate(keep_rows):
        selected = np.flatnonzero(warm[old_row])
        size = len(selected)
        for key in ["qids", "responses", "concepts", "timestamps", "valid", "scoremask"]:
            out[key][new_row] = -1 if key == "concepts" else 0
            out[key][new_row, :size] = arrays[key][old_row, selected]
        out["scoremask"][new_row, 0] = False
    if not len(keep_rows):
        raise ValueError("No warm training sequences after cold-item filtering")
    return SequenceDataset(out)


class ColdTargetDataset(Dataset):
    """One query per sample; its context includes strictly past warm responses only.

    Cold targets in a row never become historical inputs for another target.
    The original official segment boundaries are retained.
    """
    def __init__(self, arrays: dict, row_indices: np.ndarray, warm_items: list[int],
                 target_items: list[int]):
        self.arrays = arrays
        self.warm = np.isin(arrays["qids"], warm_items) & arrays["valid"]
        # Cold protocol intentionally includes a cold target at position zero.
        target_mask = np.isin(arrays["qids"][row_indices], target_items) & arrays["valid"][row_indices]
        local_rows, positions = np.nonzero(target_mask)
        self.targets = np.stack([row_indices[local_rows], positions], axis=1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        row, target = self.targets[index]
        history = np.flatnonzero(self.warm[row, :target])[-(SEQUENCE_LENGTH - 1):]
        selected = np.concatenate([history, [target]])
        count = len(selected)
        output = {}
        for key in ["qids", "responses", "concepts", "timestamps"]:
            shape = (SEQUENCE_LENGTH, MAX_CONCEPTS) if key == "concepts" else (SEQUENCE_LENGTH,)
            array = np.full(shape, -1 if key == "concepts" else 0, dtype=np.int64)
            array[:count] = self.arrays[key][row, selected]
            output[key] = torch.from_numpy(array)
        output["valid"] = torch.arange(SEQUENCE_LENGTH) < count
        output["scoremask"] = torch.arange(SEQUENCE_LENGTH) == count - 1
        output["sequence_row"] = torch.tensor(int(row), dtype=torch.long)
        output["target_position"] = torch.tensor(int(target), dtype=torch.long)
        return output


def prepare(data_root: str | Path, fold: int = 0, protocol: str = "standard",
            cold_seed: int = 42) -> dict:
    if fold not in range(5) or protocol not in {"standard", "cold"}:
        raise ValueError("fold must be 0..4; protocol must be standard or cold")
    root = Path(data_root).resolve()
    question_dir = _question_directory(root)
    embedding_path = root / "baim_embeddings.pt"
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True).float()
    if embeddings.ndim != 3 or embeddings.shape[1] != 4:
        raise ValueError("BAIM embedding cache must have shape [questions, 4, dimension]")
    train_path = question_dir / "train_valid_sequences_quelevel.csv"
    arrays = _read_sequences(train_path, len(embeddings))
    if set(np.unique(arrays["folds"])) != set(range(5)):
        raise ValueError("Training CSV must contain official folds 0,1,2,3,4")
    train_rows = np.flatnonzero(arrays["folds"] != fold)
    val_rows = np.flatnonzero(arrays["folds"] == fold)
    if set(arrays["uids"][train_rows]) & set(arrays["uids"][val_rows]):
        raise ValueError("Student IDs overlap across training and validation folds")
    train, val = SequenceDataset(arrays, train_rows), SequenceDataset(arrays, val_rows)
    partition = None
    metadata = {"dataset": "XES3G5M", "representation": "question-level",
                "protocol": protocol, "fold": fold, "sequence_length": SEQUENCE_LENGTH,
                "baim_commit": BAIM_COMMIT, "num_items": len(embeddings),
                "num_concepts": NUM_CONCEPTS, "embedding_shape": list(embeddings.shape),
                "embedding_semantics": "four-stage Qwen3-VL-32B author cache; all-item PCA768",
                "sourcehash": {"train_valid_csv": sha256(train_path),
                               "baim_embeddings": sha256(embedding_path)},
                "test_file": "test_sequences_quelevel.csv", "test_loaded": False,
                "metric": "question-level ROC-AUC; all selected events pooled",
                "mask_rule": "valid and selectmasks != -1; position zero excluded in standard",
                "paper_comparison": "single fold is not five-fold mean; causal-fixed public implementation"}
    if protocol == "cold":
        partition = _cold_partition(len(embeddings), cold_seed)
        train = _warm_only(arrays, train_rows, partition["warm_items"])
        warm_history_items = torch.nonzero(_counts(train, len(embeddings)) > 0).flatten().tolist()
        val = ColdTargetDataset(arrays, val_rows, warm_history_items, partition["cold_val_items"])
        metadata["cold_partition"] = partition
        metadata["coldsplit_hash"] = partition["sha256"]
        metadata["cold_warm_history_items"] = warm_history_items
        metadata["cold_definition"] = ("preknown item catalog and all-item text PCA; zero held-out-item "
            "training responses; each query uses only prior warm responses in its original segment; "
            "not strict inductive unseen-text evaluation")
        if not len(val):
            raise ValueError("Cold validation contains no targets")
    metadata["train"] = _summary(train)
    metadata["val"] = (_summary(val) if isinstance(val, SequenceDataset) else
                       {"sequences": len(val), "scored_events": len(val), "query_windows": len(val)})
    metadata["mask_counts"] = {"train": metadata["train"]["scored_events"],
                               "val": metadata["val"]["scored_events"]}
    item_counts = _counts(train, len(embeddings))
    metadata["padding_qid"] = 0 if item_counts[0] > 0 else int(item_counts.argmax())
    test_cache = None

    def test_factory() -> Dataset:
        nonlocal test_cache
        if test_cache is None:
            test_path = question_dir / "test_sequences_quelevel.csv"
            test_arrays = _read_sequences(test_path, len(embeddings))
            if set(test_arrays["uids"]) & set(arrays["uids"]):
                raise ValueError("Student IDs overlap between train/validation and official test")
            if partition is None:
                test_cache = SequenceDataset(test_arrays)
                metadata["test"] = _summary(test_cache)
            else:
                test_cache = ColdTargetDataset(test_arrays, np.arange(len(test_arrays["qids"])),
                    warm_history_items, partition["cold_test_items"])
                metadata["test"] = {"sequences": len(test_cache), "scored_events": len(test_cache)}
            metadata["sourcehash"]["test_csv"] = sha256(test_path)
            metadata["mask_counts"]["test"] = metadata["test"]["scored_events"]
            metadata["test_loaded"] = True
        return test_cache

    return {"train": train, "val": val, "test_factory": test_factory,
            "embeddings": embeddings, "item_counts": item_counts, "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "inspect"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--protocol", choices=["standard", "cold"], default="standard")
    args = parser.parse_args()
    result = download(args.root) if args.command == "download" else prepare(args.root, args.fold, args.protocol)["metadata"]
    print(json.dumps(result, indent=2))
