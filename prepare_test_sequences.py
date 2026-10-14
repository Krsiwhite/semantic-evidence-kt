"""Materialize the missing main-test CSV using the pinned BAIM function.

This is deterministic data preparation, with no model import, fitting, split,
ID remapping, or inference. The official raw test remains unchanged.
Requires pandas and numpy; the training environment need not install pandas
when preparation is run with a separate Python interpreter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

COMMIT = "a9501b5e44dafc0e81d2f17e717766a4106d0122"
BASE_URL = (f"https://raw.githubusercontent.com/sjin4861/BAIM/{COMMIT}/"
            "src/pykt-toolkit/pykt/preprocess/")
SOURCES = {
    "split_datasets.py": "83d6047474ec01d32750c48c8e506883eb6bf233e81aa9e895807f565e60a2b7",
    "split_datasets_que.py": "1f5e7a4fdd3e0db26aa69bf5dc63296b615fb7174e4435ef046eef63d9941101",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def official_function(root):
    cache = root / "preprocess_sources"
    cache.mkdir(parents=True, exist_ok=True)
    namespace = {"np": np, "pd": pd}
    for filename, expected in SOURCES.items():
        path = cache / filename
        if not path.exists():
            with urllib.request.urlopen(BASE_URL + filename, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Upstream preprocessing checksum mismatch")
            path.write_bytes(payload)
        if digest(path) != expected:
            raise ValueError(f"Cached upstream source changed: {filename}")
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        requested = ({"ALL_KEYS", "ONE_KEYS", "save_dcur"}
                     if filename == "split_datasets.py" else {"generate_sequences"})
        selected, found = [], set()
        for node in parsed.body:
            names = ({node.name} if isinstance(node, ast.FunctionDef) else
                     {t.id for t in node.targets if isinstance(t, ast.Name)}
                     if isinstance(node, ast.Assign) else set())
            if names & requested:
                selected.append(node)
                found.update(names)
        if found != requested:
            raise ValueError("Unexpected pinned preprocessing definitions")
        # Only the reviewed, checksum-bound function and its two constants are
        # loaded. Upstream module imports and split/remap routines never run.
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["generate_sequences"]


def main(root):
    root = Path(root).resolve()
    directory = root / "XES3G5M" / "question_level"
    original = directory / "test_quelevel.csv"
    destination = directory / "test_sequences_quelevel.csv"
    manifest_path = root / "test_sequences_manifest.json"
    source_hash = digest(original)
    if destination.exists():
        if not manifest_path.exists():
            raise FileExistsError("Existing test CSV has no preparation manifest; will not overwrite")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["input_sha256"] != source_hash or
                manifest["output_sha256"] != digest(destination)):
            raise ValueError("Existing test CSV or source differs from its manifest")
        print(json.dumps({"status": "already_prepared", **manifest}, indent=2))
        return
    source = pd.read_csv(original)
    if "selectmasks" in source.columns or set(source["fold"]) != {-1}:
        raise ValueError("Expected the official raw fixed-test rows")
    generate_sequences = official_function(root)
    result = generate_sequences(source, list(source.columns), min_seq_len=3, maxlen=200, pad_val=-1)
    raw_lengths = source["responses"].str.count(",") + 1
    masks = result["selectmasks"].str.split(",")
    lengths = masks.map(lambda row: row.count("1"))
    for column in set(source.columns) - {"fold", "uid"}:
        if not (result[column].str.count(",") == 199).all():
            raise ValueError(f"Generated sequence column is not length 200: {column}")
    counts = {"raw_students": len(source), "raw_events": int(raw_lengths.sum()),
              "segments": len(result), "valid_events": int(lengths.sum()),
              "dropped_short_tail_events": int(raw_lengths.sum() - lengths.sum()),
              "standard_scored_targets": int((lengths - 1).sum()),
              "padding_positions": int(len(result) * 200 - lengths.sum())}
    expected = {"raw_students": 3613, "raw_events": 1102810, "segments": 7711,
                "valid_events": 1102701, "dropped_short_tail_events": 109,
                "standard_scored_targets": 1094990, "padding_positions": 439499}
    if counts != expected:
        raise ValueError(f"Official asset differs from the independently audited counts: {counts}")
    temporary = destination.with_suffix(".csv.part")
    result.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(destination)
    manifest = {"baim_commit": COMMIT, "method": "unchanged upstream generate_sequences",
                "parameters": {"min_seq_len": 3, "maxlen": 200, "pad_val": -1},
                "input": original.relative_to(root).as_posix(), "input_sha256": source_hash,
                "output": destination.relative_to(root).as_posix(),
                "output_sha256": digest(destination), "source_sha256": SOURCES,
                "source_urls": {name: BASE_URL + name for name in SOURCES},
                "counts": counts, "pandas_version": pd.__version__, "numpy_version": np.__version__,
                "split_or_id_remapping_performed": False, "model_inference_performed": False,
                "reason": "The official downloaded archive omits the main segmented test CSV."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    main(parser.parse_args().data_root)
