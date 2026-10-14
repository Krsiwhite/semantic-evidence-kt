"""Budgeted, resumable, validation-selected SemanticEvidenceKT experiments.

No downloads occur here. The official fold-0 four-way ablation is frozen before
any test data are requested. Further folds use those same prespecified settings;
cold baseline/full start from scratch. Test inference is resumable by immutable
chunks for one selected checkpoint, never an opportunity to select another one.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn import functional as F


VARIANTS = ("baseline", "residual", "evidence", "full")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def log_event(path, event):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


class Budget:
    """Persist active runner wall time, conservatively including CPU work.

An unclosed prior lease is charged until restart; this may overcount downtime
after an abrupt kill but cannot silently grant a new GPU allowance on resume.
"""
    def __init__(self, path, hours):
        self.path = Path(path)
        self.limit = hours * 3600
        old = read_json(path) if self.path.exists() else {}
        if old and abs(old["limit_seconds"] - self.limit) > 1:
            raise ValueError("Resume with the original --budget-hours; allowances are not reset")
        self.base = old.get("charged_seconds", 0.0)
        if old.get("lease_open"):
            self.base += max(0, time.time() - old["updated_unix"])
        self.started = time.monotonic()
        self.last_flush = 0.0
        self.flush()

    def elapsed(self):
        return self.base + time.monotonic() - self.started

    def remaining(self, deadline=None):
        return min(self.limit, self.limit if deadline is None else deadline) - self.elapsed()

    def flush(self, closed=False):
        atomic_json(self.path, {"limit_seconds": self.limit, "charged_seconds": self.elapsed(),
                    "updated_unix": time.time(), "lease_open": not closed,
                    "scope": "active_runner_wall_time_including_cpu_and_inference"})
        self.last_flush = time.monotonic()

    def heartbeat(self):
        if time.monotonic() - self.last_flush >= 20:
            self.flush()


class StatefulBatchSampler(Sampler):
    """num_workers=0 prevents prefetched, unconsumed batches moving the cursor."""
    def __init__(self, length, batch_size, seed):
        self.length, self.batch_size = int(length), int(batch_size)
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor, self.permutation = 0, 0, None
        self.start_epoch(0)

    def start_epoch(self, epoch):
        self.epoch, self.cursor = epoch, 0
        self.permutation = torch.randperm(self.length, generator=self.generator)

    def __iter__(self):
        while self.cursor < self.length:
            stop = min(self.cursor + self.batch_size, self.length)
            indices = self.permutation[self.cursor:stop].tolist()
            self.cursor = stop
            yield indices

    def __len__(self):
        return math.ceil((self.length - self.cursor) / self.batch_size)

    def state_dict(self):
        return {"length": self.length, "batch_size": self.batch_size, "epoch": self.epoch,
                "cursor": self.cursor, "permutation": self.permutation,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if (state["length"], state["batch_size"]) != (self.length, self.batch_size):
            raise ValueError("Resume dataset length or micro-batch size changed")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.permutation = state["permutation"].cpu()
        self.generator.set_state(state["generator"].cpu())


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def scores(batch):
    mask = batch["scoremask"].bool() & batch["valid"].bool()
    return mask, batch["responses"][mask].float()


def normalize_accumulated_gradients(model, target_count):
    """Turn summed target losses into one effective-batch target mean.

Each micro-batch contributes BCE_sum + target_count * auxiliary_mean.
Dividing once here is algebraically identical to dividing every contribution
by the effective batch's total scored targets before backpropagation.
"""
    if target_count <= 0:
        raise ValueError("Cannot normalize a group without scored targets")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(target_count)


def metrics(probability, labels):
    p, y = np.asarray(probability, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if len(y) == 0 or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Invalid evaluation predictions")
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels or probabilities")
    positives = int(y.sum())
    if positives in (0, len(y)):
        raise ValueError("AUC requires both response classes")
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    stops = np.r_[starts[1:], len(y)]
    positives_by_tie = np.add.reduceat(sorted_y, starts)
    rank_sum = np.sum(positives_by_tie * ((starts + 1 + stops) / 2))
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * (len(y) - positives))
    clipped = p.clip(1e-7, 1 - 1e-7)
    bin_ids = np.minimum((p * 15).astype(int), 14)
    ece = sum(abs(p[bin_ids == i].mean() - y[bin_ids == i].mean()) * np.mean(bin_ids == i)
              for i in range(15) if np.any(bin_ids == i))
    return {"auc": float(auc), "accuracy": float(np.mean((p >= .5) == y)),
            "nll": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
            "brier": float(np.square(p - y).mean()), "ece15": float(ece),
            "targets": len(y), "positives": positives}


def make_loader(dataset, args, bundle, sampler=None, generator=None):
    options = dict(num_workers=0, pin_memory=args.device.startswith("cuda"),
                   collate_fn=bundle.get("collate_fn"), generator=generator)
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, **options)


def target_identity(batch):
    """Scored (source sequence row, source/local target position, question ID)."""
    mask, labels = scores(batch)
    qids = batch["qids"]
    rows = batch["sequence_row"].reshape(-1, 1).expand_as(qids)
    if "target_position" in batch:
        positions = batch["target_position"].reshape(-1, 1).expand_as(qids)
    else:
        positions = torch.arange(qids.shape[1], device=qids.device)[None, :].expand_as(qids)
    identity = torch.stack((rows[mask], positions[mask], qids[mask]), dim=-1)
    return identity.cpu().numpy().astype(np.int64), labels.cpu().numpy().astype(np.int64)


def contribution_snapshot(batch, output, mask, identity, limit=128):
    """Persist a bounded explanation from the already computed test forward."""
    target_positions = mask.nonzero(as_tuple=False)[:limit]
    rows, positions = target_positions[:, 0], target_positions[:, 1]
    indices = output["contribution_indices"][rows, positions]
    valid = output["contribution_valid"][rows, positions].bool()
    contribution = output["contributions"][rows, positions]
    weights = output["contribution_weights"][rows, positions]
    history_rows = rows[:, None].expand_as(indices)
    history_qids = batch["qids"][history_rows, indices]
    history_responses = batch["responses"][history_rows, indices]
    history_base = output["base_probs"][history_rows, indices]
    base = output["base_probs"][rows, positions].double()
    final_logit = output["logits"][rows, positions].double()
    final_probability = output["probabilities"][rows, positions].double()
    contribution_sum = contribution.double().sum(-1)
    # The model exposes base probabilities, not the pre-sigmoid base logits.
    # Record clipping explicitly instead of treating sigmoid inversion as exact.
    clipped = (base < 1e-7) | (base > 1 - 1e-7)
    base_logit = torch.logit(base.clamp(1e-7, 1 - 1e-7))
    reconstructed_logit = base_logit + contribution_sum
    reconstructed_probability = reconstructed_logit.sigmoid()
    logit_error = (reconstructed_logit - final_logit).abs()
    probability_error = (reconstructed_probability - final_probability).abs()
    base_probability_error = ((final_logit - contribution_sum).sigmoid() - base).abs()

    def array(tensor):
        return tensor.detach().cpu().numpy()

    def maximum(tensor):
        return float(tensor.max().item()) if tensor.numel() else 0.0

    summary = {
        "scored_targets_saved": len(target_positions), "limit": limit,
        "source": "first_test_batch_same_forward_as_scored_predictions",
        "target_identity_columns": ["sequence_row", "target_position", "question_id"],
        "history_index_semantics": "zero_based_position_in_model_input_window; invalid=-1",
        "decomposition": "final_logit = logit(base_probability) + sum(signed_history_contributions)",
        "base_probability_clamp_for_inverse_sigmoid": [1e-7, 1 - 1e-7],
        "base_probabilities_clipped": int(clipped.sum().item()),
        "max_logit_reconstruction_error": maximum(logit_error),
        "max_final_probability_reconstruction_error": maximum(probability_error),
        "max_base_probability_reconstruction_error": maximum(base_probability_error),
        "rounding_note": "Inverse sigmoid uses stored base probabilities; saturation/rounding can affect logit error.",
        "real_student_uids_saved": False,
    }
    payload = {
        "audit_target_identity": identity[:limit],
        "audit_target_input_position": array(positions),
        "audit_base_probability": array(base),
        "audit_final_logit": array(final_logit),
        "audit_final_probability": array(final_probability),
        "audit_history_local_index": array(indices.masked_fill(~valid, -1)),
        "audit_history_question_id": array(history_qids.masked_fill(~valid, -1)),
        "audit_history_response": array(history_responses.masked_fill(~valid, -1)),
        "audit_history_base_probability": array(history_base.masked_fill(~valid, 0)),
        "audit_signed_contribution": array(contribution),
        "audit_history_weight": array(weights),
        "audit_history_valid": array(valid),
        "audit_contribution_sum": array(contribution_sum),
        "audit_positive_contribution_sum": array(contribution.double().clamp_min(0).sum(-1)),
        "audit_negative_contribution_sum": array(contribution.double().clamp_max(0).sum(-1)),
        "audit_reconstructed_logit": array(reconstructed_logit),
        "audit_reconstructed_probability": array(reconstructed_probability),
        "audit_logit_reconstruction_error": array(logit_error),
        "audit_probability_reconstruction_error": array(probability_error),
        "contribution_audit_summary_json": np.asarray(json.dumps(summary, allow_nan=False)),
    }
    return payload, summary


@torch.inference_mode()
def evaluate(model, loader, device, budget, deadline, chunk_dir=None,
             checkpoint_sha=None, test_binding_sha=None):
    model.eval()
    prediction_parts, label_parts = [], []
    valid_count, scored_count = 0, 0
    mean_batch_seconds = 1.0
    contribution_audit = None
    for index, batch in enumerate(loader):
        budget.heartbeat()
        chunk = None if chunk_dir is None else Path(chunk_dir) / f"{index:07d}.npz"
        capture_contributions = (chunk is not None and index == 0
                                 and getattr(model, "variant", None) in {"evidence", "full"})
        if chunk is not None:
            identity, current_labels = target_identity(batch)
        if chunk is not None and chunk.exists():
            with np.load(chunk, allow_pickle=False) as saved:
                if (str(saved["checkpoint_sha"]) != checkpoint_sha
                        or str(saved["test_binding_sha"]) != test_binding_sha
                        or not np.array_equal(saved["target_identity"], identity)
                        or not np.array_equal(saved["labels"], current_labels)):
                    raise ValueError("Saved test chunk checkpoint, source/order binding or targets changed")
                p, y = saved["probabilities"].copy(), saved["labels"].copy()
                valid_count += int(saved["valid_count"])
                if capture_contributions:
                    contribution_audit = json.loads(str(saved["contribution_audit_summary_json"]))
        else:
            if budget.remaining(deadline) < max(30, 2 * mean_batch_seconds):
                return None
            started = time.monotonic()
            batch = move_batch(batch, device)
            mask, labels = scores(batch)
            output = (model(batch, return_contributions=True) if capture_contributions else model(batch))
            audit_payload = {}
            if capture_contributions:
                audit_payload, contribution_audit = contribution_snapshot(batch, output, mask, identity)
            p = output["probabilities"][mask].detach().float().cpu().numpy()
            y = labels.detach().cpu().numpy().astype(np.int64)
            batch_valid = int(batch["valid"].sum().item())
            valid_count += batch_valid
            if chunk is not None:
                temporary = chunk.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, probabilities=p, labels=y, valid_count=np.int64(batch_valid),
                             checkpoint_sha=np.asarray(checkpoint_sha),
                             test_binding_sha=np.asarray(test_binding_sha), target_identity=identity,
                             **audit_payload)
                os.replace(temporary, chunk)
            mean_batch_seconds = .8 * mean_batch_seconds + .2 * (time.monotonic() - started)
        prediction_parts.append(p)
        label_parts.append(y)
        scored_count += len(y)
    result = metrics(np.concatenate(prediction_parts), np.concatenate(label_parts))
    result["valid_positions"] = valid_count
    result["unscored_valid_positions"] = valid_count - scored_count
    if contribution_audit is not None:
        result["contribution_audit"] = contribution_audit
    return result


def model_for(bundle, variant, args):
    from models import SemanticEvidenceKT
    return SemanticEvidenceKT(bundle["embeddings"], bundle["item_counts"],
                              variant=variant, hidden_dim=args.hidden_dim).to(args.device)


def run_config(args, bundle, protocol, fold, variant):
    source = Path(__file__).resolve().parent
    source_files = [source / name for name in ("train.py", "models.py", "data.py")]
    vendor_files = sorted((source / "vendor").rglob("*.py"))
    if not vendor_files:
        raise ValueError("Vendored causal backbone source is missing")
    source_files.extend(vendor_files)
    return {"protocol": protocol, "fold": fold, "variant": variant, "seed": args.seed,
            "cold_seed": args.cold_seed, "hidden_dim": args.hidden_dim,
            "micro_batch_size": args.micro_batch_size, "effective_batch_size": args.effective_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr, "max_epochs": args.max_epochs, "patience": args.patience,
            "min_delta": args.min_delta, "gradient_clip": args.gradient_clip,
            "optimizer": "Adam", "scheduler": "constant_learning_rate",
            "effective_batch_loss": "sum(BCE_sum + scored_targets * microbatch_auxiliary_mean) / total_scored_targets",
            "auxiliary_accumulation_weight": "microbatch_scored_target_count",
            "best_update_rule": "validation_auc > saved_best_auc + min_delta; save and reset patience together",
            "dataset": bundle["metadata"], "causal_correction_all_variants": True,
            "paper_auc_reference_only": 0.8322,
            "source_sha256": {path.relative_to(source).as_posix(): sha256(path) for path in source_files}}


def train_one(args, bundle, run_dir, protocol, fold, variant, budget, deadline):
    run_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(args, bundle, protocol, fold, variant)
    config_sha = json_hash(config)
    config_path = run_dir / "config.json"
    if config_path.exists() and json_hash(read_json(config_path)) != config_sha:
        raise ValueError(f"Configuration/source/data drift on resume: {run_dir}")
    atomic_json(config_path, config)
    if (run_dir / "complete.json").exists():
        return read_json(run_dir / "complete.json")
    if budget.remaining(deadline) < 90:
        return None
    seed_all(args.seed)
    model = model_for(bundle, variant, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    sampler = StatefulBatchSampler(len(bundle["train"]), args.micro_batch_size, args.seed)
    loader_rng = torch.Generator().manual_seed(args.seed + 1000)
    loader = make_loader(bundle["train"], args, bundle, sampler, loader_rng)
    val_loader = make_loader(bundle["val"], args, bundle)
    state = {"epoch": 0, "global_step": 0, "best_auc": -1.0,
             "bad_epochs": 0, "epoch_loss_sum": 0.0, "epoch_targets": 0,
             "epoch_valid_positions": 0, "epoch_batches": 0,
             "last_validation_seconds": 30.0, "last_epoch_seconds": 0.0}
    latest = run_dir / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location=args.device, weights_only=False)
        if saved["config_sha256"] != config_sha:
            raise ValueError("Checkpoint configuration hash mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        sampler.load_state_dict(saved["sampler"])
        loader_rng.set_state(saved["loader_generator"].cpu())
        state = saved["training"]
        restore_rng(saved["rng"])

    def save_latest():
        atomic_torch(latest, {"config_sha256": config_sha, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "sampler": sampler.state_dict(), "loader_generator": loader_rng.get_state(),
                     "rng": rng_state(), "training": state})
        budget.flush()

    accumulation = args.effective_batch_size // args.micro_batch_size
    stop_reason = "max_epochs"
    while state["epoch"] < args.max_epochs:
        if (state["epoch"] > 0 and sampler.cursor == 0
                and budget.remaining(deadline) < 1.15 * state["last_epoch_seconds"] + 1.3 * state["last_validation_seconds"] + 45):
            stop_reason = "budget_before_next_epoch"
            break
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        accumulated_targets = 0
        interrupted = False
        for batch in loader:
            budget.heartbeat()
            if accumulated == 0 and budget.remaining(deadline) < max(45, 1.3 * state["last_validation_seconds"] + 20):
                # DataLoader has already requested this one batch. Rewind it
                # before checkpointing so a resume cannot silently skip it.
                sampler.cursor -= len(batch["qids"])
                interrupted = True
                break
            batch = move_batch(batch, args.device)
            mask, labels = scores(batch)
            state["epoch_valid_positions"] += int(batch["valid"].sum().item())
            state["epoch_batches"] += 1
            accumulated += 1
            count = labels.numel()
            if count:
                output = model(batch)
                primary = F.binary_cross_entropy_with_logits(output["logits"][mask], labels)
                auxiliary = output.get("auxiliary_loss", primary.new_zeros(()))
                loss = primary + auxiliary
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {run_dir.name}")
                (loss * count).backward()
                accumulated_targets += count
                state["epoch_loss_sum"] += float(loss.detach()) * count
                state["epoch_targets"] += count
            last_batch = sampler.cursor == sampler.length
            if accumulated == accumulation or last_batch:
                if accumulated_targets:
                    normalize_accumulated_gradients(model, accumulated_targets)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradient norm")
                    optimizer.step()
                    state["global_step"] += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                accumulated_targets = 0
                if state["global_step"] and state["global_step"] % args.checkpoint_steps == 0:
                    save_latest()
        if interrupted:
            save_latest()
            stop_reason = "budget_mid_epoch"
            break
        validation_started = time.monotonic()
        validation = evaluate(model, val_loader, args.device, budget, deadline)
        state["last_validation_seconds"] = time.monotonic() - validation_started
        state["last_epoch_seconds"] = validation_started - epoch_started
        if validation is None:
            save_latest()
            stop_reason = "budget_before_validation_complete"
            break
        epoch_number = state["epoch"] + 1
        if validation["auc"] > state["best_auc"] + args.min_delta:
            state["best_auc"] = validation["auc"]
            state["bad_epochs"] = 0
            atomic_torch(run_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch_number,
                         "validation": validation, "config_sha256": config_sha})
            atomic_json(run_dir / "best.json", {"epoch": epoch_number, "validation": validation,
                        "config_sha256": config_sha})
        else:
            state["bad_epochs"] += 1
        log_event(run_dir / "epochs.jsonl", {"event": "epoch", "run": run_dir.name, "epoch": epoch_number,
                  "loss": state["epoch_loss_sum"] / max(1, state["epoch_targets"]),
                  "training_scored_positions": state["epoch_targets"],
                  "training_valid_positions": state["epoch_valid_positions"],
                  "training_batches": state["epoch_batches"], "validation": validation,
                  "bad_epochs": state["bad_epochs"], "budget_seconds_used": budget.elapsed(),
                  "config_sha256": config_sha})
        state["epoch"] += 1
        state.update(epoch_loss_sum=0.0, epoch_targets=0, epoch_valid_positions=0, epoch_batches=0)
        scheduler.step()
        sampler.start_epoch(state["epoch"])
        save_latest()
        if state["bad_epochs"] >= args.patience:
            stop_reason = "early_stopping"
            break
    if not (run_dir / "best.pt").exists():
        atomic_json(run_dir / "pending.json", {"reason": stop_reason, "config_sha256": config_sha})
        return None
    best = read_json(run_dir / "best.json")
    result = {"status": "complete", "run": run_dir.name, "stop_reason": stop_reason,
              "completed_epochs": state["epoch"], "best": best,
              "checkpoint_sha256": sha256(run_dir / "best.pt"), "config_sha256": config_sha,
              "test_used_for_selection": False}
    atomic_json(run_dir / "complete.json", result)
    return result


def test_once(args, bundle, run_dir, budget, deadline):
    completed = read_json(run_dir / "complete.json")
    config = read_json(run_dir / "config.json")
    checksum = sha256(run_dir / "best.pt")
    if checksum != completed["checkpoint_sha256"]:
        raise ValueError("The validation-selected checkpoint changed before testing")
    if json_hash(config) != completed["config_sha256"]:
        raise ValueError("The selected run configuration changed before testing")
    if config["eval_batch_size"] != args.eval_batch_size:
        raise ValueError("Test chunk ordering requires the frozen evaluation batch size")
    if budget.remaining(deadline) < 60:
        return None
    # Called only after the initial four-way selection has been frozen. The
    # factory appends test provenance to bundle metadata; never write that
    # expanded metadata back to the training-only config.json.
    dataset = bundle["test_factory"]()
    test_metadata = bundle["metadata"]
    marker = {"checkpoint_sha256": checksum, "config_sha256": completed["config_sha256"],
              "checkpoint_selection": "validation_auc_with_minimum_improvement_only",
              "one_immutable_checkpoint": True,
              "test_csv_sha256": test_metadata["sourcehash"]["test_csv"],
              "coldsplit_sha256": test_metadata.get("coldsplit_hash"),
              "eval_batch_size": args.eval_batch_size, "dataset_length": len(dataset),
              "dataset_class": type(dataset).__name__,
              "target_order": "SequentialSampler; dataset index ascending; scored positions ascending"}
    binding_sha = json_hash(marker)
    marker_path = run_dir / "test-lock.json"
    if marker_path.exists() and read_json(marker_path) != marker:
        raise ValueError("An already locked test checkpoint, source file, split or ordering changed")
    if (run_dir / "test.json").exists():
        result = read_json(run_dir / "test.json")
        if result["test_binding_sha256"] != binding_sha:
            raise ValueError("Completed test result no longer matches its source/order binding")
        return result
    atomic_json(marker_path, marker)
    loader = make_loader(dataset, args, bundle)
    model = model_for(bundle, config["variant"], args)
    saved = torch.load(run_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(saved["model"])
    chunks = run_dir / "test_chunks"
    chunks.mkdir(exist_ok=True)
    result = evaluate(model, loader, args.device, budget, deadline, chunks, checksum, binding_sha)
    if result is None:
        return None
    result = {**marker, "test_binding_sha256": binding_sha, "run": run_dir.name, "metrics": result,
              "inference_chunks": len(list(chunks.glob("*.npz"))),
              "paper_comparison": "83.22 is an author-reported reference; causal corrections and protocols differ"}
    atomic_json(run_dir / "test.json", result)
    budget.flush()
    return result


@contextlib.contextmanager
def exclusive_queue(path):
    import fcntl
    with Path(path).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process already owns this experiment queue") from error
        yield


def main(args):
    from data import prepare
    if args.effective_batch_size % args.micro_batch_size:
        raise ValueError("effective-batch-size must be divisible by micro-batch-size")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device unavailable; no silent CPU training fallback")
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_queue(output / "queue.lock"):
        budget = Budget(output / "budget.json", args.budget_hours)
        try:
            final_reserve = args.final_reserve_minutes * 60
            warm_deadline = min(args.warm_hours * 3600, budget.limit - args.cold_hours * 3600 - final_reserve)
            cold_deadline = budget.limit - final_reserve
            cache = {}

            def get_bundle(protocol, fold):
                key = (protocol, fold)
                if key not in cache:
                    cache.clear()
                    cache[key] = prepare(args.data_root, fold=fold, protocol=protocol, cold_seed=args.cold_seed)
                return cache[key]

            def directory(protocol, fold, variant):
                return output / f"{protocol}-f{fold}-{variant}-s{args.seed}"

            core = [("standard", 0, variant) for variant in VARIANTS]
            for index, (protocol, fold, variant) in enumerate(core):
                run_dir = directory(protocol, fold, variant)
                remaining = sum(not (directory(*entry) / "complete.json").exists() for entry in core[index:])
                deadline = budget.elapsed() + max(0, budget.remaining(warm_deadline)) / max(1, remaining)
                train_one(args, get_bundle(protocol, fold), run_dir, protocol, fold, variant, budget, deadline)
            frozen_path = output / "fold0-selection-frozen.json"
            if all((directory(*entry) / "complete.json").exists() for entry in core):
                core_results = {directory(*entry).name: read_json(directory(*entry) / "complete.json") for entry in core}
                frozen = {"selection_rule": "validation_auc_with_minimum_improvement; tests_unopened_until_all_four_completed",
                          "runs": core_results,
                          "winner_by_validation_auc": max(core_results, key=lambda name: core_results[name]["best"]["validation"]["auc"]),
                          "followup_variants_prespecified": ["baseline", "full"]}
                if frozen_path.exists() and read_json(frozen_path) != frozen:
                    raise ValueError("The frozen initial ablation changed")
                atomic_json(frozen_path, frozen)
            else:
                atomic_json(output / "queue-summary.json", {"status": "incomplete_initial_ablation",
                            "tests_requested": False, "budget_seconds_used": budget.elapsed()})
                return

            for entry in core:
                test_once(args, get_bundle(entry[0], entry[1]), directory(*entry), budget, warm_deadline)
            extras = [("standard", fold, variant) for fold in args.folds if fold != 0
                      for variant in ("baseline", "full")]
            for index, (protocol, fold, variant) in enumerate(extras):
                if budget.remaining(warm_deadline) < 90:
                    break
                remaining = sum(not (directory(*entry) / "complete.json").exists() for entry in extras[index:])
                deadline = budget.elapsed() + budget.remaining(warm_deadline) / max(1, remaining)
                train_one(args, get_bundle(protocol, fold), directory(protocol, fold, variant),
                          protocol, fold, variant, budget, deadline)
            # cold_hours is a minimum reservation, not a separate hard cap:
            # unused standard-run time carries forward to this global deadline.
            cold = [("cold", 0, variant) for variant in ("baseline", "full")]
            for index, (protocol, fold, variant) in enumerate(cold):
                remaining = sum(not (directory(*entry) / "complete.json").exists() for entry in cold[index:])
                deadline = min(cold_deadline, budget.elapsed() + max(0, budget.remaining(cold_deadline)) / max(1, remaining))
                train_one(args, get_bundle(protocol, fold), directory(protocol, fold, variant),
                          protocol, fold, variant, budget, deadline)
            queue = core + extras + cold
            for entry in queue:
                if (directory(*entry) / "complete.json").exists():
                    test_once(args, get_bundle(entry[0], entry[1]), directory(*entry), budget, budget.limit - 15)
            summary = {"status": "finished_within_budget", "budget_seconds_used": budget.elapsed(),
                       "selection_frozen": str(frozen_path), "runs": {}}
            for entry in queue:
                run_dir = directory(*entry)
                summary["runs"][run_dir.name] = {
                    "training": read_json(run_dir / "complete.json") if (run_dir / "complete.json").exists() else "not_completed",
                    "test": read_json(run_dir / "test.json") if (run_dir / "test.json").exists() else "not_completed"}
            if any(value["training"] == "not_completed" or value["test"] == "not_completed" for value in summary["runs"].values()):
                summary["status"] = "budget_limited_partial_results"
            atomic_json(output / "queue-summary.json", summary)
        finally:
            budget.flush(closed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-hours", type=float, required=True)
    parser.add_argument("--warm-hours", type=float, default=15)
    parser.add_argument("--cold-hours", type=float, default=3)
    parser.add_argument("--final-reserve-minutes", type=float, default=30)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold-seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=2e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.budget_hours <= 0 or arguments.max_epochs <= 0 or arguments.checkpoint_steps <= 0:
        parser.error("budget-hours, max-epochs and checkpoint-steps must be positive")
    main(arguments)
