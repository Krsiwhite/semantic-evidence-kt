"""Small CPU-only invariant checks; does not download data or train a model."""
from __future__ import annotations

import os
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import prepare
from models import SemanticEvidenceKT, SemanticResidualEmbedder


def run() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    embeddings = torch.randn(40, 4, 8)
    counts = torch.arange(40).float()
    batch = {"qids": torch.arange(16).reshape(2, 8),
             "responses": torch.tensor([[0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0]]),
             "valid": torch.ones(2, 8, dtype=torch.bool),
             "scoremask": torch.tensor([[False] + [True] * 7] * 2),
             "concepts": torch.zeros(2, 8, 7, dtype=torch.long)}
    record = {"device": "cpu", "variants": {}}
    for variant in ["baseline", "residual", "evidence", "full"]:
        torch.manual_seed(42)
        model = SemanticEvidenceKT(embeddings, counts, variant, hidden_dim=16,
                dropout=0, n_blocks=2, n_heads=4, seq_len=8, adapter_dim=8, top_k=4)
        model.eval()
        with torch.no_grad():
            original = model(batch, return_contributions=True)
            largest = 0.0
            # Every t is tested: predictions through t cannot use response[t].
            for position in range(8):
                changed = {**batch, "responses": batch["responses"].clone()}
                changed["responses"][:, position] = 1 - changed["responses"][:, position]
                alternative = model(changed)
                error = (original["probabilities"][:, :position + 1] -
                         alternative["probabilities"][:, :position + 1]).abs().max().item()
                largest = max(largest, error)
            assert largest < 1e-7, (variant, largest)
            if variant in {"evidence", "full"}:
                actual_correction = original["logits"] - torch.logit(original["base_probs"])
                decomposition_error = (actual_correction - original["contributions"].sum(-1)).abs().max().item()
                assert decomposition_error < 3e-7
                assert original["contributions"][:, 0].abs().sum().item() == 0
                index = original["contribution_indices"]
                allowed = original["contribution_valid"]
                assert ((index < torch.arange(8)[None, :, None]) | ~allowed).all()
            else:
                decomposition_error = 0.0
        model.train()
        output = model(batch)
        loss = F.binary_cross_entropy_with_logits(output["logits"][batch["scoremask"]],
                        batch["responses"][batch["scoremask"]].float()) + output["auxiliary_loss"]
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        record["variants"][variant] = {"max_current_or_future_response_effect": largest,
                                     "decomposition_error": decomposition_error,
                                     "finite_backward": True}

    embedder = SemanticResidualEmbedder(embeddings, counts, adapter_dim=8).eval()
    before = embedder(torch.tensor([[0, 1]]))[0].detach().clone()
    with torch.no_grad():
        embedder.delta[0].fill_(999)
    after = embedder(torch.tensor([[0, 1]]))[0].detach()
    assert torch.equal(before[:, 0], after[:, 0])
    record["zero_support_residual_invariant"] = True

    pad_counts = torch.tensor([0., 3., 10., 2., 1.])
    pad_model = SemanticEvidenceKT(embeddings[:5], pad_counts, hidden_dim=16,
                                   n_blocks=1, seq_len=4, dropout=0)
    pad_batch = {"qids": torch.tensor([[1, 2, 0, 0]]),
                 "responses": torch.tensor([[1, 0, 0, 0]]),
                 "valid": torch.tensor([[True, True, False, False]])}
    pad_output = pad_model(pad_batch)
    (pad_output["logits"][:, :2].sum() + pad_output["auxiliary_loss"]).backward()
    cold_gradient = pad_model.backbone.polya_emb.que_emb.que_emb.weight.grad[:4].abs().sum().item()
    assert pad_model.padding_qid == 2 and cold_gradient == 0
    record["cold_padding"] = {"selected_warm_qid": pad_model.padding_qid,
                              "held_out_item_zero_embedding_gradient": cold_gradient}

    with tempfile.TemporaryDirectory(prefix="semantic-evidence-cpu-check-") as temporary:
        root = Path(temporary)
        qdir = root / "XES3G5M" / "question_level"
        qdir.mkdir(parents=True)
        torch.save(embeddings, root / "baim_embeddings.pt")
        fieldnames = ["fold", "uid", "questions", "responses", "concepts", "selectmasks"]
        def write_csv(path, folds):
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for fold in folds:
                    writer.writerow({"fold": fold, "uid": 100 + fold,
                        "questions": ",".join(map(str, range(40))),
                        "responses": ",".join(str(i % 2) for i in range(40)),
                        "concepts": ",".join("0" for _ in range(40)),
                        "selectmasks": ",".join("1" for _ in range(40))})
        write_csv(qdir / "train_valid_sequences_quelevel.csv", range(5))
        normal = prepare(root)
        assert not normal["metadata"]["test_loaded"]
        assert normal["metadata"]["mask_counts"] == {"train": 156, "val": 39}
        assert len(normal["train"][0]["qids"]) == 200
        cold = prepare(root, protocol="cold")
        partition = cold["metadata"]["cold_partition"]
        assert partition["item_counts"] == {"warm": 28, "cold_val": 6, "cold_test": 6}
        groups = [set(partition[key]) for key in ["warm_items", "cold_val_items", "cold_test_items"]]
        assert len(set.union(*groups)) == 40 and sum(map(len, groups)) == 40
        held_out = partition["cold_val_items"] + partition["cold_test_items"]
        assert cold["item_counts"][held_out].sum().item() == 0
        for sample in cold["train"]:
            assert all(int(x) not in held_out for x in sample["qids"][sample["valid"]])
        for sample in cold["val"]:
            ids = sample["qids"][sample["valid"]]
            assert all(int(x) in partition["warm_items"] for x in ids[:-1])
            assert int(ids[-1]) in partition["cold_val_items"]
            assert sample["scoremask"].sum().item() == 1
        # Successful prepare before the test CSV exists proves lazy test loading.
        write_csv(qdir / "test_sequences_quelevel.csv", [-1])
        assert len(normal["test_factory"]()) == 1
        cold_test = cold["test_factory"]()
        assert len(cold_test) == len(partition["cold_test_items"])
        assert normal["metadata"]["test_loaded"] and "test_csv" in normal["metadata"]["sourcehash"]
        sample_batch = next(iter(DataLoader(cold_test, batch_size=2)))
        assert sample_batch["qids"].shape == (2, 200)
        record["data"] = {"test_loading": "lazy", "fixed_length": 200,
                          "cold_train_responses_removed": True,
                          "cold_query_history_warm_only": True,
                          "cold_split_sha256": partition["sha256"]}
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
