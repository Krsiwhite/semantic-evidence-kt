"""Causal BAIM reference and two explicitly scoped experimental mechanisms.

The evidence decomposition explains the additive correction, not the BAIM core.
No student-response data are used in the frozen semantic prior.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from vendor.baim_core import SparseKTBAIMNet


class SemanticResidualEmbedder(nn.Module):
    """One shared adapter plus support-weighted, item-specific residuals."""

    def __init__(self, embeddings: Tensor, counts: Tensor, adapter_dim: int = 256,
                 item_mask_rate: float = 0.15, tau: float = 100.0):
        super().__init__()
        self.register_buffer("prior", embeddings.detach().float().clone())
        self.register_buffer("item_counts", counts.detach().float().clone())
        dim = embeddings.shape[-1]
        self.adapter = nn.Sequential(nn.Linear(dim, adapter_dim), nn.ReLU(),
                                     nn.Linear(adapter_dim, dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.delta = nn.Parameter(torch.zeros_like(self.prior))
        self.tau_raw = nn.Parameter(torch.tensor(math.log(math.expm1(tau - 1))))
        self.item_mask_rate = item_mask_rate

    def forward(self, qids: Tensor) -> tuple[Tensor, bool]:
        unique_ids, inverse = torch.unique(qids, return_inverse=True)
        prior = self.prior[unique_ids]
        tau = 1 + F.softplus(self.tau_raw)
        n = self.item_counts[unique_ids]
        support = n / (n + tau)
        if self.training and self.item_mask_rate:
            # All occurrences of an item share the same mask, including targets.
            support = support * (torch.rand_like(n) >= self.item_mask_rate)
        value = prior + self.adapter(prior) + support[:, None, None] * self.delta[unique_ids]
        value = value[inverse].reshape(*qids.shape, *self.prior.shape[1:])
        return value, True


class HistoryEvidenceHead(nn.Module):
    """Causal retrieval of observed historical prediction errors."""

    def __init__(self, embeddings: Tensor, num_concepts: int = 865,
                 retrieval_dim: int = 64, top_k: int = 16, temperature: float = 0.2):
        super().__init__()
        prior = F.normalize(embeddings.detach().float().mean(dim=1), dim=-1)
        self.register_buffer("semantic_prior", prior)
        self.project = nn.Linear(prior.shape[-1], retrieval_dim, bias=False)
        self.kc_bias_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.decay_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        # A nonnegative bounded scale preserves the sign of each contribution.
        self.amplitude_raw = nn.Parameter(torch.tensor(math.log(0.025 / 0.975)))
        self.num_concepts = num_concepts
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, batch: dict[str, Tensor], base_probs: Tensor,
                return_contributions: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        qids, valid = batch["qids"], batch["valid"].bool()
        b, length = qids.shape
        ids, inverse = torch.unique(qids, return_inverse=True)
        features = F.normalize(self.project(self.semantic_prior[ids]), dim=-1)
        features = features[inverse].reshape(b, length, -1)
        scores = torch.bmm(features, features.transpose(1, 2)) / self.temperature

        concepts = batch.get("concepts")
        if concepts is not None:
            present = (concepts >= 0) & (concepts < self.num_concepts)
            multi_hot = scores.new_zeros((b, length, self.num_concepts))
            multi_hot.scatter_add_(-1, concepts.clamp(0, self.num_concepts - 1),
                                   present.to(scores.dtype))
            overlap = torch.bmm(multi_hot, multi_hot.transpose(1, 2)) > 0
            scores = scores + F.softplus(self.kc_bias_raw) * overlap.to(scores.dtype)

        positions = torch.arange(length, device=qids.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        scores = scores - F.softplus(self.decay_raw) * torch.log1p(lag.to(scores.dtype))
        allowed = (lag > 0)[None] & valid[:, :, None] & valid[:, None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        k = min(self.top_k, length)
        selected, indices = scores.topk(k, dim=-1)
        selected_valid = allowed.gather(-1, indices)
        has_history = selected_valid.any(dim=-1, keepdim=True)
        safe_selected = torch.where(has_history, selected, torch.zeros_like(selected))
        weights = F.softmax(safe_selected, dim=-1) * selected_valid.to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        residual = batch["responses"].to(base_probs.dtype) - base_probs.detach()
        selected_residuals = residual[:, None, :].expand(-1, length, -1).gather(-1, indices)
        effective_n = 1 / weights.square().sum(-1).clamp_min(1e-12)
        support_gate = (effective_n / (effective_n + 4)) * has_history.squeeze(-1)
        amplitude = 2 * self.amplitude_raw.sigmoid()
        contributions = amplitude * support_gate[:, :, None] * weights * selected_residuals
        correction = contributions.sum(-1)
        detail = {}
        if return_contributions:
            detail = {"contributions": contributions,
                      "contribution_indices": indices,
                      "contribution_weights": weights,
                      "contribution_valid": selected_valid,
                      "evidence_support": support_gate,
                      "evidence_amplitude": amplitude}
        return correction, detail


class SemanticEvidenceKT(nn.Module):
    """The baseline differs from the public BAIM core only by the causal repair.

    Inputs have shape [batch, sequence]; concepts has a final KC-list dimension.
    Padding must be on the right. Loss selection is the caller's scoremask.
    """

    def __init__(self, embeddings: Tensor, item_counts: Tensor,
                 variant: str = "baseline", hidden_dim: int = 256,
                 num_concepts: int = 865, dropout: float = 0.1,
                 n_blocks: int = 2, n_heads: int = 4, seq_len: int = 200,
                 adapter_dim: int = 256, item_mask_rate: float = 0.15,
                 tau: float = 100.0, retrieval_dim: int = 64, top_k: int = 16,
                 **kwargs: Any):
        super().__init__()
        if variant not in {"baseline", "residual", "evidence", "full"}:
            raise ValueError(f"Unknown variant {variant}")
        if embeddings.ndim != 3 or embeddings.shape[1] != 4:
            raise ValueError("Expected BAIM embeddings [items, 4, dimension]")
        if item_counts.numel() != embeddings.shape[0]:
            raise ValueError("Item count vector must align with embedding rows")
        # Gate balancing includes right-padding, as in BAIM. Reuse a seen item
        # there so a held-out item zero cannot receive synthetic padding updates.
        padding_qid = 0 if item_counts[0] > 0 else int(item_counts.argmax())
        self.padding_qid = padding_qid
        self.variant = variant
        self.backbone = SparseKTBAIMNet(
            n_question=embeddings.shape[0], n_pid=0, d_model=hidden_dim,
            n_blocks=n_blocks, dropout=dropout, d_ff=256,
            num_attn_heads=n_heads, seq_len=seq_len, emb_type="qid_sparseattn",
            emb_path=embeddings, flag_load_emb=True, flag_emb_freezed=False,
            pretrain_dim=embeddings.shape[-1], final_fc_dim=256,
            final_fc_dim2=256, separate_qa=False, sparse_ratio=0.8, k_index=5)
        if variant in {"residual", "full"}:
            self.backbone.polya_emb.que_emb = SemanticResidualEmbedder(
                embeddings, item_counts, adapter_dim, item_mask_rate, tau)
        self.evidence = None
        if variant in {"evidence", "full"}:
            self.evidence = HistoryEvidenceHead(embeddings, num_concepts,
                                               retrieval_dim, top_k)

    def forward(self, batch: dict[str, Tensor], return_contributions: bool = False,
                **kwargs: Any) -> dict[str, Tensor]:
        valid = batch["valid"].bool()
        qids = batch["qids"].long().masked_fill(~valid, self.padding_qid)
        responses = batch["responses"].long().masked_fill(~valid, 0)
        # The vendor core internally shifts the router state and strictly masks
        # interaction attention, so responses[t] cannot predict response[t].
        logits, _ = self.backbone(qids, qids, responses, return_logits=True)
        base_probs = logits.sigmoid()
        gate_probs = self.backbone.polya_emb.last_gate_probs
        if gate_probs is None:
            auxiliary_loss = logits.new_zeros(())
        else:
            # This matches the published code's averaging, including padding.
            usage = gate_probs.reshape(-1, gate_probs.shape[-1]).mean(0)
            auxiliary_loss = 0.01 * (usage - 1 / gate_probs.shape[-1]).square().sum()
        detail = {}
        if self.evidence is not None:
            clean_batch = {**batch, "qids": qids, "responses": responses}
            correction, detail = self.evidence(clean_batch, base_probs, return_contributions)
            logits = logits + correction
        elif return_contributions:
            detail = {"contributions": logits.new_zeros((*logits.shape, 0)),
                      "contribution_indices": torch.empty((*logits.shape, 0),
                                                          dtype=torch.long, device=logits.device)}
        probabilities = logits.sigmoid()
        return {"logits": logits, "probabilities": probabilities, "probs": probabilities,
                "base_probs": base_probs, "auxiliary_loss": auxiliary_loss,
                **detail}
