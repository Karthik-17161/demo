"""Parameter-efficient fine-tuning: low-rank adaptation (LoRA-style).

The base encoder is **frozen**; learning happens only in a low-rank residual
``ΔW = (alpha/rank) * A @ B`` applied in the shared embedding space (a NumPy
expression of the classic LoRA idea, so the lab runs without a GPU). The
trainable-parameter *budget* is fixed by config (``lora.rank``); the module
refuses to exceed it and reports the exact parameter count for the
fair-comparison gate.

Trainable params = rank*dim (A, frozen projection) + rank*dim (B) + dim (gate).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional, Tuple

import numpy as np


class BudgetExceededError(RuntimeError):
    """Raised when fine-tuning would exceed its approved compute/param budget."""


class LoRAAdapter:
    """Frozen base + trainable low-rank residual in shared embedding space."""

    def __init__(
        self,
        dim: int,
        rank: int = 8,
        alpha: float = 8.0,
        seed: int = 167,
        gated: bool = True,
    ) -> None:
        if rank < 1 or rank > dim:
            raise ValueError(f"invalid LoRA rank {rank} for dim {dim}")
        self.dim = dim
        self.rank = rank
        self.alpha = float(alpha)
        self.gated = bool(gated)
        rng = np.random.default_rng(seed)
        # float64 throughout: contrastive gradients under a saturated softmax
        # are ~1e-6 and would be rounded away in float32 accumulation.
        self.A = (rng.standard_normal((dim, rank)) / np.sqrt(dim)).astype(np.float64)
        self.B = np.zeros((rank, dim), dtype=np.float64)  # ΔW starts at zero
        self.gate = np.ones(dim, dtype=np.float64) if gated else None

    # ---------------- budget ----------------

    def trainable_param_count(self) -> int:
        n = self.A.size + self.B.size
        if self.gate is not None:
            n += self.gate.size
        return int(n)

    def check_budget(self, max_trainable_params: int) -> None:
        used = self.trainable_param_count()
        if used > max_trainable_params:
            raise BudgetExceededError(
                f"LoRA trainable params {used} exceed approved budget {max_trainable_params}"
            )

    # ---------------- forward ----------------

    def delta(self) -> np.ndarray:
        """ΔW = (alpha/rank) · A·B (∘ gate), float64."""
        d = (self.alpha / self.rank) * (self.A @ self.B)
        if self.gate is not None:
            d = d * self.gate[None, :]
        return d

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        """Add the low-rank residual to L2-normalised embeddings, re-normalise.

        ``out = normalize(e + e @ ΔW.T)`` so ΔW = 0 is exactly the identity
        (the frozen base encoder behaviour).
        """
        e = embeddings.astype(np.float64)
        out = e + e @ self.delta().T
        n = np.linalg.norm(out, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return out / n

    # ---------------- training ----------------

    def fit(
        self,
        image_embs: np.ndarray,
        text_embs: np.ndarray,
        hard_pair_indices: Optional[List[Tuple[int, int]]] = None,
        epochs: int = 30,
        lr: float = 0.05,
        temperature: float = 0.07,
        log_cb: Optional[Any] = None,
    ) -> List[float]:
        """Contrastive alignment of paired image/text embeddings.

        ``hard_pair_indices`` supplies semantically-close negative pairs from
        the curriculum; they are pushed apart alongside the InfoNCE objective.
        Returns per-epoch loss history (logged into the run record).
        """
        img = image_embs.astype(np.float64)
        txt = text_embs.astype(np.float64)
        losses: List[float] = []
        n = len(img)
        for epoch in range(epochs):
            img_a = self.apply(img)
            txt_a = self.apply(txt)
            logits = img_a @ txt_a.T / temperature
            labels = np.arange(n)

            # _xent returns d(mean-loss)/dlogits; total objective is the
            # average of the image→text and text→image terms.
            loss_i, g_i = self._xent(logits, labels)
            loss_t, g_t = self._xent(logits.T, labels)
            grad_logits = (g_i + g_t.T) / 2.0

            grad_img = grad_logits @ txt_a
            grad_txt = grad_logits.T @ img_a

            D = self.delta()
            gD = np.zeros_like(D)
            for grads, embs, emb_out in ((grad_img, img, img_a), (grad_txt, txt, txt_a)):
                inv = 1.0 / np.maximum(
                    np.linalg.norm(emb_out, axis=1, keepdims=True), 1e-8
                )
                proj = (emb_out * grads).sum(axis=1, keepdims=True) * emb_out
                g_dot = (grads - proj) * inv
                gD += g_dot.T @ embs

            scale = self.alpha / self.rank

            # SGD on B and gate — exactly the trainable set reported by
            # trainable_param_count(). Chain rule through ΔW = scale*(A@B)*gate:
            #   dL/dB    = scale * A.T @ (dL/dΔW ∘ gate)
            #   dL/dgate = Σ_ij (dL/dΔW)_ij * (A@B)_ij   (column-wise for gate[j])
            gAB = self.A @ self.B
            if self.gate is not None:
                gate_grad = (gD * gAB).sum(axis=0)
                gD = gD * self.gate[None, :]
            self.B += (-lr * scale) * (self.A.T @ gD)
            if self.gate is not None:
                self.gate += -lr * gate_grad

            # hard-negative hinge term: push supplied close pairs apart ONLY
            # when the negative threatens the positive's ranking (its
            # similarity comes within ``rank_margin`` of the true pair's).
            # Unconditional pushing would erode the shared-class structure
            # that retrieval depends on.
            hloss = 0.0
            n_violations = 0
            if hard_pair_indices:
                img_a = self.apply(img)
                txt_a = self.apply(txt)
                hinge_lr = lr * 2e-2
                rank_margin = 0.10
                for qi, neg_i in hard_pair_indices:
                    s_pos = float(img_a[qi] @ txt_a[qi])
                    s_neg = float(img_a[qi] @ txt_a[neg_i])
                    if s_neg > s_pos - rank_margin:
                        hloss += max(s_neg - (s_pos - rank_margin), 0.0)
                        n_violations += 1
                        # descend on h = <img_q, txt_neg> − threshold;
                        # approximate dH/dΔW ≈ outer(dh/dtxt_a[ni], txt[ni])
                        g_delta = np.outer(img_a[qi], txt[neg_i])
                        gB = scale * (self.A.T @ g_delta)
                        self.B -= hinge_lr * gB
                hloss = hloss / max(len(hard_pair_indices), 1)
            losses.append((loss_i + loss_t) / 2 + hloss)

            if log_cb:
                log_cb(epoch, losses[-1])
        return losses

    @staticmethod
    def _xent(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
        m = logits.max(axis=1, keepdims=True)
        z = logits - m
        logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
        logp = z - logsumexp
        n = logits.shape[0]
        loss = -logp[np.arange(n), labels].mean()
        p = np.exp(logp)
        p[np.arange(n), labels] -= 1.0
        return float(loss), p / n

    # ---------------- artifact identity ----------------

    def state_hash(self) -> str:
        payload = {
            "dim": self.dim, "rank": self.rank, "alpha": self.alpha,
            "gated": self.gated,
            "A": self.A.round(6).tolist(), "B": self.B.round(6).tolist(),
            "gate": None if self.gate is None else self.gate.round(6).tolist(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def save(self, path: str) -> None:
        payload = {
            "dim": self.dim, "rank": self.rank, "alpha": self.alpha,
            "gated": self.gated, "state_hash": self.state_hash(),
            "A": self.A.tolist(), "B": self.B.tolist(),
            "gate": None if self.gate is None else self.gate.tolist(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "LoRAAdapter":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        obj = cls(
            dim=payload["dim"], rank=payload["rank"], alpha=payload["alpha"],
            gated=payload["gated"],
        )
        obj.A = np.array(payload["A"], dtype=np.float64)
        obj.B = np.array(payload["B"], dtype=np.float64)
        obj.gate = None if payload["gate"] is None else np.array(payload["gate"], dtype=np.float64)
        if obj.state_hash() != payload.get("state_hash"):
            raise ValueError("artifact mismatch: LoRA state hash does not match saved hash")
        return obj
