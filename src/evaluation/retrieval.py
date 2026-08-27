"""Retrieval metrics (Recall@K, MRR), linear probes and robustness checks.

All evaluators are pure NumPy, deterministic, and versioned via
``EVALUATOR_VERSION`` so the fair-comparison gate can pin the evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .. import EVALUATOR_VERSION


def recall_at_k(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    gold: Sequence[int],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Text→image retrieval recall. ``gold[i]`` is the correct corpus index."""
    sims = query_embs @ corpus_embs.T
    ranks: List[int] = []
    for i in range(len(gold)):
        order = np.argsort(-sims[i])
        ranks.append(int(np.where(order == gold[i])[0][0]))
    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([r < k for r in ranks]))
    out["mrr"] = float(np.mean([1.0 / (r + 1) for r in ranks]))
    out["median_rank"] = float(np.median(ranks))
    return out


def linear_probe_accuracy(
    train_embs: np.ndarray,
    train_labels: Sequence[int],
    test_embs: np.ndarray,
    test_labels: Sequence[int],
    ridge: float = 1.0,
) -> Dict[str, float]:
    """Closed-form ridge probe on frozen embeddings (deterministic)."""
    d = train_embs.shape[1]
    classes = sorted(set(train_labels))
    class_idx = {c: i for i, c in enumerate(classes)}

    X = np.hstack([train_embs, np.ones((len(train_embs), 1), dtype=np.float32)])
    Y = np.zeros((len(train_embs), len(classes)), dtype=np.float32)
    for row, lab in enumerate(train_labels):
        Y[row, class_idx[lab]] = 1.0
    A = X.T @ X + ridge * np.eye(d + 1, dtype=np.float32)
    W = np.linalg.solve(A, X.T @ Y)

    Xt = np.hstack([test_embs, np.ones((len(test_embs), 1), dtype=np.float32)])
    preds = Xt @ W
    pred_labels = [classes[i] for i in np.argmax(preds, axis=1)]
    acc = float(np.mean([p == t for p, t in zip(pred_labels, test_labels)]))

    per_class: Dict[str, float] = {}
    for c in classes:
        mask = [t == c for t in test_labels]
        if any(mask):
            pc = [p == t for p, t, m in zip(pred_labels, test_labels, mask) if m]
            per_class[str(c)] = float(np.mean(pc))
    return {"probe_accuracy": acc, "probe_per_class": per_class}


def robustness_perturb(
    texts: Sequence[str],
    encode_fn,
    corpus_embs: np.ndarray,
    gold: Sequence[int],
) -> Dict[str, Any]:
    """Robustness to caption perturbation: typos + word dropout.

    Reports the mean recall drop when queries are degraded — a required
    robustness result for every run.
    """
    rng = np.random.default_rng(167)
    base = recall_at_k(encode_fn(list(texts)), corpus_embs, gold, ks=(1,))["recall@1"]

    def typo(s: str) -> str:
        if len(s) > 4:
            i = len(s) // 2
            return s[:i] + s[i + 1] + s[i] + s[i + 1:]
        return s

    typos = [typo(t) for t in texts]
    dropped = []
    for t in texts:
        words = t.split()
        if len(words) > 3:
            j = int(rng.integers(0, len(words)))
            words = words[:j] + words[j + 1:]
        dropped.append(" ".join(words))

    typo_r = recall_at_k(encode_fn(typos), corpus_embs, gold, ks=(1,))["recall@1"]
    drop_r = recall_at_k(encode_fn(dropped), corpus_embs, gold, ks=(1,))["recall@1"]
    return {
        "clean_recall@1": base,
        "typo_recall@1": typo_r,
        "word_dropout_recall@1": drop_r,
        "robustness_drop": float(base - min(typo_r, drop_r)),
    }


def evaluator_identity() -> Dict[str, Any]:
    return {"evaluator_version": EVALUATOR_VERSION}
