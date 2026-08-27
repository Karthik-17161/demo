"""Evaluation-layer tests: retrieval metrics, probes, robustness, fairness."""

import numpy as np

from src.evaluation.fairness import (
    critical_regression_check,
    fairness_verdict,
    slice_metrics,
)
from src.evaluation.retrieval import (
    linear_probe_accuracy,
    recall_at_k,
    robustness_perturb,
)


def _aligned(n=60, d=32, noise=0.02, seed=167):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d)).astype(np.float32)
    txt = base + noise * rng.standard_normal((n, d)).astype(np.float32)
    img = base + noise * rng.standard_normal((n, d)).astype(np.float32)

    def l2(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return l2(txt), l2(img), list(range(n))


def test_recall_perfect_alignment():
    txt, img, gold = _aligned(noise=0.0)
    m = recall_at_k(txt, img, gold)
    assert m["recall@1"] == 1.0 and m["mrr"] == 1.0


def test_recall_random_is_low():
    rng = np.random.default_rng(7)
    txt = rng.standard_normal((50, 16)).astype(np.float32)
    img = rng.standard_normal((50, 16)).astype(np.float32)
    m = recall_at_k(txt, img, list(range(50)))
    assert m["recall@1"] < 0.2
    assert "recall@10" in m and "median_rank" in m


def test_linear_probe_separates_classes():
    rng = np.random.default_rng(167)
    c1 = rng.standard_normal((40, 16)).astype(np.float32) + 3.0
    c2 = rng.standard_normal((40, 16)).astype(np.float32) - 3.0
    X = np.vstack([c1, c2])
    y = [0] * 40 + [1] * 40
    probe = linear_probe_accuracy(X, y, X, y)
    assert probe["probe_accuracy"] > 0.95
    assert set(probe["probe_per_class"]) == {"0", "1"}


def test_robustness_reports_drops():
    txt, img, gold = _aligned()
    texts = [f"a red bicycle at scene {i}" for i in range(len(gold))]
    out = robustness_perturb(texts, lambda ts: txt, img, gold)
    for k in ("clean_recall@1", "typo_recall@1", "word_dropout_recall@1", "robustness_drop"):
        assert k in out


def test_slice_metrics_and_gaps():
    txt, img, gold = _aligned()
    # make one subgroup deliberately bad: shuffle its text embeddings
    subgroups = ["common-object"] * len(gold)
    rare_idx = list(range(0, len(gold), 5))
    for i in rare_idx:
        subgroups[i] = "rare-object"
    txt_bad = txt.copy()
    txt_bad[rare_idx] = txt_bad[rare_idx][::-1]
    slices = slice_metrics(
        txt_bad, img, gold, subgroups=subgroups,
        domains=["indoor"] * (len(gold) // 2) + ["outdoor"] * (len(gold) - len(gold) // 2),
    )
    assert slices["overall"]["n"] == len(gold)
    assert set(slices["by_subgroup"]) == {"common-object", "rare-object"}
    assert slices["disparity_subgroup"]["max_gap_recall@1"] >= 0.0


def test_fairness_verdict_flags_large_gap():
    slices = {
        "overall": {"n": 100, "recall@1": 0.6},
        "by_subgroup": {
            "a": {"n": 90, "recall@1": 0.8},
            "b": {"n": 10, "recall@1": 0.2},
        },
        "by_domain": {},
        "disparity_subgroup": {"max_gap_recall@1": 0.6},
        "disparity_domain": {},
    }
    v = fairness_verdict(slices, max_gap=0.25)
    assert not v["fairness_ok"]
    ok = fairness_verdict({**slices, "disparity_subgroup": {"max_gap_recall@1": 0.1}}, max_gap=0.25)
    assert ok["fairness_ok"]


def test_critical_regression_detected_even_when_aggregate_improves():
    baseline = {
        "overall": {"n": 200, "recall@1": 0.50},
        "by_subgroup": {
            "rare-object": {"n": 20, "recall@1": 0.40},
            "other": {"n": 180, "recall@1": 0.55},
        },
        "by_domain": {},
    }
    candidate = {
        "overall": {"n": 200, "recall@1": 0.58},   # aggregate up
        "by_subgroup": {
            "rare-object": {"n": 20, "recall@1": 0.30},  # slice down
            "other": {"n": 180, "recall@1": 0.62},
        },
        "by_domain": {},
    }
    rep = critical_regression_check(baseline, candidate)
    assert rep["verdict"] == "REVISE"
    assert rep["critical_regressions"][0]["slice"] == "by_subgroup:rare-object"


def test_no_false_positive_on_uniform_improvement():
    s = lambda r1: {
        "overall": {"n": 100, "recall@1": r1},
        "by_subgroup": {"a": {"n": 50, "recall@1": r1}, "b": {"n": 50, "recall@1": r1}},
        "by_domain": {},
    }
    rep = critical_regression_check(s(0.4), s(0.5))
    assert rep["verdict"] == "OK"
