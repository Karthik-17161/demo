"""Model-layer tests: encoders, LoRA budget/identity, hard-negative curriculum."""

import numpy as np
import pytest

from src.data.ingest import generate_synthetic_records
from src.models.encoders import HashingImageEncoder, HashingTextEncoder, build_encoder
from src.models.hard_negatives import curriculum_schedule, sample_curriculum
from src.models.lora import BudgetExceededError, LoRAAdapter


def test_text_encoder_structure_and_determinism():
    enc = HashingTextEncoder(seed=167)
    a = enc.encode_texts(["a red bicycle", "a red bicycle", "a stone bridge"])
    assert a.shape == (3, 128)
    assert np.allclose(a[0], a[1])
    assert not np.allclose(a[0], a[2])
    norms = np.linalg.norm(a, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_image_encoder_scene_consistency():
    enc = HashingImageEncoder(seed=167)
    v = enc.encode_images(
        ["images/outdoor/s1/a.jpg", "images/outdoor/s1/b.jpg", "images/indoor/s2/c.jpg"],
        ["outdoor-scene-0001", "outdoor-scene-0001", "indoor-scene-0002"],
    )
    same_scene_sim = float(v[0] @ v[1])
    diff_scene_sim = float(v[0] @ v[2])
    assert same_scene_sim > diff_scene_sim


def test_build_encoder_fallback_is_deterministic():
    cfg = {"encoder": {"prefer_real": False, "seed": 167}}
    e1 = build_encoder(cfg)
    e2 = build_encoder(cfg)
    x = e1["text"].encode_texts(["hello world"])
    y = e2["text"].encode_texts(["hello world"])
    assert np.allclose(x, y)


def test_lora_param_budget_enforced():
    adapter = LoRAAdapter(dim=128, rank=8, seed=167)
    n = adapter.trainable_param_count()
    assert n == 8 * 128 + 8 * 128 + 128  # A + B + gate
    adapter.check_budget(n)
    with pytest.raises(BudgetExceededError):
        adapter.check_budget(n - 1)


def test_lora_delta_starts_zero():
    adapter = LoRAAdapter(dim=64, rank=4, seed=167)
    x = np.eye(64, dtype=np.float32)[:4]
    assert np.allclose(adapter.apply(x), x, atol=1e-6)  # ΔW=0 ⇒ identity


def test_lora_training_reduces_loss():
    # Noisy paired embeddings: shared direction + large per-modality noise.
    # Loss starts high (~5) so there is real optimisation headroom — a
    # near-perfectly-aligned pair would saturate the softmax and leave no
    # gradient signal to demonstrate learning.
    rng = np.random.default_rng(167)
    d, n = 64, 40

    def l2n(x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm[norm == 0] = 1.0
        return x / norm

    base = l2n(rng.standard_normal((n, d)))
    img = (base + 1.0 * rng.standard_normal((n, d))).astype(np.float32)
    txt = (base + 1.0 * rng.standard_normal((n, d))).astype(np.float32)

    def contrastive_loss(ad):
        ia, ta = ad.apply(img), ad.apply(txt)
        s = ia @ ta.T / 0.07
        m = s.max(axis=1, keepdims=True)
        p = np.exp(s - m)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(n), np.arange(n)] + 1e-12).mean())

    ad = LoRAAdapter(dim=d, rank=8, seed=167)
    before = contrastive_loss(ad)
    hist = ad.fit(img, txt, epochs=10, lr=0.05)
    after = contrastive_loss(ad)
    assert before > 1.0                      # headroom exists
    assert hist[-1] < hist[0]                # fit() objective decreases
    assert after < before                    # held-out recompute agrees


def test_lora_state_hash_changes_after_training():
    ad = LoRAAdapter(dim=32, rank=4, seed=167)
    h0 = ad.state_hash()
    img = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    ad.fit(img, img, epochs=2)
    assert ad.state_hash() != h0


def test_lora_save_load_roundtrip(tmp_path):
    ad = LoRAAdapter(dim=32, rank=4, seed=167)
    path = tmp_path / "ad.json"
    ad.save(str(path))
    loaded = LoRAAdapter.load(str(path))
    assert loaded.state_hash() == ad.state_hash()
    x = np.random.default_rng(2).standard_normal((5, 32)).astype(np.float32)
    assert np.allclose(ad.apply(x), loaded.apply(x))


def test_tampered_checkpoint_refused(tmp_path):
    import json

    ad = LoRAAdapter(dim=32, rank=4, seed=167)
    path = tmp_path / "ad.json"
    ad.save(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["A"][0][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        LoRAAdapter.load(str(path))


def test_curriculum_schedule_order():
    sched = curriculum_schedule(6)
    assert sched == ["easy", "easy", "medium", "medium", "hard", "hard"]


def test_hard_negatives_log_every_pair():
    records = generate_synthetic_records(200, seed=167)
    for stage in ("easy", "medium", "hard"):
        cur = sample_curriculum(records, stage=stage, seed=167, max_pairs=50)
        assert len(cur.pair_indices) <= 50
        assert len(cur.negative_log) == len(cur.pair_indices)
        for entry in cur.negative_log:
            assert entry["query"] and entry["negative"] and entry["reason"]
    # determinism
    c1 = sample_curriculum(records, "hard", seed=167, max_pairs=20)
    c2 = sample_curriculum(records, "hard", seed=167, max_pairs=20)
    assert c1.pair_indices == c2.pair_indices


def test_hard_stage_prefers_same_class_different_scene():
    records = generate_synthetic_records(200, seed=167)
    cur = sample_curriculum(records, stage="hard", seed=167, max_pairs=100)
    assert cur.pair_indices
    for qi, ni in cur.pair_indices:
        q, n_ = records[qi], records[ni]
        # same object class (confusable), different scene instance
        assert q["object_class"] == n_["object_class"]
        assert q["scene_id"] != n_["scene_id"]
