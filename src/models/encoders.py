"""Vision-language encoders.

Two interchangeable backends:

* ``HashingTextEncoder`` / ``HashingImageEncoder`` — a deterministic, seeded
  feature-hashing encoder used so the *entire* lab (training, evaluation,
  telemetry, acceptance, negative tests) runs on any machine with zero GPU and
  zero downloads. It is genuinely structured: captions that share words land
  near each other in cosine space, so retrieval metrics are real.
* OpenCLIP backend — if ``open_clip_torch`` + torch are importable and the
  config sets ``prefer_real: true``, the same interface drives a real pinned
  OpenCLIP model. Everything downstream is unchanged.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence

import numpy as np

_DIM = 128


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HashingTextEncoder:
    """Seeded feature-hashing text encoder."""

    name = "hashing-text-v1"
    dim = _DIM

    def __init__(self, seed: int = 167) -> None:
        self.seed = seed

    def _vec(self, tokens: Sequence[str]) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            h = hashlib.sha256(f"{self.seed}:{tok}".encode()).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            v[idx] += sign
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(_tokenize(t)) for t in texts])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_texts(texts)


class HashingImageEncoder:
    """Deterministic pseudo-image encoder derived from image URI + scene.

    Real deployments swap this for an OpenCLIP visual tower. The URI encodes
    domain/scene; tokens are hashed through the *same* token→dimension map as
    :class:`HashingTextEncoder` so that words shared between a caption and its
    image's scene/domain land in the same dimensions — giving the retrieval
    task genuine cross-modal structure (shared vocabulary ⇒ high cosine).
    """

    name = "hashing-image-v1"
    dim = _DIM

    def __init__(self, seed: int = 167) -> None:
        self.seed = seed

    def encode_images(
        self, uris: Sequence[str], scenes: Sequence[str] | None = None
    ) -> np.ndarray:
        vecs = []
        for i, uri in enumerate(uris):
            scene = scenes[i] if scenes is not None else ""
            toks = _tokenize(uri.replace("/", " ").replace(".jpg", "")) + _tokenize(scene)
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in toks:
                h = hashlib.sha256(f"{self.seed}:{tok}".encode()).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                v[idx] += sign
            n = float(np.linalg.norm(v))
            if n > 0:
                v /= n
            vecs.append(v)
        return np.stack(vecs)


def build_encoder(config: Dict[str, Any]):
    """Factory honouring ``encoder.prefer_real``; falls back deterministically."""
    enc_cfg = config.get("encoder", {})
    prefer_real = bool(enc_cfg.get("prefer_real", False))
    if prefer_real:
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore

            model_name = enc_cfg.get("openclip_model", "ViT-B-32")
            pretrained = enc_cfg.get("openclip_pretrained", "laion2b_s34b_b79k")

            class _OpenCLIPWrapper:
                name = f"openclip-{model_name}-{pretrained}"

                def __init__(self) -> None:
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained
                    )
                    self.tokenizer = open_clip.get_tokenizer(model_name)
                    self.dim = int(self.model.text_projection.shape[1])

                def encode_texts(self, texts):
                    with torch.no_grad():
                        feats = self.model.encode_text(self.tokenizer(list(texts)))
                    return _l2n(feats.float().cpu().numpy())

                def encode_images(self, uris, scenes=None):
                    feats = []
                    with torch.no_grad():
                        for uri in uris:
                            img = Image.open(uri).convert("RGB")
                            x = self.preprocess(img).unsqueeze(0)
                            feats.append(self.model.encode_image(x))
                    return _l2n(torch.cat(feats).float().cpu().numpy())

            wrapper = _OpenCLIPWrapper()
            return {
                "text": wrapper,
                "image": wrapper,
            }
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[encoders] OpenCLIP unavailable ({exc}); using hashing fallback.")
    seed = int(enc_cfg.get("seed", 167))
    return {
        "text": HashingTextEncoder(seed=seed),
        "image": HashingImageEncoder(seed=seed),
    }


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n
