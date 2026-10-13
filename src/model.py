"""CLIP loading and embedding helpers.

The model is Transformers' ``CLIPModel`` with ``AutoProcessor``. In
Transformers 5.x ``get_image_features`` / ``get_text_features`` return a
``BaseModelOutputWithPooling`` rather than a bare tensor, so :func:`_to_tensor`
accepts either shape and pulls the projected vector out of ``pooler_output``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from config import DEFAULT_BATCH_SIZE
from dataset import InvalidImageError, open_image


class ModelLoadError(RuntimeError):
    """Raised when the CLIP checkpoint cannot be loaded."""


def resolve_device(requested: str = "auto") -> str:
    """``auto`` picks CUDA when available, otherwise CPU."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[model] CUDA requested but unavailable - falling back to CPU")
        return "cpu"
    return requested


def default_batch_size(device: str) -> int:
    return DEFAULT_BATCH_SIZE["cuda"] if device.startswith("cuda") else DEFAULT_BATCH_SIZE["cpu"]


def _to_tensor(output: object) -> "object":
    """Pull the projected embedding tensor out of a CLIP feature output."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    for attribute in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    raise ModelLoadError(f"cannot read an embedding tensor from {type(output).__name__}")


class CLIPEmbedder:
    """Loads CLIP once and encodes images and texts into unit-norm vectors."""

    def __init__(
        self,
        model_id: str,
        checkpoint_dir: Path | None = None,
        device: str = "auto",
        batch_size: int | None = None,
        verbose: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelLoadError(
                "PyTorch and Transformers are required; create the environment with "
                "'python -m venv .venv' and install requirements.txt"
            ) from exc

        self._torch = torch
        self.device = resolve_device(device)
        self.batch_size = batch_size or default_batch_size(self.device)
        self.source = str(checkpoint_dir) if checkpoint_dir else model_id
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        if verbose:
            print(f"[model] loading {self.source} on {self.device}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.source)
            self.model = CLIPModel.from_pretrained(self.source)
        except Exception as exc:  # noqa: BLE001 - surface any download/config failure clearly
            raise ModelLoadError(f"failed to load {self.source}: {exc}") from exc
        self.model.eval()
        self.model.to(self.device)
        self.embed_dim = int(self.model.config.projection_dim)
        # The processor's image size drives the cache fingerprint.
        image_size = getattr(getattr(self.processor, "image_processor", None), "size", {})
        self.image_size = tuple(sorted(image_size.items())) if isinstance(image_size, dict) else ()

    # ------------------------------------------------------------------ helpers
    @property
    def uses_cuda(self) -> bool:
        return str(self.device).startswith("cuda")

    def _autocast(self):
        torch = self._torch
        if self.uses_cuda:
            return torch.autocast("cuda", dtype=torch.float16)
        import contextlib

        return contextlib.nullcontext()

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        return (array / np.clip(norms, 1e-12, None)).astype(np.float32)

    def _text_batches(self, texts: Sequence[str]) -> Iterable[object]:
        for start in range(0, len(texts), 256):
            chunk = list(texts[start : start + 256])
            yield self.processor(
                text=chunk, padding=True, truncation=True, max_length=77, return_tensors="pt"
            )

    # ------------------------------------------------------------------- encode
    def encode_texts(self, texts: Sequence[str], verbose: bool = False) -> np.ndarray:
        """Encode a list of strings into ``(N, embed_dim)`` unit-norm vectors."""
        torch = self._torch
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        cleaned = [text if isinstance(text, str) and text.strip() else " " for text in texts]
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in self._text_batches(cleaned):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with self._autocast():
                    features = self.model.get_text_features(**batch)
                tensor = _to_tensor(features)
                chunks.append(tensor.detach().float().cpu().numpy())
                if verbose:
                    print(f"[model] encoded {sum(len(c) for c in chunks)}/{len(texts)} texts", end="\r")
        if verbose:
            print()
        stacked = np.concatenate(chunks, axis=0)
        return self._normalize(stacked)

    def encode_images(
        self,
        paths: Sequence[Path],
        batch_size: int | None = None,
        verbose: bool = True,
        skip_invalid: bool = True,
    ) -> tuple[np.ndarray, list[int], list[str]]:
        """Encode image files.

        Returns ``(embeddings, kept_positions, failures)`` where
        ``kept_positions`` indexes into ``paths`` so callers can drop the files
        that could not be decoded.
        """
        torch = self._torch
        size = batch_size or self.batch_size
        vectors: list[np.ndarray] = []
        kept: list[int] = []
        failures: list[str] = []
        done = 0
        with torch.inference_mode():
            for start in range(0, len(paths), size):
                window = paths[start : start + size]
                images = []
                positions = []
                for offset, path in enumerate(window):
                    try:
                        images.append(open_image(Path(path)))
                        positions.append(start + offset)
                    except InvalidImageError as exc:
                        failures.append(str(exc))
                        if not skip_invalid:
                            raise
                if not images:
                    continue
                batch = self.processor(images=images, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with self._autocast():
                    features = self.model.get_image_features(**batch)
                tensor = _to_tensor(features)
                vectors.append(tensor.detach().float().cpu().numpy())
                kept.extend(positions)
                done += len(images)
                if verbose:
                    print(f"[model] encoded {done}/{len(paths)} images", end="\r")
        if verbose:
            print()
        if not vectors:
            return np.zeros((0, self.embed_dim), dtype=np.float32), kept, failures
        return self._normalize(np.concatenate(vectors, axis=0)), kept, failures
