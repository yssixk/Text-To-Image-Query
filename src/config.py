"""Central configuration for the butterfly text-to-image retrieval project.

Every path the project touches is derived from this module, so the same code
runs from a local checkout or from a Colab notebook: point ``BF_DATA_DIR`` (or
pass ``--data-dir``) at the folder holding ``training_image/``,
``Caption_blip.txt`` and ``Training_set.csv``.

The source dataset is only ever *read*. Derived artefacts (the combined CSV,
splits, embeddings, models, reports) are written under ``project_root``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

DATA_DIR_ENV = "BF_DATA_DIR"

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CAPTION_FILE = "Caption_blip.txt"
DEFAULT_LABELS_FILE = "Training_set.csv"
DEFAULT_IMAGES_DIR = "training_image"

COMBINED_CSV_NAME = "descriptions.csv"
SPLITS_NAME = "splits.json"

#: Image formats the loader accepts (lowercase, with the dot).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

#: Default batch sizes, tuned to stay inside a 4 GB laptop GPU.
DEFAULT_BATCH_SIZE = {"cuda": 32, "cpu": 8}


@dataclass(frozen=True)
class Config:
    """Resolved paths and model settings for one run."""

    project_root: Path
    data_dir: Path
    model_id: str = DEFAULT_MODEL_ID
    checkpoint_dir: Path | None = None
    device: str = "auto"
    batch_size: int | None = None

    # ------------------------------------------------------------------ sources
    @property
    def images_dir(self) -> Path:
        return self.data_dir / DEFAULT_IMAGES_DIR

    @property
    def caption_file(self) -> Path:
        return self.data_dir / DEFAULT_CAPTION_FILE

    @property
    def labels_file(self) -> Path:
        return self.data_dir / DEFAULT_LABELS_FILE

    # ------------------------------------------------------------- derived data
    @property
    def data_out_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def combined_csv(self) -> Path:
        return self.data_out_dir / COMBINED_CSV_NAME

    @property
    def splits_file(self) -> Path:
        return self.data_out_dir / SPLITS_NAME

    @property
    def embeddings_dir(self) -> Path:
        return self.project_root / "embeddings"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    # ------------------------------------------------------------------ helpers
    def model_label(self) -> str:
        """Short human-readable identifier for the model in use."""
        if self.checkpoint_dir is not None:
            return f"{self.model_id} (fine-tuned: {self.checkpoint_dir})"
        return self.model_id

    def ensure_dirs(self) -> None:
        for path in (
            self.data_out_dir,
            self.embeddings_dir,
            self.models_dir,
            self.reports_dir,
            self.outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **kwargs: object) -> "Config":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    # ------------------------------------------------------------------- load
    @classmethod
    def load(
        cls,
        data_dir: str | os.PathLike[str] | None = None,
        project_root: str | os.PathLike[str] | None = None,
        model_id: str | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> "Config":
        """Build a config, resolving defaults from the environment.

        ``data_dir`` falls back to ``$BF_DATA_DIR`` and then to the project
        root, which is where this repository keeps its dataset today.
        """
        root = Path(project_root) if project_root else default_project_root()
        resolved_data = data_dir or os.environ.get(DATA_DIR_ENV) or root
        resolved_ckpt = Path(checkpoint_dir) if checkpoint_dir else None
        return cls(
            project_root=root,
            data_dir=Path(resolved_data),
            model_id=model_id or DEFAULT_MODEL_ID,
            checkpoint_dir=resolved_ckpt,
            device=device or "auto",
            batch_size=batch_size,
        )


def default_project_root() -> Path:
    """The repository root (the parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the flags every entry point shares."""
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"folder holding training_image/, {DEFAULT_CAPTION_FILE} and "
        f"{DEFAULT_LABELS_FILE} (default: ${DATA_DIR_ENV} or the project root)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"CLIP checkpoint id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="local fine-tuned checkpoint directory (overrides --model)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="compute device (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="encode/train batch size (default: 32 on GPU, 8 on CPU)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Config from parsed ``add_common_args`` flags."""
    return Config.load(
        data_dir=getattr(args, "data_dir", None),
        model_id=getattr(args, "model", None),
        checkpoint_dir=getattr(args, "checkpoint", None),
        device=getattr(args, "device", "auto"),
        batch_size=getattr(args, "batch_size", None),
    )
