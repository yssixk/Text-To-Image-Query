"""Dataset handling for the butterfly retrieval project.

This module reads the two source files (never writes them) and produces the
derived artefacts used downstream:

* ``Caption_blip.txt`` - blocks of the form::

      Image_1.jpg:
      1. A photo of a SOUTHERN DOGFACE butterfly. a yellow butterfly is on a small flower
      2. a yellow butterfly rests on a small flower
      ...

* ``Training_set.csv`` - ``filename,label`` rows mapping image to species.

Outputs: the combined long-format CSV (one row per caption), a validation
report, and species-stratified train/val/test splits keyed by image so all
captions of an image always land in the same split.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from config import IMAGE_EXTENSIONS

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DatasetError(RuntimeError):
    """Raised when a source file cannot be parsed as expected."""


class InvalidImageError(RuntimeError):
    """Raised when an image file cannot be opened as a readable RGB image."""


# --------------------------------------------------------------------------- #
# Source parsers
# --------------------------------------------------------------------------- #

_HEADER_RE = re.compile(r"^Image_(\d+)\.([A-Za-z0-9]+)\s*:?\s*$")
_NUMBERED_RE = re.compile(r"^\d+\s*[.):]\s*(.*)$")
_INDEX_RE = re.compile(r"(\d+)")


@dataclass
class CaptionBook:
    """Captions keyed by image index, plus the filename each header declared."""

    #: image index -> list of caption strings, in file order
    entries: dict[int, list[str]] = field(default_factory=dict)
    #: image index -> filename exactly as written in the header (e.g. ``Image_1.jpg``)
    filenames: dict[int, str] = field(default_factory=dict)
    #: non-fatal oddities ("line 12: caption without a number")
    warnings: list[str] = field(default_factory=list)
    #: headers seen more than once, with the number of blocks
    duplicate_blocks: dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def caption_count(self) -> int:
        return sum(len(v) for v in self.entries.values())


def parse_captions(path: Path, max_index: int | None = None) -> CaptionBook:
    """Parse a ``Caption_blip.txt`` style file.

    Blank lines separate blocks; a block starts at an ``Image_<n>.<ext>:``
    header and continues with numbered caption lines. Numbering prefixes such
    as ``1. `` are stripped and the caption text is kept verbatim.
    """
    if not path.is_file():
        raise DatasetError(f"caption file not found: {path}")

    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    book = CaptionBook()
    current: int | None = None
    for lineno, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.strip()
        if not line:
            current = None
            continue

        header = _HEADER_RE.match(line)
        if header:
            index = int(header.group(1))
            current = index
            book.duplicate_blocks[index] = book.duplicate_blocks.get(index, 0) + 1
            book.entries.setdefault(index, [])
            book.filenames.setdefault(index, line.rstrip(":").strip())
            continue

        if current is None:
            book.warnings.append(f"line {lineno}: text outside any image block, ignored: {line[:60]!r}")
            continue

        numbered = _NUMBERED_RE.match(line)
        if numbered:
            caption = numbered.group(1).strip()
        else:
            caption = line
            book.warnings.append(f"line {lineno}: caption without a number prefix")
        if caption:
            book.entries[current].append(caption)

    if not book.entries:
        raise DatasetError(f"no image blocks found in {path}")

    duplicated = {k: v for k, v in book.duplicate_blocks.items() if v > 1}
    book.duplicate_blocks = duplicated

    if max_index is not None:
        book.entries = {k: v for k, v in book.entries.items() if k <= max_index}
        book.filenames = {k: v for k, v in book.filenames.items() if k <= max_index}
    return book


@dataclass
class LabelBook:
    """Species labels keyed by image index, as read from ``Training_set.csv``."""

    labels: dict[int, str] = field(default_factory=dict)
    filenames: dict[int, str] = field(default_factory=dict)
    duplicated_filenames: list[str] = field(default_factory=list)

    def label_at(self, index: int) -> str | None:
        return self.labels.get(index)

    @property
    def species(self) -> set[str]:
        return set(self.labels.values())


def parse_labels(path: Path) -> LabelBook:
    """Read ``filename,label`` rows from ``Training_set.csv``."""
    if not path.is_file():
        raise DatasetError(f"label file not found: {path}")

    book = LabelBook()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{path} is empty")
        lowered = {name.strip().lower(): name for name in reader.fieldnames if name}
        if "filename" not in lowered or "label" not in lowered:
            raise DatasetError(
                f"{path} must have 'filename' and 'label' columns, found {reader.fieldnames}"
            )
        filename_key, label_key = lowered["filename"], lowered["label"]
        for row in reader:
            filename = (row.get(filename_key) or "").strip()
            label = (row.get(label_key) or "").strip()
            if not filename:
                continue
            match = _INDEX_RE.search(filename)
            if match is None:
                raise DatasetError(f"cannot read an image index from filename {filename!r}")
            index = int(match.group(1))
            if filename in seen:
                book.duplicated_filenames.append(filename)
            seen.add(filename)
            book.labels[index] = label
            book.filenames[index] = filename
    if not book.labels:
        raise DatasetError(f"no label rows parsed from {path}")
    return book


# --------------------------------------------------------------------------- #
# Image discovery
# --------------------------------------------------------------------------- #


@dataclass
class ImageInventory:
    """Image files found on disk, keyed by the numeric index in the filename."""

    by_index: dict[int, Path] = field(default_factory=dict)
    duplicates: dict[int, list[Path]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    extension_counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.by_index)


def discover_images(images_dir: Path) -> ImageInventory:
    """Find images in ``images_dir``, accepting JPG, JPEG and PNG."""
    if not images_dir.is_dir():
        raise DatasetError(f"image folder not found: {images_dir}")

    inventory = ImageInventory()
    for entry in sorted(images_dir.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            inventory.unmatched.append(entry.name)
            continue
        inventory.extension_counts[suffix] = inventory.extension_counts.get(suffix, 0) + 1
        match = _INDEX_RE.search(entry.stem)
        if match is None:
            inventory.unmatched.append(entry.name)
            continue
        index = int(match.group(1))
        if index in inventory.by_index:
            inventory.duplicates.setdefault(index, [inventory.by_index[index]]).append(entry)
            continue
        inventory.by_index[index] = entry
    return inventory


def open_image(path: Path) -> "object":
    """Open an image as RGB, raising :class:`InvalidImageError` when unreadable."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as handle:
            handle.load()
            return handle.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"{path.name}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Combined CSV
# --------------------------------------------------------------------------- #

CSV_COLUMNS = ("filename", "label", "caption", "sentence_id", "source")
CAPTION_SOURCE = "blip"
TEMPLATE_PREFIX = "A photo of a "


def sentence_id(filename: str, position: int) -> str:
    """``Image_3251.jpg`` + 1 -> ``Image_3251_c1``."""
    return f"{Path(filename).stem}_c{position}"


def build_rows(
    captions: CaptionBook,
    labels: LabelBook,
    max_index: int | None = None,
) -> list[dict[str, str]]:
    """One row per caption, combining the caption text with its species label."""
    rows: list[dict[str, str]] = []
    for index in sorted(captions.entries):
        if max_index is not None and index > max_index:
            continue
        filename = captions.filenames.get(index) or labels.filenames.get(index)
        if filename is None:
            continue
        label = labels.label_at(index) or ""
        for position, caption in enumerate(captions.entries[index], start=1):
            rows.append(
                {
                    "filename": filename,
                    "label": label,
                    "caption": caption,
                    "sentence_id": sentence_id(filename, position),
                    "source": CAPTION_SOURCE,
                }
            )
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write the combined CSV (quoted properly for the captions containing commas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read the combined CSV written by :func:`write_rows`."""
    if not path.is_file():
        raise DatasetError(
            f"combined CSV not found: {path} - run 'python src/build_dataset.py' first"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DatasetError(f"{path} has no data rows")
    missing = [c for c in CSV_COLUMNS if c not in rows[0]]
    if missing:
        raise DatasetError(f"{path} is missing columns: {missing}")
    return rows


def group_captions_by_image(rows: Iterable[dict[str, str]]) -> dict[str, list[str]]:
    """``{filename: [caption, ...]}`` preserving the row order."""
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["filename"], []).append(row["caption"])
    return grouped


def label_by_image(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    """``{filename: species}`` taken from the combined CSV."""
    return {row["filename"]: row.get("label", "") for row in rows}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass
class ValidationReport:
    """Everything the loader noticed about the dataset."""

    caption_images: int = 0
    caption_strings: int = 0
    label_rows: int = 0
    image_files: int = 0
    rows: int = 0
    unique_sentence_ids: int = 0
    duplicate_sentence_ids: list[str] = field(default_factory=list)
    label_mismatches: list[str] = field(default_factory=list)
    captioned_without_label: list[int] = field(default_factory=list)
    labeled_without_caption: list[int] = field(default_factory=list)
    captioned_without_image: list[int] = field(default_factory=list)
    image_without_row: list[str] = field(default_factory=list)
    duplicate_image_indices: dict[int, list[str]] = field(default_factory=dict)
    unmatched_files: list[str] = field(default_factory=list)
    captions_missing_from_csv: int = 0
    duplicate_caption_strings: int = 0
    templated_captions: int = 0
    source_warnings: list[str] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.label_mismatches

    def as_dict(self) -> dict[str, object]:
        data = {k: v for k, v in self.__dict__.items()}
        data["ok"] = self.ok
        return data

    def render_markdown(self, title: str = "Dataset report") -> str:
        lines = [f"# {title}", ""]
        lines.append("## Sources (as read)")
        lines.append("")
        lines.append("| Source | Count |")
        lines.append("|---|---|")
        lines.append(f"| Images with captions | {self.caption_images} |")
        lines.append(f"| Caption strings | {self.caption_strings} |")
        lines.append(f"| Label rows (`Training_set.csv`) | {self.label_rows} |")
        lines.append(f"| Image files on disk | {self.image_files} |")
        lines.append("")
        lines.append("## Combined CSV")
        lines.append("")
        lines.append("| Check | Value |")
        lines.append("|---|---|")
        lines.append(f"| Data rows | {self.rows} |")
        lines.append(f"| Unique sentence ids | {self.unique_sentence_ids} |")
        lines.append(f"| Duplicate sentence ids | {len(self.duplicate_sentence_ids)} |")
        lines.append(f"| Source captions missing from CSV | {self.captions_missing_from_csv} |")
        lines.append(f"| Label mismatches | {len(self.label_mismatches)} |")
        lines.append(f"| Repeated caption strings | {self.duplicate_caption_strings} |")
        lines.append(f"| Templated 'A photo of a ...' captions | {self.templated_captions} |")
        lines.append("")
        lines.append("## Coverage")
        lines.append("")
        lines.append("| Check | Value |")
        lines.append("|---|---|")
        lines.append(f"| Captioned images without a label | {len(self.captioned_without_label)} |")
        lines.append(f"| Labelled images without captions | {len(self.labeled_without_caption)} |")
        lines.append(f"| Captioned images with no file on disk | {len(self.captioned_without_image)} |")
        lines.append(f"| Image files with no CSV row | {len(self.image_without_row)} |")
        lines.append(f"| Files not matching Image_<n>.<ext> | {len(self.unmatched_files)} |")
        lines.append(f"| Repeated image indices on disk | {len(self.duplicate_image_indices)} |")
        lines.append("")
        if self.extra:
            lines.append("## Extras")
            lines.append("")
            for key, value in self.extra.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        if self.problems:
            lines.append("## Problems")
            lines.append("")
            lines.extend(f"- {problem}" for problem in self.problems)
            lines.append("")
        if self.source_warnings:
            lines.append("## Parser warnings")
            lines.append("")
            lines.extend(f"- {warning}" for warning in self.source_warnings[:20])
            lines.append("")
        return "\n".join(lines) + "\n"


def validate(
    rows: Sequence[dict[str, str]],
    captions: CaptionBook,
    labels: LabelBook,
    inventory: ImageInventory,
) -> ValidationReport:
    """Cross-check the combined CSV against all three sources."""
    report = ValidationReport(
        caption_images=len(captions),
        caption_strings=captions.caption_count,
        label_rows=len(labels.labels),
        image_files=len(inventory),
        rows=len(rows),
        source_warnings=list(captions.warnings),
        duplicate_image_indices={
            index: [path.name for path in paths] for index, paths in inventory.duplicates.items()
        },
        unmatched_files=list(inventory.unmatched),
    )

    seen_ids: set[str] = set()
    caption_pool: dict[str, int] = {}
    for row in rows:
        sid = row.get("sentence_id", "")
        if sid in seen_ids:
            report.duplicate_sentence_ids.append(sid)
        seen_ids.add(sid)
        caption_pool[row.get("caption", "")] = caption_pool.get(row.get("caption", ""), 0) + 1
        if row.get("caption", "").startswith(TEMPLATE_PREFIX):
            report.templated_captions += 1
    report.unique_sentence_ids = len(seen_ids)
    report.duplicate_caption_strings = sum(count - 1 for count in caption_pool.values() if count > 1)

    # every source caption must appear exactly once in the CSV
    source_captions: dict[str, int] = {}
    for index, caption_list in captions.entries.items():
        filename = captions.filenames.get(index, f"Image_{index}.jpg")
        for caption in caption_list:
            key = f"{filename}\u0000{caption}"
            source_captions[key] = source_captions.get(key, 0) + 1
    csv_captions: dict[str, int] = {}
    for row in rows:
        key = f"{row['filename']}\u0000{row['caption']}"
        csv_captions[key] = csv_captions.get(key, 0) + 1
    report.captions_missing_from_csv = sum(
        max(0, count - csv_captions.get(key, 0)) for key, count in source_captions.items()
    )

    for row in rows:
        filename = row["filename"]
        match = _INDEX_RE.search(filename)
        index = int(match.group(1)) if match else -1
        if row.get("label", "") != (labels.label_at(index) or ""):
            report.label_mismatches.append(
                f"{filename}: csv={row.get('label')!r} source={labels.label_at(index)!r}"
            )

    caption_indices = set(captions.entries)
    report.captioned_without_label = sorted(caption_indices - set(labels.labels))
    caption_max = max(caption_indices) if caption_indices else 0
    report.labeled_without_caption = sorted(
        index for index in labels.labels if index <= caption_max and index not in caption_indices
    )
    report.captioned_without_image = sorted(caption_indices - set(inventory.by_index))
    csv_indices = {
        int(_INDEX_RE.search(row["filename"]).group(1))
        for row in rows
        if _INDEX_RE.search(row["filename"])
    }
    report.image_without_row = sorted(
        inventory.by_index[index].name
        for index in set(inventory.by_index) - csv_indices
    )

    if report.rows != report.caption_strings:
        report.problems.append(
            f"CSV has {report.rows} rows but the caption source holds {report.caption_strings} captions"
        )
    if report.duplicate_sentence_ids:
        report.problems.append(f"{len(report.duplicate_sentence_ids)} duplicate sentence ids")
    if report.captions_missing_from_csv:
        report.problems.append(f"{report.captions_missing_from_csv} source captions missing from the CSV")
    if report.label_mismatches:
        report.problems.append(f"{len(report.label_mismatches)} label mismatches against Training_set.csv")
    if report.captioned_without_image:
        report.problems.append(
            f"{len(report.captioned_without_image)} captioned images have no file on disk"
        )
    return report


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def make_splits(
    rows: Sequence[dict[str, str]],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    max_index: int | None = None,
) -> dict[str, object]:
    """Species-stratified splits keyed by image, so captions never leak.

    Each image appears in exactly one split and all of its captions follow it.
    Groups are shuffled with a seed derived from the species name, which keeps
    the split reproducible while still mixing the file order.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")

    grouped = group_captions_by_image(rows)
    labels = label_by_image(rows)
    names = ["train", "val", "test"]
    splits: dict[str, list[str]] = {name: [] for name in names}

    by_label: dict[str, list[str]] = {}
    for filename, label in labels.items():
        by_label.setdefault(label, []).append(filename)

    for label, filenames in sorted(by_label.items()):
        if max_index is not None:
            filenames = [
                name
                for name in filenames
                if (_INDEX_RE.search(name) and int(_INDEX_RE.search(name).group(1)) <= max_index)
            ]
        if not filenames:
            continue
        if label not in ("", None):
            filenames = sorted(filenames, key=lambda name: int(_INDEX_RE.search(name).group(1)))
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(filenames)

        total = len(filenames)
        raw = [total * ratio for ratio in ratios]
        counts = [int(value) for value in raw]
        remainder = total - sum(counts)
        order = sorted(range(len(ratios)), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:remainder]:
            counts[i] += 1

        cursor = 0
        for name, count in zip(names, counts):
            splits[name].extend(filenames[cursor : cursor + count])
            cursor += count

    # A dataset small enough to starve a split would silently produce no
    # validation signal, so borrow an image from train instead.
    for name in ("val", "test"):
        if not splits[name] and splits["train"]:
            borrowed = splits["train"].pop()
            splits[name].append(borrowed)

    overlap = (set(splits["train"]) & set(splits["val"])) | (
        set(splits["train"]) & set(splits["test"])
    ) | (set(splits["val"]) & set(splits["test"]))
    if overlap:
        raise DatasetError(f"split leakage detected: {sorted(overlap)[:5]}")

    covered = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    missing = set(grouped) - covered
    if missing:
        raise DatasetError(f"{len(missing)} images were assigned to no split")

    for name in names:
        splits[name] = sorted(splits[name])

    return {
        "seed": seed,
        "ratios": list(ratios),
        "key": "image",
        "counts": {name: len(splits[name]) for name in names},
        "caption_counts": {
            name: sum(len(grouped.get(filename, [])) for filename in splits[name]) for name in names
        },
        "species_counts": {
            name: len({labels.get(filename, "") for filename in splits[name]}) for name in names
        },
        "splits": splits,
    }


def save_splits(path: Path, splits: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")


def load_splits(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise DatasetError(f"split file not found: {path} - run 'python src/build_dataset.py' first")
    return json.loads(path.read_text(encoding="utf-8"))


def split_members(splits: dict[str, object], name: str) -> list[str]:
    """Filenames of one split; ``all`` means every image in the split file."""
    if name == "all":
        members: list[str] = []
        for key in ("train", "val", "test"):
            members.extend(splits["splits"].get(key, []))  # type: ignore[index]
        return sorted(members)
    if name not in ("train", "val", "test"):
        raise DatasetError(f"unknown split {name!r}; expected train, val, test or all")
    return list(splits["splits"].get(name, []))  # type: ignore[index]


def dataset_fingerprint(entries: Iterable[tuple[str, int, int]]) -> str:
    """Stable hash over (filename, size, mtime_ns) triples."""
    import hashlib

    digest = hashlib.sha1()
    for filename, size, mtime_ns in sorted(entries):
        digest.update(f"{filename}|{size}|{mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #


class CaptionImageDataset:
    """``(image, caption)`` pairs for fine-tuning.

    torch is imported lazily so the data-building and validation scripts still
    run in an environment that has no PyTorch installed.
    """

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        images_dir: Path,
        processor: object,
        filenames: Sequence[str] | None = None,
        seed: int = 0,
        max_length: int = 77,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DatasetError("PyTorch is required for the training dataset") from exc

        wanted = set(filenames) if filenames is not None else None
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.samples = [
            (row["filename"], row["caption"], row.get("label", ""))
            for row in rows
            if wanted is None or row["filename"] in wanted
        ]
        if not self.samples:
            raise DatasetError("no (image, caption) pairs selected for the dataset")
        self.seed = seed
        self.epoch = 0
        # CLIP has a fixed text context length; pad to it so every sample has the
        # same token count and batches can be stacked without custom collation.
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, item: int) -> dict[str, object]:
        filename, caption, label = self.samples[item]
        image = open_image(self.images_dir / filename)
        encoded = self.processor(  # type: ignore[operator]
            text=[caption],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value[0] for key, value in encoded.items()}
        batch["filename"] = filename
        batch["label"] = label
        return batch
