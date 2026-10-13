"""Spec acceptance checks that need fixtures rather than the real dataset.

Covers: JPG/JPEG/PNG discovery and decoding, unreadable-image handling, the
loader's reporting of missing descriptions / missing files / stray files, the
image-level split partition (no leakage), and the CUDA->CPU device fallback.
Runs against a throw-away temp dataset; the real dataset is never touched.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset import (  # noqa: E402
    InvalidImageError,
    build_rows,
    discover_images,
    load_splits,
    open_image,
    parse_captions,
    parse_labels,
    read_rows,
    split_members,
    validate,
)

FAILURES: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    suffix = "" if ok else f"   (expected {want!r})"
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {got!r}{suffix}")
    if not ok:
        FAILURES.append(label)


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    images = root / "training_image"
    images.mkdir(parents=True)
    Image.new("RGB", (24, 24), (10, 90, 30)).save(images / "Image_1.jpg", "JPEG")
    Image.new("RGB", (24, 24), (200, 200, 40)).save(images / "Image_2.jpeg", "JPEG")
    Image.new("RGB", (24, 24), (240, 240, 240)).save(images / "Image_3.png", "PNG")
    (images / "Image_4.jpg").write_bytes(b"this is not a jpeg")
    Image.new("RGB", (24, 24), (60, 60, 200)).save(images / "Image_5.jpg", "JPEG")
    (images / "notes.txt").write_text("stray file\n", encoding="utf-8")

    captions = root / "Caption_blip.txt"
    captions.write_text(
        "Image_1.jpg:\n"
        "1. a blue butterfly on a leaf\n"
        "2. a butterfly with spotted wings\n\n"
        "Image_2.jpeg:\n"
        "1. a yellow butterfly\n\n"
        "Image_3.png:\n"
        "1. a white butterfly flying\n\n"
        "Image_9.jpg:\n"
        "1. a butterfly whose file is missing\n",
        encoding="utf-8",
    )

    labels = root / "Training_set.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label"])
        for name, label in (
            ("Image_1.jpg", "ALPHA"),
            ("Image_2.jpeg", "BETA"),
            ("Image_3.png", "GAMMA"),
            ("Image_5.jpg", "EPSILON"),
            ("Image_9.jpg", "DELTA"),
        ):
            writer.writerow([name, label])
    return images, captions, labels


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        images_dir, captions_path, labels_path = make_fixture(Path(tmp))

        print("1. image discovery: .jpg/.jpeg/.png accepted, stray files flagged")
        inventory = discover_images(images_dir)
        check(
            "image files found",
            sorted(path.name for path in inventory.by_index.values()),
            ["Image_1.jpg", "Image_2.jpeg", "Image_3.png", "Image_4.jpg", "Image_5.jpg"],
        )
        check("extension counts", inventory.extension_counts, {".jpg": 3, ".jpeg": 1, ".png": 1})
        check("non-image file flagged", inventory.unmatched, ["notes.txt"])
        check("duplicate indices", inventory.duplicates, {})

        print("2. decoding: jpg/jpeg/png open, corrupt file raises")
        for name in ("Image_1.jpg", "Image_2.jpeg", "Image_3.png"):
            image = open_image(images_dir / name)
            check(f"{name} decodes to RGB", (image.mode, image.size), ("RGB", (24, 24)))
        try:
            open_image(images_dir / "Image_4.jpg")
        except InvalidImageError as exc:
            check("corrupt image raises InvalidImageError", "Image_4.jpg" in str(exc), True)
        else:
            check("corrupt image raises InvalidImageError", "no exception raised", "InvalidImageError")

        print("3. loader reporting: missing descriptions, missing files, stray files")
        captions = parse_captions(captions_path)
        labels = parse_labels(labels_path)
        rows = build_rows(captions, labels)
        check("one CSV row per caption", len(rows), 5)
        check(
            "labels joined onto captions",
            sorted({(row["filename"], row["label"]) for row in rows}),
            [
                ("Image_1.jpg", "ALPHA"),
                ("Image_2.jpeg", "BETA"),
                ("Image_3.png", "GAMMA"),
                ("Image_9.jpg", "DELTA"),
            ],
        )
        report = validate(rows, captions, labels, inventory)
        check("captioned images", report.caption_images, 4)
        check("captions read", report.caption_strings, 5)
        check("captioned image with no file on disk", report.captioned_without_image, [9])
        check(
            "image present but has no description",
            sorted(report.image_without_row),
            ["Image_4.jpg", "Image_5.jpg"],
        )
        check("label with no caption in caption range", report.labeled_without_caption, [5])
        check("no label mismatches", report.label_mismatches, [])
        check("stray file surfaced", report.unmatched_files, ["notes.txt"])
        check("report ok", report.ok, False)
        check("problem text", report.problems, ["1 captioned images have no file on disk"])

        print("4. encoder skips the unreadable image and keeps jpeg/png")
        from model import CLIPEmbedder

        embedder = CLIPEmbedder("openai/clip-vit-base-patch32", device="cuda", verbose=False)
        paths = [
            images_dir / name
            for name in ("Image_1.jpg", "Image_2.jpeg", "Image_3.png", "Image_4.jpg", "Image_5.jpg")
        ]
        vectors, kept, failures = embedder.encode_images(paths, verbose=False)
        check("encoded matrix", vectors.shape, (4, 512))
        check("kept positions (corrupt index 3 dropped)", kept, [0, 1, 2, 4])
        check("failure reported", len(failures) == 1 and "Image_4.jpg" in failures[0], True)
        check(
            "embeddings unit-norm",
            bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)),
            True,
        )

        print("5. splits: image-level partition, complete, species-stratified")
        from config import Config

        config = Config.load()
        splits = load_splits(config.splits_file)
        groups = {name: set(split_members(splits, name)) for name in ("train", "val", "test")}
        check("train n val", groups["train"] & groups["val"], set())
        check("train n test", groups["train"] & groups["test"], set())
        check("val n test", groups["val"] & groups["test"], set())
        check(
            "no file listed twice",
            sum(len(members) for members in groups.values()),
            len(set().union(*groups.values())),
        )
        all_rows = read_rows(config.combined_csv)
        every_image = {row["filename"] for row in all_rows}
        check("splits cover every captioned image", set().union(*groups.values()), every_image)
        label_by_name: dict[str, str] = {}
        for row in all_rows:
            label_by_name.setdefault(row["filename"], row["label"])
        check(
            "every split holds every species",
            {name: len({label_by_name[n] for n in members}) for name, members in groups.items()},
            {"train": 75, "val": 75, "test": 75},
        )
        check("split counts recorded", splits.get("counts"), {"train": 957, "val": 137, "test": 106})

        print("6. device resolution and CPU fallback")
        import torch

        from model import resolve_device

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        check("auto picks the available device", resolve_device("auto"), expected)
        real_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False  # simulate a machine without CUDA
        try:
            check("cuda requested but unavailable -> cpu", resolve_device("cuda"), "cpu")
            check("auto without cuda -> cpu", resolve_device("auto"), "cpu")
        finally:
            torch.cuda.is_available = real_available

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL SPEC ACCEPTANCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
