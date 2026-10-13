"""Build and validate ``data/descriptions.csv`` from the source dataset.

Combines ``Caption_blip.txt`` with ``Training_set.csv`` into one long-format
CSV (one row per caption) and writes a validation report plus leakage-free
splits. Source files are only read.

    python src/build_dataset.py
    python src/build_dataset.py --max-index 1200 --report reports/dataset_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import Config, add_common_args, config_from_args
from dataset import (
    DatasetError,
    build_rows,
    discover_images,
    load_splits,
    make_splits,
    parse_captions,
    parse_labels,
    read_rows,
    save_splits,
    validate,
    write_rows,
    dataset_fingerprint,
)

DEFAULT_MAX_INDEX = 1200


def build(config: Config, max_index: int | None, write_splits: bool = True) -> tuple[list[dict[str, str]], object]:
    captions = parse_captions(config.caption_file, max_index=max_index)
    labels = parse_labels(config.labels_file)
    inventory = discover_images(config.images_dir)

    rows = build_rows(captions, labels, max_index=max_index)
    if not rows:
        raise DatasetError("no captions in range - check --max-index and the caption file")

    write_rows(config.combined_csv, rows)

    # Re-read from disk so validation checks the artefact that was written,
    # not the in-memory list.
    written = read_rows(config.combined_csv)
    report = validate(written, captions, labels, inventory)

    split_info = make_splits(written, max_index=None)
    if write_splits:
        save_splits(config.splits_file, split_info)
    report.extra = {
        "images": len({row["filename"] for row in written}),
        "species": len({row["label"] for row in written}),
        "split_images": split_info["counts"],  # type: ignore[index]
        "split_captions": split_info["caption_counts"],  # type: ignore[index]
        "split_species": split_info["species_counts"],  # type: ignore[index]
        "dataset_fingerprint": dataset_fingerprint(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in inventory.by_index.values()
        ),
    }
    return written, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument(
        "--max-index",
        type=int,
        default=DEFAULT_MAX_INDEX,
        help=f"highest image index to include (default: {DEFAULT_MAX_INDEX}; 0 means no limit)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="markdown report path (default: reports/dataset_report.md)",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="skip writing data/splits.json",
    )
    args = parser.parse_args(argv)

    config = config_from_args(args)
    config.ensure_dirs()
    max_index = args.max_index or None

    try:
        rows, report = build(config, max_index, write_splits=not args.no_splits)
    except DatasetError as exc:
        print(f"[build_dataset] error: {exc}", file=sys.stderr)
        return 1

    report_path = Path(args.report) if args.report else config.reports_dir / "dataset_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.render_markdown(), encoding="utf-8")
    config.data_out_dir.joinpath("dataset_report.json").write_text(
        __import__("json").dumps(report.as_dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[build_dataset] wrote {len(rows)} rows -> {config.combined_csv}")
    print(f"[build_dataset] splits -> {config.splits_file} {report.extra.get('split_images', {})}")
    print(f"[build_dataset] report -> {report_path}")
    print(
        f"[build_dataset] images={report.extra.get('images')} "
        f"species={report.extra.get('species')} "
        f"duplicate caption strings={report.duplicate_caption_strings} "
        f"templated={report.templated_captions}"
    )
    if not report.ok:
        print("[build_dataset] PROBLEMS FOUND:", file=sys.stderr)
        for problem in report.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    print("[build_dataset] validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
