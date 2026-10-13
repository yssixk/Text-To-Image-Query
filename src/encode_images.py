"""Precompute CLIP image embeddings and cache them on disk.

    python src/encode_images.py                 # encode, reusing the cache when valid
    python src/encode_images.py --force         # re-encode regardless
    python src/encode_images.py --no-faiss      # skip the FAISS index

The cache is keyed by a fingerprint over the image set (name, size, mtime) and
the model, so embeddings are recomputed only when the dataset or the model
changes.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import Config, add_common_args, config_from_args
from dataset import (
    DatasetError,
    dataset_fingerprint,
    discover_images,
    read_rows,
)

EMBEDDINGS_NAME = "image_embeds.npy"
FILENAMES_NAME = "filenames.json"
META_NAME = "meta.json"
FAISS_NAME = "index.faiss"


def load_filenames(embeddings_dir: Path) -> list[str]:
    payload = json.loads((embeddings_dir / FILENAMES_NAME).read_text(encoding="utf-8"))
    return list(payload["filenames"])


def load_labels(embeddings_dir: Path) -> list[str]:
    payload = json.loads((embeddings_dir / FILENAMES_NAME).read_text(encoding="utf-8"))
    return list(payload.get("labels", [""] * len(payload["filenames"])))


def read_meta(embeddings_dir: Path) -> dict[str, object]:
    meta_path = embeddings_dir / META_NAME
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def model_mismatch(config: Config, embedder: object | None = None) -> str | None:
    """Why the cached vectors come from a different model, or ``None``.

    Two models put their embeddings in unrelated vector spaces, so ranking one
    against the other compares numbers that were never meant to be compared --
    a silent way to get wrong results instead of an error.
    """
    meta = read_meta(config.embeddings_dir)
    cached_model = meta.get("model_id")
    if cached_model and cached_model != config.model_id:
        return f"cache was built with {cached_model}, not {config.model_id}"
    if str(meta.get("checkpoint_dir") or "") != str(config.checkpoint_dir or ""):
        return "cache was built with a different checkpoint"
    if embedder is not None and int(meta.get("embed_dim", -1)) != int(embedder.embed_dim):
        return "embedding dimension changed"
    return None


def cache_is_valid(config: Config, embedder: object | None = None) -> tuple[bool, str]:
    """Is the cached embedding set usable for the current dataset and model?"""
    embeddings_dir = config.embeddings_dir
    for name in (EMBEDDINGS_NAME, FILENAMES_NAME, META_NAME):
        if not (embeddings_dir / name).is_file():
            return False, f"missing {name}"
    mismatch = model_mismatch(config, embedder)
    if mismatch:
        return False, mismatch
    if read_meta(embeddings_dir).get("fingerprint") != current_fingerprint(config):
        return False, "image set changed"
    return True, "cache is current"


def current_fingerprint(config: Config) -> str:
    inventory = discover_images(config.images_dir)
    rows = read_rows(config.combined_csv)
    wanted = {row["filename"] for row in rows}
    entries = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in inventory.by_index.values()
        if path.name in wanted
    ]
    return dataset_fingerprint(entries)


def build_faiss_index(embeddings: np.ndarray, path: Path) -> tuple[bool, str]:
    try:
        import faiss  # type: ignore
    except ImportError:
        return False, "faiss not installed"
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(path))
    return True, f"{index.ntotal} vectors"


def run(
    config: Config,
    force: bool = False,
    limit: int | None = None,
    use_faiss: bool = True,
    quiet: bool = False,
) -> dict[str, object]:
    """Encode every image that has a caption, writing the cache."""
    from model import CLIPEmbedder

    rows = read_rows(config.combined_csv)
    inventory = discover_images(config.images_dir)
    labels = {row["filename"]: row.get("label", "") for row in rows}

    wanted: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = row["filename"]
        if name in seen:
            continue
        seen.add(name)
        wanted.append(name)
    if limit:
        wanted = wanted[:limit]

    present = [name for name in wanted if _index_of(name) in inventory.by_index]
    missing = [name for name in wanted if name not in set(present)]
    paths = [inventory.by_index[_index_of(name)] for name in present]

    if not force and not limit:
        cached, reason = cache_is_valid(config)
        if cached:
            meta = read_meta(config.embeddings_dir)
            if not quiet:
                print(f"[encode_images] {reason}: reusing {meta.get('count')} embeddings "
                      f"in {config.embeddings_dir}")
            return {"reused": True, "count": meta.get("count"), "meta": meta}

    embedder = CLIPEmbedder(
        config.model_id,
        checkpoint_dir=config.checkpoint_dir,
        device=config.device,
        batch_size=config.batch_size,
        verbose=not quiet,
    )
    if not quiet:
        print(f"[encode_images] encoding {len(paths)} images "
              f"(missing files: {len(missing)}, batch size: {embedder.batch_size})")

    started = time.perf_counter()
    embeddings, kept, failures = embedder.encode_images(paths, verbose=not quiet)
    elapsed = time.perf_counter() - started
    if embeddings.shape[0] == 0:
        raise DatasetError("no images could be encoded")

    filenames = [present[position] for position in kept]
    filename_labels = [labels.get(name, "") for name in filenames]

    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.embeddings_dir / EMBEDDINGS_NAME, embeddings.astype(np.float32))
    (config.embeddings_dir / FILENAMES_NAME).write_text(
        json.dumps({"filenames": filenames, "labels": filename_labels}, indent=2) + "\n",
        encoding="utf-8",
    )

    faiss_ok, faiss_note = (False, "skipped")
    if use_faiss:
        faiss_ok, faiss_note = build_faiss_index(embeddings, config.embeddings_dir / FAISS_NAME)

    meta = {
        "model_id": config.model_id,
        "checkpoint_dir": str(config.checkpoint_dir) if config.checkpoint_dir else None,
        "fingerprint": current_fingerprint(config),
        "dataset_fingerprint": dataset_fingerprint(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths
        ),
        "embed_dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "device": str(embedder.device),
        "batch_size": int(embedder.batch_size),
        "image_size": list(embedder.image_size),
        "normalized": True,
        "faiss": {"built": faiss_ok, "note": faiss_note},
        "missing_images": missing,
        "invalid_images": failures,
        "seconds": round(elapsed, 2),
        "images_per_second": round(len(filenames) / elapsed, 2) if elapsed else None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.embeddings_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if not quiet:
        print(
            f"[encode_images] encoded {len(filenames)} images in {elapsed:.1f}s "
            f"({meta['images_per_second']}/s) -> {config.embeddings_dir / EMBEDDINGS_NAME}"
        )
        print(f"[encode_images] FAISS: {faiss_note}")
        if missing:
            print(f"[encode_images] WARNING: {len(missing)} captioned images have no file "
                  f"(e.g. {', '.join(missing[:5])})")
        if failures:
            print(f"[encode_images] WARNING: {len(failures)} unreadable images skipped: "
                  f"{'; '.join(failures[:3])}")
    return {"reused": False, "count": len(filenames), "meta": meta}


def _index_of(filename: str) -> int:
    import re

    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument("--force", action="store_true", help="re-encode even if the cache is valid")
    parser.add_argument("--limit", type=int, default=None, help="encode only the first N images")
    parser.add_argument("--no-faiss", action="store_true", help="do not build a FAISS index")
    args = parser.parse_args(argv)

    config = config_from_args(args)
    config.ensure_dirs()
    try:
        run(config, force=args.force, limit=args.limit, use_faiss=not args.no_faiss)
    except DatasetError as exc:
        print(f"[encode_images] error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
