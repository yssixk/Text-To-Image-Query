"""Interactive text-to-image retrieval.

    python src/search.py
    python src/search.py --query "blue butterfly with black wing edges"
    python src/search.py --topk 10 --grid --save

Image embeddings are loaded once at start-up; every query after that is a
matrix multiply, so repeat queries stay fast.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from config import Config, add_common_args, config_from_args
from dataset import DatasetError
from retrieval import EmbeddingIndex, Hit, RetrievalError

BANNER = "Butterfly Image Retrieval System"


def print_results(
    hits: list[Hit],
    images_dir: Path,
    topk: int,
    encode_ms: float,
    rank_ms: float,
    size: int,
) -> None:
    best = hits[0]
    print()
    print("Best matching image:")
    print(f"  {images_dir / best.filename}")
    print(f"Similarity score: {best.score:.4f}")
    if best.label:
        print(f"Species label:    {best.label}")
    print(
        f"Answered in {encode_ms + rank_ms:.1f} ms "
        f"(text encoding {encode_ms:.1f} ms + ranking {size} cached images {rank_ms:.1f} ms)"
    )
    if topk > 1:
        print()
        print(f"Top {len(hits)} matches")
        print(f"  {'#':>2}  {'score':>7}  {'species':<24}  file")
        for hit in hits:
            print(f"  {hit.rank:>2}  {hit.score:>7.4f}  {hit.label[:24]:<24}  {hit.filename}")


def show_images(
    hits: list[Hit],
    images_dir: Path,
    grid: bool,
    save: Path | None,
    display: bool = True,
) -> None:
    """Display the best match (or a top-k grid) with matplotlib."""
    import matplotlib

    show_window = display and not _headless()
    if not show_window:
        # Nothing is shown: use a non-interactive backend so saving works headlessly.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    shown = hits if grid else hits[:1]
    columns = min(len(shown), 5)
    rows = (len(shown) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)
    for position, hit in enumerate(shown):
        axis = axes[position // columns][position % columns]
        path = images_dir / hit.filename
        try:
            with Image.open(path) as handle:
                axis.imshow(handle.convert("RGB"))
        except OSError as exc:
            axis.text(0.5, 0.5, f"cannot open\n{hit.filename}\n{exc}", ha="center", va="center")
        axis.set_title(f"#{hit.rank} {hit.score:.3f}\n{hit.label}\n{hit.filename}", fontsize=9)
        axis.axis("off")
    for position in range(len(shown), rows * columns):
        axes[position // columns][position % columns].axis("off")
    figure.suptitle("Butterfly retrieval", fontsize=12)
    figure.tight_layout()

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save, dpi=140)
        print(f"Result image saved to {save}")
    if show_window:
        try:
            plt.show(block=True)
        except TypeError:  # pragma: no cover - backend dependent
            plt.show()
    plt.close(figure)


def _headless() -> bool:
    import os

    return os.name != "nt" and not os.environ.get("DISPLAY")


def save_report(hits: list[Hit], query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": [hit.__dict__ for hit in hits],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "query")[:60]


def handle_query(
    query: str,
    embedder: object,
    index: EmbeddingIndex,
    config: Config,
    topk: int,
    display: bool,
    grid: bool,
    save: bool,
) -> list[Hit]:
    encode_started = time.perf_counter()
    vector = embedder.encode_texts([query])[0]  # type: ignore[attr-defined]
    encode_ms = 1000 * (time.perf_counter() - encode_started)
    rank_started = time.perf_counter()
    hits = index.search(vector, k=topk)
    rank_ms = 1000 * (time.perf_counter() - rank_started)
    print_results(hits, config.images_dir, topk, encode_ms, rank_ms, index.size)

    target = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = config.outputs_dir / "retrieval" / f"{timestamp}_{slugify(query)}.png"
    if display or save:
        show_images(hits, config.images_dir, grid=grid, save=target, display=display)
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument("--topk", type=int, default=5, help="how many matches to rank (default: 5)")
    parser.add_argument("--query", default=None, help="answer one query and exit")
    parser.add_argument("--grid", action="store_true", help="show a grid of the top-k matches")
    parser.add_argument("--no-display", action="store_true", help="do not open a plot window")
    parser.add_argument("--save", action="store_true", help="save the plot under outputs/retrieval/")
    parser.add_argument("--no-faiss", action="store_true", help="force the NumPy ranking path")
    parser.add_argument(
        "--auto-encode",
        action="store_true",
        help="encode images if the cache is missing or was built by another model",
    )
    args = parser.parse_args(argv)

    config = config_from_args(args)
    config.ensure_dirs()

    missing_paths = [
        path
        for path in (config.combined_csv, config.splits_file)
        if not path.is_file()
    ]
    if missing_paths:
        print(
            "[search] dataset artefacts missing: "
            + ", ".join(str(path) for path in missing_paths)
            + "\n         run 'python src/build_dataset.py' first"
        )
        return 1

    from encode_images import cache_is_valid, model_mismatch, run as encode_run

    valid, reason = cache_is_valid(config)
    if args.auto_encode and not valid:
        print(f"[search] embedding cache not usable ({reason}); encoding now")
        encode_run(config)
    else:
        mismatch = model_mismatch(config)
        if mismatch:
            # Cached vectors from another model live in a different space, so
            # ranking against them would silently mix unrelated numbers.
            print(
                f"[search] error: embedding cache not usable for this model ({mismatch})\n"
                "         rerun 'python src/encode_images.py' (or pass --auto-encode) "
                "to rebuild it",
                file=sys.stderr,
            )
            return 1
        if not valid and not reason.startswith("missing "):
            print(
                f"[search] warning: embedding cache not current ({reason}); "
                "results may be stale",
                file=sys.stderr,
            )

    try:
        index = EmbeddingIndex.load(config.embeddings_dir, use_faiss=not args.no_faiss)
    except RetrievalError as exc:
        print(f"[search] error: {exc}")
        return 1

    from model import CLIPEmbedder, ModelLoadError

    try:
        embedder = CLIPEmbedder(
            config.model_id,
            checkpoint_dir=config.checkpoint_dir,
            device=config.device,
            batch_size=config.batch_size,
            verbose=False,
        )
    except ModelLoadError as exc:
        print(f"[search] error: {exc}", file=sys.stderr)
        return 1

    print(BANNER)
    print(
        f"{index.size} images indexed | {index.dim}-d embeddings | "
        f"model {config.model_label()} | device {embedder.device} | backend {index.backend}"
    )

    topk = max(1, args.topk)
    if args.query is not None:
        query = args.query.strip()
        if not query:
            print("[search] the provided --query is empty")
            return 2
        hits = handle_query(
            query, embedder, index, config, topk, not args.no_display, args.grid, args.save
        )
        save_report(hits, query, config.outputs_dir / "retrieval" / "last_query.json")
        return 0

    print("Enter a description, or 'q' to quit. Commands: :topk N, :help")
    while True:
        try:
            raw = input("\nEnter a description: ")
        except (EOFError, KeyboardInterrupt):
            print("\n[search] bye")
            return 0
        query = raw.strip()
        if not query:
            print("Please enter a description (or 'q' to quit).")
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("[search] bye")
            return 0
        if query.lower().startswith(":topk"):
            parts = query.split()
            if len(parts) == 2 and parts[1].isdigit():
                topk = max(1, min(int(parts[1]), index.size))
                print(f"Showing top {topk} matches.")
            else:
                print("Usage: :topk 10")
            continue
        if query.lower() in (":help", "help", "?"):
            print(
                "Type a natural-language description of a butterfly.\n"
                "Commands: :topk N  (results to show), :help, q (quit)"
            )
            continue

        print("\nSearching dataset...")
        try:
            hits = handle_query(
                query,
                embedder,
                index,
                config,
                topk,
                not args.no_display,
                args.grid,
                args.save,
            )
        except RetrievalError as exc:
            print(f"[search] could not run that query: {exc}")
            continue
        except OSError as exc:
            print(f"[search] error while displaying the result: {exc}")
            continue
        save_report(hits, query, config.outputs_dir / "retrieval" / "last_query.json")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetError as exc:
        print(f"[search] error: {exc}")
        raise SystemExit(1) from None
