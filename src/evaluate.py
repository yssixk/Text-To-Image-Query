"""Retrieval evaluation: Recall@1 / Recall@5 / Recall@10.

Two flavours are reported for every metric:

* **exact** - the retrieved item must be the exact source image.
* **species-aware** - any image of the correct species counts as relevant.
  With ~75 species spread over 1200 images this is the meaningful signal,
  since several images of one species can be equally valid answers.

    python src/evaluate.py --split test --tag baseline
    python src/evaluate.py --split val  --tag finetuned --checkpoint models/clip-butterfly
    python src/evaluate.py --split test --compare reports/eval_baseline.json

Cached image embeddings are checked against the model in use: if they were
built by a different model (a ``--checkpoint``, say), they are re-encoded
before scoring, so fine-tuned text vectors are never ranked against
pretrained image vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import Config, add_common_args, config_from_args
from dataset import DatasetError, load_splits, read_rows, split_members
from retrieval import EmbeddingIndex, RetrievalError, summarize_ranks

KS = (1, 5, 10)


def _rank_pair(
    index: EmbeddingIndex, vector: np.ndarray, filename: str, label: str
) -> tuple[int, int]:
    """Exact-image rank and first-same-species rank for one query vector."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    scores = index.embeddings @ vector
    order = np.argsort(-scores, kind="stable")
    target = index.filenames.index(filename)
    exact = int(np.where(order == target)[0][0]) + 1
    species = len(index.filenames) + 1
    for position, candidate in enumerate(order, start=1):
        if index.labels[candidate] == label:
            species = position
            break
    return exact, species


def evaluate_split(
    config: Config,
    split: str,
    limit_queries: int | None = None,
    use_faiss: bool = True,
    rebuild_cache: bool = True,
) -> dict[str, object]:
    """Encode and score one split, returning a metrics dictionary."""
    from model import CLIPEmbedder

    rows = read_rows(config.combined_csv)
    splits = load_splits(config.splits_file)
    members = set(split_members(splits, split))
    if not members:
        raise DatasetError(f"split {split!r} is empty")

    # Cached image vectors must come from the model being evaluated. With
    # --checkpoint the cache belongs to the base model until it is rebuilt, and
    # ranking fine-tuned text vectors against base-model image vectors would
    # silently report numbers from two different vector spaces.
    from encode_images import cache_is_valid, run as encode_run

    valid, reason = cache_is_valid(config)
    if not valid:
        if not rebuild_cache:
            raise RetrievalError(
                f"embedding cache is not usable for this model ({reason}); "
                "rerun 'python src/encode_images.py' to rebuild it"
            )
        print(f"[evaluate] embedding cache not usable ({reason}); encoding images now")
        encode_run(config, quiet=True)
        valid, reason = cache_is_valid(config)
        if not valid:
            raise RetrievalError(
                f"embedding cache still not usable after encoding ({reason}); "
                "rerun 'python src/encode_images.py'"
            )

    index = EmbeddingIndex.load(config.embeddings_dir, use_faiss=use_faiss)
    candidates = index.restrict([name for name in index.filenames if name in members])
    if candidates.size == 0:
        raise DatasetError(
            f"none of the {len(members)} images in split {split!r} have cached embeddings"
        )
    label_by_name = {row["filename"]: row.get("label", "") for row in rows}

    queries = [row for row in rows if row["filename"] in members]
    if limit_queries:
        queries = queries[:limit_queries]
    unique_texts = sorted({row["caption"] for row in queries})
    repeated = len(queries) - len(unique_texts)

    embedder = CLIPEmbedder(
        config.model_id,
        checkpoint_dir=config.checkpoint_dir,
        device=config.device,
        batch_size=config.batch_size,
        verbose=False,
    )
    started = time.perf_counter()
    text_matrix = embedder.encode_texts(unique_texts)
    text_vectors = {text: text_matrix[i] for i, text in enumerate(unique_texts)}
    encode_seconds = time.perf_counter() - started

    exact_ranks: list[int] = []
    species_ranks: list[int] = []
    started = time.perf_counter()
    for row in queries:
        exact, species = _rank_pair(
            candidates, text_vectors[row["caption"]], row["filename"], row.get("label", "")
        )
        exact_ranks.append(exact)
        species_ranks.append(species)
    query_seconds = time.perf_counter() - started

    # image -> text, using the cached image vectors already in the index
    similarity = candidates.embeddings @ text_matrix.T  # (images, unique captions)
    owners = _caption_owners_for(rows, unique_texts)
    i2t_exact_ranks: list[int] = []
    i2t_species_hits = 0
    for row_index, filename in enumerate(candidates.filenames):
        order = np.argsort(-similarity[row_index], kind="stable")
        own = [position for position, text in enumerate(unique_texts) if filename in owners[text]]
        if own:
            best = min(int(np.where(order == position)[0][0]) + 1 for position in own)
            i2t_exact_ranks.append(best)
        # species-aware: does the top caption come from an image of this species?
        top_text = unique_texts[int(order[0])]
        predicted_labels = {label_by_name.get(name, "") for name in owners[top_text]}
        if label_by_name.get(filename, "") in predicted_labels:
            i2t_species_hits += 1

    metrics: dict[str, object] = {
        "split": split,
        "tag": None,
        "model": config.model_label(),
        "device": str(embedder.device),
        "candidates": candidates.size,
        "queries": len(queries),
        "unique_query_texts": len(unique_texts),
        "repeated_query_texts": repeated,
        "species_in_split": len({label_by_name.get(name, "") for name in candidates.filenames}),
        "backend": candidates.backend if candidates.index is not None else (
            "faiss" if use_faiss and index.index is not None else "numpy"
        ),
        "text_to_image": {
            "exact": summarize_ranks(exact_ranks, KS),
            "species": summarize_ranks(species_ranks, KS),
        },
        "image_to_text": {
            "exact": summarize_ranks(i2t_exact_ranks, KS),
            "species_top1_accuracy": round(
                i2t_species_hits / candidates.size, 4
            )
            if candidates.size
            else 0.0,
        },
        "timing_seconds": {
            "encode_unique_texts": round(encode_seconds, 3),
            "rank_all_queries": round(query_seconds, 3),
            "per_query_ms": round(1000 * query_seconds / max(len(queries), 1), 2),
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return metrics


def _caption_owners_for(
    rows: list[dict[str, str]], unique_texts: list[str]
) -> dict[str, set[str]]:
    """Which images use each caption string (captions repeat across images)."""
    owners: dict[str, set[str]] = {text: set() for text in unique_texts}
    for row in rows:
        if row["caption"] in owners:
            owners[row["caption"]].add(row["filename"])
    return owners


def render_markdown(metrics: dict[str, object], comparison: dict[str, object] | None = None) -> str:
    t2i = metrics["text_to_image"]  # type: ignore[index]
    i2t = metrics["image_to_text"]  # type: ignore[index]
    lines = [
        f"# Retrieval evaluation - {metrics['tag'] or metrics['split']}",
        "",
        f"- model: `{metrics['model']}`",
        f"- split: **{metrics['split']}** ({metrics['candidates']} candidate images, "
        f"{metrics['species_in_split']} species)",
        f"- queries: {metrics['queries']} caption rows "
        f"({metrics['unique_query_texts']} unique strings, {metrics['repeated_query_texts']} repeated)",
        f"- backend: {metrics['backend']} on {metrics['device']}",
        f"- generated: {metrics['created_at']}",
        "",
        "## Text -> image",
        "",
        "| Metric | Exact image | Species-aware |",
        "|---|---|---|",
    ]
    for key in ("recall@1", "recall@5", "recall@10", "mrr", "mean_rank", "median_rank"):
        lines.append(f"| {key} | {t2i['exact'][key]} | {t2i['species'][key]} |")  # type: ignore[index]
    lines += [
        "",
        "## Image -> text",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| recall@1 (exact) | {i2t['exact']['recall@1']} |",  # type: ignore[index]
        f"| recall@5 (exact) | {i2t['exact']['recall@5']} |",  # type: ignore[index]
        f"| recall@10 (exact) | {i2t['exact']['recall@10']} |",  # type: ignore[index]
        f"| species top-1 accuracy | {i2t['species_top1_accuracy']} |",  # type: ignore[index]
        "",
        "## Timing",
        "",
        f"- encode {metrics['unique_query_texts']} unique captions: "
        f"{metrics['timing_seconds']['encode_unique_texts']}s",  # type: ignore[index]
        f"- rank all queries from cached image embeddings: "
        f"{metrics['timing_seconds']['rank_all_queries']}s "
        f"({metrics['timing_seconds']['per_query_ms']} ms/query)",  # type: ignore[index]
        "",
    ]
    if comparison:
        lines += ["## Change vs baseline", "", "| Metric | Baseline | Now | Delta |", "|---|---|---|---|"]
        for section, key in (
            ("text_to_image", "recall@1"),
            ("text_to_image", "recall@5"),
            ("text_to_image", "recall@10"),
        ):
            base_value = comparison[section]["species"][key]  # type: ignore[index]
            now_value = metrics[section]["species"][key]  # type: ignore[index]
            lines.append(
                f"| text->image species {key} | {base_value} | {now_value} | "
                f"{round(now_value - base_value, 4):+} |"
            )
        base_r1 = comparison["text_to_image"]["exact"]["recall@1"]  # type: ignore[index]
        now_r1 = metrics["text_to_image"]["exact"]["recall@1"]  # type: ignore[index]
        lines.append(
            f"| text->image exact recall@1 | {base_r1} | {now_r1} | {round(now_r1 - base_r1, 4):+} |"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--tag", default="baseline", help="name for the report files")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--compare", default=None, help="previous metrics JSON to diff against")
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args(argv)

    config = config_from_args(args)
    config.ensure_dirs()
    try:
        metrics = evaluate_split(
            config, args.split, args.limit_queries, use_faiss=not args.no_faiss
        )
    except (DatasetError, RetrievalError) as exc:
        print(f"[evaluate] error: {exc}", file=sys.stderr)
        return 1
    metrics["tag"] = args.tag

    comparison = None
    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.is_file():
            comparison = json.loads(compare_path.read_text(encoding="utf-8"))
        else:
            print(f"[evaluate] warning: {compare_path} not found, skipping comparison")

    report_dir = Path(args.report_dir) if args.report_dir else config.reports_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / f"eval_{args.tag}.md"
    json_path = report_dir / f"eval_{args.tag}.json"
    markdown_path.write_text(render_markdown(metrics, comparison), encoding="utf-8")
    json_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    exact = metrics["text_to_image"]["exact"]  # type: ignore[index]
    species = metrics["text_to_image"]["species"]  # type: ignore[index]
    print(
        f"[evaluate] split={args.split} candidates={metrics['candidates']} "
        f"queries={metrics['queries']} species={metrics['species_in_split']}"
    )
    print(
        "           text->image exact   : "
        + "  ".join(f"R@{k}={exact[f'recall@{k}']:.3f}" for k in KS)  # type: ignore[index]
    )
    print(
        "           text->image species : "
        + "  ".join(f"R@{k}={species[f'recall@{k}']:.3f}" for k in KS)  # type: ignore[index]
    )
    print(f"           mean rank exact={exact['mean_rank']} species={species['mean_rank']}")
    print(f"[evaluate] report -> {markdown_path}")
    print(f"[evaluate] metrics -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
