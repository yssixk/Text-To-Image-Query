"""Optional CLIP fine-tuning for butterfly-specific image-text matching.

The model is only ever *adapted* from the pretrained checkpoint: by default
only the two projection heads are trained. Validation species-aware Recall@k
is measured before and after, and the checkpoint is written only when the
fine-tuned model is actually better (use ``--force`` to keep it regardless).

    python src/train.py --epochs 5 --batch-size 32 --tune projection
    python src/train.py --steps 10 --out models/smoke     # quick smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import Config, add_common_args, config_from_args
from dataset import (
    DatasetError,
    CaptionImageDataset,
    dataset_fingerprint,
    discover_images,
    load_splits,
    read_rows,
    split_members,
)

TUNE_SCOPES = ("projection", "last_block", "all")


def trainable_parameters(model: object, scope: str) -> list:
    """Select the parameter tensors to update for a given scope."""
    if scope == "all":
        return list(model.parameters())  # type: ignore[attr-defined]
    if scope == "projection":
        return [
            parameter
            for name, parameter in model.named_parameters()  # type: ignore[attr-defined]
            if "projection" in name
        ]
    if scope == "last_block":
        selected = list(model.visual_projection.parameters()) + list(  # type: ignore[attr-defined]
            model.text_projection.parameters()  # type: ignore[attr-defined]
        )
        for tower in ("vision_model", "text_model"):
            encoder = getattr(getattr(model, tower, None), "encoder", None)
            layers = getattr(encoder, "layers", None)
            if layers:
                selected.extend(layers[-1].parameters())
        return selected
    raise ValueError(f"unknown tune scope {scope!r}; expected one of {TUNE_SCOPES}")


def freeze_all(model: object) -> None:
    for parameter in model.parameters():  # type: ignore[attr-defined]
        parameter.requires_grad = False


def text_max_length(model: object, fallback: int = 77) -> int:
    """CLIP's fixed text context length, read from the loaded checkpoint."""
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    value = getattr(text_config, "max_position_embeddings", None)
    return int(value) if value else fallback


def validation_metrics(
    embedder: object,
    rows: list[dict[str, str]],
    members: list[str],
    images_dir: Path,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Species-aware and exact text->image recall on a split, live model."""
    from retrieval import EmbeddingIndex

    present = [name for name in members if (images_dir / name).is_file()]
    if not present:
        return {f"species_recall@{k}": 0.0 for k in ks} | {"exact_recall@1": 0.0}
    label_by_name = {row["filename"]: row.get("label", "") for row in rows}
    vectors, kept, _ = embedder.encode_images([images_dir / name for name in present], verbose=False)  # type: ignore[attr-defined]
    kept_names = [present[position] for position in kept]
    index = EmbeddingIndex(
        vectors, kept_names, [label_by_name.get(name, "") for name in kept_names]
    )
    queries = [row for row in rows if row["filename"] in set(kept_names)]
    unique_texts = sorted({row["caption"] for row in queries})
    text_matrix = embedder.encode_texts(unique_texts)  # type: ignore[attr-defined]
    vectors_by_text = {text: text_matrix[i] for i, text in enumerate(unique_texts)}

    exact_ranks: list[int] = []
    species_ranks: list[int] = []
    for row in queries:
        vector = vectors_by_text[row["caption"]]
        scores = index.embeddings @ vector
        order = np.argsort(-scores, kind="stable")
        target = index.filenames.index(row["filename"])
        exact_ranks.append(int(np.where(order == target)[0][0]) + 1)
        label = row.get("label", "")
        rank = len(index.filenames) + 1
        for position, candidate in enumerate(order, start=1):
            if index.labels[candidate] == label:
                rank = position
                break
        species_ranks.append(rank)

    metrics = {
        f"species_recall@{k}": round(sum(1 for r in species_ranks if r <= k) / len(species_ranks), 4)
        for k in ks
    }
    metrics["exact_recall@1"] = round(sum(1 for r in exact_ranks if r == 1) / len(exact_ranks), 4)
    metrics["species_mean_rank"] = round(float(np.mean(species_ranks)), 2)
    return metrics


def run(
    config: Config,
    epochs: int = 5,
    lr: float = 1e-5,
    scope: str = "projection",
    seed: int = 42,
    out_dir: Path | None = None,
    force: bool = False,
    steps: int | None = None,
    amp: bool = True,
    loss_log_every: int = 20,
) -> dict[str, object]:
    """Fine-tune the projection heads and report before/after retrieval quality."""
    from model import CLIPEmbedder

    torch = __import__("torch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows = read_rows(config.combined_csv)
    splits = load_splits(config.splits_file)
    train_images = split_members(splits, "train")
    val_images = split_members(splits, "val")

    embedder = CLIPEmbedder(
        config.model_id,
        checkpoint_dir=config.checkpoint_dir,
        device=config.device,
        batch_size=config.batch_size,
        verbose=True,
    )
    model = embedder.model
    processor = embedder.processor
    device = embedder.device

    print("[train] measuring the pretrained baseline on the val split")
    baseline = validation_metrics(embedder, rows, val_images, config.images_dir)
    print(f"[train] baseline: {baseline}")

    freeze_all(model)
    selected = trainable_parameters(model, scope)
    if not selected:
        raise DatasetError(f"no trainable parameters matched scope {scope!r}")
    for parameter in selected:
        parameter.requires_grad = True
    trainable_count = sum(p.numel() for p in selected)
    print(f"[train] scope={scope}: {trainable_count:,} trainable parameters "
          f"of {sum(p.numel() for p in model.parameters()):,}")

    dataset = CaptionImageDataset(
        rows,
        config.images_dir,
        processor,
        filenames=train_images,
        max_length=text_max_length(model),
    )
    optimizer = torch.optim.AdamW([p for p in selected], lr=lr, weight_decay=0.01)
    use_amp = amp and str(device).startswith("cuda")
    try:
        scaler = torch.amp.GradScaler(device, enabled=use_amp)
    except (AttributeError, TypeError):  # older/other torch builds
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    batch_size = config.batch_size or embedder.batch_size
    history: list[dict[str, object]] = []
    best = dict(baseline)
    best_state: dict[str, object] | None = None
    last_state: dict[str, object] | None = None
    last_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        dataset.set_epoch(epoch)
        order = list(range(len(dataset)))
        random.Random(seed + epoch).shuffle(order)
        running, seen = 0.0, 0
        started = time.perf_counter()
        for batch_number, start in enumerate(range(0, len(order), batch_size), start=1):
            if steps is not None and batch_number > steps:
                break
            chunk = order[start : start + batch_size]
            items = [dataset[i] for i in chunk]
            inputs = {
                key: torch.stack([item[key] for item in items]).to(device)
                for key in ("input_ids", "attention_mask", "pixel_values")
                if key in items[0]
            }
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
            )
            with autocast_context:
                outputs = model(**inputs, return_loss=True)
                loss = outputs.loss
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(selected, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selected, 1.0)
                optimizer.step()
            running += float(loss.detach())
            seen += 1
            if loss_log_every and batch_number % loss_log_every == 0:
                print(f"[train] epoch {epoch} batch {batch_number} loss={running / seen:.4f}")
        epoch_seconds = time.perf_counter() - started

        model.eval()
        metrics = validation_metrics(embedder, rows, val_images, config.images_dir)
        improved = metrics["species_recall@1"] > best.get("species_recall@1", -1.0)
        print(
            f"[train] epoch {epoch}: loss={running / max(seen, 1):.4f} "
            f"val={metrics} ({epoch_seconds:.1f}s) {'<-- best' if improved else ''}"
        )
        history.append(
            {
                "epoch": epoch,
                "loss": round(running / max(seen, 1), 4),
                "batches": seen,
                "seconds": round(epoch_seconds, 1),
                "val": metrics,
            }
        )
        if improved:
            best = dict(metrics)
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        last_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        last_metrics = dict(metrics)

    output_dir = Path(out_dir) if out_dir else config.models_dir / "clip-butterfly"
    beat_baseline = best.get("species_recall@1", 0.0) > baseline.get("species_recall@1", 0.0)
    state_to_save: dict[str, object] | None = None
    save_reason = ""
    if best_state is not None and beat_baseline:
        state_to_save, save_reason = best_state, "beat the pretrained baseline"
    elif force and last_state is not None:
        # --force keeps the final epoch's weights even without an improvement.
        state_to_save, save_reason = last_state, "kept because --force was given"
        if best_state is None:
            best = dict(last_metrics)
    missing_files = [
        name
        for name in train_images
        if not (config.images_dir / name).is_file()
    ]

    if state_to_save is not None:
        model.load_state_dict(state_to_save)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        config_json = {
            "base_model": config.model_id,
            "tune_scope": scope,
            "trainable_parameters": int(trainable_count),
            "epochs": epochs,
            "steps_per_epoch": steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "seed": seed,
            "device": str(device),
            "amp": use_amp,
            "dataset_fingerprint": dataset_fingerprint(
                (path.name, path.stat().st_size, path.stat().st_mtime_ns)
                for path in discover_images(config.images_dir).by_index.values()
            ),
            "split_counts": splits.get("counts"),
            "baseline_val": baseline,
            "best_val": best,
            "history": history,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (output_dir / "butterfly_config.json").write_text(
            json.dumps(config_json, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[train] saved fine-tuned model -> {output_dir} ({save_reason})")
    elif last_state is None:
        print("[train] no complete epoch was run, nothing to save")
    else:
        print(
            "[train] NOT saving: fine-tuning did not beat the pretrained baseline "
            f"({best.get('species_recall@1')} <= {baseline.get('species_recall@1')}). "
            "Pretrained retrieval stays the better option; rerun with --force to keep it anyway."
        )

    result = {
        "baseline_val": baseline,
        "best_val": best,
        "last_val": last_metrics,
        "history": history,
        "saved": state_to_save is not None
        and Path(output_dir, "butterfly_config.json").is_file(),
        "output_dir": str(output_dir),
        "trainable_parameters": int(trainable_count),
        "tune_scope": scope,
        "missing_train_images": missing_files,
        "device": str(device),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tune", default="projection", choices=TUNE_SCOPES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="checkpoint directory")
    parser.add_argument("--force", action="store_true", help="save even without improvement")
    parser.add_argument("--steps", type=int, default=None, help="max batches per epoch (smoke test)")
    parser.add_argument("--no-amp", action="store_true", help="disable fp16 autocast")
    parser.add_argument("--tag", default="finetune")
    args = parser.parse_args(argv)

    config = config_from_args(args)
    config.ensure_dirs()
    try:
        result = run(
            config,
            epochs=args.epochs,
            lr=args.lr,
            scope=args.tune,
            seed=args.seed,
            out_dir=Path(args.out) if args.out else None,
            force=args.force,
            steps=args.steps,
            amp=not args.no_amp,
        )
    except (DatasetError, RuntimeError) as exc:
        print(f"[train] error: {exc}", file=sys.stderr)
        return 1

    import json as _json

    report = config.reports_dir / f"train_{args.tag}.json"
    report.write_text(_json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[train] report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
