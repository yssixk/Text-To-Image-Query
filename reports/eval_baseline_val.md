# Retrieval evaluation - baseline_val

- model: `openai/clip-vit-base-patch32`
- split: **val** (137 candidate images, 75 species)
- queries: 685 caption rows (669 unique strings, 16 repeated)
- backend: faiss on cuda
- generated: 2026-10-13T16:08:00+00:00

## Text -> image

| Metric | Exact image | Species-aware |
|---|---|---|
| recall@1 | 0.1839 | 0.219 |
| recall@5 | 0.4759 | 0.5153 |
| recall@10 | 0.6146 | 0.6628 |
| mrr | 0.3222 | 0.3604 |
| mean_rank | 16.53 | 12.83 |
| median_rank | 6.0 | 5.0 |

## Image -> text

| Metric | Value |
|---|---|
| recall@1 (exact) | 0.3285 |
| recall@5 (exact) | 0.6204 |
| recall@10 (exact) | 0.7883 |
| species top-1 accuracy | 0.3869 |

## Timing

- encode 669 unique captions: 1.063s
- rank all queries from cached image embeddings: 0.014s (0.02 ms/query)

