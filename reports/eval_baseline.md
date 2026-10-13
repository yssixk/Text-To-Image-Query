# Retrieval evaluation - baseline

- model: `openai/clip-vit-base-patch32`
- split: **test** (106 candidate images, 75 species)
- queries: 530 caption rows (526 unique strings, 4 repeated)
- backend: faiss on cuda
- generated: 2026-10-13T16:07:10+00:00

## Text -> image

| Metric | Exact image | Species-aware |
|---|---|---|
| recall@1 | 0.2019 | 0.2358 |
| recall@5 | 0.4547 | 0.5075 |
| recall@10 | 0.6057 | 0.6472 |
| mrr | 0.3362 | 0.3722 |
| mean_rank | 14.6 | 12.44 |
| median_rank | 7.0 | 5.0 |

## Image -> text

| Metric | Value |
|---|---|
| recall@1 (exact) | 0.3113 |
| recall@5 (exact) | 0.6604 |
| recall@10 (exact) | 0.7547 |
| species top-1 accuracy | 0.3585 |

## Timing

- encode 526 unique captions: 0.875s
- rank all queries from cached image embeddings: 0.009s (0.02 ms/query)

