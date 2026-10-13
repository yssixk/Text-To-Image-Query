"""Check that search.py's display path draws the exact retrieved files.

Renders a top-k grid with matplotlib in Agg mode, keeps the figure open, and
compares each axes image against the source file's pixels.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
from model import CLIPEmbedder  # noqa: E402
from retrieval import EmbeddingIndex  # noqa: E402
from search import show_images  # noqa: E402

config = Config.load()
index = EmbeddingIndex.load(config.embeddings_dir, use_faiss=True)
embedder = CLIPEmbedder(config.model_id, device=config.device, verbose=False)

query = "blue butterfly with black wing edges"
hits = index.search(embedder.encode_texts([query])[0], k=6)
print("hits:", [(h.rank, h.filename) for h in hits])

# Keep the figure alive so the axes can be inspected.
real_close = plt.close
plt.close = lambda *args, **kwargs: None  # type: ignore[assignment]
out = ROOT / "outputs" / "retrieval" / "display_check.png"
show_images(hits, config.images_dir, grid=True, save=out, display=False)
figures = [plt.figure(num) for num in plt.get_fignums()]
plt.close = real_close  # type: ignore[assignment]
figure = figures[0]

drawn = 0
mismatches = 0
for axis, hit in zip([a for a in figure.axes if a.has_data()], hits):
    if not axis.images:
        continue
    array = np.asarray(axis.images[0].get_array())
    source = np.asarray(Image.open(config.images_dir / hit.filename).convert("RGB"))
    same = array.shape == source.shape and float(
        np.abs(array.astype(float) - source.astype(float)).mean()
    ) < 1e-6
    print(f"  {hit.filename}: drawn shape={array.shape} source shape={source.shape} identical={same}")
    drawn += 1
    mismatches += 0 if same else 1

print(f"panels with images: {drawn}/{len(hits)} | mismatches: {mismatches}")
print("saved file exists:", out.is_file(), "size:", out.stat().st_size if out.is_file() else 0)
plt.close("all")
raise SystemExit(1 if mismatches or drawn != len(hits) else 0)
