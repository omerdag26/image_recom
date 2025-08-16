# bench_search.py
# Benchmark ANN search using the exact query-encoding pipeline from search.py (encode_one).
# Preloads the deep model and (optionally) the UMAP reducer once to avoid repeated loads.

import os
import time
import sqlite3
import argparse
import statistics as stats
import numpy as np

from ann_index import ANNIndex
from search import encode_one
from similarity_measures import load_model  # for deep/umap
from joblib import load as joblib_load     # for UMAP reducer


def sample_paths(db_path: str, n: int = 10) -> list[str]:
    """Pick n random filepaths from the DB (cheap at small n)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT filepath FROM images ORDER BY RANDOM() LIMIT ?", (n,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["deep", "color", "umap"], required=True)
    ap.add_argument("--db", required=True, help="SQLite DB path (used to sample random query paths)")
    ap.add_argument("--features", required=True, help="NPZ with ids, feats (must match method)")
    ap.add_argument("--index", required=True, help="FAISS index path (must match method)")
    ap.add_argument("--n", type=int, default=10, help="Number of random queries")
    ap.add_argument("--topk", type=int, default=10, help="Top-k to retrieve")
    ap.add_argument("--warmup", type=int, default=2, help="Warm-up queries (excluded from timings)")
    ap.add_argument("--umap_model", default="models/umap_deep.joblib",
                    help="UMAP reducer path (only used when --method umap)")
    args = ap.parse_args()

    # --- preload ids + FAISS index (match the method’s feature dim) ---
    data = np.load(args.features, allow_pickle=True, mmap_mode="r")
    ids = data["ids"]
    dim = int(data["feats"].shape[1])
    ann = ANNIndex(dim, index_path=args.index)
    ann.load()

    # --- preload model/reducer once (encode_one will use these if provided) ---
    deep_model = load_model() if args.method in {"deep", "umap"} else None
    umap_reducer = joblib_load(args.umap_model) if args.method == "umap" else None

    # --- pick random queries from DB ---
    paths = sample_paths(args.db, n=args.n)
    if not paths:
        print("No query paths sampled from DB.")
        return

    # --- warm-up (build kernel caches, JITs, etc.) ---
    for p in paths[:args.warmup]:
        v = encode_one(p, method=args.method, deep_model=deep_model, reducer=umap_reducer)
        ann.query(v.reshape(1, -1), k=args.topk)

    # --- timed loop ---
    t_feat, t_ann, t_total = [], [], []
    for p in paths:
        t0 = time.perf_counter()
        v = encode_one(p, method=args.method, deep_model=deep_model, reducer=umap_reducer)
        t1 = time.perf_counter()
        ann.query(v.reshape(1, -1), k=args.topk)
        t2 = time.perf_counter()

        t_feat.append(t1 - t0)   # feature time (encode_one)
        t_ann.append(t2 - t1)    # FAISS search time
        t_total.append(t2 - t0)  # end-to-end

    def fmt(a): return f"{stats.mean(a)*1000:.1f} ms (median {stats.median(a)*1000:.1f} ms)"
    print(f"Method: {args.method}")
    print(f"Queries: {len(paths)}, topk={args.topk}")
    print(f"Feature time: {fmt(t_feat)}")
    print(f"ANN time:     {fmt(t_ann)}")
    print(f"Total time:   {fmt(t_total)}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
