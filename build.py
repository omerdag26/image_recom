# build.py

import os
import sqlite3
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from joblib import dump
from ann_index import ANNIndex
from similarity_measures import (
    load_model,            # load (and cache) ResNet50 avg-pool
    compute_color_feature, # single-image LAB/HSV histogram
    compute_embeddings_batch  # sequential loader + batched predict (fallback)
)
from helper import shut_gpu
import umap

# Keras image utilities for optional parallel loader in deep path
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Disable GPU and set OpenCV to use single thread for consistency
cv2.setNumThreads(0)
shut_gpu()


# This function ensures the parent directory of 'path' exists.
def _ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# This function saves (ids, feats) into a single NPZ file.
def _save_npz(feat_path, ids, feats):
    _ensure_parent(feat_path)
    # Save the ids and feats arrays into a .npz file.
    np.savez(feat_path, ids=ids.astype(np.int64), feats=feats.astype(np.float32))
    dim = feats.shape[1] if feats.size else 0
    print(f"✔ Saved features to '{feat_path}' (N={len(ids)}, dim={dim}).")


# This function streams rows from the SQLite DB in order, yielding (id, path) tuples.
def _stream_cursor(db_path):
    # Open the SQLite database and create a cursor to fetch image data.
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Get the total number of images in the database.
    cur.execute("SELECT COUNT(*) FROM images")
    total = int(cur.fetchone()[0])
    # Fetch all image IDs and file paths in order.
    cur.execute("SELECT image_id, filepath FROM images ORDER BY image_id")
    return conn, cur, total

# This function merges feature shards into a single NPZ file.
# We are merging shards to avoid memory issues with large datasets. 
def _merge_shards_to_npz(shards, out_feat_path, dim):
    # Merge multiple shards of features into a single NPZ file.
    if not shards:
        _save_npz(out_feat_path, np.empty((0,), np.int64), np.empty((0, dim), np.float32))
        print("⚠ No features written.")
        return
    # Load all shards and concatenate them into a single array.
    ids_all   = [np.load(p_ids,  mmap_mode="r") for (p_ids, _)   in shards]
    feats_all = [np.load(p_feat, mmap_mode="r") for (_, p_feat)   in shards]
    ids_all   = np.concatenate(ids_all)
    feats_all = np.vstack(feats_all).astype(np.float32)
    # Save the concatenated arrays to the output NPZ file.
    _save_npz(out_feat_path, ids_all, feats_all)
    # Cleanup temporary shard files.
    for a, b in shards:
        try:
            os.remove(a); os.remove(b)
        except OSError:
            pass


# This function builds an ANN index from the features stored in a NPZ file.
def build_index(feat_path, index_path):
    # Load the features from the NPZ file.
    data = np.load(feat_path, allow_pickle=True, mmap_mode="r")
    feats = np.asarray(data["feats"], dtype=np.float32)
    # Check if the features are empty.
    if feats.shape[0] == 0 or feats.shape[1] == 0:
        print("⚠ No features to index; skipping FAISS build.")
        return
    # Ensure the index path exists.
    _ensure_parent(index_path)
    # Create an ANN index with the number of dimensions equal to the feature size.
    ann = ANNIndex(int(feats.shape[1]), index_path=index_path)
    ann.build(feats)
    print(f"✔ ANN index built & saved to '{index_path}'.")


# Helper: load+preprocess a single image to array (used by parallel loader)
def _load_preproc_one(path, target=(224, 224)):
    # Try to load the image, resize, and convert to float array
    try:
        img = keras_image.load_img(path, target_size=target)
        x = keras_image.img_to_array(img)
        return x
    except Exception:
        return None


# This function builds the deep features from images in the SQLite database.
def build_deep(db_path, feat_path, batch_size=64, chunk=4096, workers=1):
    # Load model once
    model = load_model()

    # Open cursor and get total count
    conn, cur, total = _stream_cursor(db_path)

    # Temp shard folder to keep RAM flat
    tmp_dir = "features/_tmp_deep"
    os.makedirs(tmp_dir, exist_ok=True)
    # Prepare to stream rows in chunks
    part = 0
    shards = []
    skipped_total = 0
    total_written = 0
    feat_dim = 2048  # known output of ResNet50 avg-pool

    pbar = tqdm(total=total, desc="Deep features")
    # Iterate over the database in chunks
    while True:
        # Stream a chunk from DB
        rows = cur.fetchmany(chunk)
        if not rows:
            break

        # Split ids and paths
        ids = [int(r[0]) for r in rows]
        fps = [r[1] for r in rows]

        # If workers>1, parallelize image loading, then do a single batched predict
        if workers and workers > 1:
            # Parallel load to arrays
            arrays = [None] * len(fps)
            ok = [False] * len(fps)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut2i = {ex.submit(_load_preproc_one, p): i for i, p in enumerate(fps)}
                for fut in as_completed(fut2i):
                    i = fut2i[fut]
                    arr = fut.result()
                    if arr is not None:
                        arrays[i] = arr
                        ok[i] = True
                    else:
                        skipped_total += 1

            # Keep only successfully loaded indices
            good_idx = [i for i, flag in enumerate(ok) if flag]
            if good_idx:
                # Stack + preprocess
                X = np.stack([arrays[i] for i in good_idx]).astype(np.float32)
                X = preprocess_input(X)
                # Batched predict to control memory usage
                outs = []
                for s in range(0, X.shape[0], batch_size):
                    e = s + batch_size
                    pred = model.predict(X[s:e], verbose=0).astype(np.float32)
                    outs.append(pred)
                embs = np.vstack(outs)
                # L2-normalize row-wise
                embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
                embs = embs.astype(np.float32)
                feat_dim = embs.shape[1]

                # Map back kept ids in order
                kept_ids = [ids[i] for i in good_idx]
            else:
                # Nothing valid in this chunk
                embs = np.zeros((0, feat_dim), dtype=np.float32)
                kept_ids = []
        else:
            # Sequential loader + batched predict from similarity_measures
            embs, ok = compute_embeddings_batch(fps, model=model, batch_size=batch_size)
            # Keep only successfully encoded ids in the same order
            kept_ids = [id_ for id_, flag in zip(ids, ok) if flag]
            skipped_total += int((~ok).sum())
            # Track dim from output
            if embs.shape[0] > 0:
                feat_dim = embs.shape[1]

        # Write shard if we have any embeddings in this chunk
        if embs.shape[0] > 0:
            ids_path   = os.path.join(tmp_dir, f"ids_{part:05d}.npy")
            feats_path = os.path.join(tmp_dir, f"feats_{part:05d}.npy")
            np.save(ids_path,   np.asarray(kept_ids, dtype=np.int64))
            np.save(feats_path, embs.astype(np.float32))
            shards.append((ids_path, feats_path))
            part += 1
            total_written += len(kept_ids)

        # Advance progress by the number of rows attempted
        pbar.update(len(rows))

    pbar.close()
    conn.close()

    # Merge all shards into a single NPZ file
    _merge_shards_to_npz(shards, feat_path, dim=feat_dim)
    print(f"✔ Deep features saved to '{feat_path}' (N={total_written}, dim={feat_dim}).")
    # If any images were skipped, report it
    if skipped_total:
        print(f"⚠ Skipped {skipped_total} images (unreadable).")


# This function builds color features from images in the SQLite database.
def build_color(db_path, feat_path, bins=8, workers=4, chunk=4096):
    # Open cursor and get total count
    conn, cur, total = _stream_cursor(db_path)
    # Temp shard folder to keep RAM flat
    tmp_dir = "features/_tmp_color"
    os.makedirs(tmp_dir, exist_ok=True)
    # Variables for streaming rows in chunks    
    part = 0
    shards = []
    skipped_total = 0
    total_written = 0

    dim = bins * bins * bins
    pbar = tqdm(total=total, desc="Color features")
    # Iterate over the database in chunks
    while True:
        # Stream a chunk from DB
        rows = cur.fetchmany(chunk)
        if not rows:
            break
        # Split ids and paths
        ids = [int(r[0]) for r in rows]
        fps = [r[1] for r in rows]
        # Prepare to compute color features in parallel
        vecs = [None] * len(fps)
        ok = [False] * len(fps)
        # Function to compute color feature for a single image
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2i = {ex.submit(compute_color_feature, p, bins=bins): i for i, p in enumerate(fps)}
            # Collect results as they complete
            for fut in as_completed(fut2i):
                i = fut2i[fut]
                # If the feature computation raises an exception, it will be caught
                # and None will be returned.
                try:
                    v = fut.result()
                except Exception:
                    v = None
                # If the feature is valid, store it
                if v is not None:
                    vecs[i] = v
                    ok[i] = True
                # If the feature is None, it means the image was unreadable
                else:
                    skipped_total += 1
                pbar.update(1)
        # If no valid features, skip this chunk
        good = [i for i, flag in enumerate(ok) if flag]
        if not good:
            continue
        # Keep only successfully encoded ids in the same order
        kept_ids = np.asarray([ids[i] for i in good], dtype=np.int64)
        Z = np.stack([vecs[i] for i in good]).astype(np.float32)
        # Write shard if we have any embeddings in this chunk
        ids_path   = os.path.join(tmp_dir, f"ids_{part:05d}.npy")
        feats_path = os.path.join(tmp_dir, f"feats_{part:05d}.npy")
        np.save(ids_path, kept_ids)
        np.save(feats_path, Z)
        shards.append((ids_path, feats_path))
        part += 1
        total_written += len(kept_ids)
    # Close the progress bar and database connection
    pbar.close()
    conn.close()
    # Merge all shards into a single NPZ file
    _merge_shards_to_npz(shards, feat_path, dim=dim)
    print(f"✔ Color features saved to '{feat_path}' (N={total_written}, dim={dim}).")
    # If any images were skipped, report it
    if skipped_total:
        print(f"⚠ Skipped {skipped_total} images (unreadable).")


# This function builds UMAP features from precomputed deep embeddings.
# It fits a UMAP model on a random subset of deep embeddings and transforms all.
# The transformed features are saved along with the fitted UMAP model.
def build_umap_from_deep(deep_npz_path,out_feat_path,model_path="models/umap_deep.joblib",n_components=256,
                        n_neighbors=15,min_dist=0.0,metric="cosine",random_state=26,sample_size=50_000,
):
    # Ensure the output directories exist
    _ensure_parent(model_path)
    # Load the precomputed deep features from the NPZ file.
    data = np.load(deep_npz_path, allow_pickle=True, mmap_mode="r")
    ids = data["ids"]
    X = np.asarray(data["feats"], dtype=np.float32)
    N = X.shape[0]
    # If there are no features, save an empty NPZ and return.
    if N == 0:
        _save_npz(out_feat_path, np.empty((0,), np.int64), np.zeros((0, n_components), np.float32))
        return

    # Fit a UMAP model on a random subset of the deep embeddings
    rng = np.random.default_rng(random_state)
    idx = rng.choice(N, size=min(sample_size, N), replace=False)
    X_fit = X[idx]
    # Reduce the dimensionality using UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )
    reducer.fit(X_fit)
    # Transform all deep embeddings to UMAP space
    print(f"Transforming {N} deep embeddings to UMAP space...")
    Z = reducer.transform(X).astype(np.float32)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    # Save the transformed features and the fitted UMAP model
    _save_npz(out_feat_path, ids, Z)
    dump(reducer, model_path)
    # Print success message
    print(f"✔ UMAP features saved to '{out_feat_path}' (N={len(ids)}, dim={n_components}).")


# Main function to build features and indices based on command-line arguments.
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    # Method to build: deep, color, or umap and database path 
    parser.add_argument("--method", choices=["deep", "color", "umap"], required=True)
    parser.add_argument("--db", help="SQLite DB path (required for deep/color, or to auto-build deep for UMAP)")
    # Color params
    parser.add_argument("--bins", type=int, default=8)
    # UMAP params
    parser.add_argument("--deep_features", default="features/deep.npz",
                        help="Path to precomputed deep features NPZ (ids, feats)")
    parser.add_argument("--umap_model", default="models/umap_deep.joblib",
                        help="Path to save fitted UMAP model")
    parser.add_argument("--n_components", type=int, default=256)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.0)
    parser.add_argument("--sample_size", type=int, default=50_000)
    # Chunk size and workers for parallel processing
    parser.add_argument("--chunk", type=int, default=4096, help="Chunk size for processing rows in DB")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for feature computation")
    args = parser.parse_args()

    # Ensure COLOR_BINS env is in sync with build-time configuration (used by search.py encode)
    os.environ["COLOR_BINS"] = str(args.bins)

    # Ensure project folders
    os.makedirs("features", exist_ok=True)
    os.makedirs("index", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Build features and indices based on the selected method
    if args.method == "deep":
        assert args.db, "--db is required for deep"
        feat_path = "features/deep.npz"
        index_path = "index/deep.index"
        # Build deep features and index (chunk + workers are honored)
        build_deep(args.db, feat_path, batch_size=64, chunk=args.chunk, workers=args.workers)
        build_index(feat_path, index_path)

    elif args.method == "color":
        assert args.db, "--db is required for color"
        feat_path = "features/color.npz"
        index_path = "index/color.index"
        # Build color features and index (chunk + workers are honored)
        build_color(args.db, feat_path, bins=args.bins, workers=args.workers, chunk=args.chunk)
        build_index(feat_path, index_path)

    else:  
        deep_npz = args.deep_features
        # If deep features are not provided, compute them
        if not os.path.exists(deep_npz):
            if not args.db:
                raise FileNotFoundError(f"{deep_npz} not found and no --db provided to compute deep.")
            print("ℹ Deep features not found; computing now…")
            build_deep(args.db, deep_npz, batch_size=64, chunk=args.chunk, workers=args.workers)
            build_index(deep_npz, "index/deep.index")
        # Build UMAP features and index
        umap_feat_path = "features/umap.npz"
        umap_index_path = "index/umap.index"
        build_umap_from_deep(
            deep_npz, umap_feat_path,
            model_path=args.umap_model,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            sample_size=args.sample_size,
        )
        build_index(umap_feat_path, umap_index_path)