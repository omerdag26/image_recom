# search.py
import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from ann_index import ANNIndex
from database import ImageDatabase
import faiss
from similarity_measures import load_model, compute_embedding, compute_color_feature
from tensorflow.keras.applications.resnet50 import preprocess_input



# Color feature config (fixed at build time ideally)
COLOR_BINS = int(os.getenv("COLOR_BINS", 8))

# Default path to fitted UMAP reducer (can be overridden via env)
UMAP_MODEL_PATH = os.getenv("UMAP_MODEL_PATH", "models/umap_deep.joblib")

# Global caches (avoid re-loading models / indices)
CACHE = {}
GLOBAL = {
    "deep_model": None,   
    "deep_dim": 2048,    
    "umap_model": None,   
}



# Build feature file path (features/{method}.npz)
def _feat_path(method):
    return os.path.join("features", f"{method}.npz")

# Build index file path (index/{method}.index)
def _index_path(method):
    return os.path.join("index", f"{method}.index")

# Load ids and dim from NPZ (mmap to be light on RAM)
def _load_npz_ids_dim(path):
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    ids = np.asarray(data["ids"])
    dim = int(np.asarray(data["feats"]).shape[1])
    return ids, dim


# ----------------------- UMAP loader + warmups -----------------------

# Load fitted UMAP reducer once (joblib preferred; pickle fallback)
def _load_umap(model_path):
    if GLOBAL["umap_model"] is not None:
        return GLOBAL["umap_model"]
    try:
        import joblib
        reducer = joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, "rb") as f:
            reducer = pickle.load(f)
    GLOBAL["umap_model"] = reducer
    return reducer

# Warm up Keras (pay first-call overhead)
def _warmup_keras(model):
    try:
        x = np.zeros((1, 224, 224, 3), dtype=np.float32)
        x = preprocess_input(x)
        _ = model.predict(x, verbose=0)
    except Exception:
        pass

# Warm up UMAP .transform
def _warmup_umap(reducer, dim):
    try:
        _ = reducer.transform(np.zeros((1, dim), dtype=np.float32))
    except Exception:
        pass


# ----------------------- Preload (per method) -----------------------

# Preload ids, ANN index, and encoders; cache by (method, db).
def preload(method, db_path):
    key = (method, os.path.abspath(db_path))
    if key in CACHE:
        return CACHE[key]

    # Resolve feature/index files
    feat_path = _feat_path(method)
    index_path = _index_path(method)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(feat_path)
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)

    # Read ids and vector dimension
    ids, dim = _load_npz_ids_dim(feat_path)

    # Load FAISS index (mmap if supported)
    ann = ANNIndex(dim, index_path=index_path)
    try:
        ann.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    except Exception:
        ann.load()

    # Prepare optional encoders
    deep_model = None
    reducer = None

    # Deep (and UMAP) need the CNN model
    if method in {"deep", "umap"}:
        if GLOBAL["deep_model"] is None:
            GLOBAL["deep_model"] = load_model()
            _warmup_keras(GLOBAL["deep_model"])
        deep_model = GLOBAL["deep_model"]

    # UMAP additionally needs the reducer and deep dim
    if method == "umap":
        reducer = _load_umap(UMAP_MODEL_PATH)
        # Try to read deep dim from features (nice-to-have)
        try:
            _, dim_d = _load_npz_ids_dim(_feat_path("deep"))
            GLOBAL["deep_dim"] = dim_d
        except Exception:
            pass
        _warmup_umap(reducer, GLOBAL["deep_dim"])

    # Bundle and cache
    bundle = {"ids": ids, "ann": ann, "deep_model": deep_model, "reducer": reducer}
    CACHE[key] = bundle
    return bundle

# Preload multiple methods in parallel; ignore failures
def preload_many(methods, db_path):
    out = {}
    if not methods:
        return out
    with ThreadPoolExecutor(max_workers=min(3, len(methods))) as ex:
        futs = {ex.submit(preload, m, db_path): m for m in methods}
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                out[m] = fut.result()
            except Exception as e:
                print(f"[preload:{m}] {e}")
    return out



# Build a normalized feature vector for the given method
def encode_one(path, method, *, deep_model=None, reducer=None):
    if method == "deep":
        v = compute_embedding(path, model=deep_model).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v
    elif method == "color":
        v = compute_color_feature(path, bins=COLOR_BINS).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v
    else: 
        # Deep → UMAP.reduce → normalize
        if reducer is None:
            reducer = _load_umap(UMAP_MODEL_PATH)
        v = compute_embedding(path, model=deep_model).astype(np.float32)[None, :]
        z = reducer.transform(v).astype(np.float32)[0]
        z /= (np.linalg.norm(z) + 1e-12)
        return z



# Map [-1,1] similarity to [0,1] for stable averaging
def norm_score(s):
    s = float(max(-1.0, min(1.0, s)))
    return 0.5 * (s + 1.0)

# Normalize weights so selected methods sum to 1
def normalize_weights(methods, w_deep, w_color, w_umap):
    raw = {"deep": float(w_deep), "color": float(w_color), "umap": float(w_umap)}
    w = {m: raw[m] for m in methods}
    s = sum(w.values())
    if s <= 0:
        w = {m: 1.0 for m in methods}
        s = float(len(methods))
    w = {m: w[m] / s for m in methods}
    return w



# Run a search with multi-query early fusion and multi-method late fusion
def run_search(img_files,methods_selected,db_path,topk,pool,fuser,w_deep, w_color, w_umap, trim_base):
    # Validate query list
    if not img_files:
        return [], []

    # Resolve absolute query paths; fail fast for missing files
    qpaths = []
    for f in img_files:
        p = getattr(f, "name", f)
        if not os.path.isfile(p):
            return [], [[0, "-", 0.0, f"file not found: {p}"]]
        qpaths.append(os.path.abspath(p))

    # Filter to supported methods
    methods = [m for m in methods_selected if m in {"deep", "color", "umap"}]
    if not methods:
        return [], [[0, "-", 0.0, "select at least one method"]]

    # Normalize weights and preload assets
    W = normalize_weights(methods, w_deep, w_color, w_umap)
    bundles = preload_many(methods, db_path)

    # Per-method candidates: {method: [(image_id, score, rank), ...]}
    per_method_results = {}

    # Encode each query for each method and retrieve a candidate pool
    for m in methods:
        pack = bundles.get(m)
        if not pack:
            continue
        ids = pack["ids"]
        ann = pack["ann"]
        deep_model = pack["deep_model"]
        reducer = pack["reducer"]

        # Early fusion over multiple queries (average)
        vecs = []
        for qp in qpaths:
            v = encode_one(qp, m, deep_model=deep_model, reducer=reducer)
            vecs.append(v)
        q_avg = np.mean(np.stack(vecs, axis=0), axis=0)
        q_avg /= (np.linalg.norm(q_avg) + 1e-12)

        # ANN candidate retrieval
        scores, idxs = ann.query(q_avg[None, :], k=int(pool))

        # Keep (image_id, score, 1-based rank)
        rows = []
        for rank, (sc, ix) in enumerate(zip(scores, idxs), start=1):
            img_id = int(ids[ix])
            rows.append((img_id, float(sc), rank))
        per_method_results[m] = rows

    # Guard fusion type (fallback to "mean")
    if fuser not in {"mean", "max"}:
        fuser = "mean"

    # Collect weighted normalized scores per image over methods
    per_img_scores, seen = {}, set()
    for m in methods:
        if m not in per_method_results:
            continue
        w = W[m]
        for img_id, sc, _rank in per_method_results[m]:
            seen.add(img_id)
            per_img_scores.setdefault(img_id, []).append(w * norm_score(sc))

    # Late fusion: mean or max over contributions
    fused = {}
    for img_id in seen:
        arr = per_img_scores.get(img_id, [])
        if not arr:
            continue
        fused[img_id] = float(np.mean(arr) if fuser == "mean" else np.max(arr))

    # Rank by fused score and keep top-k
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:int(topk)]

    # Resolve filepaths; skip rows that have no valid path
    db = ImageDatabase(db_path)
    results = []
    for img_id, score in ranked:
        fp = db.get_filepath(img_id)
        if not fp or not os.path.isfile(fp):
            continue
        results.append((img_id, fp, float(score)))
    db.close()

    # Make trim_base safe
    trim_base = trim_base or ""

    # Build gallery (first tile: “Query xN”) and table
    gallery = [(Image.open(qpaths[0]).convert("RGB"), f"Query x{len(qpaths)}")]
    table = []
    for rank, (img_id, fp, sc) in enumerate(results, start=1):
        # Be robust if an image fails to open at display time
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:
            continue
        caption = f"{fp.replace(trim_base, '')}\nfused: {sc:.3f}"
        gallery.append((im, caption))
        table.append([rank, img_id, sc, fp])

    return gallery, table