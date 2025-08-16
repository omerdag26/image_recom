import numpy as np
from ann_index import ANNIndex
import tempfile, os

def test_ann_hnsw_top1_matches_bruteforce():
    # 2D, normalize → L2 == cosine sıralaması
    X = np.array([[1,0],[0,1],[1,1]], dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True)+1e-12)
    q = np.array([1,0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as d:
        idx_path = os.path.join(d, "x.index")
        ann = ANNIndex(2, index_path=idx_path)
        ann.build(X)
        ann.load()
        scores, inds = ann.query(q.reshape(1,-1), k=1)
        # en yakın [1,0] (index 0)
        assert int(inds[0]) == 0
        assert np.isfinite(scores[0])