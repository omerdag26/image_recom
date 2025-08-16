# tests/test_umap.py
import os, numpy as np, pytest, tempfile
from PIL import Image
from search import encode_one
from similarity_measures import load_model

UMAP_MODEL = "models/umap_deep.joblib"

@pytest.mark.skipif(not os.path.exists(UMAP_MODEL), reason="UMAP model missing")
def test_umap_encode_shape_and_norm():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "img.jpg")
        Image.new("RGB", (224,224), (50,80,120)).save(p)
        v = encode_one(p, method="umap", deep_model=load_model())
        assert v.ndim == 1 and v.size > 0
        n = np.linalg.norm(v)
        assert 0.99 < n < 1.01