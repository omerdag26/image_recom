import numpy as np, pytest, tempfile, os
from PIL import Image
pytestmark = pytest.mark.slow

from similarity_measures import compute_embedding, load_model

def test_deep_embedding_shape_and_norm():
    model = load_model()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jpg")
        Image.new("RGB",(224,224),(120,80,30)).save(p)
        v = compute_embedding(p, model=model)
        assert v.shape == (2048,)
        assert np.isfinite(v).all()
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-3)

def test_deep_shape_and_norm():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "img.jpg")
        Image.new("RGB", (224,224), (120,130,140)).save(p)
        v = compute_embedding(p, model=load_model())
        assert v.ndim == 1 and v.shape[0] == 2048
        assert np.isfinite(v).all()
        n = np.linalg.norm(v)
        assert 0.99 < n < 1.01