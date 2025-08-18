# Image Recommender (Deep · Color · UMAP)

Large-scale image similarity search with:
- **Deep** — ResNet50 avg-pool (2048-D, semantic).
- **Color** — LAB histogram (bins³, palette/style).
- **UMAP** — Deep → 256-D (smaller, faster ANN).

Features saved as `.npz`, indexed with **FAISS HNSW**. Simple **Gradio** UI. Images tracked in **SQLite**.

## Setup
conda create -n image_recom python=3.10 -y

conda activate image_recom

pip install -r requirements.txt

if FAISS missing:   pip install faiss-cpu

Apple Silicon TF:   pip install tensorflow-macos


##1) Build the image DB

python populate_db.py /path/to/images --db images.db

##2) Build features + ANN

### Deep (2048-D)
python build.py --method deep  --db images.db --chunk 4096

### Color (bins=8)
python build.py --method color --db images.db --bins 8 --workers 4 --chunk 4096

### UMAP (Deep → 256-D) — requires deep features
python build.py --method umap --deep_features features/deep.npz

Outputs:
	•	Features → features/{deep|color|umap}.npz
	•	Indexes  → index/{deep|color|umap}.index



## 3) Run the app

python app.py

open: http://0.0.0.0:7860


## Bench quick check

python bench_search.py --method deep  --db images.db --features features/deep.npz  --index index/deep.index  --n 10 --topk 10
python bench_search.py --method color --db images.db --features features/color.npz --index index/color.index --n 10 --topk 10
python bench_search.py --method umap  --db images.db --features features/umap.npz  --index index/umap.index --umap_model models/umap_deep.joblib --n 10 --topk 10




## Tests(Make sure test files are in the same directory as project)

pytest -q
