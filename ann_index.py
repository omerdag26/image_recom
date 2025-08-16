import os
import numpy as np
import faiss

# ANNIndex class for building and querying an ANN index using FAISS
class ANNIndex:
    # Initialize the ANNIndex with the dimension of the embeddings and the index path.
    def __init__(self, dim: int, index_path: str = "ann.index"):
        self.dim = dim
        self.index_path = index_path
        self.index = None

        # Use all CPU threads FAISS can see
        try:
            faiss.omp_set_num_threads(os.cpu_count() or 8)
        except Exception:
            pass

    def _make_hnsw(self, M: int = 32, ef_construction: int = 200):
        # HNSW with L2 metric
        index = faiss.IndexHNSWFlat(self.dim, M)
        index.hnsw.efConstruction = ef_construction
        return index
    
    # This function builds the ANN index from the provided embeddings.
    def build(self, embeddings: np.ndarray):
        xb = embeddings.astype(np.float32, copy=False)
        # Normalize so that L2 distance corresponds to cosine similarity ordering
        faiss.normalize_L2(xb)
        # Create the index using HNSW
        index = self._make_hnsw(M=32, ef_construction=200)
        index.add(xb)
        # Save the index to the specified path
        faiss.write_index(index, self.index_path)
        self.index = index

    # This function loads the ANN index from the specified path.
    def load(self):
        self.index = faiss.read_index(self.index_path)

    # This function queries the ANN index with a given embedding.
    def query(self, emb: np.ndarray, k: int = 5, ef_search: int = 128):
        # If the index is not loaded, raise an error
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")
        # Ensure the embedding is a 1D array and reshape it for querying
        q = emb.astype(np.float32, copy=False).reshape(1, -1)
        faiss.normalize_L2(q)

        # Set the efSearch parameter for HNSW if applicable
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = max(ef_search, k * 16)
        # Perform the search in the index
        D, I = self.index.search(q, k)
        # Convert distances to similarity scores (1 - normalized distance)
        scores = 1.0 - (D[0] / 2.0)
        # Return the similarity scores and indices of the nearest neighbors
        return scores, I[0]