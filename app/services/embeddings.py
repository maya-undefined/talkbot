import numpy as np

class Embedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v

EMBEDDER = Embedder()