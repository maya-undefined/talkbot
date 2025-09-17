import numpy as np
from .openai_client import OPENAI

EMBEDDING_MODEL = "text-embedding-3-small"  # adjust as needed

async def embed_texts(texts: list[str]) -> list[np.ndarray]:
    vecs = await OPENAI.embeddings(EMBEDDING_MODEL, texts)
    out = []
    for v in vecs:
        arr = np.array(v, dtype=np.float32)
        arr /= (np.linalg.norm(arr) + 1e-9)
        out.append(arr)
    return out