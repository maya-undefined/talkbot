from typing import List

def split_into_chunks(text: str, target_tokens: int = 400, overlap: int = 60) -> List[str]:
    words = text.split()
    chunk_size = target_tokens
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c.strip()]
