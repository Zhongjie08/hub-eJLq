import numpy as np
from typing import List, Union
from redisvl.extensions.cache.embeddings import EmbeddingsCache as RedisVLEmbeddingsCache
from redisvl.utils.vectorize import HFTextVectorizer


class EmbeddingsCache:
    def __init__(
            self,
            name: str,
            ttl: int = 3600,
            redis_url: str = "redis://localhost:6379",
            model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.name = name
        self.ttl = ttl

        self._cache = RedisVLEmbeddingsCache(
            name=name,
            redis_url=redis_url,
            ttl=ttl,
        )

        self._vectorizer = HFTextVectorizer(
            model=model,
            cache=self._cache
        )

    def store(self, text: Union[List[str], str], embedding: np.ndarray):
        if isinstance(text, str):
            text = [text]

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        results = []
        for i, t in enumerate(text):
            self._cache.store(key=t, vector=embedding[i].tolist())
            results.append(True)
        return results

    def call(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        results = []
        for t in text:
            vec = self._cache.retrieve(key=t)
            if vec is None:
                results.append(None)
            else:
                results.append(np.array(vec, dtype=np.float32))
        return results

    def delete(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        keys_deleted = 0
        for t in text:
            try:
                self._cache.delete(key=t)
                keys_deleted += 1
            except Exception:
                pass
        return keys_deleted if keys_deleted > 0 else -1

    def embed(self, text: str) -> np.ndarray:
        return np.array(self._vectorizer.embed(text), dtype=np.float32)


if __name__ == "__main__":
    embed_cache = EmbeddingsCache(
        name="embed_cache",
        ttl=360,
    )

    # 测试 embed（自动缓存）
    emb = embed_cache.embed("What is machine learning?")
    print(f"embedding shape: {emb.shape}")

    # 测试 store + call（手动存取）
    print(embed_cache.store(text="hello world", embedding=np.random.rand(768)))
    print(embed_cache.call(text="hello world"))
    print(embed_cache.delete(text="hello world"))