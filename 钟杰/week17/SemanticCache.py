import numpy as np
from typing import List, Union, Callable, Any
from redisvl.extensions.cache.llm import SemanticCache as RedisVLSemanticCache

class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any] = None,
            ttl: int = 3600 * 24,
            redis_url: str = "redis://localhost:6379",
            distance_threshold=0.1
    ):
        self.name = name
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method  # redisvl 内部自带 embedding，保留参数仅兼容旧接口

        self.llmcache = RedisVLSemanticCache(
            name=name,
            ttl=ttl,
            redis_url=redis_url,
            distance_threshold=distance_threshold
        )

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        for q, a in zip(prompt, response):
            self.llmcache.store(prompt=q, response=a)
        return True

    def call(self, prompt: str):
        if self.llmcache is None:
            return None

        results = self.llmcache.check(prompt=prompt)
        if not results:
            return None
        # redisvl 返回 [{"response": "..."}, ...]
        return [r.get("response") for r in results]

    def clear_cache(self):
        self.llmcache.clear()

    def query(self, prompt: str, llm_method: Callable[[str], str] = None):
        # 1. 尝试从缓存获取
        cached = self.call(prompt)
        if cached is not None:
            for r in cached:
                if r is not None:
                    return r

        # 2. 缓存未命中，调用 LLM
        if llm_method is None:
            return None
        response = llm_method(prompt)
        self.store(prompt, response)
        return response


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]

        return np.array([np.ones(768) for t in text])

    embed_cache = SemanticCache(
        name="semantic_ache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()

    embed_cache.store(prompt="hello world", response="hello world1232")
    print(embed_cache.call(prompt="hello world"))

    embed_cache.store(prompt="hello my bame", response="nihao")
    print(embed_cache.call(prompt="hello world"))