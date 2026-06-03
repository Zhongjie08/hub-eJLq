from typing import Optional, Union, List, Dict, Any
from redisvl.extensions.message_history import SemanticMessageHistory as RedisVLSemanticMessageHistory


class SemanticMessageHistory:
    def __init__(
            self,
            name: str,
            redis_url: str = "redis://localhost:6379",
            distance_threshold: float = 0.7,
    ):
        self._history = RedisVLSemanticMessageHistory(
            name=name,
            redis_url=redis_url,
            distance_threshold=distance_threshold,
        )

    def get_history(self):
        return self._history.get_recent(top_k=1000000)

    def add_message(self, message: Union[Dict[str, Any], List[Dict[str, Any]]]):
        if isinstance(message, dict):
            message = [message]
        self._history.add_messages(message)

    def get_recent(self, **kwargs):
        return self._history.get_recent(**kwargs)

    def get_relevant(self, content: str, top_k=10):
        return self._history.get_relevant(content, top_k=top_k)

    def delete_history(self, top_k=10):
        history = self.get_history()
        keep = history[-top_k:]
        self.clear_history()
        if keep:
            self._history.add_messages(keep)

    def clear_history(self):
        self._history.clear()


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="redis://localhost:6379",
    )
    history.clear_history()
    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "what is the weather going to be today?"},
    ])

    print("get_history", history.get_history())
    print("get_recent topk=1", history.get_recent(top_k=1))
    print("get_recent role=user", history.get_recent(role="user", top_k=1))

    print("\nget_relevant today", history.get_relevant("today", top_k=1))
    print("get_relevant thanks", history.get_relevant("thanks", top_k=1))