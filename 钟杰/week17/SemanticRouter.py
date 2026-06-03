from redisvl.extensions.router import Route, SemanticRouter as RedisVLSemanticRouter


class SemanticRouter:
    def __init__(self, name: str = "topic-router", redis_url: str = "redis://localhost:6379"):
        self.name = name
        self.redis_url = redis_url
        self.routes = []
        self._router = None

    def add_route(self, **kwargs):
        """
        添加路由规则。
        支持动态参数，例如：
            add_route(questions=["hello", "hi"], target="greeting")
        """
        target = kwargs.get("target")
        questions = kwargs.get("questions", [])

        route = Route(
            name=target,
            references=questions,
            metadata=kwargs.get("metadata", {"type": target}),
            distance_threshold=kwargs.get("distance_threshold", 0.3),
        )
        self.routes.append(route)
        self._rebuild_router()

    def _rebuild_router(self):
        self._router = RedisVLSemanticRouter(
            name=self.name,
            routes=self.routes,
            redis_url=self.redis_url,
        )

    def route(self, question: str):
        if self._router is None:
            return None
        return self._router(question)

    def __call__(self, question: str):
        return self.route(question)


if __name__ == "__main__":
    router = SemanticRouter()
    router.add_route(questions=["Hi, good morning", "Hi, good afternoon"], target="greeting")
    router.add_route(questions=["如何退货"], target="refund")

    print(router("Hi, good morning"))