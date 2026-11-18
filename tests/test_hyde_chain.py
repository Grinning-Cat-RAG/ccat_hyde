if __name__ == "__main__":
    import asyncio
    import importlib.util
    import os
    from pathlib import Path

    # The core needs its crypto env vars at import time.
    os.environ.setdefault("CAT_CRYPTO_SALT", "grinning_cat_salt")
    os.environ.setdefault("CAT_CRYPTO_KEY", "grinning_cat_key")

    plugin_dir = Path(__file__).resolve().parent.parent

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from cat import UserMessage

    hyde = _load("ccat_hyde_hyde", plugin_dir / "hyde.py")
    settings_module = _load("ccat_hyde_settings", plugin_dir / "settings.py")
    MySettings = settings_module.MySettings

    class FakeEmbedder:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    class FakePlugin:
        async def load_settings(self):
            return MySettings().model_dump()

    class FakeMadHatter:
        def get_plugin(self):
            return FakePlugin()

    class FakeCat:
        def __init__(self, llm):
            self.working_memory = {}
            self.mad_hatter = FakeMadHatter()
            self.large_language_model = llm

        async def embedder(self):
            return FakeEmbedder()

    async def main():
        # Happy path: the chain writes the HyDE answer and returns the UserMessage.
        llm = RunnableLambda(lambda _prompt_value: AIMessage(content="fake hyde answer"))
        cat = FakeCat(llm)
        user_message = UserMessage(text="what furniture is in my room?")

        returned = await hyde.before_cat_reads_message.function(user_message, cat)

        assert returned is user_message, "hook must return the UserMessage"
        assert cat.working_memory[hyde.HYDE_ANSWER] == "fake hyde answer", (
            "hyde_answer must be written to working memory"
        )

        # Failure path: an LLM error propagates and leaves no hyde_answer behind.
        def boom(_prompt_value):
            raise RuntimeError("llm failure")

        failing_cat = FakeCat(RunnableLambda(boom))
        try:
            await hyde.before_cat_reads_message.function(
                UserMessage(text="hello"), failing_cat
            )
        except RuntimeError as exc:
            assert str(exc) == "llm failure"
        else:
            raise AssertionError("LLM exception must propagate")
        assert hyde.HYDE_ANSWER not in failing_cat.working_memory, (
            "hyde_answer must not be written when the LLM fails"
        )

        print("test_hyde_chain: ALL ASSERTIONS PASSED")

    asyncio.run(main())
