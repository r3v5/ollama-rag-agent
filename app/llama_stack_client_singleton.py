from llama_stack_client import LlamaStackClient

from app.rag_server_config import RAGServerConfig


class LlamaStackClientSingleton:
    """Manages the singleton instance of the LlamaStackClient."""

    # private attribute of LlamaStackClient instance
    __instance = None

    @classmethod
    def get_instance(cls) -> LlamaStackClient:
        """
        Returns the singleton instance of the LlamaStackClient.
        Initializes it on the first call using settings from RAGServerConfig.
        """
        if not cls.__instance:
            print("Creating new LlamaStackClient instance (Singleton)...")
            rag_server_config = RAGServerConfig()
            cls.__instance = LlamaStackClient(base_url=rag_server_config.get_base_url())

        return cls.__instance
