from llama_stack_client import LlamaStackClient

from app.rag_server_config import RAGServerConfig


class LlamaStackClientSingleton:
    """Manages the singleton instance of the LlamaStackClient."""

    # private attribute of LlamaStackClient instance
    __instance = None

    @staticmethod
    def get_instance() -> LlamaStackClient:
        """
        Returns the singleton instance of the LlamaStackClient.
        Initializes it on the first call using settings from RAGServerConfig.
        """
        if not LlamaStackClientSingleton.__instance:
            print("Creating new LlamaStackClient instance (Singleton)...")
            rag_server_config = RAGServerConfig()
            LlamaStackClientSingleton.__instance = LlamaStackClient(
                base_url=rag_server_config.get_base_url()
            )

        return LlamaStackClientSingleton.__instance
