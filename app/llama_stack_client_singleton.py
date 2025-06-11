from llama_stack_client import LlamaStackClient

from app.app_config import AppConfig


class LlamaStackClientSingleton:
    """Manages the singleton instance of the LlamaStackClient."""

    # private attribute of LlamaStackClient instance
    _instance = None

    @classmethod
    def get_instance(cls) -> LlamaStackClient:
        """
        Returns the singleton instance of the LlamaStackClient.
        Initializes it on the first call using settings from AppConfig.
        """
        if cls._instance is None:
            print("Creating new LlamaStackClient instance (Singleton)...")
            config = AppConfig()
            cls._instance = LlamaStackClient(base_url=config.get_base_url())
        return cls._instance
