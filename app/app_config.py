import os
from typing import Tuple


class AppConfig:
    """Holds all static configuration for the application."""

    def __init__(self) -> None:
        """Initializes the configuration properties."""
        self._LLAMA_STACK_PORT = os.environ.get("LLAMA_STACK_PORT")
        self._INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL")
        self._EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
        self._EMBEDDING_DIM = os.environ.get("EMBEDDING_DIM")
        self._LOCAL_FILES = os.environ.get("LOCAL_FILES")
        self._VECTOR_DB_PROVIDER = os.environ.get("VECTOR_DB_PROVIDER")
        self._CHUNK_SIZE_IN_TOKENS = os.environ.get("CHUNK_SIZE_IN_TOKENS")
        self._RAG_TOP_K = os.environ.get("RAG_TOP_K")

    def get_llama_stack_port(self) -> int:
        return int(self._LLAMA_STACK_PORT) if self._LLAMA_STACK_PORT else 8321

    def get_inference_model(self) -> str:
        return self._INFERENCE_MODEL or "ollama/llama3.2:3b-instruct-fp16"

    def get_embedding_model(self) -> str:
        return self._EMBEDDING_MODEL or "ollama/all-minilm:latest"

    def get_embedding_dim(self) -> int:
        return int(self._EMBEDDING_DIM) if self._EMBEDDING_DIM else 384

    def get_local_files(self) -> Tuple[str, ...]:
        if self._LOCAL_FILES:
            file_list = [item.strip() for item in self._LOCAL_FILES.split(",")]
            return tuple(file_list)
        else:
            return ("matias.txt", "ian.txt")

    def get_vector_db_provider(self) -> str:
        return self._VECTOR_DB_PROVIDER or "milvus"

    def get_chunk_size_in_tokens(self) -> int:
        return int(self._CHUNK_SIZE_IN_TOKENS) if self._CHUNK_SIZE_IN_TOKENS else 256

    def get_rag_top_k(self) -> int:
        return int(self._RAG_TOP_K) if self._RAG_TOP_K else 3

    def get_base_url(self) -> str:
        """Constructs the base URL for the LlamaStack client."""
        return f"http://localhost:{self.get_llama_stack_port()}"
