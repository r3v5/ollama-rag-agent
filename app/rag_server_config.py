import os
import uuid


class RAGServerConfig:
    """Holds all static configuration for the RAG server."""

    def __init__(self) -> None:
        """Initializes the configuration properties."""
        self.__LLAMA_STACK_PORT = os.environ.get("LLAMA_STACK_PORT")
        self.__INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL")
        self.__EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
        self.__EMBEDDING_DIM = os.environ.get("EMBEDDING_DIM")
        self.__VECTOR_DB_PROVIDER = os.environ.get("VECTOR_DB_PROVIDER")
        self.__CHUNK_SIZE_IN_TOKENS = os.environ.get("CHUNK_SIZE_IN_TOKENS")
        self.__RAG_TOP_K = os.environ.get("RAG_TOP_K")
        self.__VECTOR_DB_NAME = os.environ.get("VECTOR_DB_NAME")

    def get_llama_stack_port(self) -> int:
        return int(self.__LLAMA_STACK_PORT) if self.__LLAMA_STACK_PORT else 8321

    def get_inference_model(self) -> str:
        return self.__INFERENCE_MODEL or "ollama/llama3.2:3b-instruct-fp16"

    def get_embedding_model(self) -> str:
        return self.__EMBEDDING_MODEL or "ollama/all-minilm:latest"

    def get_embedding_dim(self) -> int:
        return int(self.__EMBEDDING_DIM) if self.__EMBEDDING_DIM else 384

    def get_vector_db_provider(self) -> str:
        return self.__VECTOR_DB_PROVIDER or "milvus"

    def get_chunk_size_in_tokens(self) -> int:
        return int(self.__CHUNK_SIZE_IN_TOKENS) if self.__CHUNK_SIZE_IN_TOKENS else 256

    def get_rag_top_k(self) -> int:
        return int(self.__RAG_TOP_K) if self.__RAG_TOP_K else 3

    def get_vector_db_name(self) -> str:
        if self.__VECTOR_DB_NAME:
            return self.__VECTOR_DB_NAME
        else:
            # If not set, generate a unique default name to avoid collisions
            return f"rag-db-{uuid.uuid4().hex}"

    def get_base_url(self) -> str:
        """Constructs the base URL for the LlamaStack client."""
        return f"http://localhost:{self.get_llama_stack_port()}"
