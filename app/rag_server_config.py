import os
import uuid


class RAGServerConfig:
    """Holds all static configuration for the RAG server."""

    def __init__(self) -> None:
        """Initializes the configuration properties."""
        self.LLAMA_STACK_PORT = os.environ.get("LLAMA_STACK_PORT")
        self.INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL")
        self.EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
        self.EMBEDDING_DIM = os.environ.get("EMBEDDING_DIM")
        self.VECTOR_DB_PROVIDER = os.environ.get("VECTOR_DB_PROVIDER")
        self.CHUNK_SIZE_IN_TOKENS = os.environ.get("CHUNK_SIZE_IN_TOKENS")
        self.RAG_TOP_K = os.environ.get("RAG_TOP_K")
        self.VECTOR_DB_NAME = os.environ.get("VECTOR_DB_NAME")

        # VLM (Visual Language Model) configuration
        self.VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME")
        self.OLLAMA_API_URL_FOR_GENERATING = os.environ.get(
            "OLLAMA_API_URL_FOR_GENERATING"
        )
        self.MIN_OCR_TEXT_LENGTH = os.environ.get("MIN_OCR_TEXT_LENGTH")

    def get_llama_stack_port(self) -> int:
        return int(self.LLAMA_STACK_PORT) if self.LLAMA_STACK_PORT else 8321

    def get_inference_model(self) -> str:
        return self.INFERENCE_MODEL or "ollama/llama3.2:3b-instruct-fp16"

    def get_embedding_model(self) -> str:
        return self.EMBEDDING_MODEL or "ollama/all-minilm:latest"

    def get_embedding_dim(self) -> int:
        return int(self.EMBEDDING_DIM) if self.EMBEDDING_DIM else 384

    def get_vector_db_provider(self) -> str:
        return self.VECTOR_DB_PROVIDER or "milvus"

    def get_chunk_size_in_tokens(self) -> int:
        return int(self.CHUNK_SIZE_IN_TOKENS) if self.CHUNK_SIZE_IN_TOKENS else 256

    def get_rag_top_k(self) -> int:
        return int(self.RAG_TOP_K) if self.RAG_TOP_K else 3

    def get_vector_db_name(self) -> str:
        if self.VECTOR_DB_NAME:
            return self.VECTOR_DB_NAME
        else:
            # If not set, generate a unique default name to avoid collisions
            return f"rag-db-{uuid.uuid4().hex}"

    def get_vlm_model_name(self) -> str:
        return self.VLM_MODEL_NAME

    def get_ollama_api_url_for_generating(self) -> str:
        return self.OLLAMA_API_URL_FOR_GENERATING

    def get_min_ocr_text_length(self) -> int:
        return int(self.MIN_OCR_TEXT_LENGTH)

    def get_base_url(self) -> str:
        """Constructs the base URL for the LlamaStack client."""
        return f"http://localhost:{self.get_llama_stack_port()}"
