from typing import List
import uuid
from llama_stack_client.types import Document
from app.app_config import AppConfig
from app.llama_stack_client_singleton import LlamaStackClientSingleton


class VectorDBManager:
    """Manages vector database creation and data insertion."""

    def __init__(self, config: AppConfig) -> None:
        self.client = LlamaStackClientSingleton.get_instance()
        self.config = config

    def setup_and_insert(self, documents: List[Document]) -> str:
        """Registers a new vector DB and inserts documents."""

        vector_db_id = f"matias-vector-db-{uuid.uuid4().hex}"
        print(f"\n:file_cabinet: Registering vector DB: {vector_db_id}")
        self.client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=self.config.get_embedding_model(),
            embedding_dimension=self.config.get_embedding_dim(),
            provider_id=self.config.get_vector_db_provider(),
        )

        print(":inbox_tray: Inserting documents into vector DB...")
        self.client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=self.config.get_chunk_size_in_tokens(),
        )
        print(f":page_facing_up: Inserted {len(documents)} document(s).")
        return vector_db_id
