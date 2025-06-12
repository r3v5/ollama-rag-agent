from typing import List
import uuid
from llama_stack_client.types import Document
from app.rag_server_config import RAGServerConfig
from app.llama_stack_client_singleton import LlamaStackClientSingleton


class VectorDBManager:
    """Manages vector database creation and data insertion."""

    def __init__(self, rag_server_config: RAGServerConfig) -> None:
        self.llamastack_client = LlamaStackClientSingleton.get_instance()
        self.rag_server_config = rag_server_config
        self.__vector_db_id = None

    def get_vector_db_id(self) -> str:
        return self.__vector_db_id

    def set_vector_db_id(self, new_vector_db_id: str) -> None:
        self.__vector_db_id = new_vector_db_id

    def setup_and_insert(self, documents: List[Document]) -> str:
        """Registers a new vector DB and inserts documents."""

        self.set_vector_db_id(f"matias-vector-db-{uuid.uuid4().hex}")
        print(f"\n:file_cabinet: Registering vector DB: {self.get_vector_db_id()}")
        self.llamastack_client.vector_dbs.register(
            vector_db_id=self.get_vector_db_id(),
            embedding_model=self.rag_server_config.get_embedding_model(),
            embedding_dimension=self.rag_server_config.get_embedding_dim(),
            provider_id=self.rag_server_config.get_vector_db_provider(),
        )

        print(":inbox_tray: Inserting documents into vector DB...")
        self.llamastack_client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=self.get_vector_db_id(),
            chunk_size_in_tokens=self.rag_server_config.get_chunk_size_in_tokens(),
        )
        print(f":page_facing_up: Inserted {len(documents)} document(s).")
        return self.get_vector_db_id()
