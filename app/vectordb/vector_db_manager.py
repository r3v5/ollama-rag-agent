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

    def setup_and_insert(self, document: List[Document]) -> str:
        """
        Ensures a vector DB is registered and inserts a single document into it.

        This method creates a new, unique database for each session.

        Args:
            document: A single Document object to insert into the vector DB.

        Returns:
            The ID of the vector database used for the operation.

        Raises:
            RuntimeError: If there is a failure communicating with the LlamaStack server.
            ValueError: If the provided document is invalid.
        """
        if not document:
            raise ValueError("A valid document must be provided for insertion.")

        # --- Database Registration ---
        # For a single-document session, we always create a new DB.
        db_name = self.rag_server_config.get_vector_db_name()
        self.set_vector_db_id(f"{db_name}-{uuid.uuid4().hex}")

        print(f"\n:file_cabinet: Registering new vector DB: {self.get_vector_db_id()}")
        try:
            self.llamastack_client.vector_dbs.register(
                vector_db_id=self.get_vector_db_id(),
                embedding_model=self.rag_server_config.get_embedding_model(),
                embedding_dimension=self.rag_server_config.get_embedding_dim(),
                provider_id=self.rag_server_config.get_vector_db_provider(),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to register vector DB. Original error: {e}"
            ) from e

        # --- Document Insertion ---
        print(f":inbox_tray: Inserting document into {self.get_vector_db_id()}...")
        try:

            # The API client's insert method expects a list, so we wrap our single document.
            self.llamastack_client.tool_runtime.rag_tool.insert(
                documents=document,
                vector_db_id=self.get_vector_db_id(),
                chunk_size_in_tokens=self.rag_server_config.get_chunk_size_in_tokens(),
            )
            print(f":page_facing_up: Insertion complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to insert document. Original error: {e}") from e

        # Return the active DB ID for the agent to use.
        return self.get_vector_db_id()
