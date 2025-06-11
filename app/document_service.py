from pathlib import Path
from typing import List, Tuple
from llama_stack_client.types import Document


class DocumentService:
    """Handles loading documents from various sources."""

    def load_from_local_files(self, file_paths: Tuple[str, ...]) -> List[Document]:
        """Reads local text files and converts them to Document objects."""
        documents = []
        for i, file_path in enumerate(file_paths):
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                documents.append(
                    Document(
                        document_id=f"doc-{i}",
                        content=content,
                        mime_type="text/plain",
                        metadata={"source": file_path},
                    )
                )
                print(f"📄 Loaded document: {file_path}")
            except FileNotFoundError:
                print(f"⚠️  Warning: File not found at '{file_path}'. Skipping.")
        return documents
