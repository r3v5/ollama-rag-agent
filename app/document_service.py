import uuid
from pathlib import Path
from typing import List


from llama_stack_client.types import Document

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


class DocumentService:
    """
    Handles loading documents from various sources, using a customized
    docling DocumentConverter based on the official examples.
    """

    def __init__(self):
        """
        Initializes the service and a highly customized DocumentConverter client.
        """
        print("Initializing DocumentService with custom multi-format converter...")

        self._docling_converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
            },
        )

    def load_from_local_files(self, file_paths: List[str]) -> List[Document]:
        """
        Reads a list of local files, processes them in a single batch using
        the customized DocumentConverter, and converts them to Document objects.
        """
        documents = []
        input_paths = [Path(p) for p in file_paths if Path(p).exists()]

        if not input_paths:
            print("Warning: No valid file paths were provided or files do not exist.")
            return []

        print(f"-> Processing batch of {len(input_paths)} files with Docling...")
        try:
            conv_results = self._docling_converter.convert_all(input_paths)

            for res in conv_results:
                if res.document:
                    print(f"   - Successfully converted: {res.input.file.name}")
                    content = res.document.export_to_text().replace("\n", " ").strip()

                    if content:
                        documents.append(
                            Document(
                                document_id=f"doc-{uuid.uuid4().hex}",
                                content=content,
                                mime_type="text/plain",
                                metadata={"source": str(res.input.file)},
                            )
                        )
                    else:
                        print(
                            f"Warning: No text content was extracted from {res.input.file.name}."
                        )
                else:
                    print(
                        f"Warning: Conversion failed or produced no document for {res.input.file.name}."
                    )

        except Exception as e:
            print(f"An error occurred during Docling batch conversion: {e}")

        print(
            f"Successfully loaded and processed a total of {len(documents)} document(s)."
        )
        return documents

    def create_document_from_text(
        self, content: str, source: str = "user_input"
    ) -> Document:
        """Creates a Document object from a raw string of text."""
        processed_content = content.replace("\n", " ").strip()
        return Document(
            document_id=f"doc-{uuid.uuid4().hex}",
            content=processed_content,
            mime_type="text/plain",
            metadata={"source": source},
        )
