import uuid
from pathlib import Path
from typing import List


from llama_stack_client.types import Document

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ImageFormatOption,
)


from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
)

from docling.datamodel.pipeline_options import OcrOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions


class DocumentSystem:
    """
    Handles loading documents from various sources, using a customized
    docling DocumentConverter based on the official examples.
    """

    def __init__(self):
        """
        Initializes the service and a highly customized DocumentConverter client.
        """
        print("Initializing DocumentService with custom multi-format converter...")

        ocr_opts = OcrOptions(engine="tesseract", lang=["eng"])
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = True
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.table_structure_options.do_cell_matching = True

        # This setup now correctly configures a pipeline for each file type.
        self._docling_converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                    ocr=ocr_opts,
                ),
                InputFormat.IMAGE: ImageFormatOption(ocr=ocr_opts),
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
            print("⚠️ Warning: No valid file paths were provided or files do not exist.")
            return []

        print(f"-> Processing batch of {len(input_paths)} files with Docling...")
        try:
            conv_results = self._docling_converter.convert_all(input_paths)

            for res in conv_results:
                if res.document:
                    print(f"   - Successfully converted: {res.input.file.name}")
                    # Extract and pre-process text for better RAG performance
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
                            f"⚠️ Warning: No text content was extracted from {res.input.file.name}."
                        )
                else:
                    print(
                        f"⚠️ Warning: Conversion failed or produced no document for {res.input.file.name}."
                    )

        except Exception as e:
            print(f"❌ An error occurred during Docling batch conversion: {e}")

        print(
            f"📄 Successfully loaded and processed a total of {len(documents)} document(s)."
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
