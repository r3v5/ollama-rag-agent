import uuid
from pathlib import Path
import base64
import requests
from llama_stack_client.types import Document
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ImageFormatOption,
)
from app.rag_server_config import RAGServerConfig

from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
)

from docling.datamodel.pipeline_options import OcrOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions


class DocumentSystem:
    """
    Handles loading documents from various sources. It uses a smart pipeline that
    attempts OCR on images first, then falls back to a VLM if no text is found.
    """

    def __init__(self, rag_server_config: RAGServerConfig):
        """Initializes the service and specialized DocumentConverter clients."""
        print("Initializing DocumentService with smart routing capabilities...")

        self.rag_server_config = rag_server_config
        ocr_opts = OcrOptions(engine="tesseract", lang=["eng"])
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = True
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.table_structure_options.do_cell_matching = True

        # --- Specialized Converters ---
        # A converter for standard text-based documents (PDF)
        self.docling_pdf_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                    ocr=ocr_opts,
                ),
            },
        )

        # A dedicated converter just for attempting OCR on images
        self.docling_image_converter = DocumentConverter(
            format_options={InputFormat.IMAGE: ImageFormatOption(ocr=ocr_opts)}
        )

    def _get_image_description_from_vlm(self, file_path: Path) -> str:
        """
        Sends an image to a running VLM via the Ollama API to get a text description.
        """
        print(
            f"   (No significant text found via OCR. Using VLM '{self.rag_server_config.get_vlm_model_name()}' to describe image...)"
        )
        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            payload = {
                "model": self.rag_server_config.get_vlm_model_name(),
                "prompt": "Describe this image in detail. Be specific about objects, colors, setting, and any text present.",
                "stream": False,
                "images": [encoded_string],
            }

            response = requests.post(
                self.rag_server_config.get_ollama_api_url_for_generating(),
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

            description = response.json().get("response", "").strip()
            print("   -> VLM Description received.")
            return description
        except requests.exceptions.ConnectionError:
            print(
                f"Error: Could not connect to Ollama at {self.rag_server_config.get_ollama_api_url_for_generating()}. Please ensure Ollama is running."
            )
            return ""
        except Exception as e:
            print(f"An error occurred while generating the image description: {e}")
            return ""

    def load_file(self, file_path_str: str) -> Document:
        """
        Reads a single file, intelligently routing it to the best processing
        pipeline (OCR vs. VLM for images), and returns a single Document object.
        """
        file_path = Path(file_path_str)
        content = ""
        try:
            print(f"-> Processing file: {file_path.name}")
            extension = file_path.suffix.lower()

            if not file_path.exists():
                print(f"Warning: File not found: {file_path_str}")
                return None

            if extension == ".txt":
                print("   (Plain text file detected. Reading directly...)")
                content = file_path.read_text(encoding="utf-8")

            elif extension in [".png", ".jpg", ".jpeg"]:
                print("   (Image detected. Attempting OCR first...)")
                ocr_result = self.docling_image_converter.convert(file_path)
                if ocr_result and ocr_result.document:
                    content = ocr_result.document.export_to_text().strip()

                if len(content) <= self.rag_server_config.get_min_ocr_text_length():
                    content = self._get_image_description_from_vlm(file_path)
                else:
                    print(
                        f"   -> Found significant text ({len(content)} chars). Using OCR content."
                    )

            elif extension == ".pdf":
                print(f"   (Using Docling to convert '{extension}' to text...)")
                result = self.docling_pdf_converter.convert(file_path)
                if result and result.document:
                    content = result.document.export_to_text()

            else:
                print(f"Warning: Unsupported file type '{extension}'.")
                return None

            if content:
                processed_content = content.replace("\n", " ").strip()
                print(f"Successfully created document for: {file_path.name}")
                return Document(
                    document_id=f"doc-{uuid.uuid4().hex}",
                    content=processed_content,
                    mime_type="text/plain",
                    metadata={"source": str(file_path)},
                )
            else:
                print(f"Warning: No text content was created for {file_path.name}.")
                return None

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return None

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
