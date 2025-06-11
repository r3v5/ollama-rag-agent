import traceback
import sys
from dotenv import load_dotenv

from app.app_config import AppConfig
from app.document_service import DocumentService
from app.rag_agent_manager import RAGAgentManager
from app.vectordb.vector_db_manager import VectorDBManager


def main():
    """Orchestrates the interactive RAG application setup and execution."""
    try:
        # --- Initialization ---
        config = AppConfig()
        doc_service = DocumentService()
        vector_db_manager = VectorDBManager(config)
        agent_manager = RAGAgentManager(config)

        # --- Interactive Document Input ---
        print("--- Real-time RAG Document Loader (with Docling) ---")
        print("How would you like to provide the document(s)?")

        choice = input(
            "Enter '1' to provide file path(s) (pdf, docx, png, txt), or '2' to paste text directly: "
        ).strip()

        documents = []
        if choice == "1":
            # ENHANCED: Now accepts one or more comma-separated paths
            paths_input = input(
                "Please enter the full path to your file(s), separated by commas: "
            ).strip()
            # Create a list by splitting the input string and stripping whitespace from each path
            file_paths = [path.strip() for path in paths_input.split(",")]

            # The DocumentService now processes the whole list in a batch
            documents = doc_service.load_from_local_files(file_paths)

        elif choice == "2":
            print(
                "Please paste your text below. Press Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows) when done."
            )
            content = sys.stdin.read()
            if content:
                documents.append(doc_service.create_document_from_text(content))
                print("📄 Text received.")
        else:
            print("❌ Invalid choice. Exiting.")
            return

        if not documents:
            print("❌ No documents were loaded. Exiting.")
            return

        vector_db_id = vector_db_manager.setup_and_insert(documents)
        agent_instructions = "You are a helpful assistant that answers questions based only on the document provided."
        agent_manager.initialize_agent(vector_db_id, agent_instructions)

        # --- Interactive Q&A Loop ---
        print("\n--- Query Your Document(s) ---")
        while True:
            user_prompt = input("\nAsk a question (or type 'quit' to exit): ").strip()
            if user_prompt.lower() == "quit":
                print("👋 Goodbye!")
                break

            if not user_prompt:
                continue

            final_output, chunks_found = agent_manager.make_query(user_prompt)

            if final_output:
                print("\n:brain: RAG Response:")
                print(final_output)
            elif chunks_found:
                print(
                    "\n:warning: Chunks were retrieved but the model did not generate a response."
                )
            else:
                print(
                    "\n:x: No relevant chunks were found in the document for your query."
                )

    except Exception as e:
        print(f"\n:x: An unexpected error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Loading environment variables from .env.dev")
    load_dotenv(dotenv_path=".env.dev")
    main()
