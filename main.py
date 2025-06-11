import traceback


from dotenv import load_dotenv


from app.app_config import AppConfig
from app.document_service import DocumentService
from app.rag_agent_manager import RAGAgentManager
from app.vectordb.vector_db_manager import VectorDBManager


def main():
    """Orchestrates the RAG application setup and execution."""
    try:
        # Initialization
        config = AppConfig()
        doc_service = DocumentService()
        vector_db_manager = VectorDBManager(config)
        agent_manager = RAGAgentManager(config)

        # Execution Flow
        print("--- Loading documents from configuration ---")
        local_files_to_load = config.get_local_files()
        documents = doc_service.load_from_local_files(local_files_to_load)
        if not documents:
            print("❌ No documents were loaded. Exiting.")
            return

        vector_db_id = vector_db_manager.setup_and_insert(documents)

        agent_instructions = "You are a helpful assistant that answers questions based only on the provided documents."
        agent_manager.initialize_agent(vector_db_id, agent_instructions)

        user_prompt = "Who was Matias Schimuneck?"
        final_output, chunks_found = agent_manager.make_query(user_prompt)

        if final_output:
            print("\n:brain: RAG Response:")
            print(final_output)
        elif chunks_found:
            print(
                "\n:warning: Chunks were retrieved but no response was generated from the model."
            )
        else:
            print("\n:x: No documents were retrieved and no response was generated.")

    except Exception as e:
        print(f"\n:x: An unexpected error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Loading environment variables from .env.dev")
    load_dotenv(dotenv_path=".env.dev")
    main()
