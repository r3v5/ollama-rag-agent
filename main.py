import traceback
from dotenv import load_dotenv

from app.rag_server_config import RAGServerConfig
from app.document_system import DocumentSystem
from app.rag_agent_manager import RAGAgentManager
from app.vectordb.vector_db_manager import VectorDBManager


def main():
    """Orchestrates a RAG session for a single document."""
    try:
        # --- One-Time Initialization ---
        rag_server_config = RAGServerConfig()
        doc_sys = DocumentSystem(rag_server_config)
        vector_db_manager = VectorDBManager(rag_server_config)
        agent_manager = RAGAgentManager(rag_server_config)
        document = None

        # --- Single Document Input ---
        print("--- RAG Document Loader ---")
        print("Please provide a single document to start a session.")

        file_path = input(
            "Please enter the full path to your file (pdf, png, jpg, txt): "
        ).strip()

        if file_path:
            # The updated load_file method now returns a single document or None
            document = doc_sys.load_file(file_path)
        else:
            print("No path entered. Exiting.")
            return

        # The logic now checks for a single document object
        if not document:
            print("No document was loaded. Exiting.")
            return

        # --- Setup RAG with the single document ---
        # The setup_and_insert method expects a list, so we wrap the document in one.
        vector_db_id = vector_db_manager.setup_and_insert([document])
        # agent_instructions = "You are a helpful assistant that answers questions based only on the document provided. Use the RAG tool."
        agent_instructions = "You are a helpful assistant that answers questions based only on the document provided."
        agent_manager.initialize_agent(vector_db_id, agent_instructions)

        # --- Interactive Q&A Loop ---
        print("\n--- Query Your Document ---")
        while True:
            user_prompt = input("\nAsk a question (or type 'quit' to exit): ").strip()
            if user_prompt.lower() == "quit":
                print("👋 Goodbye!")
                break

            if not user_prompt:
                continue

            final_output, chunks_found = agent_manager.make_query(user_prompt)

            if final_output:
                print("\nRAG Response:")
                print(final_output)
            elif chunks_found:
                print(
                    "\nwarning: Chunks were retrieved but the model did not generate a response."
                )
            else:
                print("\nNo relevant chunks were found in the document for your query.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Loading environment variables from .env.dev")
    load_dotenv(dotenv_path=".env.dev")
    main()
