import uuid
from pathlib import Path
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig

# Your local llama-stack server
LLAMA_STACK_PORT = 8321
INFERENCE_MODEL = "ollama/llama3.2:3b-instruct-fp16"
EMBEDDING_MODEL = "ollama/all-minilm:latest"
EMBEDDING_DIM = 384
# Paths to your local files
local_files = ["matias.txt", "ian.txt"]


def create_http_client():
    return LlamaStackClient(base_url=f"http://localhost:{LLAMA_STACK_PORT}")


client = create_http_client()
# Read your local documents
documents = []
for i, file_path in enumerate(local_files):
    content = Path(file_path).read_text(encoding="utf-8")
    documents.append(
        Document(
            document_id=f"doc-{i}", content=content, mime_type="text/plain", metadata={}
        )
    )
# Register vector DB in llama-stack
vector_db_id = f"matias-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=EMBEDDING_MODEL,
    embedding_dimension=EMBEDDING_DIM,
    provider_id="milvus",
)
print(":inbox_tray: Inserting documents into vector DB...")
client.tool_runtime.rag_tool.insert(
    documents=documents, vector_db_id=vector_db_id, chunk_size_in_tokens=256
)
print(
    f":page_facing_up: Inserted {len(documents)} document(s) with approx {len(documents[0].content.split())} words."
)
# Define RAG agent config
agent_config = AgentConfig(
    model=INFERENCE_MODEL,
    instructions="You are a helpful assistant that uses only the provided documents. When asked a question, first search the documents and then answer based on what you find.",
    enable_session_persistence=False,
    toolgroups=[
        {
            "name": "builtin::rag",
            "args": {
                "vector_db_ids": [vector_db_id],
                "top_k": 3,
                "similarity_threshold": 0.0,  # Remove threshold completely
            },
        }
    ],
)
# Create RAG agent and run
try:
    rag_agent = Agent(client, agent_config)
    session_id = rag_agent.create_session("matias-rag-session")
    print(":white_check_mark: Agent initialized.")
    print(f":books: Using vector DB: {vector_db_id}")
    user_prompt = "Who was Ian Miller?"
    print(f"\n:thinking_face: Query: {user_prompt}")
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": user_prompt}],
        session_id=session_id,
        stream=True,
    )
    print("\n:mag: Retrieved Chunks:")
    final_output = None
    chunks_found = False
    for event in response:
        # Tool call results
        if hasattr(event, "tool_calls") and event.tool_calls:
            print("\n:memo: Tool calls detected:")
            for call in event.tool_calls:
                print(f"  Tool: {call.name}")
                if call.name == "builtin::rag/knowledge_search":
                    results = call.args.get("results", [])
                    print(f"  Number of results: {len(results)}")
                    if results:
                        chunks_found = True
                    for i, doc in enumerate(results, 1):
                        print(f"\n  [{i}] Score: {doc.get('score', 0):.3f}")
                        content_preview = doc.get("content", "")[:200].replace(
                            "\n", " "
                        )
                        print(f"      Content: {content_preview}...")
                if not call.args.get("results"):
                    print(":warning: No documents retrieved by vector search.")
                    print(f"  Tool args: {call.args}")
        # Final model response
        if hasattr(event, "message") and event.message:
            if hasattr(event.message, "content") and event.message.content:
                final_output = event.message.content
                print("\n:speech_balloon: Model response received")
        # Handle streaming text chunks from AgentTurnResponseStreamChunk
        if hasattr(event, "event") and hasattr(event.event, "payload"):
            payload = event.event.payload
            if hasattr(payload, "delta") and hasattr(payload.delta, "text"):
                if final_output is None:
                    final_output = ""
                final_output += payload.delta.text
    if final_output:
        print("\n:brain: RAG Response:")
        print(final_output)
    elif chunks_found:
        print("\n:mag: Chunks were retrieved but no response was generated")
    else:
        print("\n:x: No response received from the model")
except Exception as e:
    print(f"\n:x: Error occurred: {str(e)}")
    import traceback

    traceback.print_exc()
