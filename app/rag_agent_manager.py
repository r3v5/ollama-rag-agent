from typing import Optional, Tuple
from app.rag_server_config import RAGServerConfig
from app.llama_stack_client_singleton import LlamaStackClientSingleton
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig


class RAGAgentManager:
    """Manages the creation, session, and querying of the RAG agent."""

    def __init__(self, rag_server_config: RAGServerConfig):
        self.llamastack_client = LlamaStackClientSingleton.get_instance()
        self.rag_server_config = rag_server_config
        self.rag_agent = None
        self.session_id = None

    def initialize_agent(self, vector_db_id: str, agent_instructions: str):
        """Creates the RAG agent with a specified vector DB."""

        agent_config = AgentConfig(
            model=self.rag_server_config.get_inference_model(),
            instructions=agent_instructions,
            enable_session_persistence=False,
            toolgroups=[
                {
                    "name": "builtin::rag",
                    "args": {
                        "vector_db_ids": [vector_db_id],
                        "top_k": self.rag_server_config.get_rag_top_k(),
                        "similarity_threshold": 0.0,
                    },
                }
            ],
        )
        self.rag_agent = Agent(self.llamastack_client, agent_config)
        self.session_id = self.rag_agent.create_session("matias-rag-session")
        print(":white_check_mark: Agent initialized.")
        print(f":books: Using vector DB: {vector_db_id}")

    def make_query(self, user_prompt: str) -> Tuple[Optional[str], bool]:
        """Runs a query against the RAG agent and processes the response stream."""

        if not self.rag_agent or not self.session_id:
            raise RuntimeError(
                "Agent has not been initialized. Call initialize_agent() first."
            )

        print(f"\n:thinking_face: Query: {user_prompt}")
        response_stream = self.rag_agent.create_turn(
            messages=[{"role": "user", "content": user_prompt}],
            session_id=self.session_id,
            stream=True,
        )

        final_output = None
        chunks_found = False
        print("\n:mag: Retrieved Chunks:")
        for event in response_stream:
            if hasattr(event, "tool_calls") and event.tool_calls:
                for call in event.tool_calls:
                    if call.name == "builtin::rag/knowledge_search":
                        results = call.args.get("results", [])
                        if results:
                            chunks_found = True
                        for i, doc in enumerate(results, 1):
                            content_preview = doc.get("content", "")[:200].replace(
                                "\n", " "
                            )
                            print(
                                f"  [{i}] Score: {doc.get('score', 0):.3f} | Content: {content_preview}..."
                            )
            if hasattr(event, "event") and hasattr(event.event, "payload"):
                payload = event.event.payload
                if hasattr(payload, "delta") and hasattr(payload.delta, "text"):
                    if final_output is None:
                        final_output = ""
                    final_output += payload.delta.text

        return final_output, chunks_found
