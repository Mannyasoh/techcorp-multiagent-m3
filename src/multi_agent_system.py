from typing import Any, Dict

from langfuse import Langfuse, observe

from .agents.evaluator import EvaluatorAgent
from .agents.orchestrator import OrchestratorAgent
from .agents.rag_agent import FinanceAgent, HRAgent, TechAgent
from .config import Settings
from .logger import get_logger
from .vector_store import VectorStoreManager

logger = get_logger("multi_agent_system")


class MultiAgentSystem:
    def __init__(self, settings: Settings, data_dir: str = "data"):
        logger.info("Initializing MultiAgentSystem")
        self.settings = settings

        # Initialize Langfuse
        logger.debug("Setting up Langfuse tracing")
        self.langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

        # Initialize vector stores
        logger.debug("Initializing vector stores")
        self.vector_manager = VectorStoreManager(settings.openai_api_key)
        self.vector_manager.setup_all_stores(data_dir)

        # Initialize agents
        logger.debug("Initializing agents")
        self.orchestrator = OrchestratorAgent(settings.openai_api_key)

        self.hr_agent = HRAgent(
            settings.openai_api_key, self.vector_manager.get_vector_store("hr")
        )

        self.tech_agent = TechAgent(
            settings.openai_api_key, self.vector_manager.get_vector_store("tech")
        )

        self.finance_agent = FinanceAgent(
            settings.openai_api_key, self.vector_manager.get_vector_store("finance")
        )

        # Initialize evaluator
        self.evaluator = EvaluatorAgent(settings.openai_api_key, self.langfuse)

        logger.success("MultiAgentSystem initialization completed")

    @observe(name="multi-agent-query")
    def process_query(self, query: str, evaluate: bool = False) -> Dict[str, Any]:
        logger.info(
            f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}"
        )

        # Step 1: Classify intent
        routing = self.orchestrator.route_query(query)
        logger.debug(
            f"Query routed to: {routing['intent']} (confidence: {routing['confidence']})"
        )

        # Step 2: Route to appropriate agent
        if routing["intent"] == "hr":
            response = self.hr_agent.answer_query(query)
        elif routing["intent"] == "tech":
            response = self.tech_agent.answer_query(query)
        elif routing["intent"] == "finance":
            response = self.finance_agent.answer_query(query)
        else:
            response = {
                "answer": "I'm sorry, I couldn't determine which department "
                "can best help you with that question. Please contact our "
                "general support team or try rephrasing your question with "
                "more specific details about whether it's related to HR, IT, "
                "or Finance.",
                "source_documents": [],
                "agent": "general",
            }

        result = {
            "query": query,
            "routing": routing,
            "response": response,
            "evaluation": None,
        }

        # Step 3: Optional evaluation
        if evaluate and response["agent"] != "general":
            try:
                evaluation = self.evaluator.evaluate_response(
                    query, response["answer"], response["agent"]
                )
                result["evaluation"] = {"evaluation": evaluation.dict()}
            except Exception as e:
                result["evaluation"] = {"error": f"Evaluation failed: {str(e)}"}

        return result

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "agents": [
                "orchestrator",
                "hr_agent",
                "tech_agent",
                "finance_agent",
                "evaluator",
            ],
            "vector_stores": list(self.vector_manager.vector_stores.keys()),
            "langfuse_configured": bool(
                self.settings.langfuse_public_key and self.settings.langfuse_secret_key
            ),
        }
