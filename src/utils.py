from typing import TYPE_CHECKING, Dict, Optional

from .config import Settings, validate_environment
from .logger import get_logger

if TYPE_CHECKING:
    from .multi_agent_system import MultiAgentSystem

logger = get_logger("utils")


def initialize_system() -> Optional["MultiAgentSystem"]:
    try:
        validate_environment()
        settings = Settings()
        from .multi_agent_system import MultiAgentSystem

        logger.info("System initialization completed successfully")
        return MultiAgentSystem(settings)
    except ValueError as e:
        logger.error(f"Environment setup error: {e}")
        return None


def safe_process_query(
    system: "MultiAgentSystem", query: str, evaluate: bool = False
) -> Optional[Dict]:
    try:
        logger.debug(
            f"Processing query: '{query[:50]}{'...' if len(query) > 50 else ''}'"
        )
        result = system.process_query(query, evaluate=evaluate)
        logger.info(
            f"Query processed successfully - Intent: {result['routing']['intent']}"
        )
        return result
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return None


def format_query_result(result: Dict, show_details: bool = True) -> str:
    if not result:
        return "Query failed"

    output = []
    output.append(f"Intent: {result['routing']['intent']}")
    output.append(f"Agent: {result['response']['agent']}")

    if show_details:
        answer = result["response"]["answer"]
        truncated = answer[:100] + "..." if len(answer) > 100 else answer
        output.append(f"Response: {truncated}")

        if result.get("evaluation"):
            if "evaluation" in result["evaluation"]:
                score = result["evaluation"]["evaluation"]["overall_score"]
                output.append(f"Quality: {score}/10")
            elif "error" in result["evaluation"]:
                output.append(f"Evaluation error: {result['evaluation']['error']}")

    return "\n   ".join([""] + output)


def log_system_header(title: str, level: str = "INFO") -> None:
    separator = "=" * len(title)
    logger.info(f"{title}")
    logger.debug(separator)


def log_performance_summary(results: Dict) -> None:
    total = results["total_queries"]
    logger.info("Performance Metrics:")
    logger.info(
        f"   - Routing Accuracy: {results['correct_routing']}/{total} "
        f"({results['correct_routing']/total*100:.1f}%)"
    )
    logger.info(
        f"   - Response Generation: {results['responses_generated']}/{total} "
        f"({results['responses_generated']/total*100:.1f}%)"
    )
    logger.info(
        f"   - Source Retrieval: {results['sources_found']}/{total} "
        f"({results['sources_found']/total*100:.1f}%)"
    )


def validate_and_initialize() -> Optional["MultiAgentSystem"]:
    try:
        validate_environment()
        logger.success("Environment validation passed")

        logger.info("Initializing multi-agent system...")
        settings = Settings()
        from .multi_agent_system import MultiAgentSystem

        system = MultiAgentSystem(settings)
        logger.success("System ready for queries")

        return system
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return None
