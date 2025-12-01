from .logger import get_logger

logger = get_logger("src")


def llm_orchestator() -> None:
    logger.info("LLM Orchestrator initialized")
