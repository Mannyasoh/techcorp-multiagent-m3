#!/usr/bin/env python3

from src.logger import get_logger, setup_logger
from src.utils import (
    format_query_result,
    log_system_header,
    safe_process_query,
    validate_and_initialize,
)

setup_logger()
logger = get_logger("demo")


def quick_demo() -> None:
    log_system_header("Multi-Agent System Quick Demo")

    system = validate_and_initialize()
    if not system:
        return

    demo_queries = [
        "How many sick days do I get per year?",
        "My password expired, how do I reset it?",
        "What's the limit for client dinner expenses?",
        "Can I work from home permanently?",
    ]

    for i, query in enumerate(demo_queries, 1):
        logger.info(f"\n{i}. Query: {query}")
        result = safe_process_query(system, query, evaluate=True)
        if result:
            logger.info(format_query_result(result, show_details=False))
        else:
            logger.warning(f"Query {i} failed to process")


def interactive_mode() -> None:
    log_system_header("Interactive Multi-Agent Assistant")
    logger.info("Ask questions about HR, IT, or Finance policies")
    logger.info("Type 'quit' to exit")

    system = validate_and_initialize()
    if not system:
        return

    while True:
        try:
            query = input("\nYour question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                logger.info("Goodbye!")
                break

            if not query:
                continue

            result = safe_process_query(system, query, evaluate=True)
            if result:
                logger.info(f"\nDepartment: {result['routing']['intent'].upper()}")
                logger.info(f"Answer: {result['response']['answer']}")

                if result.get("evaluation", {}).get("evaluation"):
                    score = result["evaluation"]["evaluation"]["overall_score"]
                    logger.info(f"Quality Score: {score}/10")

                logger.info("\n" + "-" * 50)

        except KeyboardInterrupt:
            logger.info("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Interactive session error: {e}")


def test_single_query(query: str) -> None:
    log_system_header(f"Testing: {query}")

    system = validate_and_initialize()
    if not system:
        return

    result = safe_process_query(system, query, evaluate=True)
    if not result:
        return

    logger.info(f"Intent: {result['routing']['intent']}")
    logger.info(f"Confidence: {result['routing']['confidence']}")
    logger.info(f"Agent: {result['response']['agent']}")
    logger.info(f"Answer: {result['response']['answer']}")

    if result.get("evaluation", {}).get("evaluation"):
        eval_data = result["evaluation"]["evaluation"]
        logger.info("\nQuality Evaluation:")
        logger.info(f"  Overall: {eval_data['overall_score']}/10")
        logger.info(f"  Relevance: {eval_data['relevance_score']}/10")
        logger.info(f"  Completeness: {eval_data['completeness_score']}/10")
        logger.info(f"  Accuracy: {eval_data['accuracy_score']}/10")

    logger.info(f"\nSources ({len(result['response']['source_documents'])}):")
    for doc in result["response"]["source_documents"]:
        logger.info(f"  - {doc['filename']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            test_single_query(" ".join(sys.argv[1:]))
    else:
        quick_demo()
