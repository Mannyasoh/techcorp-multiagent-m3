#!/usr/bin/env python3

import json

# from src.config import Settings, validate_environment
from src.logger import get_logger, setup_logger
from src.multi_agent_system import MultiAgentSystem
from src.utils import (
    log_performance_summary,
    log_system_header,
    validate_and_initialize,
)

# from typing import Dict


setup_logger()
logger = get_logger("main")


def test_intent_classification(system: MultiAgentSystem) -> None:
    logger.info("Testing Intent Classification:")
    logger.debug("=" * 50)

    test_queries = [
        "How many vacation days do I get?",
        "My laptop won't start, can you help?",
        "What's the expense limit for business dinners?",
        "What's the weather like today?",
    ]

    for query in test_queries:
        routing = system.orchestrator.route_query(query)
        logger.info(f"\nQuery: {query}")
        logger.info(f"Intent: {routing['intent']}")
        logger.info(f"Confidence: {routing['confidence']}")
        logger.info(f"Route to: {routing['route_to']}")


def test_specialized_agents(system: MultiAgentSystem) -> None:
    logger.info("\n\nTesting Specialized RAG Agents:")
    logger.debug("=" * 50)

    test_cases = [
        (
            "HR Agent",
            "What health insurance plans does the company offer?",
            system.hr_agent,
        ),
        (
            "Tech Agent",
            "How do I set up VPN access for remote work?",
            system.tech_agent,
        ),
        (
            "Finance Agent",
            "What's the approval process for purchases over $25,000?",
            system.finance_agent,
        ),
    ]

    for agent_name, query, agent in test_cases:
        logger.info(f"\n{agent_name}:")
        logger.info(f"Query: {query}")
        logger.debug("-" * 30)

        response = agent.answer_query(query)
        logger.info(f"Answer: {response['answer'][:200]}...")
        logger.info(f"Sources: {len(response['source_documents'])} documents")


def test_full_workflow(system: MultiAgentSystem) -> None:
    logger.info("\n\nComplete Multi-Agent Workflow:")
    logger.debug("=" * 60)

    demo_queries = [
        "I need to submit a travel expense report. What's the process?",
        "My computer is running slowly. How can I troubleshoot this?",
        "What's the company policy on working remotely?",
    ]

    for query in demo_queries:
        logger.info(f"\nProcessing Query: {query}")
        logger.debug("-" * 40)

        result = system.process_query(query)

        logger.info(
            f"Intent: {result['routing']['intent']} "
            f"(confidence: {result['routing']['confidence']})"
        )
        logger.info(f"Agent: {result['response']['agent']}")
        logger.info(f"Response: {result['response']['answer'][:150]}...")
        logger.info(f"Sources: {len(result['response']['source_documents'])} documents")


def validate_test_queries(system: MultiAgentSystem) -> None:
    logger.info("\n\nValidating Test Queries:")
    logger.debug("=" * 40)

    with open("test_queries.json", "r") as f:
        test_queries = json.load(f)

    correct_classifications = 0
    total_tested = min(5, len(test_queries))

    for test_case in test_queries[:total_tested]:
        query = test_case["query"]
        expected = test_case["expected_intent"]

        routing = system.orchestrator.route_query(query)
        actual = routing["intent"]

        is_correct = actual == expected
        if is_correct:
            correct_classifications += 1

        status = "PASS" if is_correct else "FAIL"
        logger.info(f"{status} Query: {query[:50]}...")
        logger.info(f"    Expected: {expected} | Actual: {actual}")

    accuracy = correct_classifications / total_tested * 100
    logger.info(
        f"\nClassification Accuracy: {accuracy:.1f}% "
        f"({correct_classifications}/{total_tested})"
    )


def test_evaluation_system(system: MultiAgentSystem) -> None:
    logger.info("\n\nResponse Quality Evaluation:")
    logger.debug("=" * 50)

    evaluation_queries = [
        "How much vacation time do I get in my first year?",
        "What should I do if my laptop screen is cracked?",
        "What are the meal limits for business travel?",
    ]

    for query in evaluation_queries:
        logger.info(f"\nEvaluating Query: {query}")
        logger.debug("-" * 30)

        result = system.process_query(query, evaluate=True)

        logger.info(f"Agent: {result['response']['agent']}")
        logger.info(f"Response: {result['response']['answer'][:100]}...")

        if result["evaluation"]:
            eval_data = result["evaluation"]["evaluation"]
            logger.info("Quality Scores:")
            logger.info(f"   - Overall: {eval_data['overall_score']}/10")
            logger.info(f"   - Relevance: {eval_data['relevance_score']}/10")
            logger.info(f"   - Completeness: {eval_data['completeness_score']}/10")
            logger.info(f"   - Accuracy: {eval_data['accuracy_score']}/10")
            logger.info("Scores submitted to Langfuse")


def interactive_demo(system: MultiAgentSystem) -> None:
    logger.info("\n\nInteractive Demo:")
    logger.debug("=" * 30)

    demo_queries = [
        "What software can I install without IT approval?",
        "Can I get reimbursed for a business lunch with a client?",
        "What's the process for reporting a security incident?",
    ]

    for query in demo_queries:
        logger.info(f"\n{query}")
        result = system.process_query(query, evaluate=True)
        logger.info(f"Routed to: {result['routing']['intent']} agent")
        logger.info(f"Response: {result['response']['answer'][:120]}...")

        if result["evaluation"]:
            score = result["evaluation"]["evaluation"]["overall_score"]
            logger.info(f"Quality Score: {score}/10")


def performance_summary(system: MultiAgentSystem) -> None:
    logger.info("\n\nSystem Performance Summary:")
    logger.debug("=" * 40)

    performance_queries = [
        ("How do I apply for parental leave?", "hr"),
        ("My VPN keeps disconnecting. What should I check?", "tech"),
        ("Can I get reimbursed for a business lunch?", "finance"),
        ("What's the process for reporting a security incident?", "tech"),
        ("When are performance reviews conducted?", "hr"),
    ]

    results = {
        "total_queries": len(performance_queries),
        "correct_routing": 0,
        "responses_generated": 0,
        "sources_found": 0,
    }

    for query, expected_intent in performance_queries:
        result = system.process_query(query)

        if result["routing"]["intent"] == expected_intent:
            results["correct_routing"] += 1

        if result["response"]["answer"] and len(result["response"]["answer"]) > 50:
            results["responses_generated"] += 1

        if result["response"]["source_documents"]:
            results["sources_found"] += 1

    results_data = {
        "total_queries": results["total_queries"],
        "correct_routing": results["correct_routing"],
        "responses_generated": results["responses_generated"],
        "sources_found": results["sources_found"],
    }
    log_performance_summary(results_data)

    info = system.get_system_info()
    logger.info("\nSystem Configuration:")
    logger.info(f"   - Vector Stores: {len(info['vector_stores'])} domains")
    logger.info(f"   - Agents: {len(info['agents'])} total")
    logger.info(
        f"   - Langfuse Tracing: "
        f"{'Enabled' if info['langfuse_configured'] else 'Disabled'}"
    )


def main() -> None:
    log_system_header("Multi-Agent System with LangChain and Langfuse")

    system = validate_and_initialize()
    if not system:
        return

    info = system.get_system_info()
    logger.success("System initialized with:")
    logger.info(f"   - Agents: {', '.join(info['agents'])}")
    logger.info(f"   - Vector Stores: {', '.join(info['vector_stores'])}")
    logger.info(f"   - Langfuse configured: {info['langfuse_configured']}")

    # Run all test suites
    test_intent_classification(system)
    test_specialized_agents(system)
    test_full_workflow(system)
    validate_test_queries(system)
    test_evaluation_system(system)
    interactive_demo(system)
    performance_summary(system)

    logger.success(
        "\nAll tests completed! Check Langfuse dashboard for detailed traces."
    )


if __name__ == "__main__":
    main()
