#!/usr/bin/env python3

import json
from typing import Any, Dict, cast

from src.logger import get_logger
from src.multi_agent_system import MultiAgentSystem
from src.utils import safe_process_query, validate_and_initialize

logger = get_logger("evaluate")


def run_evaluation_suite(system: MultiAgentSystem) -> Dict[str, Any]:
    with open("test_queries.json", "r") as f:
        test_queries = json.load(f)

    results = {
        "total_queries": len(test_queries),
        "correct_classifications": 0,
        "successful_responses": 0,
        "quality_scores": [],
        "by_domain": {
            "hr": {"total": 0, "correct": 0},
            "tech": {"total": 0, "correct": 0},
            "finance": {"total": 0, "correct": 0},
            "general": {"total": 0, "correct": 0},
        },
    }

    logger.info("Running comprehensive evaluation...")
    logger.info("=" * 50)

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_intent = test_case["expected_intent"]

        logger.info(f"{i:2d}. {query[:60]}...")

        result = safe_process_query(system, query, evaluate=True)
        if result.is_successful:
            actual_intent = result.intent

            # Track domain statistics
            domain_stats = cast(
                Dict[str, int], results["by_domain"][expected_intent]
            )  # type: ignore
            domain_stats["total"] = domain_stats["total"] + 1  # type: ignore

            # Check classification accuracy
            is_correct = actual_intent == expected_intent
            if is_correct:
                results["correct_classifications"] = (
                    cast(int, results["correct_classifications"]) + 1
                )
                domain_stats["correct"] = domain_stats["correct"] + 1

            # Check response quality
            has_response = bool(result.answer and len(result.answer) > 20)
            if has_response:
                results["successful_responses"] = (
                    cast(int, results["successful_responses"]) + 1
                )

            # Collect quality scores
            if result.has_evaluation:
                score = result.evaluation_score
                quality_scores = cast(list, results["quality_scores"])
                quality_scores.append(score)

            status = "PASS" if is_correct else "FAIL"
            logger.info(
                f"    {status} Expected: {expected_intent} | Got: {actual_intent}"
            )

            if result.has_evaluation:
                score = result.evaluation_score
                logger.info(f"    Quality: {score}/10")
        else:
            logger.warning("    Query failed")

    return results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    total = results["total_queries"]

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    # Overall metrics
    classification_accuracy = results["correct_classifications"] / total * 100
    response_success_rate = results["successful_responses"] / total * 100

    logger.info("Overall Performance:")
    logger.info(
        f"   Classification Accuracy: "
        f"{results['correct_classifications']}/{total} "
        f"({classification_accuracy:.1f}%)"
    )
    logger.info(
        f"   Response Success Rate: "
        f"{results['successful_responses']}/{total} "
        f"({response_success_rate:.1f}%)"
    )

    # Quality scores
    quality_scores = results["quality_scores"]
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)

        logger.info(f"   Average Quality Score: {avg_quality:.1f}/10")
        logger.info(f"   Quality Range: {min_quality}-{max_quality}/10")

    # Per-domain breakdown
    logger.info("\nPer-Domain Performance:")
    by_domain = results["by_domain"]
    for domain, stats in by_domain.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            logger.info(
                f"   {domain.upper()}: "
                f"{stats['correct']}/{stats['total']} "
                f"({accuracy:.1f}%)"
            )

    # Performance grading
    logger.info("\nPerformance Grade:")
    if classification_accuracy >= 90:
        grade = "A (Excellent)"
    elif classification_accuracy >= 80:
        grade = "B (Good)"
    elif classification_accuracy >= 70:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Improvement)"

    logger.info(f"   Overall Grade: {grade}")


def benchmark_performance(system: MultiAgentSystem) -> None:
    logger.info("Performance Benchmarking")
    logger.info("=" * 40)

    import time

    benchmark_queries = [
        "How many vacation days do I get?",
        "My laptop won't connect to WiFi",
        "What's the expense limit for meals?",
        "Can I install Slack on my computer?",
        "What's the parental leave policy?",
    ]

    total_time: float = 0
    successful_queries: int = 0

    for query in benchmark_queries:
        start_time = time.time()

        result = safe_process_query(system, query)
        end_time = time.time()

        query_time = end_time - start_time
        total_time += query_time

        if result.is_successful and result.answer:
            successful_queries += 1
            logger.info(f"PASS {query[:40]}... ({query_time:.2f}s)")
        else:
            logger.warning(f"FAIL {query[:40]}... ({query_time:.2f}s) - Failed")

    avg_time = total_time / len(benchmark_queries)
    success_rate = successful_queries / len(benchmark_queries) * 100

    logger.info("\nPerformance Results:")
    logger.info(f"   Average Response Time: {avg_time:.2f} seconds")
    logger.info(f"   Success Rate: {success_rate:.1f}%")
    logger.info(f"   Total Time: {total_time:.2f} seconds")


def stress_test(system: MultiAgentSystem) -> None:
    logger.info("\nStress Test (10 rapid queries)")
    logger.info("=" * 40)

    stress_queries = [
        "How do I reset my password?",
        "What's the vacation policy?",
        "Can I expense this meal?",
        "My computer is slow",
        "When are reviews done?",
        "How do I submit expenses?",
        "What software is approved?",
        "How much sick leave do I have?",
        "Can I work remotely?",
        "What's the IT helpdesk number?",
    ]

    import time

    start_time = time.time()

    successful: int = 0
    for i, query in enumerate(stress_queries, 1):
        result = safe_process_query(system, query)
        if result.is_successful and result.answer:
            successful += 1
            logger.info(f"PASS Query {i:2d}: {result.intent}")
        else:
            logger.warning(f"FAIL Query {i:2d}: Failed")

    end_time = time.time()
    total_time = end_time - start_time

    logger.info("\nStress Test Results:")
    logger.info(f"   Successful Queries: {successful}/{len(stress_queries)}")
    logger.info(f"   Total Time: {total_time:.2f} seconds")
    logger.info(
        f"   Average Time per Query: {total_time/len(stress_queries):.2f} seconds"
    )


def print_system_header(title: str, width: int = 50) -> None:
    separator = "=" * width
    logger.info(f"{title}")
    logger.info(separator)


def main() -> None:
    print_system_header("Multi-Agent System Evaluation", 50)

    system = validate_and_initialize()
    if not system:
        return

    # Run comprehensive evaluation
    results = run_evaluation_suite(system)
    print_evaluation_summary(results)

    # Run performance benchmarks
    benchmark_performance(system)

    # Run stress test
    stress_test(system)

    logger.success("\nEvaluation complete! Check Langfuse for detailed analytics.")


if __name__ == "__main__":
    main()
