#!/usr/bin/env python3

import json
from typing import Dict

from src.multi_agent_system import MultiAgentSystem
from src.utils import safe_process_query, validate_and_initialize


def run_evaluation_suite(system: MultiAgentSystem) -> Dict[str, float]:
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

    print("Running comprehensive evaluation...")
    print("=" * 50)

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_intent = test_case["expected_intent"]

        print(f"{i:2d}. {query[:60]}...")

        result = safe_process_query(system, query, evaluate=True)
        if result:
            actual_intent = result["routing"]["intent"]

            # Track domain statistics
            results["by_domain"][expected_intent]["total"] += 1

            # Check classification accuracy
            is_correct = actual_intent == expected_intent
            if is_correct:
                results["correct_classifications"] += 1
                results["by_domain"][expected_intent]["correct"] += 1

            # Check response quality
            has_response = bool(
                result["response"]["answer"] and len(result["response"]["answer"]) > 20
            )
            if has_response:
                results["successful_responses"] += 1

            # Collect quality scores
            if result["evaluation"]:
                score = result["evaluation"]["evaluation"]["overall_score"]
                results["quality_scores"].append(score)

            status = "PASS" if is_correct else "FAIL"
            print(f"    {status} Expected: {expected_intent} | Got: {actual_intent}")

            if result.get("evaluation", {}).get("evaluation"):
                score = result["evaluation"]["evaluation"]["overall_score"]
                print(f"    Quality: {score}/10")
        else:
            print("    Query failed")

    return results


def print_evaluation_summary(results: Dict[str, float]) -> None:
    total = results["total_queries"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Overall metrics
    classification_accuracy = results["correct_classifications"] / total * 100
    response_success_rate = results["successful_responses"] / total * 100

    print("Overall Performance:")
    print(
        f"   Classification Accuracy: "
        f"{results['correct_classifications']}/{total} "
        f"({classification_accuracy:.1f}%)"
    )
    print(
        f"   Response Success Rate: "
        f"{results['successful_responses']}/{total} "
        f"({response_success_rate:.1f}%)"
    )

    # Quality scores
    if results["quality_scores"]:
        avg_quality = sum(results["quality_scores"]) / len(results["quality_scores"])
        min_quality = min(results["quality_scores"])
        max_quality = max(results["quality_scores"])

        print(f"   Average Quality Score: {avg_quality:.1f}/10")
        print(f"   Quality Range: {min_quality}-{max_quality}/10")

    # Per-domain breakdown
    print("\nPer-Domain Performance:")
    for domain, stats in results["by_domain"].items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            print(
                f"   {domain.upper()}: "
                f"{stats['correct']}/{stats['total']} "
                f"({accuracy:.1f}%)"
            )

    # Performance grading
    print("\nPerformance Grade:")
    if classification_accuracy >= 90:
        grade = "A (Excellent)"
    elif classification_accuracy >= 80:
        grade = "B (Good)"
    elif classification_accuracy >= 70:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Improvement)"

    print(f"   Overall Grade: {grade}")


def benchmark_performance(system: MultiAgentSystem) -> None:
    print("Performance Benchmarking")
    print("=" * 40)

    import time

    benchmark_queries = [
        "How many vacation days do I get?",
        "My laptop won't connect to WiFi",
        "What's the expense limit for meals?",
        "Can I install Slack on my computer?",
        "What's the parental leave policy?",
    ]

    total_time = 0
    successful_queries = 0

    for query in benchmark_queries:
        start_time = time.time()

        result = safe_process_query(system, query)
        end_time = time.time()

        query_time = end_time - start_time
        total_time += query_time

        if result and result["response"]["answer"]:
            successful_queries += 1
            print(f"PASS {query[:40]}... ({query_time:.2f}s)")
        else:
            print(f"FAIL {query[:40]}... ({query_time:.2f}s) - Failed")

    avg_time = total_time / len(benchmark_queries)
    success_rate = successful_queries / len(benchmark_queries) * 100

    print("\nPerformance Results:")
    print(f"   Average Response Time: {avg_time:.2f} seconds")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Total Time: {total_time:.2f} seconds")


def stress_test(system: MultiAgentSystem) -> None:
    print("\nStress Test (10 rapid queries)")
    print("=" * 40)

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

    successful = 0
    for i, query in enumerate(stress_queries, 1):
        result = safe_process_query(system, query)
        if result and result["response"]["answer"]:
            successful += 1
            print(f"PASS Query {i:2d}: {result['routing']['intent']}")
        else:
            print(f"FAIL Query {i:2d}: Failed")

    end_time = time.time()
    total_time = end_time - start_time

    print("\nStress Test Results:")
    print(f"   Successful Queries: {successful}/{len(stress_queries)}")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Average Time per Query: {total_time/len(stress_queries):.2f} seconds")


def print_system_header(title: str, width: int = 50) -> None:
    separator = "=" * width
    print(f"{title}")
    print(separator)


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

    print("\nEvaluation complete! Check Langfuse for detailed analytics.")


if __name__ == "__main__":
    main()
