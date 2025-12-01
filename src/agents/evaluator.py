# from typing import Dict

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse, observe
from pydantic import BaseModel

from ..llm_factory import create_llm


class EvaluationResult(BaseModel):
    relevance_score: int
    completeness_score: int
    accuracy_score: int
    overall_score: int
    reasoning: str


class EvaluationOutputParser(BaseOutputParser):
    def parse(self, text: str) -> EvaluationResult:
        return self._parse_evaluation_text(text)

    def _parse_evaluation_text(self, text: str) -> EvaluationResult:
        try:
            lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

            scores = self._extract_scores(lines)
            reasoning = self._extract_reasoning(lines)

            return EvaluationResult(**scores, reasoning=reasoning)
        except (ValueError, IndexError) as e:
            return self._default_evaluation(f"Failed to parse evaluation: {str(e)}")

    def _extract_scores(self, lines: list[str]) -> dict[str, int]:
        scores = {
            "relevance_score": 5,
            "completeness_score": 5,
            "accuracy_score": 5,
            "overall_score": 5,
        }

        for line in lines:
            for score_type in ["relevance", "completeness", "accuracy", "overall"]:
                if line.lower().startswith(f"{score_type}:"):
                    try:
                        scores[f"{score_type}_score"] = int(
                            line.split(":", 1)[1].strip()
                        )
                    except ValueError:
                        pass

        return scores

    def _extract_reasoning(self, lines: list[str]) -> str:
        for line in lines:
            if line.lower().startswith("reasoning:"):
                return line.split(":", 1)[1].strip()
        return "No reasoning provided"

    def _default_evaluation(self, reasoning: str) -> EvaluationResult:
        return EvaluationResult(
            relevance_score=5,
            completeness_score=5,
            accuracy_score=5,
            overall_score=5,
            reasoning=reasoning,
        )


class EvaluatorAgent:
    def __init__(self, openai_api_key: str, langfuse_client: Langfuse):
        self.llm = create_llm(openai_api_key, temperature=0)
        self.langfuse = langfuse_client
        self.parser = EvaluationOutputParser()

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an AI response evaluator for TechCorp's customer support system.
Evaluate the quality of the AI agent's response based on the user's original question.

Original Question: {question}
Agent Response: {response}
Agent Type: {agent_type}

Rate the response on a scale of 1-10 for each dimension:

1. Relevance: How well does the response address the specific question asked?
2. Completeness: Does the response provide comprehensive information or guidance?
3. Accuracy: Is the information provided correct and consistent with company policies?

Overall Score: Calculate the average of the three scores above.

Provide your evaluation in this exact format:
Relevance: [1-10]
Completeness: [1-10]
Accuracy: [1-10]
Overall: [1-10]
Reasoning: [Brief explanation of your scoring]
"""
        )

    @observe(name="evaluator-score")
    def evaluate_response(
        self, question: str, response: str, agent_type: str
    ) -> EvaluationResult:
        messages = self.prompt.format_messages(
            question=question, response=response, agent_type=agent_type
        )

        llm_response = self.llm.invoke(messages)
        evaluation = self.parser.parse(llm_response.content)

        return evaluation
