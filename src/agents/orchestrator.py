from typing import Dict, Literal

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from pydantic import BaseModel

from src.llm_factory import create_llm


class IntentClassification(BaseModel):
    intent: Literal["hr", "tech", "finance", "general"]
    confidence: float
    reasoning: str


class IntentOutputParser(BaseOutputParser):
    def parse(self, text: str) -> IntentClassification:
        return self._parse_classification_text(text)

    def _parse_classification_text(self, text: str) -> IntentClassification:
        try:
            lines = text.strip().split("\n")

            intent = self._extract_field(lines, "intent:", "general")
            confidence = self._extract_confidence(lines)
            reasoning = self._extract_field(
                lines, "reasoning:", "Unable to determine reasoning"
            )

            return IntentClassification(
                intent=intent, confidence=confidence, reasoning=reasoning
            )
        except Exception:
            return IntentClassification(
                intent="general", confidence=0.1, reasoning="Failed to parse response"
            )

    def _extract_field(self, lines: list, prefix: str, default: str) -> str:
        line = next((line for line in lines if line.lower().startswith(prefix)), "")
        return line.split(":", 1)[1].strip().lower() if ":" in line else default

    def _extract_confidence(self, lines: list) -> float:
        confidence_line = next(
            (line for line in lines if line.lower().startswith("confidence:")), ""
        )
        confidence_str = (
            confidence_line.split(":", 1)[1].strip()
            if ":" in confidence_line
            else "0.5"
        )
        try:
            return float(confidence_str)
        except ValueError:
            return 0.5


class OrchestratorAgent:
    def __init__(self, openai_api_key: str):
        self.llm = create_llm(openai_api_key, temperature=0)
        self.parser = IntentOutputParser()
        self.prompt = ChatPromptTemplate.from_template(
            """
You are an intent classification agent for TechCorp's customer support system.
Analyze the user query and classify it into one of these categories:

- hr: Questions about HR policies, benefits, employment, time off, "
"performance reviews, workplace conduct
- tech: Questions about IT support, software, hardware, security "
"policies, technical issues, system access
- finance: Questions about expenses, procurement, budgets, "
"financial policies, reimbursements, purchasing
- general: Questions that don't fit the above categories or are unclear

User Query: {query}

Respond with exactly this format:
Intent: [hr|tech|finance|general]
Confidence: [0.0-1.0]
Reasoning: [brief explanation for your classification]
"""
        )

    @observe(name="orchestrator-classify")
    def classify_intent(self, query: str) -> IntentClassification:
        messages = self.prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        return self.parser.parse(response.content)

    def route_query(self, query: str) -> Dict[str, str]:
        classification = self.classify_intent(query)

        return {
            "intent": classification.intent,
            "confidence": str(classification.confidence),
            "reasoning": classification.reasoning,
            "route_to": f"{classification.intent}_agent"
            if classification.intent != "general"
            else "general_response",
        }
