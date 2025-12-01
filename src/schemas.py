from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    filename: str
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RoutingInfo(BaseModel):
    intent: str
    confidence: float = 0.0
    reasoning: str = ""
    route_to: str = ""


class ResponseInfo(BaseModel):
    agent: str
    answer: str = ""
    source_documents: List[SourceDocument] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    overall_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0


class QueryResult(BaseModel):
    query: str
    routing: Optional[RoutingInfo] = None
    response: Optional[ResponseInfo] = None
    evaluation: Optional[Dict[str, Any]] = None

    @property
    def is_successful(self) -> bool:
        """Check if the query was processed successfully."""
        return (
            self.routing is not None
            and self.response is not None
            and bool(self.response.answer)
        )

    @property
    def intent(self) -> str:
        """Safely get the intent or return 'unknown'."""
        return self.routing.intent if self.routing else "unknown"

    @property
    def confidence(self) -> float:
        """Safely get the confidence or return 0.0."""
        return self.routing.confidence if self.routing else 0.0

    @property
    def agent(self) -> str:
        """Safely get the agent or return 'none'."""
        return self.response.agent if self.response else "none"

    @property
    def answer(self) -> str:
        """Safely get the answer or return empty string."""
        return self.response.answer if self.response else ""

    @property
    def source_documents(self) -> List[SourceDocument]:
        """Safely get source documents or return empty list."""
        return self.response.source_documents if self.response else []

    @property
    def evaluation_score(self) -> float:
        """Safely get evaluation score or return 0.0."""
        if (
            self.evaluation
            and "evaluation" in self.evaluation
            and "overall_score" in self.evaluation["evaluation"]
        ):
            return self.evaluation["evaluation"]["overall_score"]
        return 0.0

    @property
    def has_evaluation(self) -> bool:
        """Check if evaluation data exists."""
        return (
            self.evaluation is not None
            and "evaluation" in self.evaluation
            and self.evaluation["evaluation"] is not None
        )

    def get_evaluation_data(self) -> Optional[Dict[str, Any]]:
        """Safely get evaluation data or return None."""
        if self.has_evaluation and self.evaluation:
            return self.evaluation["evaluation"]
        return None


class EmptyQueryResult(QueryResult):
    """Represents a failed or empty query result."""

    def __init__(self, query: str = "", error_message: str = "Query failed"):
        super().__init__(query=query, routing=None, response=None, evaluation=None)
        self._error_message = error_message

    @property
    def error_message(self) -> str:
        return self._error_message


def create_safe_result(data: Optional[Dict[str, Any]], query: str = "") -> QueryResult:
    """Create a safe QueryResult from potentially None data."""
    if not data:
        return EmptyQueryResult(query, "Query processing failed")

    try:
        # Convert routing data
        routing_data = data.get("routing")
        routing = RoutingInfo(**routing_data) if routing_data else None

        # Convert response data
        response_data = data.get("response")
        response = None
        if response_data:
            # Convert source documents
            source_docs = []
            for doc_data in response_data.get("source_documents", []):
                if isinstance(doc_data, dict):
                    source_docs.append(SourceDocument(**doc_data))

            response = ResponseInfo(
                agent=response_data.get("agent", "unknown"),
                answer=response_data.get("answer", ""),
                source_documents=source_docs,
            )

        return QueryResult(
            query=data.get("query", query),
            routing=routing,
            response=response,
            evaluation=data.get("evaluation"),
        )

    except Exception as e:
        return EmptyQueryResult(query, f"Schema conversion failed: {str(e)}")
