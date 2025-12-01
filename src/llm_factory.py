from langchain_openai import ChatOpenAI

from src.config import Settings


def create_llm(
    api_key: str, temperature: float = 0, model: str | None = None
) -> ChatOpenAI:
    if model is None:
        settings = Settings()
        model = settings.model_name

    return ChatOpenAI(model=model, openai_api_key=api_key, temperature=temperature)
