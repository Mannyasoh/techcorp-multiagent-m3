import os

from pydantic_settings import BaseSettings

from .logger import get_logger

logger = get_logger("config")


class Settings(BaseSettings):
    openai_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"
    model_name: str = "gpt-3.5-turbo"


def validate_environment() -> None:
    logger.debug("Validating environment variables")
    required_vars = ["OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    logger.success("Environment validation completed successfully")
    # MODEL_NAME is optional with default value
