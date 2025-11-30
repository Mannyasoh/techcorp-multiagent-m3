from unittest.mock import MagicMock, patch

import pytest

from src.agents.orchestrator import OrchestratorAgent
from src.config import Settings, validate_environment
from src.vector_store import VectorStoreManager

# from langchain_classic.chains.llm import LLMChain


class TestConfiguration:
    def test_settings_creation(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "LANGFUSE_PUBLIC_KEY": "test-public",
                "LANGFUSE_SECRET_KEY": "test-secret",
            },
        ):
            settings = Settings()
            assert settings.openai_api_key == "test-key"
            assert settings.langfuse_public_key == "test-public"
            assert settings.langfuse_secret_key == "test-secret"

    def test_environment_validation_success(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "LANGFUSE_PUBLIC_KEY": "test-public",
                "LANGFUSE_SECRET_KEY": "test-secret",
            },
        ):
            validate_environment()

    def test_environment_validation_failure(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(
                ValueError, match="Missing required environment variables"
            ):
                validate_environment()


class TestVectorStoreManager:
    @patch("src.vector_store.OpenAIEmbeddings")
    def test_initialization(self, mock_embeddings: MagicMock) -> None:
        manager = VectorStoreManager("test-key")
        assert manager.embeddings is not None
        assert manager.text_splitter is not None
        assert manager.vector_stores == {}

    @patch("src.vector_store.OpenAIEmbeddings")
    @patch("builtins.open", create=True)
    @patch("pathlib.Path.glob")
    def test_load_documents(
        self, mock_glob: MagicMock, mock_open: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        mock_file = MagicMock()
        mock_file.read.return_value = "test content"
        mock_open.return_value.__enter__.return_value = mock_file

        mock_path = MagicMock()
        mock_path.name = "test.txt"
        mock_glob.return_value = [mock_path]

        manager = VectorStoreManager("test-key")
        docs = manager.load_documents_from_directory("test_dir")

        assert len(docs) == 1
        assert docs[0].page_content == "test content"
        assert docs[0].metadata["filename"] == "test.txt"


class TestOrchestratorAgent:
    @patch("src.llm_factory.ChatOpenAI")
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "LANGFUSE_PUBLIC_KEY": "test-public",
            "LANGFUSE_SECRET_KEY": "test-secret",
        },
    )
    def test_initialization(self, mock_llm: MagicMock) -> None:
        agent = OrchestratorAgent("test-key")
        assert agent.llm is not None
        assert agent.parser is not None
        assert agent.prompt is not None

    @patch("src.llm_factory.ChatOpenAI")
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "LANGFUSE_PUBLIC_KEY": "test-public",
            "LANGFUSE_SECRET_KEY": "test-secret",
        },
    )
    def test_classify_intent_format(self, mock_llm_class: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.content = (
            "Intent: hr\nConfidence: 0.9\nReasoning: Question about vacation policies"
        )

        # Mock the LLM instance and its invoke method
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm_instance

        agent = OrchestratorAgent("test-key")
        result = agent.classify_intent("How many vacation days do I get?")

        assert result.intent == "hr"
        assert result.confidence == 0.9
        assert "vacation policies" in result.reasoning

    @patch("src.llm_factory.ChatOpenAI")
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "LANGFUSE_PUBLIC_KEY": "test-public",
            "LANGFUSE_SECRET_KEY": "test-secret",
        },
    )
    def test_route_query(self, mock_llm_class: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.content = (
            "Intent: tech\nConfidence: 0.8\nReasoning: IT support question"
        )

        # Mock the LLM instance and its invoke method
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm_instance

        agent = OrchestratorAgent("test-key")
        routing = agent.route_query("My laptop won't start")

        assert routing["intent"] == "tech"
        assert routing["confidence"] == "0.8"
        assert routing["route_to"] == "tech_agent"


class TestSystemIntegration:
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "LANGFUSE_PUBLIC_KEY": "test-public",
            "LANGFUSE_SECRET_KEY": "test-secret",
        },
    )
    @patch("src.multi_agent_system.Langfuse")
    @patch("src.vector_store.FAISS")
    @patch("src.vector_store.OpenAIEmbeddings")
    @patch("src.llm_factory.ChatOpenAI")
    @patch("langchain_classic.chains.retrieval_qa.base.RetrievalQA.from_chain_type")
    def test_system_initialization(
        self,
        mock_retrieval_qa: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
        mock_faiss: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        from src.multi_agent_system import MultiAgentSystem

        # Setup mock vector store with as_retriever method
        mock_faiss.as_retriever.return_value = MagicMock()

        # Mock vector store manager to have the required stores
        with patch(
            "src.vector_store.VectorStoreManager.setup_all_stores"
        ), patch.object(
            VectorStoreManager, "get_vector_store", return_value=mock_faiss
        ):
            system = MultiAgentSystem(Settings())
            assert system.orchestrator is not None
            assert system.hr_agent is not None
            assert system.tech_agent is not None
            assert system.finance_agent is not None
            assert system.evaluator is not None
