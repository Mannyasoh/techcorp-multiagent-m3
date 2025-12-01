from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores.faiss import FAISS

# from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langfuse import observe

from src.llm_factory import create_llm


class BaseRAGAgent(ABC):
    def __init__(self, openai_api_key: str, vector_store: FAISS, agent_name: str):
        self.llm = create_llm(openai_api_key, temperature=0.2)
        self.vector_store = vector_store
        self.agent_name = agent_name
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.get_prompt_template()},
        )

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        pass

    def _create_domain_template(
        self, role: str, domain: str, redirect: str, guidance: str
    ) -> PromptTemplate:
        return PromptTemplate(
            template=f"""You are TechCorp's {role}. Use the provided context to "
            f"answer {domain}.

If the question is not related to your domain or you cannot find "
            f"relevant information in the context, politely redirect the user to "
            f"{redirect}.

Context: {{context}}

Question: {{question}}

{guidance}

Answer:""",
            input_variables=["context", "question"],
        )

    @observe()
    def answer_query(self, query: str) -> Dict[str, Any]:
        result = self.qa_chain.invoke({"query": query})

        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown"),
                }
                for doc in result["source_documents"]
            ],
            "agent": self.agent_name,
        }


class HRAgent(BaseRAGAgent):
    def __init__(self, openai_api_key: str, vector_store: FAISS):
        super().__init__(openai_api_key, vector_store, "hr_agent")

    def get_prompt_template(self) -> PromptTemplate:
        return self._create_domain_template(
            role="HR assistant",
            domain="HR-related questions about company policies, benefits, "
            "employment procedures, and workplace guidelines",
            redirect="appropriate department",
            guidance="Provide a helpful, accurate answer based on the "
            "company policies. If specific procedures are mentioned, "
            "include the relevant steps. Be professional and empathetic "
            "in your response.",
        )


class TechAgent(BaseRAGAgent):
    def __init__(self, openai_api_key: str, vector_store: FAISS):
        super().__init__(openai_api_key, vector_store, "tech_agent")

    def get_prompt_template(self) -> PromptTemplate:
        return self._create_domain_template(
            role="IT Support assistant",
            domain="technical questions about software, hardware, IT "
            "policies, system access, security procedures, and troubleshooting",
            redirect="submit an IT ticket or contact the helpdesk",
            guidance="Provide clear, step-by-step technical guidance when "
            "applicable. Include relevant contact information or escalation "
            "procedures when appropriate. Prioritize security and "
            "compliance in your responses.",
        )


class FinanceAgent(BaseRAGAgent):
    def __init__(self, openai_api_key: str, vector_store: FAISS):
        super().__init__(openai_api_key, vector_store, "finance_agent")

    def get_prompt_template(self) -> PromptTemplate:
        return self._create_domain_template(
            role="Finance assistant",
            domain="questions about expense policies, procurement "
            "procedures, budget guidelines, reimbursements, and financial "
            "compliance",
            redirect="contact the finance department directly",
            guidance="Provide accurate information about financial policies "
            "and procedures. Include relevant approval processes, limits, "
            "and contact information. Emphasize compliance requirements "
            "when applicable.",
        )
