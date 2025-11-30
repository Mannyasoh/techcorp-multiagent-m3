import os
from pathlib import Path
from typing import Dict, List

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .logger import get_logger

logger = get_logger("vector_store")


class VectorStoreManager:
    def __init__(self, openai_api_key: str):
        logger.info("Initializing VectorStoreManager")
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.vector_stores: Dict[str, FAISS] = {}

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        logger.debug(f"Loading documents from directory: {directory_path}")
        documents = []
        directory = Path(directory_path)

        for file_path in directory.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            doc = Document(
                page_content=content,
                metadata={"source": str(file_path), "filename": file_path.name},
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents

    def create_vector_store(self, documents: List[Document], store_name: str) -> FAISS:
        logger.info(
            f"Creating vector store '{store_name}' with {len(documents)} documents"
        )
        chunks = self.text_splitter.split_documents(documents)
        logger.debug(f"Split into {len(chunks)} chunks")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_stores[store_name] = vector_store
        logger.success(f"Vector store '{store_name}' created successfully")
        return vector_store

    def get_vector_store(self, store_name: str) -> FAISS:
        if store_name not in self.vector_stores:
            logger.error(
                f"Vector store '{store_name}' not found. Available stores: {list(self.vector_stores.keys())}"
            )
            raise ValueError(f"Vector store '{store_name}' not found")
        logger.debug(f"Retrieved vector store: {store_name}")
        return self.vector_stores[store_name]

    def setup_all_stores(self, data_dir: str) -> None:
        logger.info(f"Setting up all vector stores from data directory: {data_dir}")
        stores = {"hr": "hr_docs", "tech": "tech_docs", "finance": "finance_docs"}

        for store_name, folder_name in stores.items():
            docs_path = os.path.join(data_dir, folder_name)
            documents = self.load_documents_from_directory(docs_path)
            self.create_vector_store(documents, store_name)

        logger.success(f"All vector stores setup completed: {list(stores.keys())}")
