from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import Qdrant

from src.config import Config
from src.model import create_embeddings, create_reranker
from logger.logging import logging

def create_retriever(llm: BaseLanguageModel, vector_store: Optional[VectorStore] = None) -> VectorStoreRetriever:
    logging.info("Starting retriever creation")

    try:
        if not vector_store:
            logging.info("No vector store provided; creating a new one.")
            vector_store = Qdrant.from_existing_collection(
                embedding=create_embeddings(),
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
                path=Config.Path.DATABASE_DIR,
            )
            logging.info("Vector store created successfully.")

        logging.info("Creating base retriever.")
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        logging.info("Base retriever created.")

        if Config.Retriever.USE_RERANKER:
            logging.info("Using reranker to enhance retriever.")
            retriever = ContextualCompressionRetriever(
                base_compressor=create_reranker(), base_retriever=retriever
            )
            logging.info("Reranker applied successfully.")

        if Config.Retriever.USE_CHAIN_FILTER:
            logging.info("Using chain filter to enhance retriever.")
            retriever = ContextualCompressionRetriever(
                base_compressor=LLMChainFilter.from_llm(llm), base_retriever=retriever
            )
            logging.info("Chain filter applied successfully.")

        logging.info("Retriever creation completed.")
        return retriever

    except Exception as e:
        logging.error("Error during retriever creation: %s", e)
        raise