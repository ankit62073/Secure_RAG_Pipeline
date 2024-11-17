from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config
from logger.logging import logging

class IngestionPipeline:
    # Initializing the Embedding model
    def __init__(self):
        try:
            logging.info("Initializing FastEmbedEmbeddings...")
            self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
            logging.info("FastEmbedEmbeddings initialized successfully.")

            logging.info("Initializing SemanticChunker...")
            self.semantic_splitter = SemanticChunker(
                self.embeddings, breakpoint_threshold_type="interquartile"
            )
            logging.info("SemanticChunker initialized successfully.")

            logging.info("Initializing RecursiveCharacterTextSplitter...")
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                add_start_index=True
            )
            logging.info("RecursiveCharacterTextSplitter initialized successfully.")
        
        except Exception as e:
            logging.exception("Failed to initialize components in IngestionPipeline: %s", e)

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        documents = []
        for doc_path in doc_paths:
            try:
                logging.info(f"Loading documents from {doc_path}...")
                loaded_documents = PyPDFium2Loader(doc_path).load()
                document_text = "\n".join([doc.page_content for doc in loaded_documents])
                logging.info(f"Loaded {len(loaded_documents)} documents from {doc_path}.")

                logging.info("Chunking documents...")
                chunked_documents = self.recursive_splitter.split_documents(
                    self.semantic_splitter.create_documents([document_text])
                )
                documents.extend(chunked_documents)
                logging.info("Chunking is complete for %s", doc_path)

            except Exception as e:
                logging.exception("Error processing document %s: %s", doc_path, e)
        
        try:
            logging.info("Creating Qdrant vector store...")
            vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                path=Config.Path.DATABASE_DIR,
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
            )
            logging.info("Qdrant vector store created successfully.")
            return vector_store
        
        except Exception as e:
            logging.exception("Failed to create Qdrant vector store: %s", e)
