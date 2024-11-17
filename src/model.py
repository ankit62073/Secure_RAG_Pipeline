from langchain_ollama import ChatOllama
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from src.config import Config
from langchain_community.llms import Ollama

def create_llm() -> BaseLanguageModel:
    try:
        llm = ChatOllama(
            model= Config.Model.LOCAL_LLM,
            # base_url="http://localhost:11434/",
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h", 
            max_tokens=Config.Model.MAX_TOKENS
        )
        print("LLM has been created")
        return llm
    except Exception as e:
        print(f"Error creating LLM: {e}")
        return None

def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)

def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)
