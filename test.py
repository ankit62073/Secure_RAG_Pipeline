# from logger.logging import logging

# logging.info("This is first logging test")

# from src.uploader import upload_files
from src.ingestor import IngestionPipeline
from src.retriever import create_retriever
from src.model import create_llm
from src.uploader import upload_files
from src.chain import ask_question, create_chain
from typing import List


import re
from operator import itemgetter
from typing import List

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever


from src.config import Config
from src.session_history import get_session_history

SYSTEM_PROMPT = """
Utilize the provided contextual information to respond to the user question. If the answer is not found within the context, state that the answer cannot be found. Prioritize concise responsed (maximum of 3 sentences) and use a list where applicable. The contextual information is organized with the most relevant source appearing first. Each source is seperated by a horizontal rule (---).

Context: {context}

Use markdown formatting where appropriate.
"""

if __name__ == "__main__":
    def remove_links(text: str) -> str:
        url_pattern = r"https?://\S+|www\.\S+"
        return re.sub(url_pattern, "", text)


    def format_documents(documents: List[Document]) -> str:
        texts = []
        for doc in documents:
            texts.append(doc.page_content)
            texts.append("----")

        return remove_links("\n".join(texts))

    question = "what is social control"

    def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        chain = (
            RunnablePassthrough.assign(
                context = itemgetter("question")
                | retriever.with_config({"run_name": "context_retriever"})
                | format_documents
            )
            | prompt
            | llm
        )

        print(RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key = "question",
            # history_messages_key = "chat_history",  
        ).with_config({"run_name": "chain_answer"}))

    ingestion_obj = IngestionPipeline()
    ingestion_obj.ingest(["data2/data.pdf"])
    llm = create_llm()
    retriever = create_retriever(llm)
    create_chain(llm,retriever)
    # if chain:
    #     print("Chain has been created")
