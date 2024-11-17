import asyncio
import random
import streamlit as st
from dotenv import load_dotenv
import os 
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

from src.chain import ask_question, create_chain
from src.config import Config
from src.ingestor import IngestionPipeline
from src.model import create_llm
from src.retriever import create_retriever
from src.uploader import upload_files
import time 

import requests
import sqlite3

from src.advanced_query_classifier import AdvancedQueryClassifier

load_dotenv()


def run_demo(query):
    # Initialize classifier
    classifier = AdvancedQueryClassifier()
    
    # query1 = "what is the salary of virat kohli"
    type1, conf1, evidence1 = classifier.classify_with_context(query)

    return type1

@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
    vector_store = IngestionPipeline().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)

async def stream_response(text: str, message_placeholder):
    """Helper function to stream text word by word"""
    full_response = ""
    # Split text into words and add spaces back
    words = text.split(' ')
    for i, word in enumerate(words):
        full_response += word
        if i < len(words) - 1:  # Don't add space after last word
            full_response += " "
        message_placeholder.markdown(full_response + "▌")
        await asyncio.sleep(0.05)  # Adjust speed of typing
    return full_response

async def ask_chain(question: str, chain=None):
    case = run_demo(question)
    print(case)
    if case == 'RAG':
        st.markdown(case)
        full_response = ""
        assistant = st.chat_message(
            "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.webp")
        )
        with assistant:
            message_placeholder = st.empty()
            documents = []
            
            if chain:
                # Use RAG if chain exists (PDF uploaded)
                async for event in ask_question(chain, question, session_id="session-id-42"):
                    if isinstance(event, str):
                        full_response += event
                        message_placeholder.markdown(full_response)
                    if isinstance(event, list):
                        documents.extend(event)
            else:
                # Use direct LLM if no PDFs uploaded
                llm = create_llm()
                try:
                    # Create a proper ChatOllama message
                    messages = [HumanMessage(content=question)]
                    response = await llm.agenerate([messages])
                    # Stream the complete response
                    full_response = await stream_response(
                        response.generations[0][0].text,
                        message_placeholder
                    )
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    full_response = "I apologize, but I encountered an error processing your request."
                    message_placeholder.markdown(full_response)

            # Remove the cursor after streaming is complete
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        st.markdown(case)
        
        OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"  # Default Ollama endpoint

        def get_ollama_response(question, prompt):
            """Function to get response from Ollama model"""
            try:
                # Combine prompt and question
                full_prompt = f"{prompt[0]}\n\nQuestion: {question}\nSQL Query:"
                
                # Prepare the request payload
                payload = {
                    "model": "llama3.2:3b",  # Using LLaMA2 3B model
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": 0.7
                }
                
                # Make request to Ollama API
                response = requests.post(OLLAMA_ENDPOINT, json=payload)
                response.raise_for_status()  # Raise exception for bad status codes
                
                # Parse response
                response_data = response.json()
                sql_query = response_data['response'].strip()
                
                # Basic validation to ensure it's a SQL query
                if not any(keyword in sql_query.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    raise ValueError("Generated response is not a valid SQL query")
                    
                return sql_query
            
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to Ollama API: {str(e)}")
                return None
            except Exception as e:
                st.error(f"Error generating SQL query: {str(e)}")
                return None

        def read_sql_query(sql, db):
            """Function to retrieve query from the database"""
            try:
                conn = sqlite3.connect(db)
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                conn.commit()
                conn.close()
                return rows
            except sqlite3.Error as e:
                st.error(f"Database error: {str(e)}")
                return []
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                return []

        # Define Prompt
        prompt = [
            """
            You are an expert in converting English questions to SQL query! The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION

            For example, 
            Example 1 - How many entries of records are present?, the SQL command will be something like this SELECT COUNT(*) from STUDENT;
            Example 2 - Tell me all the students studying in Data Science class?, the SQL command will be something like this SELECT * from STUDENT where CLASS="Data Science";
            
            Return only the SQL query without any additional text or formatting.
            """
        ]

      
        response = requests.get("http://localhost:11434/")
            
        if question:
            with st.spinner("Generating SQL query..."):
                response = get_ollama_response(question, prompt)
                    
                if response:
                    st.code(response, language="sql")
                        
                    with st.spinner("Executing query..."):
                        result = read_sql_query(response, "student.db")
                            
                        if result:
                            st.subheader("Query Results:")
                            for row in result:
                                st.write(row)
                        else:
                            st.info("No results found for this query.")
           

def show_upload_documents():
    with st.sidebar:
        # Display Stratlytics logo
        # st.image("images/logo.png", width=220)

        st.header("RagBase")
        st.subheader("Get answers from your documents")

        uploaded_files = st.file_uploader(
            label="Upload PDF files (optional)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Analyzing your document(s)..."):
                chain = build_qa_chain(uploaded_files)
                st.success(f"✅ {len(uploaded_files)} document(s) successfully processed! You can now ask questions about your documents.")
        
        return chain if uploaded_files else None


def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.webp"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.jpeg"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])

def run_async(coro):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

def show_chat_input(chain):
    if prompt := st.chat_input("Ask any question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.jpeg")
        ):
            st.markdown(prompt)
        run_async(ask_chain(prompt, chain))

# Page setup
st.set_page_config(
    page_title="RagBase", 
    page_icon="",
    layout="wide"
)

# Apply custom CSS for button and logo styling
st.markdown(
    """
    <style>
        /* Avatar container styling */
        .st-emotion-cache-p4micv {
            width: 2.75rem !important;
            height: 2.75rem !important;
            border-radius: 25%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Avatar image styling */
        .st-emotion-cache-p4micv img {
            width: 100% !important;
            height: 100% !important;
            object-fit: cover !important;
            border-radius: 50%;
        }
        
        
        /* Other existing styles */
        button { 
            background-color: #007BFF; 
            color: white; 
            border: none; 
            border-radius: 50%; 
        }
        .main .block-container {
            padding-top: 2rem;
            max-width: 800px;
            margin: 0 auto;
        }

        /* Expander styling */
        .streamlit-expander {
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .streamlit-expander .streamlit-expanderHeader {
            background-color: #f8f9fa;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize message history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ready to assist you.",
        }
    ]

# Main content area title
st.title("Secure Chatbot")

# Add position selection dropdown
position = st.selectbox(
    "Select Position",
    ["Junior", "Senior"],
    key="position_selector"
)

# Display upload interface, message history, and chat input
chain = show_upload_documents()
show_message_history()
show_chat_input(chain)