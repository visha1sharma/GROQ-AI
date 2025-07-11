import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
 
import os
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
 
# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
# Streamlit UI
st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload a PDF and chat with its content!")
 
# Groq API key input
api_key = st.text_input("Enter your Groq API Key:", type="password")
 
# Chat session state
if 'store' not in st.session_state:
    st.session_state.store = {}
 
# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
 
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
 
    session_id = st.text_input("Session ID:", value="default")
 
    if uploaded_file:
        # Save uploaded file temporarily
        temppdf = './temp.pdf'
        with open(temppdf, "wb") as f:
            f.write(uploaded_file.getvalue())
 
        # Load and split the document
        loader = PyPDFLoader(temppdf)
        documents = loader.load()
 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
 
        # Vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
 
        # Prompts
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do not answer the question. Just reformulate it if needed, or return as is."
        )
 
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
 
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
 
        system_prompt = (
            "You are an assistant for question answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )
 
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
 
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
 
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
 
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
 
        user_input = st.text_input("Your question:")
 
        if user_input:
            session_history = get_session_history(session_id)
 
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
 
            st.success("Assistant: " + response['answer'])
            st.write("Chat History:", session_history.messages)
 
else:
    st.warning("Please enter your Groq API key to start.")