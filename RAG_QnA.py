import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# import chromadb
from langchain_community.vectorstores import FAISS
import streamlit as st
# chromadb.api.client.SharedSystemClient.clear_system_cache()

# Set API keys using st.secrets
os.environ['LANGSMITH_API_KEY'] = st.secrets["credentials"]["LANGSMITH_API_KEY"]
os.environ['GROQ_API_KEY'] = st.secrets["credentials"]["GROQ_API_KEY"]
os.environ['LANGSMITH_TRACING_V2'] = "true"
os.environ['LANGSMITH_PROJECT'] = "Q&A using RAG and GROQ"
os.environ['HF_TOKEN'] = st.secrets["credentials"]["HF_TOKEN"]

llm = ChatGroq(model="Gemma2-9b-It")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question according to the context only.
    Please provide the most correct answer according to the question.
    If you Don't know simple say you don't know
    <context>
    {context}
    <context>
    Question:{input}
    """
)

st.title("PDF Q&A using RAG")
uploaded_document=st.file_uploader("Upload your Document")

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        if uploaded_document is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_document.read()) 
                temp_file_path = tmp_file.name
            st.session_state.loader = PyPDFLoader(temp_file_path).load()
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(st.session_state.loader)
            st.session_state.db=FAISS.from_documents(st.session_state.text_splitter,st.session_state.embeddings)



if uploaded_document is not None:
    query=st.text_input(f"Ask your question from {uploaded_document.name}")
    with st.spinner("Processing the Document..."):
        create_vector_embeddings()
        if query:
            with st.spinner("Retrieving Answers..."):
                document_chain=create_stuff_documents_chain(llm,prompt)
                retriever=st.session_state.db.as_retriever()
                retrieval_chain=create_retrieval_chain(retriever,document_chain)
                response=retrieval_chain.invoke({"input":query})
                st.write(response['answer'])
