from src.ingestion import load_all_documents,load_uploaded_documents
from src.vectorStore import FaissVectorStore
from src.search import RAGSearch
import streamlit as st

st.set_page_config(page_title="AutoRAG", layout="wide")
st.title("AutoRAG Chatbot")

uploaded_files=st.sidebar.file_uploader(
    "Upload your documents (PDF, TXT, CSV, Excel, Word, JSON)", 
    accept_multiple_files=True
    )

if uploaded_files:
    with st.spinner("Processing documents..."):
        docs = load_uploaded_documents(uploaded_files)

        vectorstore = FaissVectorStore()
        vectorstore.build_from_documents(docs)   
        st.success("Documents processed successfully!")
query = st.text_input("Ask a question:")

if query:
    search_engine = RAGSearch(vectorstore)
    response = search_engine.search_and_summarize(query)

    st.markdown("### 🤖 Answer")
    
    #st.write(type(response))
    st.write(response)

   

