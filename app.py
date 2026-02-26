from src.ingestion import load_all_documents,load_uploaded_documents
from src.vectorStore import FaissVectorStore
from src.search import RAGSearch
import streamlit as st
import uuid

st.set_page_config(page_title="AutoRAG", layout="wide")
st.title("AutoRAG Chatbot")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_files=st.sidebar.file_uploader(
    "Upload your documents (PDF, TXT, CSV, Excel, Word, JSON)", 
    accept_multiple_files=True
    )

if uploaded_files:
    with st.spinner("Processing documents..."):
        docs = load_uploaded_documents(uploaded_files)

        session_folder = f"faiss_store{st.session_state.session_id}"
        vectorstore = FaissVectorStore(persist_dir=session_folder)
        vectorstore.build_from_documents(docs)  
        st.session_state.vectorstore = vectorstore 
        st.success("Documents processed successfully!")
query = st.text_input("Ask a question:")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload documents first !!")
    else:
        search_engine = RAGSearch(st.session_state.vectorstore)
        response = search_engine.search_and_summarize(query)
        st.markdown("### 🤖 Answer")
    
        #st.write(type(response))
        st.write(response)
    

    
   

