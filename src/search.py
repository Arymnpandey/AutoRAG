import os
from dotenv import load_dotenv
from src.vectorStore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self,vectorstore,data_path="data", llm_model="llama-3.1-8b-instant"):
        self.vectorstore = vectorstore
        self.data_path = data_path
        self.llm_model = llm_model
        # Load or build vectorstore
        faiss_path = os.path.join(vectorstore.persist_dir, "faiss.index")
        meta_path = os.path.join(vectorstore.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from ingestion import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key= os.getenv("groq_api_key")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke(prompt)
        return response.content

