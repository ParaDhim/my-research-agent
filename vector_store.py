import os
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
FAISS_INDEX_PATH = "faiss_index_scientific_papers"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Use Streamlit's cache to load the model and vector store only once
@st.cache_resource
def get_vector_store():
    """Creates or loads the FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if os.path.exists(FAISS_INDEX_PATH):
        st.info("Loading existing FAISS vector store...")
        # Allow dangerous deserialization as we are sure of the source
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        st.success("Vector store loaded successfully.")
    else:
        st.info("Creating new FAISS vector store. This may take a few minutes...")
        
        # Load a subset of the dataset
        with st.spinner("Downloading dataset..."):
            full_dataset = load_dataset("franz96521/scientific_papers", split='train', streaming=True)
            subset_dataset_iterable = full_dataset.take(100) # Using 100 papers for the demo
            papers_data = list(subset_dataset_iterable)
        
        # Chunk the documents
        with st.spinner("Chunking documents..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            all_chunks = []
            for paper in papers_data:
                chunks = text_splitter.split_text(paper['full_text'])
                for chunk in chunks:
                    doc = Document(page_content=chunk, metadata={"paper_id": paper['id']})
                    all_chunks.append(doc)
        
        # Create and save the vector store
        with st.spinner(f"Embedding {len(all_chunks)} chunks and building index..."):
            vector_store = FAISS.from_documents(all_chunks, embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
        
        st.success("Vector store created and saved.")
    
    return vector_store