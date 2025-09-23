import os
import streamlit as st
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
FAISS_INDEX_PATH = "faiss_index_scientific_papers"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def get_embeddings():
    """Get and cache the embedding model"""
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name, 
        show_progress=False  # Disable progress bar in production
    )

def load_papers_dataset(num_papers=100):
    """Load and process the scientific papers dataset"""
    try:
        full_dataset = load_dataset("franz96521/scientific_papers", split='train', streaming=True)
        subset_dataset_iterable = full_dataset.take(num_papers)
        papers_data = list(subset_dataset_iterable)
        return papers_data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []

def create_document_chunks(papers_data):
    """Create document chunks from papers data"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    
    for i, paper in enumerate(papers_data):
        try:
            # Get the full text, handle missing data gracefully
            full_text = paper.get('full_text', '')
            if not full_text:
                continue
                
            # Split text into chunks
            chunks = text_splitter.split_text(full_text)
            
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only keep substantial chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "paper_id": paper.get('id', f'paper_{i}'),
                            "chunk_id": f'chunk_{i}_{j}',
                            "title": paper.get('title', 'Unknown Title')[:100]
                        }
                    )
                    all_chunks.append(doc)
                    
        except Exception as e:
            st.warning(f"Error processing paper {i}: {e}")
            continue
    
    return all_chunks

@st.cache_resource
def get_vector_store():
    """Creates or loads the FAISS vector store with caching."""
    embeddings = get_embeddings()
    
    # Try to load existing vector store
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            st.info("📂 Loading existing FAISS vector store...")
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.success("✅ Vector store loaded successfully!")
            return vector_store
            
        except Exception as e:
            st.warning(f"⚠️ Error loading existing vector store: {e}")
            st.info("🔄 Creating new vector store...")
    
    # Create new vector store
    try:
        progress_bar = st.progress(0, "🚀 Creating new FAISS vector store...")
        
        # Step 1: Load dataset
        progress_bar.progress(20, "📚 Loading scientific papers dataset...")
        papers_data = load_papers_dataset(num_papers=100)
        
        if not papers_data:
            st.error("❌ No papers loaded from dataset")
            return None
        
        # Step 2: Create chunks
        progress_bar.progress(40, "📄 Processing and chunking documents...")
        all_chunks = create_document_chunks(papers_data)
        
        if not all_chunks:
            st.error("❌ No document chunks created")
            return None
        
        st.info(f"📊 Created {len(all_chunks)} document chunks from {len(papers_data)} papers")
        
        # Step 3: Create embeddings and vector store
        progress_bar.progress(70, f"🧠 Creating embeddings for {len(all_chunks)} chunks...")
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        
        # Step 4: Save vector store
        progress_bar.progress(90, "💾 Saving vector store...")
        try:
            vector_store.save_local(FAISS_INDEX_PATH)
            st.success("✅ Vector store created and saved successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not save vector store: {e}")
        
        progress_bar.progress(100, "✅ Vector store ready!")
        progress_bar.empty()
        
        return vector_store
        
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        return None

def get_vector_store_info(vector_store):
    """Get information about the vector store"""
    if vector_store is None:
        return "Vector store not initialized"
    
    try:
        # Get the number of documents
        doc_count = vector_store.index.ntotal
        return f"Vector store contains {doc_count} document embeddings"
    except:
        return "Vector store information unavailable"