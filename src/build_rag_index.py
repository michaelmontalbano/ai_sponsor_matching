# build_rag_index.py
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import json
import re

# ---------------------------
# 1. Config
# ---------------------------
DATA_DIR = Path("../data")  # directory with PDFs, TXT, JSON, etc
PAPERS_DIR = DATA_DIR / "papers"
INDEX_SAVE_PATH = Path("../data/llama_index_storage")

# Use sentence-transformers embedding model (free, local)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ---------------------------
# 2. Simple Text Chunking
# ---------------------------
def create_simple_text_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Create simple text chunks using character-based splitting"""
    
    # Clean up text first
    text = text.replace('\n\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            sentence_end = text.rfind('.', start, end + 100)
            if sentence_end > start + chunk_size // 2:  # Don't break too early
                end = sentence_end + 1
            else:
                # Look for space boundaries
                space_pos = text.rfind(' ', start, end)
                if space_pos > start + chunk_size // 2:
                    end = space_pos
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + chunk_size - chunk_overlap, end)
    
    return chunks

# ---------------------------
# 3. Process Documents
# ---------------------------
def process_documents_for_chatbot(documents):
    """Process documents into optimal chunks for chatbot RAG"""
    
    processed_docs = []
    
    print(f"🔄 Processing {len(documents)} documents...")
    
    for doc_id, doc in enumerate(documents):
        print(f"   Processing document {doc_id + 1}/{len(documents)}: {doc.metadata.get('file_name', 'unknown')}")
        
        # Add document metadata
        doc_metadata = {
            'file_name': doc.metadata.get('file_name', 'unknown'),
            'file_type': doc.metadata.get('file_type', 'unknown'),
            'document_id': doc_id,
            'processing_timestamp': str(Path(__file__).stat().st_mtime)
        }
        
        # Split document into chunks
        try:
            chunks = create_simple_text_chunks(doc.text)
        except Exception as e:
            print(f"   ⚠️  Error processing document {doc_id}: {e}")
            print(f"   📄 Skipping problematic document")
            continue
        
        for chunk_id, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
                
            # Create chunk metadata
            chunk_metadata = doc_metadata.copy()
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_size': len(chunk),
                'word_count': len(chunk.split()),
                'chunk_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
            
            # Create new document for each chunk
            from llama_index.core import Document
            chunk_doc = Document(
                text=chunk,
                metadata=chunk_metadata,
                id_=f"doc_{doc_id}_chunk_{chunk_id}"
            )
            
            processed_docs.append(chunk_doc)
    
    return processed_docs

# ---------------------------
# 4. Build RAG Index
# ---------------------------
def build_chatbot_rag_index():
    """Build a general-purpose RAG index for chatbot"""
    
    print("🚀 Building intelligent RAG index for general chatbot...")
    
    # Setup embedding model
    print(f"📥 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    
    # Load documents
    print(f"📁 Scanning for documents in {DATA_DIR}...")
    
    if not DATA_DIR.exists():
        print(f"❌ Data directory {DATA_DIR} does not exist!")
        return None
    
    documents = SimpleDirectoryReader(
        str(DATA_DIR),
        filename_as_id=True,
        recursive=True
    ).load_data()
    
    print(f"📄 Found {len(documents)} documents")
    
    if len(documents) == 0:
        print("❌ No documents found to process!")
        return None
    
    # Process documents into optimal chunks
    print("🧠 Processing documents with intelligent chunking...")
    processed_docs = process_documents_for_chatbot(documents)
    
    print(f"🔍 Created {len(processed_docs)} semantic chunks")
    
    # Build vector index
    print("🏗️  Building vector index...")
    index = VectorStoreIndex(processed_docs)
    
    # Configure and save index
    print(f"💾 Saving index to {INDEX_SAVE_PATH}...")
    INDEX_SAVE_PATH.mkdir(exist_ok=True)
    
    storage_context = StorageContext.from_defaults()
    storage_context.persist(persist_dir=str(INDEX_SAVE_PATH))
    
    # Save index metadata
    index_metadata = {
        'total_documents': len(documents),
        'total_chunks': len(processed_docs),
        'embedding_model': EMBEDDING_MODEL_NAME,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'processing_timestamp': str(Path(__file__).stat().st_mtime),
        'index_version': '3.0',
        'purpose': 'general_chatbot_rag'
    }
    
    metadata_path = INDEX_SAVE_PATH / "index_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(index_metadata, f, indent=2)
    
    print(f"✅ RAG index built successfully!")
    print(f"   📍 Location: {INDEX_SAVE_PATH}")
    print(f"   📊 Documents processed: {len(documents)}")
    print(f"   🔍 Chunks created: {len(processed_docs)}")
    print(f"   🤖 Embedding model: {EMBEDDING_MODEL_NAME}")
    
    return index

# ---------------------------
# 5. Load Existing Index
# ---------------------------
def load_chatbot_index():
    """Load an existing chatbot RAG index"""
    
    if not INDEX_SAVE_PATH.exists():
        print("🔄 No existing index found. Building new index...")
        return build_chatbot_rag_index()
    
    print(f"📂 Loading existing index from {INDEX_SAVE_PATH}...")
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_SAVE_PATH))
        index = load_index_from_storage(storage_context)
        
        # Load and display metadata
        metadata_path = INDEX_SAVE_PATH / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📊 Index loaded: {metadata.get('total_chunks', 'unknown')} chunks, v{metadata.get('index_version', 'unknown')}")
        
        return index
        
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        print("🔄 Building new index...")
        return build_chatbot_rag_index()

# ---------------------------
# 6. Main Function
# ---------------------------
def main():
    """Main function to build or load the RAG index"""
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_SAVE_PATH.mkdir(exist_ok=True)
    
    # Check if force rebuild is requested
    force_rebuild = os.getenv("REBUILD_INDEX", "false").lower() == "true"
    
    if force_rebuild:
        print("🔄 Force rebuild enabled...")
        index = build_chatbot_rag_index()
    else:
        index = load_chatbot_index()
    
    return index

if __name__ == "__main__":
    index = main()
    print(f"🎯 RAG index ready for chatbot queries!")
