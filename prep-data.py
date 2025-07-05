import os
import shutil # Import shutil for directory operations
import sys # For sys.exit()

# Import all modular components
from src.config_loader import load_config
from src.document_loader import load_documents_from_sources
from src.text_splitter import get_text_splitter
from src.embedding_model import get_ollama_embeddings
from src.vector_store import get_chroma_vector_store, add_documents_to_vector_store

def prepare_data(clear_existing_db: bool = True):
    """
    Prepares the data for the RAG chatbot by loading, chunking, embedding,
    and storing it in ChromaDB.

    Args:
        clear_existing_db (bool): If True, deletes the existing ChromaDB
                                  persistence directory before starting.
                                  Useful for a full data refresh.
    """
    print("\n--- Starting Data Preparation ---")
    try:
        config = load_config()
        ollama_config = config['ollama']
        data_ingestion_config = config['data_ingestion']
        vector_store_config = data_ingestion_config['vector_store']
        chunking_config = data_ingestion_config['chunking']
        app_name = config['app_name']
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please ensure config.yaml is valid and accessible in the root directory.")
        sys.exit(1) # Exit if configuration cannot be loaded

    print(f"Application: {app_name}")
    print(f"Ollama Host: {ollama_config['host']}")
    print(f"LLM Model (for reference): {ollama_config['llm_model']}")
    print(f"Embedding Model: {ollama_config['embedding_model']}")
    print(f"ChromaDB Persistence Directory: {vector_store_config['persist_directory']}")
    print(f"ChromaDB Collection Name: {vector_store_config['collection_name']}")
    print(f"Chunk Size: {chunking_config['chunk_size']}, Chunk Overlap: {chunking_config['chunk_overlap']}")

    # --- Full Refresh Logic ---
    persist_directory = vector_store_config['persist_directory']
    if clear_existing_db:
        if os.path.exists(persist_directory):
            print(f"Clearing existing ChromaDB at: {persist_directory}...")
            try:
                shutil.rmtree(persist_directory)
                print("ChromaDB directory cleared successfully.")
            except Exception as e:
                print(f"Error clearing ChromaDB directory '{persist_directory}': {e}")
                print("Please ensure the directory is not in use by another process.")
                sys.exit(1) # Abort if we can't clear the DB as requested
        else:
            print("ChromaDB directory does not exist. No need to clear.")
    else:
        print("Skipping existing ChromaDB clearing (incremental update assumed).") # Will just append if docs are new

    # 1. Load Documents
    try:
        documents = load_documents_from_sources(data_ingestion_config['document_sources'])
        if not documents:
            print("No documents loaded. Please check 'document_sources' in config.yaml and ensure data paths are correct.")
            print("--- Data Preparation Aborted: No documents to process ---")
            return # Exit if no documents are found
    except Exception as e:
        print(f"Failed to load documents: {e}")
        print("--- Data Preparation Aborted ---")
        sys.exit(1) # Exit if document loading fails

    # 2. Split Documents
    try:
        text_splitter = get_text_splitter(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap']
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Successfully split {len(documents)} documents into {len(chunks)} chunks.")
    except Exception as e:
        print(f"Failed to split documents: {e}")
        print("--- Data Preparation Aborted ---")
        sys.exit(1) # Exit if document splitting fails

    # 3. Initialize Embedding Model
    try:
        embeddings = get_ollama_embeddings(
            model_name=ollama_config['embedding_model'],
            base_url=ollama_config['host']
        )
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        print("Please ensure Ollama is running and the embedding model is pulled.")
        print("--- Data Preparation Aborted ---")
        sys.exit(1) # Exit if embedding model fails

    # 4. Initialize or Load ChromaDB and Add Documents
    try:
        vector_db = get_chroma_vector_store(
            persist_directory=persist_directory,
            collection_name=vector_store_config['collection_name'],
            embedding_function=embeddings
        )
        add_documents_to_vector_store(vector_db, chunks)
    except Exception as e:
        print(f"Failed to interact with vector store: {e}")
        print("--- Data Preparation Aborted ---")
        sys.exit(1) # Exit if vector store interaction fails

    print("--- Data Preparation Complete ---")

if __name__ == "__main__":
    # To perform a full refresh, set clear_existing_db=True (default behavior)
    # To attempt an incremental update (append only), set clear_existing_db=False
    prepare_data(clear_existing_db=True)