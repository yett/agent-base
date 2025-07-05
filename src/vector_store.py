import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings # For type hinting/consistency
from langchain_core.documents import Document
from typing import List, Any

def get_chroma_vector_store(
    persist_directory: str,
    collection_name: str,
    embedding_function: Any # Should be a callable embedding function, e.g., OllamaEmbeddings instance
) -> Chroma:
    """
    Initializes or loads a ChromaDB vector store.

    Args:
        persist_directory (str): The directory where the ChromaDB data will be stored.
        collection_name (str): The name of the collection within ChromaDB.
        embedding_function (Any): The embedding function to use with ChromaDB
                                  (e.g., an instance of OllamaEmbeddings).

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    print(f"Attempting to connect to ChromaDB at: '{persist_directory}' with collection: '{collection_name}'")
    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)

    try:
        # This will load an existing collection or create a new one if it doesn't exist
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
        print("ChromaDB vector store initialized successfully.")
        return vector_store
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        raise # Re-raise the exception to stop execution if vector store is critical

def add_documents_to_vector_store(
    vector_store: Chroma,
    documents: List[Document]
):
    """
    Adds a list of documents to the given ChromaDB vector store.

    Args:
        vector_store (Chroma): The ChromaDB instance.
        documents (List[Document]): A list of LangChain Document objects to add.
    """
    if not documents:
        print("No documents to add to the vector store.")
        return

    print(f"Adding {len(documents)} documents to the vector store...")
    try:
        # Chroma's add_documents will handle embedding the documents using the
        # embedding_function provided during its initialization.
        # It also handles de-duplication if the document IDs are stable.
        vector_store.add_documents(documents)
        vector_store.persist() # Important to save changes to disk
        print(f"Successfully added {len(documents)} documents to vector store.")
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        raise

# Example usage (for testing)
if __name__ == "__main__":
    # This test requires Ollama running and 'nomic-embed-text' pulled
    # and a temporary directory for ChromaDB
    from src.embedding_model import get_ollama_embeddings
    from langchain_core.documents import Document # Ensure Document is imported

    TEST_DB_DIR = "./temp_chroma_db_test"
    TEST_COLLECTION = "test_collection"

    # Clean up previous test run
    if os.path.exists(TEST_DB_DIR):
        import shutil
        shutil.rmtree(TEST_DB_DIR)
        print(f"Cleaned up previous test directory: {TEST_DB_DIR}")

    try:
        embeddings_test = get_ollama_embeddings(model_name="nomic-embed-text")
        
        chroma_test = get_chroma_vector_store(
            persist_directory=TEST_DB_DIR,
            collection_name=TEST_COLLECTION,
            embedding_function=embeddings_test
        )

        dummy_documents = [
            Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "sentence1"}),
            Document(page_content="Artificial intelligence is transforming many industries.", metadata={"source": "ai_info"}),
            Document(page_content="Dogs are known for their loyalty and playful nature.", metadata={"source": "animals"})
        ]

        add_documents_to_vector_store(chroma_test, dummy_documents)
        
        print("\nPerforming a similarity search...")
        query = "What is AI?"
        results = chroma_test.similarity_search(query, k=2)
        print(f"Query: '{query}'")
        for doc in results:
            print(f"  Retrieved: '{doc.page_content[:50]}...' (Source: {doc.metadata.get('source', 'N/A')})")
        
        print(f"\nVector store size after adding: {chroma_test._collection.count()} documents.")

    except Exception as e:
        print(f"Vector store test failed: {e}")
    finally:
        # Clean up after test
        if os.path.exists(TEST_DB_DIR):
            import shutil
            shutil.rmtree(TEST_DB_DIR)
            print(f"Cleaned up test directory: {TEST_DB_DIR}")