from langchain_community.embeddings import OllamaEmbeddings
from typing import List

def get_ollama_embeddings(model_name: str, base_url: str = "http://localhost:11434") -> OllamaEmbeddings:
    """
    Initializes and returns an OllamaEmbeddings instance.

    Args:
        model_name (str): The name of the embedding model to use (e.g., 'nomic-embed-text').
        base_url (str): The URL of the Ollama server.

    Returns:
        OllamaEmbeddings: An instance of the Ollama embedding model.
    """
    print(f"Initializing OllamaEmbeddings with model: '{model_name}' at '{base_url}'")
    try:
        embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        # Test a small embedding to ensure connectivity
        _ = embeddings.embed_query("test embedding connection")
        print("OllamaEmbeddings initialized and connected successfully.")
        return embeddings
    except Exception as e:
        print(f"Error initializing OllamaEmbeddings: {e}")
        print("Please ensure Ollama is running and the specified embedding model is pulled.")
        raise # Re-raise the exception to stop execution if embeddings are critical

# Example usage (for testing)
if __name__ == "__main__":
    # Ensure Ollama is running and 'nomic-embed-text' is pulled: ollama pull nomic-embed-text
    try:
        embeddings_instance = get_ollama_embeddings(model_name="nomic-embed-text")
        
        test_text_single = "What is the capital of France?"
        embedding_single = embeddings_instance.embed_query(test_text_single)
        print(f"\nEmbedding for '{test_text_single}' (first 5 values): {embedding_single[:5]}...")
        print(f"Embedding dimension: {len(embedding_single)}")

        test_texts_multiple = ["Hello world", "Artificial Intelligence"]
        embeddings_multiple = embeddings_instance.embed_documents(test_texts_multiple)
        print(f"\nEmbeddings for multiple texts (first 5 values of first embedding): {embeddings_multiple[0][:5]}...")
        print(f"Number of embeddings for multiple texts: {len(embeddings_multiple)}")

    except Exception as e:
        print(f"Embedding model test failed: {e}")