from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter.

    Args:
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        RecursiveCharacterTextSplitter: An instance of the text splitter.
    """
    print(f"Initializing RecursiveCharacterTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Default separators are usually good: ["\n\n", "\n", " ", ""]
        # You can customize them if your documents have specific structural breaks
    )
    return text_splitter

# Example usage (for testing)
if __name__ == "__main__":
    sample_text = """
    This is a long piece of text that needs to be split into smaller chunks.
    It contains multiple sentences and might span across paragraphs.
    The goal is to ensure that related information stays together within a chunk,
    while also providing some overlap between chunks to maintain context during retrieval.
    This helps the LLM generate more coherent and relevant responses.
    """
    
    # Create a dummy Document object for testing
    sample_document = Document(page_content=sample_text, metadata={"source": "test_text"})

    splitter = get_text_splitter(chunk_size=100, chunk_overlap=20)
    
    # We can split raw text or a list of Document objects
    # For Document objects, use split_documents
    chunks: List[Document] = splitter.split_documents([sample_document])

    print(f"\nOriginal text length: {len(sample_text)} characters")
    print(f"Number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} (length: {len(chunk.page_content)}) ---")
        print(chunk.page_content)
        print(f"Metadata: {chunk.metadata}")