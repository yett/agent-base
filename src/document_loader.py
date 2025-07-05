import os
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader, # To load from directories
    TextLoader, # For plain text files
)
from langchain_core.documents import Document

def load_documents_from_sources(sources_config: List[Dict[str, Any]]) -> List[Document]:
    """
    Loads documents from various configured sources.

    Args:
        sources_config (List[Dict[str, Any]]): A list of dictionaries,
                                                each describing a document source
                                                as defined in config.yaml.

    Returns:
        List[Document]: A list of loaded LangChain Document objects.
    """
    all_documents = []
    print("Loading documents from configured sources...")

    for source in sources_config:
        source_type = source.get('type')
        source_path = source.get('path')
        source_urls = source.get('urls')

        try:
            if source_type == "pdf" and source_path:
                if os.path.isdir(source_path):
                    print(f"  Loading PDFs from directory: {source_path}")
                    loader = DirectoryLoader(source_path, glob="*.pdf", loader_cls=PyPDFLoader)
                    all_documents.extend(loader.load())
                elif os.path.isfile(source_path):
                    print(f"  Loading PDF file: {source_path}")
                    loader = PyPDFLoader(source_path)
                    all_documents.extend(loader.load())
                else:
                    print(f"  Warning: PDF path '{source_path}' is not a valid file or directory. Skipping.")

            elif source_type == "csv" and source_path:
                if os.path.isfile(source_path):
                    print(f"  Loading CSV file: {source_path}")
                    loader = CSVLoader(file_path=source_path, encoding="utf-8") # Adjust encoding if needed
                    all_documents.extend(loader.load())
                else:
                    print(f"  Warning: CSV file '{source_path}' not found. Skipping.")

            elif source_type == "website" and source_urls:
                print(f"  Loading documents from websites: {source_urls}")
                loader = WebBaseLoader(web_paths=source_urls)
                all_documents.extend(loader.load())

            elif source_type == "text" and source_path: # Added support for plain text files
                if os.path.isdir(source_path):
                    print(f"  Loading TXT files from directory: {source_path}")
                    loader = DirectoryLoader(source_path, glob="*.txt", loader_cls=TextLoader)
                    all_documents.extend(loader.load())
                elif os.path.isfile(source_path):
                    print(f"  Loading TXT file: {source_path}")
                    loader = TextLoader(source_path)
                    all_documents.extend(loader.load())
                else:
                    print(f"  Warning: Text path '{source_path}' is not a valid file or directory. Skipping.")

            else:
                print(f"  Warning: Unknown or incomplete document source configuration: {source}. Skipping.")
        except Exception as e:
            print(f"  Error loading source {source_type} from '{source_path or source_urls}': {e}")

    print(f"Finished loading documents. Total documents loaded: {len(all_documents)}")
    return all_documents

# Example usage (for testing)
if __name__ == "__main__":
    # This example requires a 'data' folder with dummy files for testing
    # Create a 'data' folder and put some dummy.pdf, dummy.csv, dummy.txt files inside.
    # Also, ensure 'pyyaml', 'pypdf', 'pandas', 'beautifulsoup4' are installed.
    dummy_config_sources = [
        {"type": "pdf", "path": "./data/pdfs/sample.pdf"}, # Assuming you have data/pdfs/sample.pdf
        {"type": "csv", "path": "./data/csvs/sample.csv"}, # Assuming you have data/csvs/sample.csv
        {"type": "text", "path": "./data/texts/"}, # Assuming you have data/texts/ (folder with .txt files)
        {"type": "website", "urls": ["https://www.paulgraham.com/greatwork.html"]}, # Example public URL
        {"type": "invalid_type", "path": "./non_existent.txt"}
    ]

    # Create dummy directories and files for testing if they don't exist
    os.makedirs("./data/pdfs", exist_ok=True)
    os.makedirs("./data/csvs", exist_ok=True)
    os.makedirs("./data/texts", exist_ok=True)
    
    # Create dummy PDF (requires pypdf, but just a placeholder file)
    with open("./data/pdfs/sample.pdf", "w") as f:
        f.write("This is a dummy PDF content. Please replace with a real PDF for actual testing.")
    
    # Create dummy CSV
    with open("./data/csvs/sample.csv", "w") as f:
        f.write("col1,col2\nvalue1,value2\nvalue3,value4")
        
    # Create dummy text file
    with open("./data/texts/sample.txt", "w") as f:
        f.write("This is a dummy text file. It contains some sample content for testing.")

    try:
        documents = load_documents_from_sources(dummy_config_sources)
        print(f"\nSuccessfully loaded {len(documents)} documents.")
        if documents:
            print("First document content (first 200 chars):")
            print(documents[0].page_content[:200])
            print("First document metadata:")
            print(documents[0].metadata)
    except Exception as e:
        print(f"Error during document loading test: {e}")