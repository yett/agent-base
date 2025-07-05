# **LLM-Powered RAG Chatbot Agent**

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot agent powered by a locally hosted Large Language Model (LLM) using Ollama, LangChain, and Streamlit. The chatbot can answer questions based on a custom knowledge base (your documents), providing context-aware and grounded responses.

## **🚀 Purpose & Tech Stack**

The primary purpose of this project is to provide a conceptual and extensible codebase for building RAG-based chatbots. It showcases how to integrate various open-source tools to create a functional AI agent that can interact with your private or domain-specific data.

**Key Technologies Used:**

- **Python:** The core programming language.
- **Streamlit:** For building an interactive and user-friendly web interface.
- **Ollama:** To run Large Language Models (LLMs) and embedding models locally.
- **LangChain:** A powerful framework for orchestrating the RAG pipeline, handling document loading, text splitting, embeddings, and LLM interactions.
- **ChromaDB:** A lightweight, open-source vector database for efficient storage and retrieval of document embeddings.
- **YAML:** For externalizing and managing application configurations.

## **💡 High-Level Overview**

The RAG chatbot operates on a simple yet powerful principle:

1. **Data Ingestion (prep-data.py):**
    - Your custom documents (PDFs, CSVs, text files, web pages) are loaded.
    - These documents are split into smaller, manageable "chunks."
    - Each chunk is converted into a numerical representation called an "embedding" using a local embedding model (via Ollama).
    - These embeddings and their corresponding text chunks are stored in a vector database (ChromaDB) for efficient similarity search.
2. **Chatbot Interaction (app.py):**
    - When a user asks a question, the question is also converted into an embedding.
    - This query embedding is used to search the ChromaDB for the most "similar" (relevant) document chunks.
    - The retrieved relevant chunks are then provided as "context" to a local LLM (via Ollama) along with the original user's question.
    - The LLM generates an answer, grounded in the provided context, reducing hallucinations and improving factual accuracy.
    - The conversation takes place within a Streamlit web interface.

## **📁 Project Structure**


```


├── config.yaml # Centralized configuration for the entire application  
├── prep-data.py # Script to prepare and ingest data into the vector database  
├── app.py # Streamlit web application for the RAG chatbot UI  
└── src/ # Source code for modular components  
├── \__init_\_.py # Makes 'src' a Python package  
├── config_loader.py # Handles loading and parsing of config.yaml  
├── document_loader.py # Manages loading documents from various sources (PDF, CSV, Web, Text)  
├── text_splitter.py # Encapsulates logic for splitting documents into chunks  
├── embedding_model.py # Initializes and provides the Ollama embedding model  
├── llm_model.py # Initializes and provides the Ollama LLM for generation  
├── rag_chain.py # Builds and orchestrates the LangChain RAG pipeline  
└── vector_store.py # Manages ChromaDB connection and document operations  
└── data/ # Directory to store your raw source documents (create this)  
├── pdfs/ # Example: Place your PDF files here  
├── csvs/ # Example: Place your CSV files here  
└── texts/ # Example: Place your plain text files here  
└── chroma_db/ # Directory where ChromaDB will persist its data (created by prep-data.py)  

```

## **⚙️ Setting Up the Project**

Follow these steps to get the RAG chatbot running on your local machine.

### **Prerequisites**

- **Python 3.9+:** Ensure Python is installed on your system.
- **Git:** For cloning the repository.
- **Ollama:** Download and install Ollama from [ollama.ai](https://ollama.ai/).

### **1\. Clone the Repository**

```
git clone git@github.com:yett/agent-base.git
cd rag-chatbot  
```

### **2\. Create and Activate a Virtual Environment**

It's highly recommended to use a virtual environment to manage project dependencies.

- **On Windows:**  
```
    python -m venv venv  
    .\\venv\\Scripts\\activate  
```

- **On macOS/Linux:**  
```
    python3 -m venv venv  
    source venv/bin/activate  
```

### **3\. Install Python Dependencies**

With your virtual environment activated, install all required Python packages:

```
pip install pyyaml langchain-community pypdf pandas beautifulsoup4 ollama chromadb  
```

### **4\. Pull Ollama Models**

You need to download the LLM and embedding models that your application will use. These are specified in config.yaml. By default, the configuration uses llama3 for the LLM and mixedbread for embeddings.

Run these commands in your terminal:

```
ollama pull llama3.2:latest  
ollama pull mxbai-embed-large  
```

_(If you change the llm_model or embedding_model in config.yaml, make sure to pull those specific models.)_

### **5\. Prepare Your Data**

Create the data/ directory in your project root and place your documents inside the respective subfolders (pdfs/, csvs/, texts/).

**Example data/ structure:**

```
data/  
├── pdfs/  
│ └── my_document.pdf  
├── csvs/  
│ └── sales_data.csv  
└── texts/  
└── faq.txt  
```

Update config.yaml:

Ensure the data_ingestion.document_sources section in your config.yaml accurately points to the paths of your data files or directories.
```
\# config.yaml (excerpt)  
data_ingestion:  
document_sources:  
\- type: "pdf"  
path: "./data/pdfs/"  
\- type: "csv"  
path: "./data/csvs/my_data.csv"  
\- type: "text"  
path: "./data/texts/faq.txt"  
\# - type: "website" # Uncomment and configure if needed  
\# urls:  
\# - "<https://example.com/some_article>"  
```

## **▶️ Running the Application**

### **1\. Start the Ollama Server**

Open a **separate terminal window** and start the Ollama server. This server must be running for your Python application to interact with the LLMs and embedding models.

```
ollama serve  
```

Leave this terminal window open and running throughout your session.

### **2\. Prepare/Refresh the RAG Data**

Open another terminal window, activate your virtual environment, and run the data preparation script. This will load your documents, chunk them, create embeddings, and store them in ChromaDB.

```
python prep-data.py  
```

This command will perform a **full refresh**: it will delete any existing ChromaDB data in ./chroma_db and then re-ingest all documents specified in your config.yaml.

### **3\. Launch the Chatbot UI**

In the **same terminal** where you ran prep-data.py (with the virtual environment still active), launch the Streamlit application:

```
streamlit run app.py  
```

This command will open a new tab in your default web browser, displaying the RAG chatbot interface.

### **Stopping the Streamlit Service**

To stop the Streamlit application, go to the terminal window where streamlit run app.py is executing and press Ctrl + C.

- **Troubleshooting Ctrl + C:** If it doesn't stop immediately (common on Windows if the browser tab was closed), try opening the Streamlit app's URL (<http://localhost:8501>) in your browser again, then quickly return to the terminal and press Ctrl + C.

You are now ready to interact with your LLM-powered RAG chatbot agent!
