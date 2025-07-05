# app.py (Complete Refactored Version)
import streamlit as st
import os

# Import all modular components
from src.config_loader import load_config
from src.llm_model import get_ollama_llm
from src.embedding_model import get_ollama_embeddings # Needed for vector store connection
from src.vector_store import get_chroma_vector_store
from src.rag_chain import build_rag_chain

# --- Load Configuration (cached to run once) ---
@st.cache_resource
def load_application_config():
    """Loads the application configuration from config.yaml."""
    try:
        return load_config()
    except Exception as e:
        st.error(f"Failed to load application configuration: {e}")
        st.stop() # Stop the Streamlit app if config can't be loaded

config = load_application_config()
app_name = config['app_name']
ollama_config = config['ollama']
vector_store_config = config['data_ingestion']['vector_store']
rag_config = config['rag']

# --- Streamlit UI Setup ---
st.set_page_config(page_title=app_name, layout="wide")
st.title(app_name)

# Display configurations for transparency
with st.expander("Configuration Details (from config.yaml)"):
    st.markdown(f"- **Ollama Host:** `{ollama_config['host']}`")
    st.markdown(f"- **LLM Model:** `{ollama_config['llm_model']}`")
    st.markdown(f"- **Embedding Model:** `{ollama_config['embedding_model']}`")
    st.markdown(f"- **ChromaDB Dir:** `{vector_store_config['persist_directory']}`")
    st.markdown(f"- **ChromaDB Collection:** `{vector_store_config['collection_name']}`")
    st.markdown(f"- **Retrieval K:** `{rag_config['retrieval_k']}`")
    st.markdown(f"- **RAG Chain Type:** `{rag_config['chain_type']}`")


# --- Initialize LLM, Embeddings, Vector Store, and RAG Chain (Cached for performance) ---
@st.cache_resource
def setup_rag_system(ollama_cfg, rag_cfg, vector_store_cfg):
    """
    Sets up and caches the RAG system components: LLM, Embeddings, Vector Store, and RAG Chain.
    This function is cached to run only once per session or until inputs change.
    """
    print("\n--- Setting up RAG System ---")
    
    # 1. Initialize LLM
    llm = get_ollama_llm(
        model_name=ollama_cfg['llm_model'],
        base_url=ollama_cfg['host']
    )
    
    # 2. Initialize Embedding Model (for connecting to ChromaDB)
    embeddings = get_ollama_embeddings(
        model_name=ollama_cfg['embedding_model'],
        base_url=ollama_cfg['host']
    )

    # 3. Connect to ChromaDB
    # Ensure the persistence directory exists (prep-data.py should have created it)
    if not os.path.exists(vector_store_cfg['persist_directory']):
        st.error(f"ChromaDB persistence directory not found: {vector_store_cfg['persist_directory']}.")
        st.error("Please run `python prep-data.py` first to prepare the data.")
        st.stop()

    vector_db = get_chroma_vector_store(
        persist_directory=vector_store_cfg['persist_directory'],
        collection_name=vector_store_cfg['collection_name'],
        embedding_function=embeddings
    )
    
    # Get retriever from vector store
    retriever = vector_db.as_retriever(search_kwargs={"k": rag_cfg['retrieval_k']})

    # 4. Build RAG chain
    rag_chain = build_rag_chain(
        llm=llm,
        retriever=retriever,
        chain_type=rag_cfg['chain_type'],
        return_source_documents=True # Always return sources for display in PoC
    )
    print("--- RAG System Setup Complete ---")
    return rag_chain

# Setup the RAG system (this will run once and be cached)
try:
    rag_chain = setup_rag_system(ollama_config, rag_config, vector_store_config)
except Exception as e:
    st.error(f"Failed to set up RAG system: {e}")
    st.stop()

# --- Initialize Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Generation ---
if prompt := st.chat_input("Ask me about the loaded data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        sources_info = ""

        try:
            # Invoke the RAG chain
            # RetrievalQA.invoke returns a dictionary with 'result' and 'source_documents'
            result = rag_chain.invoke({"query": prompt})
            response_content = result.get("result", "Sorry, I couldn't find an answer based on the available data.")
            source_documents = result.get("source_documents", [])
            
            # Simulate streaming effect for the main response
            for word in response_content.split():
                full_response_content += word + " "
                message_placeholder.markdown(full_response_content + "▌")
            
            # Display source documents if found
            if source_documents:
                sources_info = "\n\n**Sources:**\n"
                for i, doc in enumerate(source_documents):
                    # Prefer 'source' or 'file_path' in metadata, otherwise show content snippet
                    source_name = doc.metadata.get('source', doc.metadata.get('file_path', f"Document {i+1}"))
                    page_number = doc.metadata.get('page', None)
                    if page_number is not None:
                        source_name += f" (Page: {page_number})"
                    
                    sources_info += f"- {source_name}\n"
                    # Optionally, show a snippet of the source content in an expander
                    # with st.expander(f"Snippet from {source_name}"):
                    #     st.markdown(doc.page_content[:500] + "...") # Show first 500 chars

            final_response = full_response_content + sources_info
            message_placeholder.markdown(final_response)

        except Exception as e:
            final_response = f"An error occurred during response generation: {e}"
            st.error(final_response)
        
        # Add assistant's final response (including sources) to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_response})