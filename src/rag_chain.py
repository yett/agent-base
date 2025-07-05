# src/rag_chain.py
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from typing import Any, Union

def build_rag_chain(
    llm: Union[BaseLLM, BaseChatModel],
    retriever: BaseRetriever,
    chain_type: str = "stuff",
    return_source_documents: bool = True
) -> Any:
    """
    Builds and returns a LangChain RetrievalQA chain.
    """
    print(f"Building RAG chain with chain_type='{chain_type}'...")
    
    # --- CUSTOMIZATION POINT: Modified prompt for bullet points ---
    # This prompt instructs the LLM on how to answer, what tone to use, and how to format.
    # {context} will be replaced by the retrieved document chunks.
    # {question} will be replaced by the user's query.
    custom_prompt_template = """Based on the context provided, answer the question clearly and concisely.
    Present your answer using bullet points if multiple distinct facts are available, or a single paragraph otherwise.
    If the answer is not found in the context, politely state that the information is not available in the provided documents.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:""" # The "Answer:" line implicitly guides the format. You can also explicitly state: "Answer (in bullet points):" or "Answer as a list:"
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt_template)

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=return_source_documents,
            # Pass the custom prompt to the chain
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
        )
        print("RetrievalQA chain built successfully with custom prompt.")
        return qa_chain
    except Exception as e:
        print(f"Error building RAG chain: {e}")
        raise

# Example usage (for testing - highly simplified as it needs a live LLM and retriever)
if __name__ == "__main__":
    print("This module is primarily for integration. A full test requires a live LLM and vector store.")
    print("Please run `app.py` after `prep-data.py` to see this in action.")