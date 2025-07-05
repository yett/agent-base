from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama # Often preferred for chat models
from langchain_core.language_models import BaseChatModel, BaseLLM
from typing import Union

def get_ollama_llm(model_name: str, base_url: str = "http://localhost:11434") -> Union[BaseLLM, BaseChatModel]:
    """
    Initializes and returns an Ollama LLM or ChatOllama instance.
    Prefers ChatOllama for conversational purposes if available.

    Args:
        model_name (str): The name of the LLM to use (e.g., 'llama3', 'mistral').
        base_url (str): The URL of the Ollama server.

    Returns:
        Union[BaseLLM, BaseChatModel]: An instance of the Ollama LLM or ChatOllama model.
    """
    print(f"Initializing Ollama LLM with model: '{model_name}' at '{base_url}'")
    try:
        # Try to use ChatOllama first for better conversational capabilities
        llm = ChatOllama(model=model_name, base_url=base_url)
        # You can add a small test to ensure connectivity, e.g.,
        # llm.invoke("Hi") # This might be too heavy for a quick check.
        # A lighter check could be just instantiation.
        print("ChatOllama model initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing ChatOllama (falling back to Ollama LLM if possible): {e}")
        try:
            llm = Ollama(model=model_name, base_url=base_url)
            print("Ollama LLM model initialized successfully (using BaseLLM).")
            return llm
        except Exception as fallback_e:
            print(f"Error initializing Ollama LLM (even fallback failed): {fallback_e}")
            print("Please ensure Ollama is running and the specified LLM model is pulled.")
            raise # Re-raise the exception if no LLM can be initialized

# Example usage (for testing)
if __name__ == "__main__":
    # Ensure Ollama is running and 'llama3' (or another model) is pulled
    try:
        llm_instance = get_ollama_llm(model_name="llama3")
        print(f"\nModel type: {type(llm_instance)}")
        
        # Test a simple invocation
        test_query = "What is the capital of France?"
        print(f"Invoking LLM with query: '{test_query}'")
        response = llm_instance.invoke(test_query)
        print(f"LLM Response: {response}")

    except Exception as e:
        print(f"LLM model test failed: {e}")