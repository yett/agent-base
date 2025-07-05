import yaml
import os

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads and parses the YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {exc}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading config: {e}")

# Example usage (for testing, not part of the main application flow)
if __name__ == "__main__":
    try:
        app_config = load_config()
        print("Configuration loaded successfully:")
        print(app_config)
        # Access specific values
        print(f"\nApp Name: {app_config['app_name']}")
        print(f"LLM Model: {app_config['ollama']['llm_model']}")
        print(f"ChromaDB Persistent Directory: {app_config['data_ingestion']['vector_store']['persist_directory']}")
    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"Error: {e}")