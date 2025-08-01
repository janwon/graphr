import os
import json
from llm_interface import LLMClient, ExtractedData # Import ExtractedData Pydantic model
from dotenv import load_dotenv
from pydantic import BaseModel, Field # Import BaseModel and Field for defining test schema
from typing import List

# Load environment variables from .env file
load_dotenv()

# Define a simple Pydantic model for the test structured output
class TestCompanyInfo(BaseModel):
    company_name: str = Field(..., description="The full name of the company.")
    founders: List[str] = Field(..., description="A list of the company's founders.")
    headquarters: str = Field(..., description="The location of the company's headquarters.")
    products: List[str] = Field(..., description="A list of key products the company is known for.")

def test_llm_client():
    """
    Tests the LLMClient for text generation, structured output, and embedding generation.
    """
    print("--- Testing LLMClient ---")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Error: AZURE_OPENAI_API_KEY not found in environment variables.")
        print("Please ensure your .env file is correctly set up with AZURE_OPENAI_API_KEY.")
        return

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    if not azure_endpoint or not azure_api_version:
        print("Error: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_VERSION not found in environment variables.")
        print("Please ensure your .env file is correctly set up with these values.")
        return

    model_deployment_name = "gpt-4o-mini-20240718-gs" # <<< VERIFY THIS MATCHES YOUR AZURE DEPLOYMENT NAME
    embedding_model_deployment_name = os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-ada-002") # Get embedding model from env
    
    print(f"Attempting to use LLM model deployment: '{model_deployment_name}'")
    print(f"Attempting to use Embedding model deployment: '{embedding_model_deployment_name}'")


    llm = LLMClient(
        api_key=api_key,
        model_name=model_deployment_name,
        temperature=0.7,
        system_prompt="You are a helpful assistant that provides concise answers.",
        embedding_model_name=embedding_model_deployment_name # Pass embedding model name
    )

    # Test Text Generation
    print("\n--- Testing Text Generation ---")
    text_prompt = "Explain the concept of quantum entanglement in simple terms."
    print(f"Prompt: {text_prompt}")
    generated_text = llm.generate_text(text_prompt)
    if generated_text:
        print("\nGenerated Text:")
        print(generated_text)
    else:
        print("Text generation failed. Check LLM client logs for details.")

    llm.clear_chat_history()

    # Test Structured Output Generation with Pydantic
    print("\n--- Testing Structured Output Generation with Pydantic ---")
    base_structured_prompt = "Extract details about 'Apple Inc.' from this text: 'Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne. It is headquartered in Cupertino, California. They are known for their iPhone and Mac computers.'"
    print(f"Base Prompt: {base_structured_prompt}")

    # Pass the Pydantic model class directly
    structured_output = llm.generate_structured_output(base_structured_prompt, TestCompanyInfo)
    
    if structured_output:
        print("\nGenerated Structured Output (Pydantic):")
        try:
            print(json.dumps(structured_output, indent=2))
        except TypeError as e:
            print(f"Error: Structured output is not a valid dictionary for JSON serialization. Type: {type(structured_output)}. Error: {e}")
            print(f"Raw structured_output content: {structured_output}")
        except Exception as e:
            print(f"An unexpected error occurred during structured output processing: {e}")
    else:
        print("Structured output generation failed. This could be due to:")
        print("1. LLM failing to generate valid JSON or adhere to the schema.")
        print("2. API call failure (e.g., incorrect model deployment name, rate limits).")
        print("Check the console output from 'llm_interface.py' for more specific errors (e.g., 'OpenAI API error', 'Error decoding JSON').")

    llm.clear_chat_history() # Clear history for the next test

    # Test Embedding Generation
    print("\n--- Testing Embedding Generation ---")
    embedding_text = "The quick brown fox jumps over the lazy dog."
    print(f"Text for embedding: '{embedding_text}'")
    embedding = llm.get_embedding(embedding_text)
    if embedding:
        print(f"\nGenerated Embedding (first 5 dimensions): {embedding[:5]}...")
        print(f"Embedding length: {len(embedding)}")
        assert isinstance(embedding, list)
        assert all(isinstance(x, (float, int)) for x in embedding)
        assert len(embedding) > 0 # Embeddings should not be empty
        print("Embedding generation test PASSED.")
    else:
        print("Embedding generation failed. Check LLM client logs for details.")
        print("Common reasons: Incorrect embedding model deployment name, or API key issues.")

    print("\n--- LLMClient Test Complete ---")

if __name__ == "__main__":
    test_llm_client()
