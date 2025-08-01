import os
import sys
import json
from dotenv import load_dotenv
import time # Import time for delays if needed

# Add the parent directory to the Python path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import GraphRAGConfig # To get prompt templates
from data_loader import get_data_loader
from llm_interface import LLMClient, ExtractedData # Import ExtractedData
from processing_pipeline import TextProcessor, Extractor
from vector_store import VectorStoreManager # Import VectorStoreManager

def run_processing_pipeline_tests():
    print("--- Testing Processing Pipeline with CosmosDB Input ---")

    # Load environment variables
    load_dotenv()

    # --- Configuration Setup ---
    # Temporarily override config for CosmosDB input for this test
    os.environ["GRAPHRAG_DATA_SOURCE_TYPE"] = "cosmosdb"
    os.environ["GRAPHRAG_COSMOSDB_CONN_STR"] = os.getenv("GRAPHRAG_COSMOSDB_CONN_STR", "")
    os.environ["GRAPHRAG_COSMOSDB_DB_NAME"] = os.getenv("GRAPHRAG_COSMOSDB_DB_NAME", "")
    os.environ["GRAPHRAG_COSMOSDB_CONTAINER_NAME"] = os.getenv("GRAPHRAG_COSMOSDB_CONTAINER_NAME", "")
    os.environ["GRAPHRAG_CHUNK_SIZE"] = "500" # Smaller chunk size for more chunks
    os.environ["GRAPHRAG_CHUNK_OVERLAP"] = "50"
    os.environ["GRAPHRAG_LLM_MODEL"] = os.getenv("GRAPHRAG_LLM_MODEL", "gpt-4o") # Ensure this is set for LLMClient
    os.environ["GRAPHRAG_EMBEDDING_MODEL"] = os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-ada-002") # Ensure this is set for LLMClient

    config = GraphRAGConfig.load_from_env()

    # Check CosmosDB credentials
    if not all([config.data_input.cosmosdb_connection_string,
                config.data_input.cosmosdb_database_name,
                config.data_input.cosmosdb_container_name]):
        print("Skipping processing pipeline test: CosmosDB environment variables not set. "
              "Please set GRAPHRAG_COSMOSDB_CONN_STR, GRAPHRAG_COSMOSDB_DB_NAME, and GRAPHRAG_COSMOSDB_CONTAINER_NAME in your .env file.")
        return

    # Initialize LLM Client
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Error: AZURE_OPENAI_API_KEY not found in environment variables. Cannot test extractor.")
        return

    llm_client = LLMClient(
        api_key=api_key,
        model_name=config.llm.model_name, # Use model from config
        temperature=config.llm.temperature,
        max_output_tokens=config.llm.max_output_tokens,
        embedding_model_name=config.llm.embedding_model_name # Pass embedding model name
    )

    # Initialize Vector Store Manager
    vector_store = VectorStoreManager()

    # --- 1. Load Data from CosmosDB ---
    print("\n--- Step 1: Loading Data from CosmosDB ---")
    try:
        data_loader = get_data_loader(config.data_input.__dict__)
        raw_documents = data_loader.load_data()
        if not raw_documents:
            print("No documents loaded from CosmosDB. Ensure your container has data. Exiting test.")
            return
        
        # Combine content from all documents into a single corpus for chunking
        full_text_corpus = ""
        print("Loaded Document Titles:")
        for doc in raw_documents:
            title = doc.get("title", "Untitled Document")
            content = doc.get("content", "")
            print(f"- {title}")
            full_text_corpus += content + "\n\n" # Add content and separation
        
        print(f"Combined content from {len(raw_documents)} documents for chunking.")

    except Exception as e:
        print(f"Error during data loading from CosmosDB: {e}")
        return

    # --- 2. Create Text Chunks ---
    print("\n--- Step 2: Creating Text Chunks ---")
    text_processor = TextProcessor(config.processing.chunk_size, config.processing.chunk_overlap)
    # text_chunks now returns (chunk_id, chunk_text) tuples
    text_chunks_with_ids = text_processor.create_text_units(full_text_corpus)
    print(f"Created {len(text_chunks_with_ids)} text chunks.")

    print("\n--- Sample Text Chunks (first 3) ---")
    for i, (chunk_id, chunk_text) in enumerate(text_chunks_with_ids[:3]):
        print(f"--- Chunk {i+1} (ID: {chunk_id}, Length: {len(chunk_text)}) ---")
        print(chunk_text)
        print("-" * 30)
    if len(text_chunks_with_ids) > 3:
        print(f"... and {len(text_chunks_with_ids) - 3} more chunks.")


    # --- 3. Extract Entities, Relationships, and Claims ---
    print("\n--- Step 3: Extracting Entities, Relationships, and Claims from Chunks ---")
    # Pass vector_store to Extractor
    extractor = Extractor(llm_client, config.processing.entity_extraction_prompt, vector_store)
    
    extracted_data_samples = []
    # Test only first 2 chunks for extraction to save tokens/time
    num_chunks_to_test = min(len(text_chunks_with_ids), 2) 

    for i in range(num_chunks_to_test):
        chunk_id, chunk_text = text_chunks_with_ids[i]
        print(f"\n--- Extracting from Chunk {i+1}/{len(text_chunks_with_ids)} (ID: {chunk_id}) ---")
        print(f"Chunk content preview: '{chunk_text[:100]}...'") # Show a snippet of the chunk

        # Pass chunk_id to the extractor method
        extracted_data = extractor.extract_entities_relationships_claims(chunk_id, chunk_text)
        
        if extracted_data:
            print(f"Extracted Data for Chunk {i+1}:")
            print(json.dumps(extracted_data, indent=2))
            extracted_data_samples.append(extracted_data)
        else:
            print(f"Extraction failed for Chunk {i+1}. Check LLM client logs for errors.")
        
        # Add a small delay to avoid hitting rate limits for LLM calls
        time.sleep(1) # Uncomment if facing rate limits

    print("\n--- Processing Pipeline Test Complete ---")

if __name__ == "__main__":
    run_processing_pipeline_tests()
