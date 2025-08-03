import os
import sys
import json
from dotenv import load_dotenv
import time # Import time for delays if needed
from typing import Dict, Any

# Add the parent directory to the Python path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import GraphRAGConfig # To get prompt templates
from data_loader import get_data_loader
from llm_interface import LLMClient, ExtractedData # Import ExtractedData
from processing_pipeline import TextProcessor, Extractor, GraphBuilder
from vector_store import VectorStoreManager, Document # Import VectorStoreManager and Document

def run_processing_pipeline_tests():
    print("--- Testing Processing Pipeline with CosmosDB Input ---")

    # Load environment variables
    load_dotenv()

    # --- Configuration Setup for Source CosmosDB (for reading) ---
    os.environ["GRAPHRAG_DATA_SOURCE_TYPE"] = "cosmosdb"
    source_cosmos_conn_str = os.getenv("GRAPHRAG_COSMOSDB_CONN_STR", "")
    source_cosmos_db_name = os.getenv("GRAPHRAG_COSMOSDB_DB_NAME", "")
    source_cosmos_container_name = os.getenv("GRAPHRAG_COSMOSDB_CONTAINER_NAME", "")

    # --- NEW: Configuration Setup for Destination CosmosDB (for writing) ---
    # These environment variables should be set in your .env file
    dest_cosmos_conn_str = os.getenv("GRAPHRAG_DEST_COSMOSDB_CONN_STR", source_cosmos_conn_str)
    dest_cosmos_db_name = os.getenv("GRAPHRAG_DEST_COSMOSDB_DB_NAME", source_cosmos_db_name)
    dest_text_unit_container_name = os.getenv("GRAPHRAG_DEST_TEXT_UNIT_CONTAINER", "text_units")
    dest_extracted_data_container_name = os.getenv("GRAPHRAG_DEST_EXTRACTED_DATA_CONTAINER", "extracted_data")

    # --- 1. Load Data ---
    print("\n--- Step 1: Loading Data from CosmosDB ---")
    data_loader_config = {
        "source_type": os.environ["GRAPHRAG_DATA_SOURCE_TYPE"],
        "cosmosdb_connection_string": source_cosmos_conn_str,
        "cosmosdb_database_name": source_cosmos_db_name,
        "cosmosdb_container_name": source_cosmos_container_name
    }
    loader = get_data_loader(data_loader_config)
    documents = loader.load_data()
    if not documents:
        print("Failed to load any documents from CosmosDB. Aborting test.")
        return

    raw_text = documents[0].get("content", "")
    if not raw_text:
        print("Loaded document has no 'content' field. Aborting test.")
        return
    print(f"Loaded {len(raw_text)} characters of text from a single document.")


    # --- 2. Process Text into Chunks ---
    print("\n--- Step 2: Processing Text into Chunks ---")
    config = GraphRAGConfig()
    text_processor = TextProcessor(config.processing.chunk_size, config.processing.chunk_overlap)
    text_chunks_with_ids = text_processor.create_text_units(raw_text)
    print(f"Text processed into {len(text_chunks_with_ids)} chunks.")
    if not text_chunks_with_ids:
        print("No chunks were created. Aborting test.")
        return

    # --- Setup LLM Client and Vector Store ---
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    llm_client = LLMClient(api_key, config.llm.model_name)
    
    # Initialize separate VectorStoreManagers for different document types
    # This simulates storing text units and extracted data in different containers
    text_unit_vector_store = VectorStoreManager(
        source_cosmos_conn_str,
        source_cosmos_db_name,
        source_cosmos_container_name,
        dest_cosmosdb_conn_str=dest_cosmos_conn_str,
        dest_cosmosdb_db_name=dest_cosmos_db_name,
        dest_cosmosdb_container_name=dest_text_unit_container_name
    )
    extracted_data_vector_store = VectorStoreManager(
        source_cosmos_conn_str,
        source_cosmos_db_name,
        source_cosmos_container_name,
        dest_cosmosdb_conn_str=dest_cosmos_conn_str,
        dest_cosmosdb_db_name=dest_cosmos_db_name,
        dest_cosmosdb_container_name=dest_extracted_data_container_name
    )

    # --- 3. Extract Entities, Relationships, and Claims from Chunks ---
    print("\n--- Step 3: Extracting Entities, Relationships, and Claims from Chunks ---")
    extractor = Extractor(llm_client, config.processing.entity_extraction_prompt, text_unit_vector_store)
    graph_builder = GraphBuilder()

    extracted_data_samples = []
    
    # Test only first 2 chunks for extraction to save tokens/time
    num_chunks_to_test = min(len(text_chunks_with_ids), 2) 

    # Prepare a list of documents to add to the text unit vector store
    text_unit_docs_to_add = []

    for i in range(num_chunks_to_test):
        chunk_id, chunk_text = text_chunks_with_ids[i]
        print(f"\n--- Extracting from Chunk {i+1}/{len(text_chunks_with_ids)} (ID: {chunk_id}) ---")
        print(f"Chunk content preview: '{chunk_text[:100]}...'") # Show a snippet of the chunk

        try:
            # Generate embedding for the chunk and prepare the document for the vector store
            print("Generating embedding for chunk...")
            chunk_embedding = llm_client.get_embedding(chunk_text)
            if chunk_embedding:
                text_unit_docs_to_add.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "embedding": chunk_embedding,
                    "type": "text_unit"
                })
                print("Generated embedding and prepared document.")
            else:
                print(f"Warning: Failed to generate embedding for chunk {chunk_id}. Skipping.")
                continue

            # Pass chunk_id to the extractor method
            extracted_data = extractor.extract_entities_relationships_claims(chunk_id, chunk_text)
          
            if extracted_data:
                print(f"Extracted Data for Chunk {i+1}:")
                
                if isinstance(extracted_data, dict):
                    extracted_data_dict = extracted_data
                elif hasattr(extracted_data, 'model_dump'):
                    extracted_data_dict = extracted_data.model_dump()
                elif hasattr(extracted_data, 'dict'):
                    extracted_data_dict = extracted_data.dict()
                else:
                    extracted_data_dict = {}

                print(json.dumps(extracted_data_dict, indent=2))
                extracted_data_samples.append(extracted_data_dict)

                # Add the extracted data to the graph builder
                # The GraphBuilder should have an 'add_extracted_data' method
                # If it doesn't, this line will cause the error.
                # Assuming the method exists based on the original code's intent.
                graph_builder.add_extracted_data_to_graph(extracted_data_dict)

                # Store the extracted data in its own vector store
                extracted_data_vector_store.add_documents([
                    {
                        "id": f"extracted_data_{chunk_id}",
                        "text": json.dumps(extracted_data_dict),
                        "embedding": chunk_embedding, # Using the same embedding as the chunk for simplicity
                        "type": "extracted_data"
                    }
                ])
                print(f"Stored extracted data for chunk {chunk_id} in '{dest_extracted_data_container_name}'.")

            else:
                print(f"Extraction failed for Chunk {i+1}. Check LLM client logs for errors.")
            
        except Exception as e:
            print(f"Error during vector production/storage/retrieval for Chunk {i+1}: {e}")
        
        # Add a small delay to avoid hitting rate limits for LLM calls
        time.sleep(1) # Uncomment if facing rate limits

    # Now, add all the prepared text unit documents to the vector store at once
    print(f"\n--- Adding all {len(text_unit_docs_to_add)} text unit documents to CosmosDB ---")
    text_unit_vector_store.add_documents(text_unit_docs_to_add)
    print(f"Documents successfully added to '{dest_text_unit_container_name}'.")

    # --- 4. Build the Knowledge Graph ---
    print("\n--- Step 4: Building Knowledge Graph from extracted data ---")
    final_graph = graph_builder.get_graph()
    print(f"Final Graph has {len(final_graph.nodes)} nodes and {len(final_graph.edges)} edges.")

    print("\n--- Example Graph Nodes (first 5) ---")
    for i, node in enumerate(final_graph.nodes(data=True)):
        if i >= 5:
            break
        print(f"- Node: {node[0]}, Attributes: {node[1]}")

    # --- Final Verification: Retrieve all documents from the destination Cosmos DBs ---
    print("\n--- Final Verification: Retrieving all stored documents from destination Cosmos DBs ---")

    # Verify text unit documents
    all_text_unit_docs = text_unit_vector_store.get_all_stored_data()
    if all_text_unit_docs:
        print(f"\nFound {len(all_text_unit_docs)} documents in '{dest_text_unit_container_name}':")
        for doc_id, doc_data in all_text_unit_docs.items():
            print(f"- ID: {doc_id}, Type: {doc_data.get('type')}, Text Preview: '{doc_data.get('text', '')[:50]}...'")
    else:
        print(f"No documents found in '{dest_text_unit_container_name}'.")

    # Verify extracted data documents
    all_extracted_data_docs = extracted_data_vector_store.get_all_stored_data()
    if all_extracted_data_docs:
        print(f"\nFound {len(all_extracted_data_docs)} documents in '{dest_extracted_data_container_name}':")
        for doc_id, doc_data in all_extracted_data_docs.items():
            print(f"- ID: {doc_id}, Type: {doc_data.get('type')}, Text Preview: '{doc_data.get('text', '')[:50]}...'")
    else:
        print(f"No documents found in '{dest_extracted_data_container_name}'.")


    print("\n--- Processing Pipeline Test Complete ---")


if __name__ == "__main__":
    run_processing_pipeline_tests()