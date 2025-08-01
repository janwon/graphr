import os
import time # Import time for delays if needed
from config import GraphRAGConfig
from data_loader import get_data_loader
from llm_interface import LLMClient
from processing_pipeline import TextProcessor, Extractor, GraphBuilder
from graph_analysis import GraphAnalyzer
from query_engine import GraphQueryEngine
from output_handler import OutputHandler
from vector_store import VectorStoreManager # Import VectorStoreManager

def run_indexing_pipeline(config: GraphRAGConfig, llm_client: LLMClient, vector_store: VectorStoreManager) -> tuple:
    """
    Executes the GraphRAG indexing pipeline:
    Data Loading -> Text Unit Creation -> Entity/Relationship/Claim Extraction ->
    Graph Building -> Community Detection -> Community Summarization.
    """
    print("\n--- Starting GraphRAG Indexing Pipeline ---")

    # 1. Data Loading
    try:
        data_loader = get_data_loader(config.data_input.__dict__)
        raw_documents = data_loader.load_data()
        if not raw_documents:
            print("No documents loaded. Exiting indexing pipeline.")
            return None, None, None, None # Added None for vector_store
        
        full_text_corpus = ""
        print("Loaded Document Titles:")
        for doc in raw_documents:
            title = doc.get("title", "Untitled Document")
            content = doc.get("content", "")
            print(f"- {title}")
            full_text_corpus += content + "\n\n"
        
        print(f"Combined content from {len(raw_documents)} documents for chunking.")

    except Exception as e:
        print(f"Error during data loading: {e}")
        return None, None, None, None

    # 2. Text Unit Creation
    text_processor = TextProcessor(config.processing.chunk_size, config.processing.chunk_overlap)
    text_units_with_ids = text_processor.create_text_units(full_text_corpus)
    print(f"Created {len(text_units_with_ids)} text units (chunks).")

    # 3. Entity, Relationship, and Claim Extraction & Chunk Embedding
    extractor = Extractor(llm_client, config.processing.entity_extraction_prompt, vector_store) # Pass vector_store
    graph_builder = GraphBuilder()
    
    print("Extracting entities, relationships, and claims from text units and generating chunk embeddings...")
    for i, (chunk_id, unit_text) in enumerate(text_units_with_ids):
        print(f"  Processing text unit {i+1}/{len(text_units_with_ids)} (ID: {chunk_id})...")
        extracted_data = extractor.extract_entities_relationships_claims(chunk_id, unit_text)
        graph_builder.add_extracted_data_to_graph(extracted_data)
        # Add a small delay to avoid hitting rate limits for LLM calls (both chat and embeddings)
        time.sleep(1) # Adjust as needed based on your Azure OpenAI quotas

    knowledge_graph = graph_builder.get_graph()
    print(f"Knowledge graph built with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges.")

    # 4. Graph Partitioning (Community Detection)
    graph_analyzer = GraphAnalyzer(llm_client, config.processing.community_summarization_prompt, vector_store) # Pass vector_store
    node_to_community_map = graph_analyzer.detect_communities(knowledge_graph)
    
    # 5. Community Summarization & Community Embedding
    community_summaries = graph_analyzer.summarize_communities(knowledge_graph, node_to_community_map)
    print(f"Generated summaries for {len(community_summaries)} communities and their embeddings.")

    print("--- GraphRAG Indexing Pipeline Completed ---")
    return knowledge_graph, node_to_community_map, community_summaries, vector_store

def run_querying_pipeline(config: GraphRAGConfig, llm_client: LLMClient,
                          knowledge_graph, node_to_community_map, community_summaries, vector_store: VectorStoreManager):
    """
    Executes the GraphRAG querying pipeline.
    """
    if knowledge_graph is None or node_to_community_map is None or community_summaries is None or vector_store is None:
        print("Indexing pipeline did not complete successfully. Cannot run querying pipeline.")
        return

    print("\n--- Starting GraphRAG Querying Pipeline ---")
    query_engine = GraphQueryEngine(llm_client, config.processing.answer_generation_prompt, vector_store) # Pass vector_store

    while True:
        user_query = input("\nEnter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        answer = query_engine.query(user_query, knowledge_graph, community_summaries)
        print("\n--- Answer ---")
        print(answer)
        print("--------------")

    print("--- GraphRAG Querying Pipeline Completed ---")

def main():
    """Main function to run the GraphRAG application."""
    config = GraphRAGConfig.load_from_env()
    print("Configuration loaded.")

    # Initialize LLM Client
    llm_client = LLMClient(
        api_key=config.llm.api_key,
        model_name=config.llm.model_name,
        temperature=config.llm.temperature,
        max_output_tokens=config.llm.max_output_tokens,
        embedding_model_name=config.llm.embedding_model_name # Pass embedding model name
    )

    # Initialize Vector Store Manager
    vector_store = VectorStoreManager()

    # Run Indexing Pipeline
    knowledge_graph, node_to_community_map, community_summaries, vector_store = run_indexing_pipeline(config, llm_client, vector_store)

    # Save outputs
    output_handler = OutputHandler(config.output.output_dir)
    if knowledge_graph:
        output_handler.save_graph(knowledge_graph, config.output.graph_file)
    if node_to_community_map:
        output_handler.save_communities(node_to_community_map, config.output.communities_file)
    if community_summaries:
        output_handler.save_community_summaries(community_summaries, config.output.communities_file.replace('.json', '_summaries.json'))
    if vector_store:
        output_handler.save_vectors(vector_store, config.output.vectors_file) # Save all embeddings

    # Run Querying Pipeline
    run_querying_pipeline(config, llm_client, knowledge_graph, node_to_community_map, community_summaries, vector_store)

if __name__ == "__main__":
    # Create a dummy input file for demonstration (if using file input)
    if not os.path.exists("input"):
        os.makedirs("input")
    with open("input/sample_document.txt", "w") as f:
        f.write("""
        Dr. Alice Smith is a leading researcher at Quantum Innovations. She specializes in quantum computing and cryptography.
        Her recent paper, "Quantum Entanglement for Secure Communications," was published in Nature Physics.
        Quantum Innovations is based in Silicon Valley and is known for its groundbreaking work in advanced technologies.
        Dr. Smith frequently collaborates with Dr. Bob Johnson from the Institute of Advanced AI, located in New York.
        Dr. Johnson's team recently developed a new AI algorithm for optimizing quantum circuits.
        The Institute of Advanced AI focuses on artificial intelligence and machine learning research.
        Their work often involves large datasets and high-performance computing.
        """)
    
    # Set environment variables for CosmosDB input for demonstration
    # You MUST set these in your actual .env file for real connection
    os.environ["GRAPHRAG_DATA_SOURCE_TYPE"] = "cosmosdb"
    # os.environ["GRAPHRAG_FILE_PATH"] = "input/sample_document.txt" # Uncomment for file input
    os.environ["GRAPHRAG_OUTPUT_DIR"] = "output"
    os.environ["GRAPHRAG_LLM_MODEL"] = "gpt-4o" # Ensure this matches your deployment
    os.environ["GRAPHRAG_EMBEDDING_MODEL"] = "text-embedding-ada-002" # Ensure this matches your deployment

    main()
