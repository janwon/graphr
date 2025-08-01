1. data_loader.py
2. llm_interface.py
3. processing_pipeline.py_ 

so I basically want to create a microsoft graphrag from scratch. What is the best way to structure my python so that I can do the 
text unit creation, 
entity relationships, and claim extraction, graph partitioning into communities/community detection, community summarization and a querying method to search through the graphs

GraphRAG Indexer
1. Loading input text
2. Create_base_text_units
1. create_final_documents
1. extract_graph
1. finalize_graph
1. create_communities
1. create_final_text_units
1. create_community_reports
1. generate_text_embeddings


1. DONE Update llm_interface.py (Add Embedding Generation)
We need the LLMClient to be able to generate embeddings. Your atlas.py had a vectorize method, so we'll adapt that.

2. DONE Update config.py (Add Embedding Model Configuration)
We'll add a setting for the embedding model name.

3. DONE Create vector_store.py (New File for Embedding Management)
This new file will handle generating, storing (in-memory for now, or to a file), and searching embeddings.

4. DONE Update processing_pipeline.py (Generate Entity/Chunk Embeddings)
Integrate the VectorStoreManager to generate embeddings for extracted entities and/or the text chunks themselves.

5. DONE Update graph_analysis.py (Generate Community Embeddings)
Integrate the VectorStoreManager to generate embeddings for the community summaries.

6. DONE Update query_engine.py (Leverage Embeddings for Search)
Modify the query logic to perform semantic search using these embeddings.

7. DONE Update main.py (Orchestrate Embedding Steps)
Integrate the new VectorStoreManager into the main workflow.

8. DONE Update output_handler.py (Save Embeddings)
Add functionality to save the generated embeddings to a file.