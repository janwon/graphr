import networkx as nx
from typing import Dict, Any, List, Optional
from llm_interface import LLMClient
from vector_store import VectorStoreManager # Import VectorStoreManager

class GraphQueryEngine:
    """
    Engine for querying the knowledge graph, leveraging community summaries and graph structure.
    Now also uses vector embeddings for semantic search.
    """
    def __init__(self, llm_client: LLMClient, answer_prompt_template: str, vector_store: VectorStoreManager):
        self.llm_client = llm_client
        self.answer_prompt_template = answer_prompt_template
        self.vector_store = vector_store # Inject VectorStoreManager

    def _retrieve_local_context(self, question: str, graph: nx.DiGraph) -> str:
        """
        Retrieves context by performing semantic search on chunk embeddings
        and then fanning out to their neighbors/properties.
        """
        context_parts = []
        
        # Generate embedding for the query
        query_embedding = self.llm_client.get_embedding(question)
        if not query_embedding:
            print("Warning: Could not generate embedding for the query. Falling back to keyword search for local context.")
            return self._fallback_keyword_local_context(question, graph)

        # Semantic search for relevant chunks
        top_chunks = self.vector_store.search_chunks(query_embedding, top_k=5) # Get top 5 relevant chunks
        
        relevant_chunk_ids = [chunk_id for chunk_id, _ in top_chunks]
        
        # For this simplified example, we don't have the original chunk text stored directly here.
        # In a real system, you'd retrieve the original text of these chunks from a storage layer
        # (e.g., a document store, or store chunk text in the vector store alongside embeddings).
        # For now, we'll try to find graph nodes related to these chunks or use the graph structure.

        # The current implementation of _retrieve_local_context primarily relies on graph traversal
        # based on entities mentioned in the question. To truly leverage chunk embeddings,
        # you'd need to map chunk_ids back to their original text and then extract entities
        # from those specific chunks to guide graph traversal.
        # For this update, we'll keep the existing graph traversal but ensure the query embedding
        # is used to *prioritize* which graph nodes to focus on, or to retrieve related information.
        
        # Identify entities in the question to guide graph traversal
        question_lower = question.lower()
        relevant_nodes = [
            node for node in graph.nodes
            if node.lower() in question_lower or
               (graph.nodes[node].get('description') and node.lower() in graph.nodes[node]['description'].lower())
        ]
        
        # Add context from relevant nodes and their neighbors
        for node in relevant_nodes:
            context_parts.append(f"--- Entity: {node} ---")
            node_attrs = graph.nodes[node]
            for attr, value in node_attrs.items():
                if attr != 'claims' and isinstance(value, (str, int, float)):
                    context_parts.append(f"  {attr.capitalize()}: {value}")
            if 'claims' in node_attrs and node_attrs['claims']:
                context_parts.append("  Claims:")
                for claim in node_attrs['claims']:
                    context_parts.append(f"    - {claim}")

            context_parts.append("  Relationships:")
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:
                    context_parts.append(f"    - {node} --({edge_data.get('type', 'UNKNOWN')})--> {neighbor} ({edge_data.get('description', '')})")
            for predecessor in graph.predecessors(node):
                edge_data = graph.get_edge_data(predecessor, node)
                if edge_data:
                    context_parts.append(f"    - {predecessor} --({edge_data.get('type', 'UNKNOWN')})--> {node} ({edge_data.get('description', '')})")

        return "\n".join(context_parts)

    def _fallback_keyword_local_context(self, question: str, graph: nx.DiGraph) -> str:
        """Fallback to keyword-based local context retrieval if embedding fails."""
        context_parts = []
        question_lower = question.lower()
        relevant_nodes = [
            node for node in graph.nodes
            if node.lower() in question_lower or
               (graph.nodes[node].get('description') and node.lower() in graph.nodes[node]['description'].lower())
        ]
        for node in relevant_nodes:
            context_parts.append(f"--- Entity: {node} ---")
            node_attrs = graph.nodes[node]
            for attr, value in node_attrs.items():
                if attr != 'claims' and isinstance(value, (str, int, float)):
                    context_parts.append(f"  {attr.capitalize()}: {value}")
            if 'claims' in node_attrs and node_attrs['claims']:
                context_parts.append("  Claims:")
                for claim in node_attrs['claims']:
                    context_parts.append(f"    - {claim}")
            # Add immediate neighbors and their relationships
            context_parts.append("  Relationships:")
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:
                    context_parts.append(f"    - {node} --({edge_data.get('type', 'UNKNOWN')})--> {neighbor} ({edge_data.get('description', '')})")
            for predecessor in graph.predecessors(node):
                edge_data = graph.get_edge_data(predecessor, node)
                if edge_data:
                    context_parts.append(f"    - {predecessor} --({edge_data.get('type', 'UNKNOWN')})--> {node} ({edge_data.get('description', '')})")
        return "\n".join(context_parts)


    def _retrieve_global_context(self, question: str, community_summaries: Dict[int, str]) -> str:
        """
        Retrieves context by performing semantic search on community summary embeddings.
        """
        context_parts = []
        
        query_embedding = self.llm_client.get_embedding(question)
        if not query_embedding:
            print("Warning: Could not generate embedding for the query. Falling back to keyword search for communities.")
            return self._fallback_keyword_global_context(question, community_summaries)

        # Semantic search for relevant communities
        top_communities = self.vector_store.search_communities(query_embedding, top_k=2) # Get top 2 relevant communities
        
        for comm_id_str, score in top_communities:
            comm_id = int(comm_id_str) # Convert back to int if needed for dict key
            summary = community_summaries.get(comm_id)
            if summary:
                context_parts.append(f"--- Community Summary {comm_id} (Score: {score:.2f}) ---\n{summary}")
        
        if not context_parts:
            print("No relevant community summaries found via semantic search. Including all available summaries as fallback.")
            # Fallback to including all summaries if semantic search yields nothing
            for comm_id, summary in community_summaries.items():
                context_parts.append(f"--- Community Summary {comm_id} ---\n{summary}")

        return "\n\n".join(context_parts)

    def _fallback_keyword_global_context(self, question: str, community_summaries: Dict[int, str]) -> str:
        """Fallback to keyword-based global context retrieval if embedding fails."""
        relevant_summaries = []
        question_lower = question.lower()
        for comm_id, summary in community_summaries.items():
            if any(keyword in summary.lower() for keyword in question_lower.split()):
                relevant_summaries.append(f"--- Community Summary {comm_id} ---\n{summary}")
        
        if not relevant_summaries:
            print("No specific community summaries matched via keyword. Including all available summaries.")
            for comm_id, summary in community_summaries.items():
                relevant_summaries.append(f"--- Community Summary {comm_id} ---\n{summary}")
        return "\n\n".join(relevant_summaries)


    def query(self, question: str, graph: nx.DiGraph, community_summaries: Dict[int, str]) -> str:
        """
        Answers a question by retrieving relevant context from the graph and summaries,
        then using the LLM to generate a response.
        """
        print(f"Processing query: '{question}'")

        # Combine local and global context
        local_context = self._retrieve_local_context(question, graph)
        global_context = self._retrieve_global_context(question, community_summaries)

        combined_context = ""
        if local_context:
            combined_context += "\n\n--- Local Graph Context (Semantic Search) ---\n" + local_context
        if global_context:
            combined_context += "\n\n--- Global Community Summaries (Semantic Search) ---\n" + global_context
        
        if not combined_context:
            return "No relevant context found in the knowledge graph or community summaries to answer your question."

        prompt = self.answer_prompt_template.format(question=question, context=combined_context)
        self.llm_client.clear_chat_history()
        answer = self.llm_client.generate_text(prompt)

        if answer:
            return answer
        else:
            return "Failed to generate an answer based on the retrieved context."

