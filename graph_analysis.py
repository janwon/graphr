import networkx as nx
import networkx.algorithms.community as nx_comm
from typing import Dict, List, Any, Tuple
from llm_interface import LLMClient
from vector_store import VectorStoreManager # Import VectorStoreManager

class GraphAnalyzer:
    """Performs community detection and summarization on a knowledge graph."""
    def __init__(self, llm_client: LLMClient, summarization_prompt_template: str, vector_store: VectorStoreManager):
        self.llm_client = llm_client
        self.summarization_prompt_template = summarization_prompt_template
        self.vector_store = vector_store # Inject VectorStoreManager

    def detect_communities(self, graph: nx.DiGraph) -> Dict[str, int]:
        """
        Detects communities within the graph using the Louvain method.
        Returns a dictionary mapping node names to community IDs.
        For more advanced hierarchical detection, libraries like `graspologic`
        with algorithms like Leiden could be used, but they have complex dependencies.
        """
        print("Detecting communities using Louvain method...")
        # Convert to undirected graph for community detection algorithms that require it
        undirected_graph = graph.to_undirected()
        
        # Ensure the graph is not empty and has edges for community detection
        if undirected_graph.number_of_nodes() == 0 or undirected_graph.number_of_edges() == 0:
            print("Graph is too sparse or empty for community detection. Returning each node as its own community.")
            return {node: i for i, node in enumerate(graph.nodes())}

        try:
            communities_sets = nx_comm.louvain_communities(undirected_graph, seed=42)
        except Exception as e:
            print(f"Error during Louvain community detection: {e}")
            print("Falling back to assigning each node to its own community.")
            return {node: i for i, node in enumerate(graph.nodes())}

        node_to_community_map = {}
        for i, community in enumerate(communities_sets):
            for node in community:
                node_to_community_map[node] = i
        print(f"Detected {len(communities_sets)} communities.")
        return node_to_community_map

    def summarize_communities(self, graph: nx.DiGraph, node_to_community_map: Dict[str, int]) -> Dict[int, str]:
        """
        Generates a summary for each detected community using the LLM and stores its embedding.
        """
        community_summaries = {}
        communities_data: Dict[int, List[str]] = {}

        # Group node information by community
        for node, community_id in node_to_community_map.items():
            if community_id not in communities_data:
                communities_data[community_id] = []
            
            node_info = f"Entity: {node}"
            node_attrs = graph.nodes[node]
            if 'type' in node_attrs:
                node_info += f" (Type: {node_attrs['type']})"
            if 'description' in node_attrs:
                node_info += f" - Description: {node_attrs['description']}"
            if 'claims' in node_attrs and node_attrs['claims']:
                node_info += f" - Claims: {'; '.join(node_attrs['claims'])}"
            
            # Add relationships involving this node within the community
            related_edges = []
            for u, v, data in graph.edges(node, data=True):
                # Only consider relationships where both ends are in the same community
                if node_to_community_map.get(u) == community_id and \
                   node_to_community_map.get(v) == community_id:
                    edge_info = f"Relationship: {u} --({data.get('type', 'UNKNOWN')})--> {v}"
                    if 'description' in data:
                        edge_info += f" - Desc: {data['description']}"
                    related_edges.append(edge_info)
            
            if related_edges:
                node_info += "\n  Related: " + "\n  ".join(related_edges)

            communities_data[community_id].append(node_info)

        print(f"Summarizing {len(communities_data)} communities...")
        for community_id, data_points in communities_data.items():
            community_text = "\n".join(data_points)
            prompt = self.summarization_prompt_template.format(text=community_text)
            self.llm_client.clear_chat_history()
            summary = self.llm_client.generate_text(prompt)
            
            if summary:
                community_summaries[community_id] = summary
                print(f"  Community {community_id} summarized.")
                # Generate and store embedding for the community summary
                summary_embedding = self.llm_client.get_embedding(summary)
                if summary_embedding:
                    self.vector_store.add_community_embedding(community_id, summary_embedding)
                else:
                    print(f"  Warning: Could not generate embedding for community {community_id} summary.")
            else:
                community_summaries[community_id] = "Could not generate summary for this community."
                print(f"  Failed to summarize community {community_id}.")
        return community_summaries

