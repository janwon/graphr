import json
import networkx as nx
import os
from typing import Dict, Any, List
from vector_store import VectorStoreManager # Import VectorStoreManager

class OutputHandler:
    """Handles saving various artifacts of the GraphRAG process."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_vectors(self, vector_store: VectorStoreManager, file_name: str = "embeddings.json"):
        """
        Saves all embeddings from the VectorStoreManager to a JSON file.
        """
        file_path = os.path.join(self.output_dir, file_name)
        all_embeddings = vector_store.get_all_embeddings()
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(all_embeddings, f, indent=2)
            print(f"All embeddings saved to {file_path}")
        except Exception as e:
            print(f"Error saving embeddings to {file_path}: {e}")

    def save_communities(self, communities: Dict[str, int], file_name: str = "communities.json"):
        """
        Saves the node-to-community mapping to a JSON file.
        """
        file_path = os.path.join(self.output_dir, file_name)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(communities, f, indent=2)
            print(f"Communities saved to {file_path}")
        except Exception as e:
            print(f"Error saving communities to {file_path}: {e}")

    def save_graph(self, graph: nx.DiGraph, file_name: str = "knowledge_graph.graphml"):
        """
        Saves the NetworkX graph to a GraphML file.
        GraphML is a common XML-based file format for graphs.
        """
        file_path = os.path.join(self.output_dir, file_name)
        try:
            nx.write_graphml(graph, file_path)
            print(f"Knowledge graph saved to {file_path}")
        except Exception as e:
            print(f"Error saving graph to {file_path}: {e}")

    def save_community_summaries(self, summaries: Dict[int, str], file_name: str = "community_summaries.json"):
        """
        Saves community summaries to a JSON file.
        """
        file_path = os.path.join(self.output_dir, file_name)
        try:
            str_keys_summaries = {str(k): v for k, v in summaries.items()}
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(str_keys_summaries, f, indent=2)
            print(f"Community summaries saved to {file_path}")
        except Exception as e:
            print(f"Error saving community summaries to {file_path}: {e}")

