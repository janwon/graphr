import numpy as np
import json
from typing import Dict, List, Tuple, Any
import math

class VectorStoreManager:
    """
    Manages the storage and retrieval of embeddings.
    For simplicity, this is an in-memory vector store.
    In a production system, this would be replaced by a dedicated vector database.
    """
    def __init__(self):
        # Stores embeddings for different types of content
        # Example: {"chunk_id": [embedding_vector], ...}
        self.chunk_embeddings: Dict[str, List[float]] = {}
        # Example: {"community_id": [embedding_vector], ...}
        self.community_embeddings: Dict[str, List[float]] = {}
        # You could add entity_embeddings as well: Dict[str, List[float]] = {}

    def add_chunk_embedding(self, chunk_id: str, embedding: List[float]):
        """Adds an embedding for a text chunk."""
        if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
            print(f"Warning: Invalid embedding format for chunk_id {chunk_id}. Expected List[float]. Got: {type(embedding)}")
            return
        self.chunk_embeddings[chunk_id] = embedding
        # print(f"Added embedding for chunk_id: {chunk_id}")

    def add_community_embedding(self, community_id: str, embedding: List[float]):
        """Adds an embedding for a community summary."""
        if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
            print(f"Warning: Invalid embedding format for community_id {community_id}. Expected List[float]. Got: {type(embedding)}")
            return
        self.community_embeddings[str(community_id)] = embedding # Ensure community_id is string for consistent dict keys
        # print(f"Added embedding for community_id: {community_id}")

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates the cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0 # Handle empty vectors
        
        # Convert to numpy arrays for efficient calculation
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)

        dot_product = np.dot(np_vec1, np_vec2)
        norm_vec1 = np.linalg.norm(np_vec1)
        norm_vec2 = np.linalg.norm(np_vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0 # Avoid division by zero
        
        return dot_product / (norm_vec1 * norm_vec2)

    def search_chunks(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Performs a similarity search over chunk embeddings.
        Returns a list of (chunk_id, similarity_score) tuples, sorted by score.
        """
        if not query_embedding:
            return []

        similarities = []
        for chunk_id, emb in self.chunk_embeddings.items():
            score = self.cosine_similarity(query_embedding, emb)
            similarities.append((chunk_id, score))
        
        # Sort by score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def search_communities(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Performs a similarity search over community embeddings.
        Returns a list of (community_id, similarity_score) tuples, sorted by score.
        """
        if not query_embedding:
            return []

        similarities = []
        for community_id, emb in self.community_embeddings.items():
            score = self.cosine_similarity(query_embedding, emb)
            similarities.append((community_id, score))
        
        # Sort by score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_all_embeddings(self) -> Dict[str, Any]:
        """Returns all stored embeddings."""
        return {
            "chunk_embeddings": self.chunk_embeddings,
            "community_embeddings": self.community_embeddings
        }

    def load_embeddings_from_file(self, file_path: str):
        """Loads embeddings from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.chunk_embeddings = data.get("chunk_embeddings", {})
                self.community_embeddings = data.get("community_embeddings", {})
            print(f"Embeddings loaded from {file_path}")
        except FileNotFoundError:
            print(f"Embeddings file not found: {file_path}. Starting with empty vector store.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from embeddings file {file_path}: {e}. Starting with empty vector store.")
        except Exception as e:
            print(f"An unexpected error occurred loading embeddings from {file_path}: {e}. Starting with empty vector store.")

