import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import math
from azure.cosmos import CosmosClient, exceptions # Import CosmosClient and exceptions

# Define a simple Document class for structured return
class Document:
    def __init__(self, id: str, text: Optional[str] = None, embedding: Optional[List[float]] = None, score: Optional[float] = None, doc_type: Optional[str] = None):
        self.id = id
        self.text = text
        self.embedding = embedding
        self.score = score
        self.type = doc_type

    def __repr__(self):
        return f"Document(id='{self.id}', type='{self.type}', score={self.score:.4f})"

class VectorStoreManager:
    """
    Manages the storage and retrieval of documents with embeddings, using CosmosDB
    as the persistent vector store. This version supports separate source (read-only)
    and destination (write-only) configurations.
    """
    def __init__(self, source_cosmosdb_conn_str: str, source_cosmosdb_db_name: str, source_cosmosdb_container_name: str, dest_cosmosdb_conn_str: Optional[str] = None, dest_cosmosdb_db_name: Optional[str] = None, dest_cosmosdb_container_name: Optional[str] = None):
        """
        Initializes the vector store with CosmosDB clients for source and destination.
        
        Args:
            source_cosmosdb_conn_str (str): The connection string for the source CosmosDB.
            source_cosmosdb_db_name (str): The database name for the source CosmosDB.
            source_cosmosdb_container_name (str): The container name for the source CosmosDB.
            dest_cosmosdb_conn_str (Optional[str]): The connection string for the destination CosmosDB.
            dest_cosmosdb_db_name (Optional[str]): The database name for the destination CosmosDB.
            dest_cosmosdb_container_name (Optional[str]): The container name for the destination CosmosDB.
        """
        self.source_container = self._initialize_cosmosdb_client(
            source_cosmosdb_conn_str,
            source_cosmosdb_db_name,
            source_cosmosdb_container_name
        )
        self.dest_container = None
        if all([dest_cosmosdb_conn_str, dest_cosmosdb_db_name, dest_cosmosdb_container_name]):
            self.dest_container = self._initialize_cosmosdb_client(
                dest_cosmosdb_conn_str,
                dest_cosmosdb_db_name,
                dest_cosmosdb_container_name
            )

    def _initialize_cosmosdb_client(self, conn_str: str, db_name: str, container_name: str):
        """Helper to create and return a CosmosDB container client."""
        try:
            client = CosmosClient.from_connection_string(conn_str)
            database = client.get_database_client(db_name)
            container = database.get_container_client(container_name)
            print(f"Successfully connected to CosmosDB container: '{container_name}' in database: '{db_name}'")
            return container
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error connecting to CosmosDB container '{container_name}': {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while connecting to CosmosDB: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Adds multiple documents (with their embeddings) to the destination vector store.
        Each document dict should have at least 'id' and 'embedding'.
        It can also have 'text' for richer storage.
        """
        if not self.dest_container:
            print("Error: Destination CosmosDB container not configured. Cannot add documents.")
            return

        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                print("Warning: Document without an 'id' cannot be added. Skipping.")
                continue

            try:
                # Use upsert_item to create or update the document
                self.dest_container.upsert_item(body=doc)
                # print(f"Upserted document with id: {doc_id}")
            except exceptions.CosmosHttpResponseError as e:
                print(f"Error upserting document {doc_id} to CosmosDB: {e}")
            except Exception as e:
                print(f"An unexpected error occurred adding document {doc_id}: {e}")
    
    def get_all_stored_data(self, doc_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves all documents from the source CosmosDB container.
        
        Args:
            doc_type (Optional[str]): Filters documents by a 'type' property if provided.
        
        Returns:
            Dict[str, Any]: A dictionary of document IDs to their data.
        """
        if not self.source_container:
            print("Error: Source CosmosDB container not configured. Cannot retrieve data.")
            return {}
        
        query = "SELECT * FROM c"
        if doc_type:
            query = f"SELECT * FROM c WHERE c.type = '{doc_type}'"
            
        stored_documents = {}
        try:
            for item in self.source_container.query_items(query=query, enable_cross_partition_query=True):
                stored_documents[item['id']] = item
            print(f"Successfully retrieved {len(stored_documents)} documents from source.")
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying CosmosDB: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during data retrieval: {e}")
        return stored_documents

    def search_documents(self, query_embedding: List[float], top_k: int = 5, doc_type: Optional[str] = None) -> List[Document]:
        """
        Performs a cosine similarity search on stored documents in the source container.
        
        Args:
            query_embedding (List[float]): The embedding of the search query.
            top_k (int): The number of top results to return.
            doc_type (Optional[str]): Filters search to a specific document type.
            
        Returns:
            List[Document]: A list of Document objects sorted by similarity score.
        """
        # Note: This is an in-memory search on all retrieved documents.
        # For a production system, this would be replaced by native vector search capabilities.
        stored_documents = self.get_all_stored_data(doc_type=doc_type)

        similarities = []
        for doc_id, doc_data in stored_documents.items():
            embedding = doc_data.get("embedding")
            if embedding:
                score = self.cosine_similarity(query_embedding, embedding)
                similarities.append(Document(id=doc_id, text=doc_data.get("text"), embedding=embedding, score=score, doc_type=doc_data.get("type")))

        # Sort by score in descending order
        similarities.sort(key=lambda x: x.score, reverse=True)
        return similarities[:top_k]

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Computes the cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def load_embeddings_from_file(self, file_path: str):
        """Loads embeddings from a JSON file. This method is retained for compatibility."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # In a persistent setup, this would load into memory, but
                # we'll assume CosmosDB is the source of truth for this version.
                print(f"Embeddings loaded from {file_path}. Note: This version of the VectorStoreManager uses CosmosDB as the primary source.")
        except FileNotFoundError:
            print(f"Embeddings file not found: {file_path}. Continuing with CosmosDB store.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from embeddings file {file_path}: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred loading embeddings from {file_path}: {e}")
