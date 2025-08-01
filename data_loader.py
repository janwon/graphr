from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
from azure.cosmos import CosmosClient, exceptions # Import CosmosClient and exceptions

class DataLoader(ABC):
    """Abstract base class for data loaders."""
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Loads data from the configured source and returns it as a list of dictionaries (documents).
        Each dictionary can represent a document with various fields like 'title', 'content', etc.
        """
        pass

class FileLoader(DataLoader):
    """Loads data from a local file."""
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified file does not exist: {file_path}")
        self.file_path = file_path

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Reads the content of the file. Assumes plain text for now.
        Returns as a list with one dictionary document.
        """
        print(f"Loading data from file: {self.file_path}")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [{"title": os.path.basename(self.file_path), "content": content}]

class CosmosDBLoader(DataLoader):
    """Loads data from Azure Cosmos DB."""
    def __init__(self, connection_string: str, database_name: str, container_name: str):
        try:
            self.client = CosmosClient.from_connection_string(connection_string)
            self.database = self.client.get_database_client(database_name)
            self.container = self.database.get_container_client(container_name)
            print(f"CosmosDBLoader initialized for DB: '{database_name}', Container: '{container_name}'")
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error initializing CosmosDB client: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during CosmosDB initialization: {e}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Fetches documents from Cosmos DB.
        Assumes documents have 'title' and 'text' fields.
        """
        print(f"Attempting to load data from Cosmos DB container: '{self.container.id}'")
        documents: List[Dict[str, Any]] = []
        try:
            # Query all items in the container
            # You can modify this query to filter or select specific fields
            query = "SELECT * FROM c"
            items = self.container.query_items(query=query, enable_cross_partition_query=True)
            
            for item in items:
                # Map 'text' field from CosmosDB document to 'content' for consistency
                # Ensure 'title' and 'content' (or 'text') exist in your CosmosDB documents
                document_data = {
                    "title": item.get("title", "No Title Provided"),
                    "content": item.get("text", item.get("content", "")) # Prioritize 'text', fallback to 'content'
                }
                documents.append(document_data)
            print(f"Successfully loaded {len(documents)} documents from CosmosDB.")
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying CosmosDB: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during CosmosDB data loading: {e}")
        return documents

# Factory function to get the appropriate data loader
def get_data_loader(config: Dict[str, Any]) -> DataLoader:
    """
    Factory function to return the correct DataLoader based on configuration.
    """
    source_type = config.get("source_type")
    if source_type == "file":
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("File path must be provided for 'file' source type.")
        return FileLoader(file_path)
    elif source_type == "cosmosdb":
        conn_str = config.get("cosmosdb_connection_string")
        db_name = config.get("cosmosdb_database_name")
        container_name = config.get("cosmosdb_container_name")
        if not all([conn_str, db_name, container_name]):
            raise ValueError("CosmosDB connection details (string, database, container) must be provided.")
        return CosmosDBLoader(conn_str, db_name, container_name)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")

