# File: processing_pipeline.py
import networkx as nx
from typing import List, Dict, Any, Tuple
from llm_interface import LLMClient, ExtractedData
from vector_store import VectorStoreManager # Import VectorStoreManager
import hashlib
import json

class TextProcessor:
    """Handles splitting raw text into manageable text units (chunks)."""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunk_id_counter = 0 # To generate unique IDs for chunks

    def create_text_units(self, text: str) -> List[Tuple[str, str]]: # Changed return type
        """
        Splits a long text into smaller chunks with overlap.
        Returns a list of (chunk_id, chunk_text) tuples.
        """
        if not text:
            return []

        chunks_with_ids = []
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunk = text[start_idx:end_idx]
            
            # Generate a unique ID for each chunk
            chunk_id = f"chunk_{self._chunk_id_counter}"
            self._chunk_id_counter += 1
            chunks_with_ids.append((chunk_id, chunk))

            if end_idx == len(text):
                break
            start_idx += (self.chunk_size - self.chunk_overlap)
            if start_idx >= len(text):
                break
        return chunks_with_ids

class Extractor:
    """
    Extracts entities, relationships, and claims from text units using an LLM.
    Also generates and stores embeddings for text chunks.
    """
    def __init__(self, llm_client: LLMClient, prompt_template: str, text_unit_vector_store: VectorStoreManager, entity_vector_store: VectorStoreManager):
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.text_unit_vector_store = text_unit_vector_store # Inject Text Unit VectorStoreManager
        self.entity_vector_store = entity_vector_store # Inject Entity VectorStoreManager

    def extract_and_store_text_unit(self, chunk_id: str, text_unit: str) -> Tuple[str, List[float]]:
        """
        Generates an embedding for a text chunk and stores it in the text unit vector store.
        Returns the chunk_id and its embedding.
        """
        chunk_embedding = self.llm_client.get_embedding(text_unit)
        if chunk_embedding:
            self.text_unit_vector_store.add_documents([
                {
                    "id": chunk_id,
                    "text": text_unit,
                    "embedding": chunk_embedding,
                    "type": "text_unit"
                }
            ])
            return chunk_id, chunk_embedding
        else:
            print(f"Warning: Could not generate embedding for chunk_id {chunk_id}.")
            return chunk_id, None

    def extract_entities_relationships_claims(self, text_unit: str) -> ExtractedData:
        """
        Uses the LLM to extract structured information.
        Returns a Pydantic ExtractedData model.
        """
        prompt = self.prompt_template.format(text=text_unit)
        self.llm_client.clear_chat_history()
        extracted_data_model = self.llm_client.generate_structured_output(prompt, ExtractedData)
        
        if extracted_data_model:
            return extracted_data_model
        return ExtractedData(entities=[], relationships=[], claims=[])

    def dedupe_and_store_entities(self, extracted_data_dict: Dict[str, Any], chunk_id: str, chunk_embedding: List[float]):
        """
        Dedupes entities and stores them individually in the entity vector store.
        Maintains a reference to the chunk_id.
        """
        if not chunk_embedding:
            print(f"Skipping entity storage for chunk {chunk_id} due to missing embedding.")
            return

        # Fetch all existing documents from the entity vector store to perform de-duplication locally
        existing_entities = self.entity_vector_store.get_all_stored_data()
        
        for entity in extracted_data_dict.get('entities', []):
            # Create a unique ID for the entity based on its name and type
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            entity_id_hash = hashlib.sha256(f"{entity_name}_{entity_type}".encode('utf-8')).hexdigest()
            entity_id = f"entity_{entity_id_hash}"

            # Check if the entity already exists in the local dictionary
            existing_doc = existing_entities.get(entity_id)

            if existing_doc:
                # Update the existing document with the new chunk reference
                references = existing_doc.get("references", [])
                if chunk_id not in references:
                    references.append(chunk_id)
                self.entity_vector_store.update_document({
                    "id": entity_id,
                    "references": references,
                    "text": existing_doc.get("text"), # Keep the original text
                    "embedding": existing_doc.get("embedding"), # Keep the original embedding
                    "type": existing_doc.get("type"), # Keep the original type
                })
            else:
                # Create a new document for the entity
                new_entity_doc = {
                    "id": entity_id,
                    "text": json.dumps(entity),
                    "embedding": chunk_embedding, # Use the chunk's embedding for context
                    "type": "entity",
                    "references": [chunk_id]
                }
                self.entity_vector_store.add_documents([new_entity_doc])

class GraphBuilder:
    """Builds a NetworkX knowledge graph from extracted entities, relationships, and claims."""
    def __init__(self, use_entity_descriptions: bool = True):
        self.graph = nx.DiGraph()
        self.use_entity_descriptions = use_entity_descriptions
    
    def add_extracted_data_to_graph(self, extracted_data_dict: Dict[str, Any]):
        """
        Adds entities, relationships, and claims to the graph.
        Entities become nodes. Relationships become edges.
        """
        entities = extracted_data_dict.get("entities", [])
        relationships = extracted_data_dict.get("relationships", [])
        claims = extracted_data_dict.get("claims", [])

        for entity in entities:
            entity_name = entity.get("name")
            if entity_name:
                if not self.graph.has_node(entity_name):
                    self.graph.add_node(entity_name, type=entity.get("type"), description=entity.get("description"))
                else:
                    current_desc = self.graph.nodes[entity_name].get("description", "")
                    new_desc = entity.get("description", "")
                    if new_desc and new_desc not in current_desc:
                        self.graph.nodes[entity_name]["description"] = f"{current_desc}; {new_desc}".strip('; ')

        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            if source and target and rel_type:
                if not self.graph.has_node(source):
                    self.graph.add_node(source, type="Unknown", description="Extracted from relationship")
                if not self.graph.has_node(target):
                    self.graph.add_node(target, type="Unknown", description="Extracted from relationship")

                self.graph.add_edge(source, target, type=rel_type, description=rel.get("description"))

        for claim in claims:
            for entity in entities:
                entity_name = entity.get("name")
                if entity_name and entity_name.lower() in claim.lower():
                    if 'claims' not in self.graph.nodes[entity_name]:
                        self.graph.nodes[entity_name]['claims'] = []
                    if claim not in self.graph.nodes[entity_name]['claims']:# Check for duplicates
                        self.graph.nodes[entity_name]['claims'].append(claim)

    def get_graph(self) -> nx.DiGraph:
        """Returns the constructed NetworkX graph."""
        return self.graph