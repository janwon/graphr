import networkx as nx
from typing import List, Dict, Any, Tuple
from llm_interface import LLMClient, ExtractedData
from vector_store import VectorStoreManager # Import VectorStoreManager

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
    def __init__(self, llm_client: LLMClient, prompt_template: str, vector_store: VectorStoreManager):
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.vector_store = vector_store # Inject VectorStoreManager

    def extract_entities_relationships_claims(self, chunk_id: str, text_unit: str) -> Dict[str, Any]:
        """
        Uses the LLM to extract structured information and generate embeddings for the chunk.
        Returns a dictionary with 'entities', 'relationships', and 'claims'.
        """
        # Generate embedding for the text chunk
        chunk_embedding = self.llm_client.get_embedding(text_unit)
        if chunk_embedding:
            self.vector_store.add_chunk_embedding(chunk_id, chunk_embedding)
        else:
            print(f"Warning: Could not generate embedding for chunk_id {chunk_id}.")

        # Proceed with structured data extraction
        prompt = self.prompt_template.format(text=text_unit)
        self.llm_client.clear_chat_history()
        extracted_data_model = self.llm_client.generate_structured_output(prompt, ExtractedData)

        if extracted_data_model:
            return extracted_data_model
        return {"entities": [], "relationships": [], "claims": []}

class GraphBuilder:
    """Builds a NetworkX knowledge graph from extracted entities, relationships, and claims."""
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_extracted_data_to_graph(self, extracted_data: Dict[str, Any]):
        """
        Adds entities, relationships, and claims to the graph.
        Entities become nodes. Relationships become edges.
        """
        entities = extracted_data.get("entities", [])
        relationships = extracted_data.get("relationships", [])
        claims = extracted_data.get("claims", [])

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

