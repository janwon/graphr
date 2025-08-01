import os
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class LLMConfig:
    """Configuration for the Large Language Model."""
    model_name: str = "gpt-4o-mini-20240718-gs"
    api_key: str = ""  # Will be populated by Canvas runtime for Gemini API
    temperature: float = 0.3
    max_output_tokens: int = 4096
    embedding_model_name: str = "text-embedding-ada-002" # Added for embeddings

@dataclass
class DataInputConfig:
    """Configuration for data input."""
    source_type: Literal["file", "cosmosdb"] = "file"
    file_path: Optional[str] = None
    cosmosdb_connection_string: Optional[str] = None
    cosmosdb_database_name: Optional[str] = None
    cosmosdb_container_name: Optional[str] = None
    # Add any other specific CosmosDB parameters like query, partition key, etc.

@dataclass
class ProcessingConfig:
    """Configuration for text and graph processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    # Prompt templates for LLM calls (can be externalized to files later)
    entity_extraction_prompt: str = (
        "Extract entities and their relationships from the following text. "
        "Entities should have a 'name', 'type' (e.g., Person, Organization, Location, Concept), and 'description'. "
        "Relationships should have 'source' (entity name), 'target' (entity name), 'type' (e.g., FOUNDED, WORKS_FOR, LOCATED_IN), and 'description'. "
        "Claims should be concise factual statements related to the text. "
        "Ensure all entities and relationships are clearly defined and relevant to the text provided."
        "\n\nExample:"
        "\nText: 'Alice works at Google. Google is based in Mountain View.'"
        "\nOutput: {{" # Escaped curly brace
        "\n  \"entities\": ["
        "\n    {{\"name\": \"Alice\", \"type\": \"Person\", \"description\": \"An individual named Alice.\"}}," # Escaped curly braces
        "\n    {{\"name\": \"Google\", \"type\": \"Organization\", \"description\": \"A technology company.\"}}," # Escaped curly braces
        "\n    {{\"name\": \"Mountain View\", \"type\": \"Location\", \"description\": \"A city in California.\"}}" # Escaped curly braces
        "\n  ],"
        "\n  \"relationships\": ["
        "\n    {{\"source\": \"Alice\", \"target\": \"Google\", \"type\": \"WORKS_AT\", \"description\": \"Alice is employed by Google.\"}}," # Escaped curly braces
        "\n    {{\"source\": \"Google\", \"target\": \"Mountain View\", \"type\": \"BASED_IN\", \"description\": \"Google has its headquarters in Mountain View.\"}}" # Escaped curly braces
        "\n  ],"
        "\n  \"claims\": ["
        "\n    \"Alice works at Google.\","
        "\n    \"Google is based in Mountain View.\""
        "\n  ]"
        "\n}}" # Escaped curly brace
        "\n\nText: {text}"
    )
    community_summarization_prompt: str = (
        "Summarize the following text, focusing on key entities, relationships, and themes. "
        "The text describes a community from a knowledge graph. "
        "\n\nCommunity Text: {text}"
    )
    answer_generation_prompt: str = (
        "Based on the provided context (knowledge graph insights and community summaries), "
        "answer the following question. If the information is not available, state that. "
        "\n\nQuestion: {question}"
        "\n\nContext: {context}"
    )

@dataclass
class OutputConfig:
    """Configuration for output files."""
    output_dir: str = "output"
    vectors_file: str = "embeddings.json"
    communities_file: str = "communities.json"
    graph_file: str = "knowledge_graph.graphml" # GraphML is a good format for NetworkX

@dataclass
class GraphRAGConfig:
    """Overall configuration for the GraphRAG system."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    data_input: DataInputConfig = field(default_factory=DataInputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(self.output.output_dir, exist_ok=True)

    @classmethod
    def load_from_env(cls):
        """Loads configuration from environment variables."""
        llm_config = LLMConfig(
            model_name=os.getenv("GRAPHRAG_LLM_MODEL", "gpt-4o-mini-20240718-gs"),
            api_key=os.getenv("GRAPHRAG_LLM_API_KEY", ""),
            temperature=float(os.getenv("GRAPHRAG_LLM_TEMPERATURE", 0.3)),
            max_output_tokens=int(os.getenv("GRAPHRAG_LLM_MAX_TOKENS", 4096)),
            embedding_model_name=os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-ada-002") # Loaded from env
        )

        data_input_config = DataInputConfig(
            source_type=os.getenv("GRAPHRAG_DATA_SOURCE_TYPE", "file"),
            file_path=os.getenv("GRAPHRAG_FILE_PATH"),
            cosmosdb_connection_string=os.getenv("GRAPHRAG_COSMOSDB_CONN_STR"),
            cosmosdb_database_name=os.getenv("GRAPHRAG_COSMOSDB_DB_NAME"),
            cosmosdb_container_name=os.getenv("GRAPHRAG_COSMOSDB_CONTAINER_NAME")
        )

        processing_config = ProcessingConfig(
            chunk_size=int(os.getenv("GRAPHRAG_CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", 100)),
            entity_extraction_prompt=ProcessingConfig.entity_extraction_prompt,
            community_summarization_prompt=ProcessingConfig.community_summarization_prompt,
            answer_generation_prompt=ProcessingConfig.answer_generation_prompt
        )

        output_config = OutputConfig(
            output_dir=os.getenv("GRAPHRAG_OUTPUT_DIR", "output"),
            vectors_file=os.getenv("GRAPHRAG_VECTORS_FILE", "embeddings.json"),
            communities_file=os.getenv("GRAPHRAG_COMMUNITIES_FILE", "communities.json"),
            graph_file=os.getenv("GRAPHRAG_GRAPH_FILE", "knowledge_graph.graphml")
        )

        return cls(llm=llm_config, data_input=data_input_config,
                   processing=processing_config, output=output_config)

