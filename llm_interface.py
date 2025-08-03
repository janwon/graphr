import os
import json
import time
from enum import Enum
from openai import AzureOpenAI, OpenAIError
from typing import Dict, Any, List, Union, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field # Import Pydantic classes

load_dotenv()

class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

class Message:
    """A simplified Message class to mimic the one used in atlas.py."""
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content

    def to_string(self) -> Dict[str, str]:
        """Converts the message to a dictionary format suitable for OpenAI API."""
        return {"role": self.role.value, "content": self.content}

# --- Pydantic Models for Structured Output ---
class Entity(BaseModel):
    """Represents an extracted entity."""
    name: str = Field(..., description="The name of the entity.")
    type: str = Field(..., description="The type of the entity (e.g., Person, Organization, Location, Concept).")
    description: str = Field(..., description="A comprehensive description of the entity's attributes and activities.")

class Relationship(BaseModel):
    """Represents an extracted relationship between two entities."""
    source: str = Field(..., description="The name of the source entity.")
    target: str = Field(..., description="The name of the target entity.")
    type: str = Field(..., description="The type of relationship (e.g., FOUNDED, WORKS_FOR, LOCATED_IN).")
    description: str = Field(..., description="A description of the relationship.")

class ExtractedData(BaseModel):
    """
    Represents the complete structured data extracted from text,
    including entities, relationships, and claims.
    """
    entities: List[Entity] = Field(..., description="A list of extracted entities.")
    relationships: List[Relationship] = Field(..., description="A list of extracted relationships.")
    claims: List[str] = Field(..., description="A list of concise factual statements or claims.")

# --- End Pydantic Models ---

class LLMClient:
    """
    A client to interact with an LLM, based on the provided Atlas class structure
    and using Azure OpenAI.
    """
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini-20240718-gs",
                 temperature: float = 0.3, max_output_tokens: int = 4096,
                 system_prompt: Optional[str] = None,
                 embedding_model_name: str = "text-embedding-ada-002"): # Added embedding model
        self.model = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.embedding_model = embedding_model_name # Store embedding model name
        self.current_token_count = 0
        
        azure_openai_api_key: str = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        print(f"LLMClient Initializing with API Version: {azure_api_version}, Endpoint: {azure_endpoint}") # Added for debugging
        
        auth_headers = {}
        auth_header_name = os.getenv("AUTH_HEADER")
        if auth_header_name and azure_openai_api_key:
            auth_headers[auth_header_name] = azure_openai_api_key

        self.client = AzureOpenAI(
                api_key = azure_openai_api_key,
                api_version = azure_api_version,
                default_headers=auth_headers,
                azure_endpoint = azure_endpoint
            )
        
        self.system_message = Message(role=Role.SYSTEM, content=system_prompt or "You are a helpful AI assistant.")
        self.chat_history: List[Message] = []

    def _prepare_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """Prepares messages including system prompt and chat history for the API call."""
        messages = [self.system_message.to_string()]
        for msg in self.chat_history:
            messages.append(msg.to_string())
        messages.append(Message(role=Role.USER, content=user_prompt).to_string())
        return messages

    def generate_text(self, prompt: str) -> Optional[str]:
        """
        Generates text based on a given prompt.
        """
        self.chat_history.append(Message(role=Role.USER, content=prompt))
        
        options = {
            "model": self.model,
            "messages": self._prepare_messages(prompt),
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens
        }

        try:
            response = self.client.chat.completions.create(**options)
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                self.chat_history.append(Message(role=Role.ASSISTANT, content=generated_text))
                return generated_text
            return None
        except OpenAIError as e:
            print(f"OpenAI API error during text generation: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during text generation: {e}")
            return None

    def generate_structured_output(self, prompt: str, pydantic_schema_class: type[BaseModel]) -> Optional[Dict[str, Any]]:
        """
        Generates structured JSON output based on a given prompt and a Pydantic schema class.
        The Pydantic schema is converted to JSON schema and embedded in the prompt.
        """
        json_schema = pydantic_schema_class.model_json_schema()

        prompt_with_schema = (
            f"{prompt}\n\n"
            f"Please generate a JSON object that strictly adheres to the following JSON schema:\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
            f"Only return the JSON object, do not include any other text or markdown formatting outside the JSON."
        )

        self.chat_history.append(Message(role=Role.USER, content=prompt_with_schema))

        options = {
            "model": self.model,
            "messages": self._prepare_messages(prompt_with_schema),
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens
        }

        try:
            response = self.client.chat.completions.create(**options)
            if response.choices and len(response.choices) > 0:
                json_string = response.choices[0].message.content
                try:
                    parsed_json = json.loads(json_string)
                    self.chat_history.append(Message(role=Role.ASSISTANT, content=json_string))
                    return parsed_json
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from LLM response: {e}. Raw response: {json_string}")
                    return None
            return None
        except OpenAIError as e:
            print(f"OpenAI API error during structured generation: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during structured generation: {e}")
            return None

    def get_embedding(self, text: str) -> Optional[List[float]]: # Renamed from get_embedding
        """
        Generates an embedding vector for the given text using the specified embedding model.
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model # Use the configured embedding model
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return None
        except OpenAIError as e:
            print(f"OpenAI API error during embedding generation: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during embedding generation: {e}")
            return None

    def clear_chat_history(self):
        """Clears the current chat history."""
        self.chat_history = []

# The ENTITY_RELATIONSHIP_SCHEMA is now represented by the ExtractedData Pydantic model.
# You will use ExtractedData directly in your processing_pipeline.py and test_llm.py.