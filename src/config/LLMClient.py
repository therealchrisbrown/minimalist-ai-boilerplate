from typing import Any, Type, List, Dict
import instructor
from config.settings import get_settings
from openai import OpenAI
from pydantic import BaseModel, Field

class LLMClient:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "llama": lambda s: instructor.from_openai(
                OpenAI(base_url=s.base_url, api_key=s.api_key), mode=instructor.Mode.JSON)
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError("Unsupported provider: {self.provider}")
    
    def create_completion(self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs)-> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages
        }
        return self.client.chat.completions.create(**completion_params)
    
if __name__ == "__main__":

    class CompletionModel(BaseModel):
        reasoning: str = Field(description="Reasoning")