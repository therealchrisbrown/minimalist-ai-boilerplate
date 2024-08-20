import instructor
from typing import Any
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
                OpenAI(
                    base_url=s.base_url,
                    api_key=s.api_key
                ),
                mode=instructor.Mode.JSON
            )
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError("Unsupported provider: {self.provider}")