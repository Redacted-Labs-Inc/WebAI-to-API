# src/models/gemini.py
from typing import Optional, List, Union
from pathlib import Path

# Try to import gemini-webapi (may not be available on ARM)
try:
    from gemini_webapi import GeminiClient as WebGeminiClient
    GEMINI_WEBAPI_AVAILABLE = True
except ImportError:
    WebGeminiClient = None
    GEMINI_WEBAPI_AVAILABLE = False

from app.config import CONFIG

class MyGeminiClient:
    """
    Wrapper for the Gemini Web API client.
    """
    def __init__(self, secure_1psid: str, secure_1psidts: str, proxy: str | None = None) -> None:
        self.client = WebGeminiClient(secure_1psid, secure_1psidts, proxy)

    async def init(self) -> None:
        """Initialize the Gemini client."""
        await self.client.init()
    async def generate_content(self, message: str, model: str, files: Optional[List[Union[str, Path]]] = None):
        """
        Generate content using the Gemini client.
        """
        return await self.client.generate_content(message, model=model, files=files)

    async def close(self) -> None:
        """Close the Gemini client."""
        await self.client.close()

    def start_chat(self, model: str):
        """
        Start a chat session with the given model.
        """
        return self.client.start_chat(model=model)
