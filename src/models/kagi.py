# src/models/kagi.py
from typing import Optional, AsyncIterator, Union
from pathlib import Path
import httpx
import json
from app.logger import logger


class KagiClient:
    """
    Client for interacting with Kagi Assistant via the web interface.
    Uses session cookies for authentication.
    """
    
    BASE_URL = "https://kagi.com"
    ASSISTANT_PROMPT_URL = f"{BASE_URL}/assistant/prompt"
    
    def __init__(self, session_token: str, proxy: Optional[str] = None):
        self.session_token = session_token
        self.proxy = proxy
        self._client: Optional[httpx.AsyncClient] = None
    
    async def init(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            proxy=self.proxy,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/event-stream,application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": self.BASE_URL,
                "Referer": f"{self.BASE_URL}/assistant",
            },
            cookies={"kagi_session": self.session_token}
        )
        
        if await self._verify_session():
            logger.info("Kagi client initialized and session verified.")
        else:
            raise Exception("Failed to verify Kagi session. Check your cookies.")
    
    async def _verify_session(self) -> bool:
        """Verify the session is valid by making a test request."""
        try:
            response = await self._client.get(f"{self.BASE_URL}/assistant")
            return response.status_code == 200 and "signup" not in str(response.url)
        except Exception as e:
            logger.error(f"Session verification failed: {e}")
            return False
    
    async def generate_content(
        self,
        message: str,
        thread_id: Optional[str] = None,
        web_access: bool = True,
        model: Optional[str] = None,
        profile_id: Optional[str] = None
    ) -> str:
        """
        Send a message to Kagi Assistant and return the response.
        """
        if not self._client:
            raise Exception("Client not initialized. Call init() first.")
        
        payload = {
            "focus": {
                "thread_id": thread_id,
                "branch_id": "00000000-0000-4000-0000-000000000000",
                "prompt": message
            },
            "profile": {
                "id": profile_id,
                "personalizations": True,
                "internet_access": web_access,
                "model": model or "ki_quick",
                "lens_id": None
            },
            "threads": [
                {
                    "tag_ids": [],
                    "saved": False,
                    "shared": False
                }
            ]
        }
        
        try:
            response = await self._client.post(
                self.ASSISTANT_PROMPT_URL,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Kagi API error: {response.status_code} - {response.text}")
                raise Exception(f"Kagi API returned status {response.status_code}")
            
            return self._parse_response(response.text)
            
        except httpx.TimeoutException:
            raise Exception("Request to Kagi timed out")
        except Exception as e:
            logger.error(f"Error calling Kagi API: {e}")
            raise

    async def generate_content_with_image(
        self,
        message: str,
        image: Union[bytes, str, Path],
        image_filename: str = "image.png",
        image_content_type: str = "image/png",
        thread_id: Optional[str] = None,
        web_access: bool = True,
        model: Optional[str] = None,
        profile_id: Optional[str] = None
    ) -> str:
        """
        Send a message with an image to Kagi Assistant.
        
        Args:
            message: The prompt text
            image: Image as bytes, file path string, or Path object
            image_filename: Filename for the image
            image_content_type: MIME type of the image
            thread_id: Optional thread ID to continue conversation
            web_access: Whether to enable web search
            model: Model to use (default: ki_quick)
            profile_id: Optional profile/assistant ID
        """
        if not self._client:
            raise Exception("Client not initialized. Call init() first.")
        
        # Load image bytes if path provided
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = image
        
        # Build the state JSON
        state = {
            "focus": {
                "thread_id": thread_id,
                "branch_id": "00000000-0000-4000-0000-000000000000",
                "prompt": message
            },
            "profile": {
                "id": profile_id,
                "personalizations": True,
                "internet_access": web_access,
                "model": model or "ki_quick",
                "lens_id": None
            },
            "files": [{}],
            "threads": [
                {
                    "tag_ids": [],
                    "saved": False,
                    "shared": False
                }
            ]
        }
        
        # Kagi expects: state + file + __kagithumbnail (thumbnail)
        files = [
            ("state", ("blob", json.dumps(state).encode(), "application/json")),
            ("file", (image_filename, image_bytes, image_content_type)),
            ("__kagithumbnail", ("blob", image_bytes, image_content_type)),
        ]
        
        # Need the special Accept header for Kagi streaming format
        headers = {"Accept": "application/vnd.kagi.stream"}
        
        try:
            response = await self._client.post(
                self.ASSISTANT_PROMPT_URL,
                files=files,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Kagi API error: {response.status_code} - {response.text}")
                raise Exception(f"Kagi API returned status {response.status_code}")
            
            return self._parse_response(response.text)
            
        except httpx.TimeoutException:
            raise Exception("Request to Kagi timed out")
        except Exception as e:
            logger.error(f"Error calling Kagi API with image: {e}")
            raise

    async def generate_content_stream(
        self,
        message: str,
        thread_id: Optional[str] = None,
        web_access: bool = True,
        model: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Send a message to Kagi Assistant and stream the response.
        """
        if not self._client:
            raise Exception("Client not initialized. Call init() first.")
        
        payload = {
            "focus": {
                "thread_id": thread_id,
                "branch_id": "00000000-0000-4000-0000-000000000000",
                "prompt": message
            },
            "profile": {
                "id": None,
                "personalizations": True,
                "internet_access": web_access,
                "model": model or "ki_quick",
                "lens_id": None
            },
            "threads": [
                {
                    "tag_ids": [],
                    "saved": False,
                    "shared": False
                }
            ]
        }
        
        try:
            async with self._client.stream(
                "POST",
                self.ASSISTANT_PROMPT_URL,
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Kagi API error: {response.status_code} - {error_text}")
                    raise Exception(f"Kagi API returned status {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line:
                        parsed = self._parse_stream_chunk(line)
                        if parsed:
                            yield parsed
                            
        except httpx.TimeoutException:
            raise Exception("Request to Kagi timed out")
        except Exception as e:
            logger.error(f"Error streaming from Kagi API: {e}")
            raise
    
    def _parse_stream_chunk(self, line: str) -> Optional[str]:
        """Parse a streaming chunk from Kagi's response."""
        import re
        import html
        
        # Look for output field in hidden divs
        if '<div hidden>' in line:
            match = re.search(r'<div hidden>({[^<]*"output"[^<]*})</div>', line)
            if match:
                try:
                    decoded = html.unescape(match.group(1))
                    data = json.loads(decoded)
                    return data.get("output", "")
                except (json.JSONDecodeError, KeyError):
                    pass
        return None
    
    def _parse_response(self, response_text: str) -> str:
        """Parse the response from Kagi's HTML/script-based streaming format."""
        import re
        import html
        
        # Look for the final message JSON containing the reply or md (markdown)
        pattern = r'<div hidden>({[^<]+})</div>\s*<script[^>]*>scriptStreamCallback\("new_message\.json"'
        matches = re.findall(pattern, response_text)
        
        for match in reversed(matches):
            try:
                decoded = html.unescape(match)
                data = json.loads(decoded)
                # Prefer markdown if available, otherwise use reply
                if data.get("md"):
                    return data["md"]
                if data.get("reply"):
                    return data["reply"]
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Fallback: look for output chunks (streaming chunks)
        output_pattern = r'<div hidden>({[^<]*"output"[^<]*})</div>'
        output_matches = re.findall(output_pattern, response_text)
        
        output_parts = []
        for match in output_matches:
            try:
                decoded = html.unescape(match)
                data = json.loads(decoded)
                if "output" in data:
                    output_parts.append(data["output"])
            except (json.JSONDecodeError, KeyError):
                continue
        
        if output_parts:
            return "".join(output_parts)
        
        # Strip HTML and return clean text
        clean = re.sub(r'<[^>]+>', '', response_text)
        clean = html.unescape(clean).strip()
        return clean if clean else response_text
    
    def _parse_sse_line(self, line: str) -> Optional[str]:
        """Parse a Server-Sent Events line."""
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                return None
            try:
                data = json.loads(data_str)
                if isinstance(data, dict):
                    return data.get("text", data.get("content", data.get("delta", "")))
                return str(data)
            except json.JSONDecodeError:
                return data_str
        return None
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

