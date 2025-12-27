# src/app/endpoints/chat.py
import time
import base64
from fastapi import APIRouter, HTTPException
from app.logger import logger
from schemas.request import GeminiRequest, OpenAIChatRequest
from app.services.gemini_client import get_gemini_client
from app.services.kagi_client import get_kagi_client
from app.services.session_manager import get_translate_session_manager

router = APIRouter()

# Map OpenAI-style Kagi model names to Kagi's internal names
KAGI_MODEL_MAP = {
    "kagi-quick": "ki_quick",
    "kagi-research": "ki_research",
    "kagi-code": "ki_code",
    "kagi-chat": "ki_chat",
}


def is_kagi_model(model: str) -> bool:
    """Check if the model is a Kagi model."""
    return model and model.startswith("kagi-")


def parse_openai_messages(messages: list) -> tuple[str, bytes | None, str | None]:
    """
    Parse OpenAI-format messages and extract text content and optional image.
    Returns (text_content, image_bytes, image_mime_type)
    """
    text_parts = []
    image_data = None
    image_mime = None
    
    for msg in messages:
        if msg.get("role") != "user":
            continue
            
        content = msg.get("content")
        
        # Simple string content
        if isinstance(content, str):
            text_parts.append(content)
            continue
        
        # Array content (OpenAI vision format)
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                    
                    # Handle base64 data URLs
                    if url.startswith("data:"):
                        # Format: data:image/png;base64,<data>
                        try:
                            header, b64_data = url.split(",", 1)
                            # Extract mime type from header
                            if ";" in header:
                                image_mime = header.split(":")[1].split(";")[0]
                            else:
                                image_mime = "image/png"
                            image_data = base64.b64decode(b64_data)
                        except Exception:
                            pass
    
    return " ".join(text_parts), image_data, image_mime

@router.post("/translate")
async def translate_chat(request: GeminiRequest):
    gemini_client = get_gemini_client()
    session_manager = get_translate_session_manager()
    if not gemini_client or not session_manager:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized.")
    try:
        # This call now correctly uses the fixed session manager
        response = await session_manager.get_response(request.model, request.message, request.files)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")

def convert_to_openai_format(response_text: str, model: str, references: list = None, reasoning: str = None, stream: bool = False):
    result = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    
    if references:
        result["references"] = references
    if reasoning:
        result["reasoning"] = reasoning
    
    return result

@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    is_stream = request.stream if request.stream is not None else False
    
    if not request.model:
        raise HTTPException(status_code=400, detail="Model not specified in the request.")
    
    # Parse messages to extract text and optional image
    text_content, image_data, image_mime = parse_openai_messages(request.messages)
    
    if not text_content:
        raise HTTPException(status_code=400, detail="No user message found.")
    
    # Route to Kagi if model starts with "kagi-"
    if is_kagi_model(request.model):
        kagi_client = get_kagi_client()
        if not kagi_client:
            raise HTTPException(status_code=503, detail="Kagi client is not initialized.")
        
        # Map model name to Kagi internal name
        kagi_model = KAGI_MODEL_MAP.get(request.model, "ki_quick")
        
        try:
            if image_data:
                result = await kagi_client.generate_content_with_image(
                    message=text_content,
                    image=image_data,
                    image_content_type=image_mime or "image/png",
                    model=kagi_model
                )
            else:
                result = await kagi_client.generate_content(
                    message=text_content,
                    model=kagi_model
                )
            
            return convert_to_openai_format(
                result["content"],
                request.model,
                result.get("references"),
                result.get("reasoning"),
                is_stream
            )
        except Exception as e:
            logger.error(f"Error in /v1/chat/completions (Kagi): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")
    
    # Default to Gemini
    gemini_client = get_gemini_client()
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized.")
    
    try:
        response = await gemini_client.generate_content(message=text_content, model=request.model, files=None)
        return convert_to_openai_format(response.text, request.model, stream=is_stream)
    except Exception as e:
        logger.error(f"Error in /v1/chat/completions (Gemini): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")
