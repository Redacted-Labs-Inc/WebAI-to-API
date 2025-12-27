# src/app/endpoints/chat.py
import time
import base64
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.logger import logger
from schemas.request import GeminiRequest, OpenAIChatRequest
from app.services.gemini_client import get_gemini_client
from app.services.kagi_client import get_kagi_client
from app.services.session_manager import get_translate_session_manager
from typing import Optional

router = APIRouter()

# In-memory file storage (for uploaded files)
_uploaded_files: dict[str, dict] = {}

# Available models for /v1/models endpoint
AVAILABLE_MODELS = [
    {"id": "kagi-quick", "object": "model", "owned_by": "kagi", "created": 1700000000},
    {"id": "kagi-research", "object": "model", "owned_by": "kagi", "created": 1700000000},
    {"id": "kagi-code", "object": "model", "owned_by": "kagi", "created": 1700000000},
    {"id": "kagi-chat", "object": "model", "owned_by": "kagi", "created": 1700000000},
]


@router.get("/v1/models")
@router.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": AVAILABLE_MODELS
    }


@router.get("/v1/models/{model_id}")
@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific model (OpenAI-compatible)."""
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@router.post("/v1/files")
@router.post("/files")
async def upload_file(file: UploadFile = File(...), purpose: str = Form("assistants")):
    """Upload a file (OpenAI-compatible)."""
    file_id = f"file-{uuid.uuid4().hex[:24]}"
    content = await file.read()
    
    _uploaded_files[file_id] = {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": file.filename,
        "purpose": purpose,
        "content": content,
        "content_type": file.content_type,
    }
    
    return {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": file.filename,
        "purpose": purpose,
    }


@router.get("/v1/files")
@router.get("/files")
async def list_files():
    """List uploaded files."""
    return {
        "object": "list",
        "data": [
            {k: v for k, v in f.items() if k != "content"}
            for f in _uploaded_files.values()
        ]
    }


@router.get("/v1/files/{file_id}")
@router.get("/files/{file_id}")
async def get_file(file_id: str):
    """Get file metadata."""
    if file_id not in _uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    f = _uploaded_files[file_id]
    return {k: v for k, v in f.items() if k != "content"}


@router.delete("/v1/files/{file_id}")
@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file."""
    if file_id not in _uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    del _uploaded_files[file_id]
    return {"id": file_id, "object": "file", "deleted": True}


def get_file_content(file_id: str) -> tuple[bytes, str] | None:
    """Get file content by ID."""
    if file_id in _uploaded_files:
        f = _uploaded_files[file_id]
        return f["content"], f.get("content_type", "application/octet-stream")
    return None

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
                        try:
                            header, b64_data = url.split(",", 1)
                            if ";" in header:
                                image_mime = header.split(":")[1].split(";")[0]
                            else:
                                image_mime = "image/png"
                            image_data = base64.b64decode(b64_data)
                        except Exception:
                            pass
                elif part.get("type") == "image_file":
                    # Handle file_id reference
                    file_id = part.get("image_file", {}).get("file_id", "")
                    file_content = get_file_content(file_id)
                    if file_content:
                        image_data, image_mime = file_content
                elif part.get("type") == "file":
                    # Alternative file format
                    file_id = part.get("file", {}).get("file_id", "")
                    file_content = get_file_content(file_id)
                    if file_content:
                        image_data, image_mime = file_content
        
        # Check for attachments field (some clients use this)
        attachments = msg.get("attachments", [])
        for attachment in attachments:
            file_id = attachment.get("file_id", "")
            file_content = get_file_content(file_id)
            if file_content:
                image_data, image_mime = file_content
    
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
