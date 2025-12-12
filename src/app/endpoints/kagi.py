# src/app/endpoints/kagi.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.logger import logger
from schemas.request import KagiRequest
from app.services.kagi_client import get_kagi_client
import json
import time
from typing import Optional

router = APIRouter()

DEFAULT_KAGI_MODEL = "ki_quick"


def to_openai_format(content: str, model: str, references: list = None, reasoning: str = None, stream: bool = False) -> dict:
    """Convert a response to OpenAI-compatible format with optional references and reasoning."""
    result = {
        "id": f"kagi-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(time.time()),
        "model": model or DEFAULT_KAGI_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
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
    
    # Add references as a custom field
    if references:
        result["references"] = references
    
    # Add reasoning/planning as a custom field (similar to OpenAI's reasoning_content)
    if reasoning:
        result["reasoning"] = reasoning
    
    return result


@router.post("/kagi")
async def kagi_generate(request: KagiRequest):
    """
    Generate a response from Kagi Assistant.
    Returns OpenAI-compatible format with references.
    """
    kagi_client = get_kagi_client()
    if not kagi_client:
        raise HTTPException(status_code=503, detail="Kagi client is not initialized.")
    
    try:
        result = await kagi_client.generate_content(
            message=request.message,
            thread_id=request.thread_id,
            web_access=request.web_access,
            model=request.model,
            profile_id=request.profile_id
        )
        return to_openai_format(result["content"], request.model, result.get("references"), result.get("reasoning"))
    except Exception as e:
        logger.error(f"Error in /kagi endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")


@router.post("/kagi/image")
async def kagi_generate_with_image(
    message: str = Form(...),
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None),
    web_access: bool = Form(True),
    model: Optional[str] = Form(None),
    profile_id: Optional[str] = Form(None)
):
    """
    Generate a response from Kagi Assistant with an image.
    Returns OpenAI-compatible format with references.
    """
    kagi_client = get_kagi_client()
    if not kagi_client:
        raise HTTPException(status_code=503, detail="Kagi client is not initialized.")
    
    try:
        image_bytes = await file.read()
        result = await kagi_client.generate_content_with_image(
            message=message,
            image=image_bytes,
            image_filename=file.filename or "image.png",
            image_content_type=file.content_type or "image/png",
            thread_id=thread_id,
            web_access=web_access,
            model=model,
            profile_id=profile_id
        )
        return to_openai_format(result["content"], model, result.get("references"), result.get("reasoning"))
    except Exception as e:
        logger.error(f"Error in /kagi/image endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")


@router.post("/kagi/stream")
async def kagi_generate_stream(request: KagiRequest):
    """
    Generate a streaming response from Kagi Assistant.
    Returns OpenAI-compatible SSE stream format.
    """
    kagi_client = get_kagi_client()
    if not kagi_client:
        raise HTTPException(status_code=503, detail="Kagi client is not initialized.")
    
    model = request.model or DEFAULT_KAGI_MODEL
    
    async def generate():
        try:
            async for chunk in kagi_client.generate_content_stream(
                message=request.message,
                thread_id=request.thread_id,
                web_access=request.web_access,
                model=model
            ):
                chunk_data = {
                    "id": f"kagi-{int(time.time() * 1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Final chunk with finish_reason
            final_chunk = {
                "id": f"kagi-{int(time.time() * 1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in /kagi/stream endpoint: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

