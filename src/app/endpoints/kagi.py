# src/app/endpoints/kagi.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.logger import logger
from schemas.request import KagiRequest
from app.services.kagi_client import get_kagi_client
import json
from typing import Optional

router = APIRouter()


@router.post("/kagi")
async def kagi_generate(request: KagiRequest):
    """
    Generate a response from Kagi Assistant.
    Each request can optionally continue a thread or start fresh.
    """
    kagi_client = get_kagi_client()
    if not kagi_client:
        raise HTTPException(status_code=503, detail="Kagi client is not initialized.")
    
    try:
        response = await kagi_client.generate_content(
            message=request.message,
            thread_id=request.thread_id,
            web_access=request.web_access,
            model=request.model,
            profile_id=request.profile_id
        )
        return {"response": response}
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
    """
    kagi_client = get_kagi_client()
    if not kagi_client:
        raise HTTPException(status_code=503, detail="Kagi client is not initialized.")
    
    try:
        image_bytes = await file.read()
        response = await kagi_client.generate_content_with_image(
            message=message,
            image=image_bytes,
            image_filename=file.filename or "image.png",
            image_content_type=file.content_type or "image/png",
            thread_id=thread_id,
            web_access=web_access,
            model=model,
            profile_id=profile_id
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in /kagi/image endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")


@router.post("/kagi/stream")
async def kagi_generate_stream(request: KagiRequest):
    """
    Generate a streaming response from Kagi Assistant.
    """
    kagi_client = get_kagi_client()
    if not kagi_client:
        raise HTTPException(status_code=503, detail="Kagi client is not initialized.")
    
    async def generate():
        try:
            async for chunk in kagi_client.generate_content_stream(
                message=request.message,
                thread_id=request.thread_id,
                web_access=request.web_access,
                model=request.model
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
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

