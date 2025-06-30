import logging
import os
import json
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, Response, HTTPException
import httpx

from app.services.chat_service import process_message

from app.services.whatsapp_utils import (
    send_whatsapp_message, 
    get_media_url, 
    download_media, 
    transcribe_audio_with_whisper
)

logger = logging.getLogger(__name__)

# Router para WhatsApp
router = APIRouter(
    prefix="/whatsapp",
    tags=["whatsapp"],
)

# Credenciales de WhatsApp API
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

@router.get("/webhook")
async def verify_webhook(request: Request):
    """
    Verificación del webhook de WhatsApp.
    """
    if request.query_params.get("hub.mode") == "subscribe" and request.query_params.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verificado exitosamente.")
        return Response(content=request.query_params["hub.challenge"])
    logger.warning("Fallo en la verificación del webhook.")
    raise HTTPException(status_code=403, detail="Invalid verification token")

@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    logger.info(f"Webhook de WhatsApp recibido: {json.dumps(data, indent=2)}")

    try:
        message_data = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender_id = message_data["from"]
        message_type = message_data["type"]
        
        message_text = ""

        if message_type == "text":
            message_text = message_data["text"]["body"]
            logger.info(f"Mensaje de texto recibido de {sender_id}: '{message_text}'")

        elif message_type == "audio":
            logger.info(f"Mensaje de audio recibido de {sender_id}.")
            await send_whatsapp_message(sender_id, "🎙️ Recibí tu audio, ¡un momento mientras lo transcribo!")
            media_id = message_data["audio"]["id"]
            media_url = await get_media_url(media_id)
            if media_url:
                audio_bytes = await download_media(media_url)
                if audio_bytes:
                    message_text = await transcribe_audio_with_whisper(audio_bytes)
                    logger.info(f"Texto transcrito: '{message_text}'")
                else:
                    logger.error("No se pudo descargar el audio.")
                    await send_whatsapp_message(sender_id, "Lo siento, tuve un problema para descargar tu audio. ¿Podrías intentarlo de nuevo?")
                    return Response(status_code=200)
            else:
                logger.error("No se pudo obtener la URL del audio.")
                await send_whatsapp_message(sender_id, "Lo siento, tuve un problema para procesar tu audio. ¿Podrías intentarlo de nuevo?")
                return Response(status_code=200)

        else:
            logger.warning(f"Tipo de mensaje no manejado: {message_type}")
            await send_whatsapp_message(sender_id, "Lo siento, solo puedo procesar mensajes de texto y audio por ahora.")
            return Response(status_code=200)

        # Si tenemos texto (ya sea original o transcrito), lo procesamos
        if message_text:
            await process_message(
                message=message_text,
                thread_id=sender_id,
            )

        return Response(status_code=200)

    except (KeyError, IndexError) as e:
        logger.info(f"Webhook recibido sin datos de mensaje válidos (probablemente una actualización de estado): {e}")
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Error crítico procesando el webhook de WhatsApp: {e}", exc_info=True)
        return Response(status_code=200)

# Alias para mantener compatibilidad
whatsapp_router = router