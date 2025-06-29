import logging
import os
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, Response, HTTPException
import httpx

from app.services.chat_service import process_message

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
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if verify_token == WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verificado exitosamente")
        return int(challenge)
    else:
        logger.warning("Token de verificación incorrecto")
        raise HTTPException(status_code=403, detail="Token de verificación incorrecto")

@router.post("/webhook")
async def receive_message(request: Request):
    """
    Recibe mensajes de WhatsApp y los procesa usando process_message.
    """
    try:
        body = await request.json()
        logger.info(f"Mensaje recibido de WhatsApp: {body}")
        
        # Verificar que el mensaje tenga la estructura esperada
        if "entry" not in body:
            return {"status": "ok"}
            
        for entry in body["entry"]:
            if "changes" not in entry:
                continue
                
            for change in entry["changes"]:
                if change.get("field") != "messages":
                    continue
                    
                value = change.get("value", {})
                messages = value.get("messages", [])
                
                for message in messages:
                    # Extraer información del mensaje
                    from_number = message.get("from")
                    message_id = message.get("id")
                    timestamp = message.get("timestamp")
                    
                    # Solo procesar mensajes de texto
                    if message.get("type") == "text":
                        text_content = message.get("text", {}).get("body", "")
                        
                        if text_content:
                            # Crear thread_id único para WhatsApp
                            thread_id = f"whatsapp_{from_number}"
                            
                            logger.info(f"Procesando mensaje de WhatsApp - Thread: {thread_id}, Mensaje: {text_content}")
                            
                            # Procesar mensaje usando la misma función que el chat
                            result = process_message(
                                message=text_content,
                                thread_id=thread_id,
                                reset_thread=False
                            )
                            
                            # Enviar respuesta por WhatsApp
                            if result.get("answer"):
                                await send_whatsapp_message(from_number, result["answer"])
                            
                            logger.info(f"Mensaje procesado exitosamente para thread: {thread_id}")
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error procesando mensaje de WhatsApp: {str(e)}")
        return {"status": "error", "message": str(e)}

async def send_whatsapp_message(to_number: str, message: str):
    """
    Envía un mensaje de respuesta por WhatsApp.
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.error("Credenciales de WhatsApp no configuradas")
        return
    
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Truncar mensaje si es muy largo (WhatsApp tiene límite de caracteres)
    max_length = 4000
    if len(message) > max_length:
        message = message[:max_length] + "..."
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {
            "body": message
        }
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"Mensaje enviado exitosamente a {to_number}")
            else:
                logger.error(f"Error enviando mensaje: {response.status_code} - {response.text}")
                
    except Exception as e:
        logger.error(f"Error en send_whatsapp_message: {str(e)}")

# Alias para mantener compatibilidad
whatsapp_router = router