import logging
import os
import httpx
import tempfile
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# --- Credenciales ---
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Clientes de API ---
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def send_whatsapp_message(to_number: str, message: str):
    """
    Envía un mensaje de texto de respuesta por WhatsApp.
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.error("Credenciales de WhatsApp no configuradas.")
        return

    url = f"https://graph.facebook.com/v20.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message},
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Mensaje enviado exitosamente a {to_number}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error enviando mensaje a WhatsApp: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error inesperado en send_whatsapp_message: {e}")

async def get_media_url(media_id: str) -> str | None:
    """
    Obtiene la URL de descarga temporal de un archivo multimedia a partir de su ID.
    """
    url = f"https://graph.facebook.com/v20.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get("url")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error obteniendo URL del medio: {e.response.status_code} - {e.response.text}")
        return None

async def download_media(media_url: str) -> bytes | None:
    """
    Descarga el contenido de un archivo multimedia desde su URL temporal.
    """
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(media_url, headers=headers)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        logger.error(f"Error descargando medio: {e.response.status_code} - {e.response.text}")
        return None

async def transcribe_audio_with_whisper(audio_bytes: bytes) -> str:
    """
    Transcribe el contenido de audio usando la API de Whisper de OpenAI.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_audio_file:
        tmp_audio_file.write(audio_bytes)
        tmp_audio_file_path = tmp_audio_file.name

    try:
        with open(tmp_audio_file_path, "rb") as audio_file:
            transcription = await aclient.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        logger.info("Audio transcrito exitosamente.")
        return str(transcription)
    except Exception as e:
        logger.error(f"Error durante la transcripción con Whisper: {e}")
        return ""
    finally:
        os.remove(tmp_audio_file_path)

