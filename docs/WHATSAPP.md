# 📱 Integración con WhatsApp

KUMAK se integra con la **API de WhatsApp Business** para ofrecer una experiencia de usuario conversacional y accesible. Esta integración permite a las PYMEs interactuar con los agentes de KUMAK directamente desde la aplicación de mensajería más popular del mundo.

## 🚀 Características de la Integración

- **Webhook Bidireccional**: El sistema está configurado para recibir mensajes entrantes y enviar respuestas a través de un webhook seguro.
- **Respuestas Interactivas**: Se pueden enviar mensajes con botones y plantillas para facilitar la interacción con el usuario.
- **Gestión de Límite de Tokens**: Para cumplir con las restricciones de WhatsApp, los mensajes largos se dividen en fragmentos de 150 tokens, asegurando que la información se entregue de manera clara y concisa.

## 🔧 Configuración

Para configurar la integración con WhatsApp, necesitas las siguientes variables de entorno en tu archivo `.env`:

```env
WHATSAPP_TOKEN=tu-token-de-api
WHATSAPP_PHONE_NUMBER_ID=tu-id-de-numero-de-telefono
WHATSAPP_VERIFY_TOKEN=tu-token-de-verificacion
```

- `WHATSAPP_TOKEN`: Token de acceso a la API de WhatsApp Business.
- `WHATSAPP_PHONE_NUMBER_ID`: ID del número de teléfono registrado en la API.
- `WHATSAPP_VERIFY_TOKEN`: Token secreto para verificar las solicitudes entrantes del webhook.

## ⚙️ Flujo del Webhook

1.  **Verificación del Webhook**: Cuando configuras el webhook en la plataforma de Meta, se envía una solicitud `GET` con un `hub.verify_token`. El servidor responde con el mismo token para confirmar la autenticidad.
2.  **Recepción de Mensajes**: Los mensajes entrantes de los usuarios se reciben a través de una solicitud `POST` al endpoint `/whatsapp/webhook`.
3.  **Procesamiento y Respuesta**: El mensaje se pasa al **Orquestador Central**, que genera una respuesta. Esta respuesta se envía de vuelta al usuario a través de la API de WhatsApp.
