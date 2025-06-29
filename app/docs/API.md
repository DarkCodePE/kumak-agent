# reference API

La API de KUMAK está construida con **FastAPI** y proporciona varios endpoints para interactuar con el sistema. A continuación se detallan los endpoints más importantes:

## Endpoints Principales

### 1. Webhook de WhatsApp

- **Endpoint**: `POST /whatsapp/webhook`
- **Descripción**: Recibe los mensajes entrantes de WhatsApp, los procesa a través del orquestador de agentes y envía las respuestas correspondientes.
- **Seguridad**: Utiliza un token de verificación (`WHATSAPP_VERIFY_TOKEN`) para asegurar que las solicitudes provengan de Meta (Facebook).

### 2. Chat Directo

- **Endpoint**: `POST /chat/message`
- **Descripción**: Permite interactuar directamente con el agente sin necesidad de usar WhatsApp. Es útil para pruebas, integraciones con otras plataformas o para ofrecer un chat web.
- **Cuerpo de la Solicitud**:
  ```json
  {
    "message": "Tu mensaje aquí",
    "thread_id": "un_identificador_unico_de_conversacion"
  }
  ```

### 3. Carga de Documentos

- **Endpoint**: `POST /documents/upload`
- **Descripción**: Sube documentos para enriquecer la base de conocimiento del agente de consultoría. Los documentos se procesan y almacenan en el vector store de **Qdrant**.
- **Cuerpo de la Solicitud**: `multipart/form-data` con el archivo a subir.

### 4. Health Check

- **Endpoint**: `GET /health`
- **Descripción**: Endpoint de monitoreo que devuelve el estado de salud del servicio. Ideal para sistemas de monitoreo y orquestadores de contenedores como Kubernetes.
- **Respuesta Exitosa**:
  ```json
  {
    "status": "ok"
  }
  ```
