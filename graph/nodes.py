import logging
import json
from typing import Dict, Any, List, Optional, Annotated

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

from config.settings import LLM_MODEL, TAVILY_API_KEY
from .state import PYMESState, BusinessInfo

logger = logging.getLogger(__name__)


# --- Modelo de Salida para la Herramienta de Análisis ---
class AnalysisResult(BaseModel):
    """Modelo de datos para la salida del análisis de conversación."""
    business_info_update: Optional[BusinessInfo] = Field(
        None, description="Diccionario con campos de BusinessInfo nuevos o actualizados."
    )
    key_insights_for_memory: Optional[List[str]] = Field(
        None, description="Lista de insights cruciales para guardar en la memoria a largo plazo."
    )
    next_topic_to_discuss: Optional[str] = Field(
        None, description="El siguiente tema estratégico a discutir con el usuario, NO una pregunta directa. Ej: 'Aclarar diferenciadores de la salsa'."
    )


# --- Herramientas Inteligentes ---
@tool
def analyze_and_synthesize(tool_call_id: Annotated[str, InjectedToolCallId], conversation_history: List[BaseMessage], current_business_info: BusinessInfo) -> Command:
    """
    Analiza la conversación, extrae información, identifica insights y sugiere el próximo tema de conversación.
    """
    logger.info("Ejecutando `analyze_and_synthesize`...")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    structured_llm = llm.with_structured_output(AnalysisResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Eres un sistema de análisis experto. Tu tarea es procesar una conversación para producir una síntesis estructurada.

<Task>
Analiza el `conversation_history` y el `current_business_info`. Luego, genera una instancia de `AnalysisResult`.
1.  **`business_info_update`**: Extrae datos nuevos o actualizados del último mensaje del usuario.
2.  **`key_insights_for_memory`**: Identifica "verdades fundamentales" para recordar a largo plazo.
3.  **`next_topic_to_discuss`**: Basado en el contexto, determina el SIGUIENTE TEMA ESTRATÉGICO a discutir. NO formules una pregunta, solo el tema. Ejemplos: "Aclarar las características de la salsa secreta", "Explorar estrategias de marketing actuales". Si no se necesita más información, déjalo en blanco.
</Task>

<Context>
**Información de Negocio Actual:** {current_business_info}
**Historial de Conversación:** {conversation_history}
</Context>"""),
    ])

    chain = prompt | structured_llm

    try:
        analysis: AnalysisResult = chain.invoke({
            "conversation_history": conversation_history,
            "current_business_info": current_business_info
        })
        
        update_command = {}
        if analysis.business_info_update:
            update_command["business_info"] = analysis.business_info_update
        if analysis.key_insights_for_memory:
            update_command["long_term_memory"] = analysis.key_insights_for_memory
        
        message_content = f"Análisis completado. Próximo tema sugerido: {analysis.next_topic_to_discuss or 'Ninguno'}."
        
        update_command["messages"] = [ToolMessage(content=message_content, tool_call_id=tool_call_id)]
        return Command(update=update_command)

    except Exception as e:
        logger.error(f"Error en `analyze_and_synthesize`: {e}", exc_info=True)
        return Command(update={"messages": [ToolMessage(content="Error durante el análisis.", tool_call_id=tool_call_id)]})


@tool
def perform_market_research(query: str) -> str:
    """
    Realiza una investigación de mercado utilizando una búsqueda web avanzada.
    Útil para encontrar tendencias del sector, análisis de competidores,
    oportunidades de mercado, o cualquier dato externo relevante para la PYME.
    """
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily.search(query=query, search_depth="advanced", max_results=5)
        formatted_results = "\n".join([f"- {r['content']}" for r in response.get('results', [])])
        return f"Resultados de la investigación de mercado para '{query}':\n{formatted_results}"
    except Exception as e:
        logger.error(f"Error en la investigación de mercado con Tavily: {e}")
        return "Error al realizar la investigación de mercado."

PYMES_TOOLS = [analyze_and_synthesize, perform_market_research]


def central_orchestrator(state: PYMESState) -> Dict[str, Any]:
    """
    El director de orquesta. Decide qué herramienta usar y CÓMO hablar con el usuario.
    """
    messages = state["messages"]
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, streaming=True)
    llm_with_tools = llm.bind_tools(PYMES_TOOLS)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """<Task>
Eres KUMAK, un consultor de IA para PYMEs. Tu trabajo es ser un socio estratégico y amigable, guiando la conversación con empatía y propósito.
</Task>

<Instructions>
1.  **Orquesta con Herramientas**: En cada turno, tu primera acción es llamar a `analyze_and_synthesize` para que procese la conversación. Si el usuario pide explícitamente una investigación, usa `perform_market_research`.

2.  **Actúa sobre el Análisis (El Paso Clave)**: La herramienta `analyze_and_synthesize` te devolverá un `ToolMessage` con el resultado de su análisis, incluyendo un "Próximo tema sugerido". Tu trabajo es convertir ese tema en una respuesta humana, amigable y con propósito.

3.  **Crea una Respuesta Amigable y con Propósito**:
    -   **NUNCA** preguntes directamente el tema sugerido.
    -   **SIEMPRE** introduce la pregunta con una frase empática que reconozca lo que el usuario acaba de decir.
    -   **SIEMPRE** justifica por qué necesitas la información, conectándolo con sus objetivos.
    -   **Ejemplo**: Si el tema sugerido es "Aclarar diferenciadores de la salsa secreta", una MALA respuesta sería: "¿Cuáles son las características de tu salsa secreta?". Una BUENA respuesta sería: *"Entendido, ¡una salsa secreta suena como un gran diferenciador! Para que podamos pensar juntos en cómo convertirla en tu arma principal contra la competencia, ¿podrías contarme un poco más sobre qué la hace tan especial?"*

4.  **Si no hay tema que discutir**: Si la herramienta de análisis no sugiere un nuevo tema, significa que tienes suficiente información. Es hora de actuar. Proporciona tu análisis, una recomendación o un plan de acción concreto basado en todo lo que has aprendido.
</Instructions>

<Context>
**Historial de Conversación:** {messages}
**Información Actual del Negocio:** {business_info}
**Memoria a Largo Plazo:** {long_term_memory}
</Context>"""),
    ])

    chain = prompt | llm_with_tools
    
    response = chain.invoke({
        "messages": messages,
        "business_info": str(state.get("business_info", {})),
        "long_term_memory": str(state.get("long_term_memory", []))
    })

    return {"messages": [response]}