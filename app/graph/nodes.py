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

from app.config.settings import LLM_MODEL, TAVILY_API_KEY
from .state import PYMESState, BusinessInfo, StrategicPlan

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
def perform_market_research(tool_call_id: Annotated[str, InjectedToolCallId], query: str) -> Command:
    """
    Realiza una investigación de mercado utilizando una búsqueda web avanzada.
    Útil para encontrar tendencias del sector, análisis de competidores,
    oportunidades de mercado, o cualquier dato externo relevante para la PYME.
    """
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily.search(query=query, search_depth="advanced", max_results=5)
        formatted_results = "\n".join([f"- {r['content']}" for r in response.get('results', [])])
        content = f"Resultados de la investigación de mercado para '{query}':\n{formatted_results}"
        return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]})
    except Exception as e:
        logger.error(f"Error en la investigación de mercado con Tavily: {e}")
        return Command(update={"messages": [ToolMessage(content="Error al realizar la investigación de mercado.", tool_call_id=tool_call_id)]})


# --- Nueva Herramienta Inteligente ---
@tool
def create_action_and_savings_plan(tool_call_id: Annotated[str, InjectedToolCallId], initiative_summary: str, business_info: BusinessInfo) -> Command:
    """
    Crea un plan de acción detallado y un plan de ahorro para una iniciativa de negocio específica.
    Usar después de que una idea ha sido validada con el usuario y se quiere pasar a la ejecución.
    """
    logger.info(f"Ejecutando `create_action_and_savings_plan` para la iniciativa: {initiative_summary}")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)  # Un poco de creatividad para los planes
    structured_llm = llm.with_structured_output(StrategicPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Eres un consultor de negocios senior especializado en PYMEs. Tu tarea es tomar una iniciativa de negocio y convertirla en un `StrategicPlan` accionable.
         El plan debe ser realista, práctico y enfocado en la autofinanciación.

<Task>
1.  **Analiza la `initiative_summary` y el `business_info`**.
2.  **Crea un `action_plan`**: Desglosa la iniciativa en pasos lógicos y secuenciales. Estima costos y tiempos realistas para una PYME.
3.  **Crea un `savings_plan`**: Piensa en formas creativas y prácticas en las que el negocio actual puede ahorrar dinero. Estos ahorros deben estar alineados con los costos del plan de acción. Sé específico. Para un restaurante, sugiere reducir desperdicio, optimizar compras, etc. Para un negocio de servicios, sugiere optimizar software, renegociar con proveedores, etc.
4.  **Escribe un `summary`**: Conecta los puntos. Explica cómo los ahorros generados permitirán ejecutar los pasos del plan de acción para lograr el crecimiento deseado.
</Task>

<Context>
**Información del Negocio:** {business_info}
**Resumen de la Iniciativa a Planificar:** {initiative_summary}
</Context>

Asegúrate de que el plan sea motivador y empodere al dueño de la PYME."""),
    ])

    chain = prompt | structured_llm

    try:
        plan: StrategicPlan = chain.invoke({
            "initiative_summary": initiative_summary,
            "business_info": business_info
        })

        plan_json = plan.model_dump_json(indent=2)
        message_content = f"Plan de acción y ahorro creado exitosamente:\n{plan_json}"
        
        return Command(update={
            "messages": [ToolMessage(content=message_content, tool_call_id=tool_call_id)],
            "current_plan": plan.model_dump()
        })

    except Exception as e:
        logger.error(f"Error en `create_action_and_savings_plan`: {e}", exc_info=True)
        return Command(update={"messages": [ToolMessage(content="Error al generar el plan de acción.", tool_call_id=tool_call_id)]})


PYMES_TOOLS = [analyze_and_synthesize, perform_market_research, create_action_and_savings_plan]


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
Eres KUMAK, un consultor de IA para PYMEs. Tu misión es ser un socio estratégico, guiando a los emprendedores desde la ideación hasta un plan de acción concreto y autofinanciable para hacer crecer su negocio.
</Task>

<Instructions>
1.  **Orquesta con Herramientas (Fase de Ideación)**:
    -   Tu primera acción en cada turno es usar `analyze_and_synthesize` para entender el contexto.
    -   Usa `perform_market_research` si el usuario lo solicita explícitamente para explorar ideas.

2.  **Actúa sobre el Análisis**: La herramienta `analyze_and_synthesize` te dará un "Próximo tema sugerido". Usa este tema para profundizar en los desafíos y oportunidades del negocio, como lo has estado haciendo.

3.  **Transición a la Planificación (El Paso CRUCIAL)**:
    -   Una vez que hayas ayudado al usuario a validar y aterrizar una estrategia o iniciativa clara (ej: "usar ingredientes locales", "lanzar campaña en redes sociales", "implementar programa de lealtad"), NO sigas en un bucle de ideación.
    -   **Realiza la transición PROACTIVAMENTE**. Pregúntale al usuario si desea convertir esa idea en un plan de acción concreto.
    -   **Ejemplo de Transición**: *"Esta idea de usar ingredientes locales suena muy prometedora y creo que tenemos una base sólida. ¿Te gustaría que trabajemos juntos en un plan de acción concreto para hacerlo realidad? Podríamos detallar los pasos, estimar costos iniciales y, lo más importante, pensar en cómo el negocio puede generar los ahorros para financiar esta iniciativa sin necesidad de préstamos."*

4.  **Llama a la Herramienta de Planificación**:
    -   Si el usuario acepta, llama a la nueva herramienta `create_action_and_savings_plan`. Deberás proporcionarle un resumen de la iniciativa.

5.  **Presenta el Plan y Refínalo**:
    -   La herramienta `create_action_and_savings_plan` te devolverá un plan estructurado.
    -   Tu trabajo es presentar este plan al usuario de forma clara y amigable. Explica los pasos de acción, el plan de ahorro y cómo se conectan.
    -   Invita al usuario a discutir y ajustar el plan.

6.  **Cierre y Empoderamiento**: Si no hay más que planificar o discutir, resume los logros y anima al usuario a poner en marcha el plan.
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