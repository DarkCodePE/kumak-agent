import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Annotated

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

from app.config.settings import LLM_MODEL, TAVILY_API_KEY
from app.graph.research_graph import create_research_graph, create_enhanced_research_graph
from .state import PYMESState, BusinessInfo, StrategicPlan

logger = logging.getLogger(__name__)

# --- Instancia del Subgrafo de Investigación ---
# Se crea dinámicamente cuando se necesita para evitar problemas de importación circular


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

1.  **`business_info_update`**: 
    - SOLO extrae información nueva o actualizada del último mensaje del usuario
    - SOLO incluye los campos que realmente tienen información nueva
    - Si no hay información nueva del negocio, deja este campo como `null`
    - Ejemplo: Si el usuario menciona "tengo una pizzería", solo incluye `{{"business_name": "pizzería", "industry": "restaurantes"}}`

2.  **`key_insights_for_memory`**: 
    - Identifica "verdades fundamentales" para recordar a largo plazo
    - Solo insights realmente importantes, no información básica
    - Si no hay insights significativos, deja como lista vacía

3.  **`next_topic_to_discuss`**: 
    - Basado en el contexto, determina el SIGUIENTE TEMA ESTRATÉGICO a discutir
    - NO formules una pregunta, solo el tema
    - Ejemplos: "Aclarar las características de la salsa secreta", "Explorar estrategias de marketing actuales"
    - Si no se necesita más información, deja en blanco
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



@tool
async def deep_market_research(topic: str) -> str:
    """
    Ejecuta una investigación de mercado profunda usando el subgrafo especializado.
    Utiliza un patrón Map-Reduce avanzado con evaluación de calidad y múltiples iteraciones.
    
    Args:
        topic: El tema específico a investigar en profundidad
        
    Returns:
        Un informe ejecutivo estructurado y detallado con análisis, insights y recomendaciones
    """
    # Crear el subgrafo mejorado
    research_graph = create_enhanced_research_graph()
    
    try:
        # Ejecutar el grafo de investigación con el tema proporcionado
        logger.info(f"[Deep Research] Iniciando investigación profunda para: {topic}")
        
        # Inicializar el estado con todos los campos requeridos
        initial_state = {
            "topic": topic,
            "search_results": []
        }
        
        result = await research_graph.ainvoke(initial_state)
        
        # Extraer el informe final del resultado
        final_report = result.get("report")
        if not final_report:
            return "Error: No se pudo generar el informe de investigación."
        
        # Formatear el informe de manera profesional
        formatted_report = f"""# {final_report.title}

## Resumen Ejecutivo
{final_report.executive_summary}

## Análisis Detallado
{final_report.detailed_analysis}

## Insights Clave
{chr(10).join([f"• {insight}" for insight in final_report.key_insights])}

## Recomendaciones
{chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(final_report.recommendations)])}

## Metodología
{final_report.methodology}

## Fuentes y Calidad de la Información
{final_report.sources_summary}

---
*Informe generado mediante investigación web avanzada con análisis automatizado de calidad*"""

        logger.info(f"[Deep Research] Investigación completada exitosamente. Informe generado con {len(final_report.key_insights)} insights y {len(final_report.recommendations)} recomendaciones.")
        return formatted_report
        
    except Exception as e:
        logger.error(f"Error en investigación profunda: {e}")
        return f"Error al ejecutar la investigación profunda: {str(e)}"


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


PYMES_TOOLS = [analyze_and_synthesize, deep_market_research, create_action_and_savings_plan, perform_market_research]


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
Eres KUMAK, un consultor de IA para PYMEs especializado en estrategias de crecimiento basadas en datos. Tu misión es ser un socio estratégico conversacional que guía a los emprendedores paso a paso, desde entender su situación hasta crear planes de acción concretos.
</Task>

<Instructions>
1.  **Conversa Naturalmente PRIMERO**: 
    - Mantén una conversación natural y empática
    - Entiende la situación del usuario antes de sugerir herramientas
    - Haz preguntas para profundizar en los desafíos específicos
    - Proporciona consejos iniciales basados en tu conocimiento

2.  **Herramientas Disponibles**:
    
    **Investigación Profunda (`deep_market_research`)**: 
    - Solo úsala DESPUÉS de entender la situación y PREGUNTAR al usuario
    - **SIEMPRE pregunta**: "Con los datos que me has proporcionado, ¿te gustaría que inicie una investigación profunda para idear una solución a tu problemática y tu propuesta de crecimiento?"
    - Es una herramienta costosa que genera informes ejecutivos completos

    **Análisis Conversacional (`analyze_and_synthesize`)**:
    - Para extraer insights de conversaciones complejas
    - Solo cuando tengas mucha información que sintetizar

    **Planificación Estratégica (`create_action_and_savings_plan`)**:
    - Cuando el usuario confirme que quiere un plan de acción concreto

3.  **Flujo de Conversación Correcto**:
    ```
    Usuario comparte problema → 
    Tú respondes con empatía y preguntas de profundización → 
    Usuario proporciona más detalles → 
    Tú ofreces consejos iniciales basados en experiencia → 
    Tú PREGUNTAS si quiere investigación profunda → 
    Usuario acepta → 
    Ejecutas deep_market_research → 
    Presentas resultados y PREGUNTAS si quiere plan de acción
    ```

4.  **CRÍTICO - No Ejecutes Herramientas Automáticamente**:
    - NUNCA uses `deep_market_research` sin preguntar primero
    - La investigación profunda cuesta tiempo y recursos
    - El usuario debe confirmar que quiere ese nivel de análisis

5.  **Cuando el Usuario Comparte un Problema**:
    - Escucha con empatía
    - Haz 2-3 preguntas específicas para entender mejor
    - Proporciona 2-3 consejos iniciales basados en buenas prácticas
    - Luego PREGUNTA: "Con los datos que me has proporcionado, ¿te gustaría que inicie una investigación profunda para idear una solución a tu problemática y tu propuesta de crecimiento?"

6.  **Sé Conversacional y Humano**: 
    - No bombardees con información técnica inmediatamente
    - Construye confianza a través de la conversación
    - Muestra que entiendes su situación antes de ofrecer soluciones complejas
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