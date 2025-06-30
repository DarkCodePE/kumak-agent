import logging
import json
import asyncio
from operator import add
from typing import List, Dict, Any, Annotated, TypedDict, Literal
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from tavily import TavilyClient

from app.config.settings import LLM_MODEL, TAVILY_API_KEY

logger = logging.getLogger(__name__)

# --- Funciones de Utilidad Asíncronas ---

async def async_tavily_search(query: str, max_results: int = 4) -> Dict[str, Any]:
    """Función asíncrona para realizar búsquedas con Tavily."""
    def sync_search():
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        return tavily.search(query=query, search_depth="advanced", max_results=max_results)
    
    # Ejecutar la operación bloqueante en un hilo separado
    return await asyncio.to_thread(sync_search)


# --- Modelos de Datos Mejorados ---

class ResearchQuery(BaseModel):
    """Una consulta de búsqueda específica y bien diseñada."""
    search_query: str = Field(
        ...,
        description="Consulta de búsqueda específica y bien enfocada para obtener información relevante"
    )
    purpose: str = Field(
        ..., 
        description="El propósito específico de esta consulta en el contexto de la investigación general"
    )

class ResearchPlan(BaseModel):
    """Un plan de investigación detallado y estratégico."""
    queries: List[ResearchQuery] = Field(
        ...,
        description="Una lista de 3 a 5 consultas de búsqueda estratégicas, específicas y complementarias"
    )
    research_focus: str = Field(
        ...,
        description="El enfoque principal de la investigación y qué aspectos se van a cubrir"
    )

class ResearchResult(BaseModel):
    """El resultado enriquecido de una búsqueda."""
    query: str
    purpose: str
    content: str
    key_insights: List[str] = Field(default_factory=list, description="Insights clave extraídos de esta búsqueda")

class QualityFeedback(BaseModel):
    """Retroalimentación sobre la calidad de la investigación."""
    grade: Literal["pass", "fail"] = Field(
        ...,
        description="Evaluación de si la investigación es suficiente (pass) o necesita más trabajo (fail)"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Aspectos específicos que faltan y necesitan más investigación"
    )
    follow_up_queries: List[ResearchQuery] = Field(
        default_factory=list,
        description="Consultas específicas para abordar las deficiencias identificadas"
    )

class FinalReport(BaseModel):
    """El informe final estructurado y profesional."""
    title: str = Field(..., description="Título claro y específico del informe")
    executive_summary: str = Field(
        ..., 
        description="Resumen ejecutivo de 100-150 palabras con los hallazgos más importantes"
    )
    detailed_analysis: str = Field(
        ...,
        description="Análisis detallado de 400-600 palabras que sintetiza toda la información"
    )
    key_insights: List[str] = Field(
        ...,
        description="Lista de 5-7 insights clave extraídos de la investigación"
    )
    recommendations: List[str] = Field(
        ...,
        description="Lista de 3-5 recomendaciones accionables basadas en los hallazgos"
    )
    methodology: str = Field(
        ...,
        description="Descripción breve de la metodología de investigación utilizada"
    )
    sources_summary: str = Field(
        ...,
        description="Resumen de las fuentes consultadas y su relevancia"
    )

# --- Estado del Subgrafo Mejorado ---

class ResearchState(TypedDict):
    """Estado completo del proceso de investigación."""
    topic: str
    plan: ResearchPlan
    search_results: Annotated[List[ResearchResult], add]
    quality_check: QualityFeedback
    research_iterations: int
    report: FinalReport

# --- Prompts Mejorados ---

PLANNER_PROMPT = """Eres un planificador de investigación experto especializado en crear estrategias de investigación profunda y estructurada.

<Objetivo de Investigación>
{topic}
</Objetivo de Investigación>

<Tarea>
Tu objetivo es generar un plan de investigación estratégico que incluya:

1. **Consultas de Búsqueda Estratégicas**: Crea 3-5 consultas específicas y complementarias que:
   - Cubran diferentes aspectos del tema desde múltiples ángulos
   - Sean lo suficientemente específicas para obtener fuentes de alta calidad
   - Eviten redundancia (no generes consultas similares como "beneficios de X" y "ventajas de X")
   - No mencionen información que pueda estar desactualizada a menos que sea específicamente relevante

2. **Enfoque de Investigación**: Define claramente qué aspectos del tema se van a investigar y por qué

Ejemplos de buenas consultas:
- "Tendencias innovadoras en gastronomía peruana 2024 ingredientes autóctonos"
- "Técnicas modernas fusión cocina tradicional peruana restaurantes exitosos"
- "Mercado internacional ingredientes peruanos exportación quinoa kiwicha"

Evita consultas vagas como "gastronomía peruana" o redundantes como "beneficios quinoa" + "ventajas quinoa".
</Tarea>

Fecha de hoy: {today}
"""

RESEARCHER_PROMPT = """Eres un investigador especializado que ejecuta búsquedas web específicas como parte de un equipo de investigación.

<Contexto de la Investigación>
Tema Principal: {topic}
Propósito de esta Búsqueda: {purpose}
Consulta Específica: {query}
</Contexto de la Investigación>

<Tarea>
1. **Ejecuta la búsqueda** usando la consulta proporcionada
2. **Analiza los resultados** cuidadosamente para extraer información relevante
3. **Identifica insights clave** que contribuyan al objetivo general de investigación
4. **Filtra y sintetiza** la información más valiosa

Tu trabajo se combinará con otros investigadores para crear un informe completo, así que enfócate en la calidad y relevancia de la información que extraes.
</Tarea>
"""

QUALITY_CHECKER_PROMPT = """Eres un evaluador de calidad especializado en investigación. Tu trabajo es determinar si la investigación realizada es suficiente para crear un informe completo y de alta calidad.

<Tema de Investigación>
{topic}
</Tema de Investigación>

<Resultados de Investigación Actuales>
{research_summary}
</Resultados de Investigación Actuales>

<Criterios de Evaluación>
Evalúa si la investigación PASA todos los criterios:

1. **Cobertura Completa**: ¿Cubre todos los aspectos importantes del tema?
2. **Profundidad Suficiente**: ¿Tiene suficiente detalle para conclusiones sólidas?
3. **Fuentes Diversas**: ¿Incluye perspectivas variadas y fuentes confiables?
4. **Información Actualizada**: ¿La información es reciente y relevante?
5. **Insights Accionables**: ¿Permite generar recomendaciones prácticas?

<Tarea>
- Si la investigación PASA todos los criterios: marca "pass"
- Si la investigación FALLA en algún criterio: marca "fail" y especifica:
  * Aspectos específicos que faltan
  * Consultas de seguimiento para abordar las deficiencias
</Tarea>

Iteración actual: {iteration} de máximo 3.
"""

SYNTHESIZER_PROMPT = """Eres un analista senior especializado en crear informes ejecutivos de alta calidad. Tu trabajo es sintetizar toda la investigación recopilada en un informe estructurado, perspicaz y accionable.

<Tema del Informe>
{topic}
</Tema del Informe>

<Metodología Utilizada>
{methodology}
</Metodología Utilizada>

<Datos de Investigación Completos>
{research_data}
</Datos de Investigación Completos>

<Especificaciones del Informe>

1. **Título**: Claro, específico y profesional
2. **Resumen Ejecutivo**: 100-150 palabras con los hallazgos más críticos
3. **Análisis Detallado**: 400-600 palabras que:
   - Sintetice (no solo enumere) la información
   - Identifique patrones y conexiones
   - Presente insights únicos y valiosos
   - Use evidencia específica de las fuentes
4. **Insights Clave**: 5-7 puntos únicos y accionables
5. **Recomendaciones**: 3-5 acciones específicas y prácticas
6. **Metodología**: Breve descripción del proceso de investigación
7. **Resumen de Fuentes**: Evaluación de la calidad y relevancia de las fuentes

<Estándares de Calidad>
- Lenguaje profesional pero accesible
- Conclusiones respaldadas por evidencia
- Insights únicos que van más allá de lo obvio
- Recomendaciones específicas y accionables
- Estructura lógica y flujo narrativo coherente
</Especificaciones del Informe>

<Fecha de Finalización>
{today}
</Fecha de Finalización>
"""

# --- Nodos Mejorados ---

async def enhanced_planner_node(state: ResearchState) -> Dict[str, Any]:
    """Planificador mejorado que crea estrategias de investigación más sofisticadas."""
    logger.info(f"[Enhanced Research] Planner ejecutándose para: {state['topic']}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_PROMPT),
    ])
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(ResearchPlan)
    chain = prompt | llm
    
    plan = await chain.ainvoke({
        "topic": state["topic"],
        "today": datetime.now().strftime("%Y-%m-%d")
    })
    
    logger.info(f"[Enhanced Research] Plan generado con {len(plan.queries)} consultas estratégicas")
    return {"plan": plan, "research_iterations": 0}

def enhanced_map_queries(state: ResearchState) -> List[Send]:
    """Mapeo mejorado de consultas con contexto enriquecido."""
    logger.info(f"[Enhanced Research] Mapeando {len(state['plan'].queries)} consultas especializadas")
    
    return [
        Send("enhanced_researcher_node", {
            "query": query.search_query,
            "purpose": query.purpose,
            "topic": state["topic"]
        })
        for query in state["plan"].queries
    ]

async def enhanced_researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Investigador mejorado con análisis de calidad y extracción de insights."""
    query = state["query"]
    purpose = state["purpose"]
    topic = state["topic"]
    
    logger.info(f"[Enhanced Research] Investigador ejecutándose: '{query}' (Propósito: {purpose})")
    
    try:
        # Usar la función asíncrona para evitar bloqueos
        response = await async_tavily_search(query=query, max_results=4)
        raw_content = "\n".join([f"- {r['content']}" for r in response.get('results', [])])
        
        # Análisis mejorado con LLM para extraer insights
        analysis_prompt = f"""Analiza el siguiente contenido de búsqueda y extrae insights clave:

Tema: {topic}
Propósito: {purpose}
Consulta: {query}

Contenido:
{raw_content}

Extrae 3-5 insights específicos y valiosos del contenido que sean relevantes para el propósito de investigación."""

        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        insights_response = await llm.ainvoke([{"role": "user", "content": analysis_prompt}])
        insights = [insight.strip() for insight in insights_response.content.split('\n') if insight.strip() and not insight.strip().startswith('#')][:5]
        
        result = ResearchResult(
            query=query,
            purpose=purpose,
            content=raw_content,
            key_insights=insights
        )
        
        logger.info(f"[Enhanced Research] Búsqueda completada con {len(insights)} insights extraídos")
        return {"search_results": [result]}
        
    except Exception as e:
        logger.error(f"Error en investigación mejorada para '{query}': {e}")
        error_result = ResearchResult(
            query=query,
            purpose=purpose,
            content=f"Error al obtener información: {e}",
            key_insights=["Error en la búsqueda - información no disponible"]
        )
        return {"search_results": [error_result]}

async def quality_checker_node(state: ResearchState) -> Dict[str, Any]:
    """Evaluador de calidad que determina si se necesita más investigación."""
    logger.info(f"[Enhanced Research] Evaluando calidad de investigación (Iteración {state.get('research_iterations', 0)})")
    
    research_summary = "\n\n".join([
        f"**Búsqueda**: {res.query}\n**Propósito**: {res.purpose}\n**Insights**: {', '.join(res.key_insights[:3])}"
        for res in state["search_results"]
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUALITY_CHECKER_PROMPT),
    ])
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(QualityFeedback)
    chain = prompt | llm
    
    feedback = await chain.ainvoke({
        "topic": state["topic"],
        "research_summary": research_summary,
        "iteration": state.get("research_iterations", 0) + 1
    })
    
    logger.info(f"[Enhanced Research] Calidad evaluada: {feedback.grade}")
    return {"quality_check": feedback, "research_iterations": state.get("research_iterations", 0) + 1}

async def enhanced_synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """Sintetizador mejorado que crea informes ejecutivos de alta calidad."""
    logger.info("[Enhanced Research] Sintetizando informe final ejecutivo")
    
    research_data = "\n\n".join([
        f"### Investigación: {res.query}\n"
        f"**Propósito**: {res.purpose}\n"
        f"**Contenido**: {res.content}\n"
        f"**Insights Clave**: {'; '.join(res.key_insights)}"
        for res in state["search_results"]
    ])
    
    methodology = f"""Metodología de Investigación:
1. Planificación estratégica con {len(state['plan'].queries)} consultas especializadas
2. Investigación web avanzada usando Tavily API con búsqueda profunda
3. Análisis automatizado de contenido para extracción de insights
4. {state['research_iterations']} iteración(es) de investigación
5. Evaluación de calidad y síntesis final
6. Enfoque: {state['plan'].research_focus}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIZER_PROMPT),
    ])
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1).with_structured_output(FinalReport)
    chain = prompt | llm
    
    report = await chain.ainvoke({
        "topic": state["topic"],
        "methodology": methodology,
        "research_data": research_data,
        "today": datetime.now().strftime("%Y-%m-%d")
    })
    
    logger.info("[Enhanced Research] Informe ejecutivo completado exitosamente")
    return {"report": report}

# --- Enrutamiento Condicional ---



def map_follow_up_queries(state: ResearchState) -> List[Send]:
    """Mapea consultas de seguimiento basadas en evaluación de calidad."""
    max_iterations = 3
    current_iteration = state.get("research_iterations", 0)
    quality_check = state.get("quality_check")
    
    # Solo hacer seguimiento si falló la calidad y no hemos alcanzado el máximo
    if (quality_check and 
        quality_check.grade == "fail" and 
        current_iteration < max_iterations and
        quality_check.follow_up_queries):
        
        logger.info(f"[Enhanced Research] Ejecutando {len(quality_check.follow_up_queries)} consultas de seguimiento")
        
        return [
            Send("enhanced_researcher_node", {
                "query": query.search_query,
                "purpose": query.purpose,
                "topic": state["topic"]
            })
            for query in quality_check.follow_up_queries
        ]
    
    # Si no hay consultas de seguimiento, retornar lista vacía
    return []

def route_after_quality_check(state: ResearchState) -> Literal["enhanced_synthesizer"]:
    """Enrutador que decide el siguiente paso después del quality check."""
    max_iterations = 3
    current_iteration = state.get("research_iterations", 0)
    quality_check = state.get("quality_check")
    
    # Si pasó la calidad o alcanzamos el máximo de iteraciones, ir a síntesis
    if (not quality_check or 
        quality_check.grade == "pass" or 
        current_iteration >= max_iterations or
        not quality_check.follow_up_queries):
        return "enhanced_synthesizer"
    
    # Si falló la calidad y tenemos consultas de seguimiento, las consultas
    # se manejan por map_follow_up_queries, pero este enrutador no se ejecuta
    # en ese caso porque map_follow_up_queries maneja el flujo
    return "enhanced_synthesizer"

# --- Compilador del Grafo Mejorado ---

def create_enhanced_research_graph():
    """Crea el subgrafo de investigación mejorado con control de calidad."""
    workflow = StateGraph(ResearchState)

    # Nodos mejorados
    workflow.add_node("enhanced_planner", enhanced_planner_node)
    workflow.add_node("enhanced_researcher_node", enhanced_researcher_node)
    workflow.add_node("quality_checker", quality_checker_node)
    workflow.add_node("enhanced_synthesizer", enhanced_synthesizer_node)

    # Flujo mejorado
    workflow.set_entry_point("enhanced_planner")
    
    # Mapeo inicial de consultas
    workflow.add_conditional_edges("enhanced_planner", enhanced_map_queries, ["enhanced_researcher_node"])
    
    # Evaluación de calidad
    workflow.add_edge("enhanced_researcher_node", "quality_checker")
    
    # Mapeo de consultas de seguimiento (si es necesario)
    workflow.add_conditional_edges("quality_checker", map_follow_up_queries, ["enhanced_researcher_node"])
    
    # Ruta principal después del quality check
    workflow.add_conditional_edges("quality_checker", route_after_quality_check, ["enhanced_synthesizer"])
    
    # Finalización
    workflow.add_edge("enhanced_synthesizer", END)

    research_graph = workflow.compile()
    logger.info("Subgrafo de investigación mejorado compilado exitosamente")
    
    return research_graph

# Crear instancia del grafo mejorado
create_research_graph = create_enhanced_research_graph 