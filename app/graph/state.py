import logging
from typing import List, Optional, Annotated
from operator import add
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class BusinessInfo(BaseModel):
    """
    Modelo de datos para la información clave de la PYME.
    """
    business_name: Optional[str] = Field(None, description="Nombre del negocio.")
    industry: Optional[str] = Field(None, description="Industria o sector del negocio.")
    """
    Información fáctica y descriptiva del negocio.
    El flag `total=False` indica que no todos los campos son requeridos.
    """
    productos_servicios_principales: Optional[List[str]]
    desafios_principales: Optional[List[str]]
    ubicacion: Optional[str]  # e.g., "Lima, Perú", "Online", "Nacional"
    descripcion_negocio: Optional[str]
    anos_operacion: Optional[int]
    num_empleados: Optional[int]
    limitaciones_recursos: Optional[List[str]]
    obstaculos_crecimiento: Optional[List[str]]
    metas_financieras: Optional[List[str]]
    objetivos_principales: Optional[List[str]]
    expansion_deseada: Optional[List[str]]
    timeline_objetivo: Optional[str]
    desafios_actuales: Optional[List[str]]
    recursos_necesarios: Optional[List[str]]
    riesgos_identificados: Optional[List[str]]


# --- Modelos para el Plan Estratégico ---
class ActionStep(BaseModel):
    """Un paso concreto dentro de un plan de acción."""
    step_number: int = Field(..., description="Número secuencial del paso.")
    title: str = Field(..., description="Título corto y claro del paso. Ej: 'Investigar proveedores locales'.")
    description: str = Field(..., description="Descripción detallada de qué hacer en este paso.")
    estimated_cost_usd: float = Field(0.0, description="Costo estimado en USD para este paso. 0.0 si no hay costo directo.")
    estimated_timeline_days: int = Field(..., description="Duración estimada en días para completar este paso.")

class SavingsTactic(BaseModel):
    """Una táctica específica para generar ahorros en el negocio."""
    tactic: str = Field(..., description="Descripción de la táctica de ahorro. Ej: 'Reducir el desperdicio de ingredientes en un 10%'.")
    estimated_monthly_savings_usd: float = Field(..., description="Ahorro mensual estimado en USD que esta táctica puede generar.")
    implementation_notes: str = Field(..., description="Cómo implementar esta táctica.")

class StrategicPlan(BaseModel):
    """Modelo completo para un plan de acción estratégico y su plan de ahorro asociado."""
    initiative_name: str = Field(..., description="Nombre claro y conciso de la iniciativa. Ej: 'Lanzamiento de Pizzas con Ingredientes Locales'.")
    action_plan: List[ActionStep] = Field(..., description="Lista secuencial de pasos a seguir para implementar la iniciativa.")
    savings_plan: List[SavingsTactic] = Field(..., description="Lista de tácticas para generar los ahorros necesarios para financiar el plan de acción.")
    summary: str = Field(..., description="Un resumen ejecutivo que explica cómo el plan de ahorro financia el plan de acción y ayuda a alcanzar el objetivo.")


class PYMESState(TypedDict):
    """
    Representa el estado completo de la conversación y el análisis de la PYME.

    Attributes:
        messages: La secuencia de mensajes de la conversación, gestionada por LangGraph.
        business_info: Un diccionario estructurado con la información de la empresa.
        long_term_memory: Una lista de insights clave para recordar entre sesiones.
        current_plan: Nuevo campo para el plan actual
    """
    messages: Annotated[List[BaseMessage], add_messages]
    business_info: BusinessInfo
    long_term_memory: Annotated[List[str], add]
    current_plan: Optional[StrategicPlan]
