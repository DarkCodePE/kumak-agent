from typing import List, Optional, Annotated
from operator import add
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class BusinessInfo(TypedDict, total=False):
    """
    Información fáctica y descriptiva del negocio.
    El flag `total=False` indica que no todos los campos son requeridos.
    """
    nombre_empresa: Optional[str]
    sector: Optional[str]  # e.g., "Restaurantes", "Software (SaaS)", "Retail de moda"
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


class PYMESState(TypedDict):
    """
    Representa el estado completo de la conversación y el análisis de la PYME.

    Attributes:
        messages: La secuencia de mensajes de la conversación, gestionada por LangGraph.
        business_info: Un diccionario estructurado con la información de la empresa.
        long_term_memory: Una lista de insights clave para recordar entre sesiones.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    business_info: BusinessInfo
    long_term_memory: Annotated[List[str], add]
