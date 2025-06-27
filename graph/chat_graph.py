import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from database.postgres import get_postgres_saver
from .state import PYMESState
from .nodes import central_orchestrator, PYMES_TOOLS

logger = logging.getLogger(__name__)


def should_continue(state: PYMESState) -> Literal["tools", "__end__"]:
    """
    Decide si continuar con herramientas o finalizar.
    """
    last_message = state["messages"][-1]
    return "tools" if isinstance(last_message, AIMessage) and last_message.tool_calls else "__end__"


def create_chat_graph():
    """
    Crea y compila el grafo del agente KUMAK.
    """
    try:
        workflow = StateGraph(PYMESState)

        # Usamos el ToolNode estándar, que sabe cómo manejar Commands.
        tool_node = ToolNode(PYMES_TOOLS)

        workflow.add_node("agent", central_orchestrator)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "__end__": END}
        )
        workflow.add_edge("tools", "agent")

        checkpointer = get_postgres_saver()
        compiled_graph = workflow.compile(checkpointer=checkpointer)

        logger.info("Grafo de chat para KUMAK compilado exitosamente con ToolNode estándar.")
        return compiled_graph

    except Exception as e:
        logger.error(f"Error fatal al crear el grafo de chat: {e}", exc_info=True)
        raise
