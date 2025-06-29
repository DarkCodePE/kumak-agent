import logging
from typing import Dict, Any, List, Optional
import traceback
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.types import Command

from app.database.postgres import get_postgres_saver, get_async_postgres_saver
from app.graph.chat_graph import create_chat_graph

logger = logging.getLogger(__name__)

def process_message(
        message: str,
        thread_id: str,
        is_resuming: bool = False,
        reset_thread: bool = False
) -> Dict[str, Any]:
    """
    Process a chat message using the LangGraph workflow.
    Simplified version that works exactly like LangGraph Studio.

    Args:
        message: The user's message
        thread_id: A unique identifier for this conversation thread
        is_resuming: Whether this is resuming after an interrupt
        reset_thread: Whether to reset the thread and start a new conversation

    Returns:
        dict: The result containing answer, status (completed/interrupted), and thread_id
    """
    try:
        # Create the chat graph
        logger.info(f"Creating chat graph for thread {thread_id}")
        graph = create_chat_graph()
        logger.info(f"Chat graph created successfully for thread {thread_id}")

        # Set up configuration with the thread_id
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Create input in the correct format - just like LangGraph Studio
        # The graph expects a simple message input
        graph_input = {"messages": [HumanMessage(content=message)]}

        # Execute the graph
        logger.info(f"Invoking graph for thread {thread_id}")
        result = graph.invoke(graph_input, config)
        logger.info(f"Graph execution completed for thread {thread_id}")

        # Get the final state to extract the response
        state = graph.get_state(config)
        
        # Extract the last AI message from the messages
        final_answer = "No se pudo obtener una respuesta."
        if state.values and "messages" in state.values:
            messages = state.values["messages"]
            # Find the last AI message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    break

        return {
            "thread_id": thread_id,
            "message": message,
            "answer": final_answer,
            "status": "completed"
        }

    except Exception as e:
        error_detail = str(e) if str(e) else "Unknown error (empty exception message)"
        stack_trace = traceback.format_exc()
        logger.error(f"Error processing message: {error_detail}")
        logger.error(f"Stack trace: {stack_trace}")
        return {
            "thread_id": thread_id,
            "message": message,
            "answer": f"Lo siento, encontré un error. Detalles técnicos: {error_detail}",
            "error": error_detail,
            "status": "error"
        }

async def get_chat_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve the chat history for a given thread.

    Args:
        thread_id: The thread ID

    Returns:
        List of message objects with role and content
    """
    try:
        # Create the chat graph to access its API
        logger.info(f"Creating chat graph for thread {thread_id}")
        graph = create_chat_graph()

        # Create a configuration for the thread
        config = {"configurable": {"thread_id": thread_id}}

        # Retrieve the state
        try:
            state_snapshot = graph.get_state(config)
            logger.info(f"Retrieved state snapshot for thread {thread_id}")
        except Exception as e:
            logger.error(f"Error retrieving state from graph: {str(e)}")
            logger.error(traceback.format_exc())
            return []

        # Extract chat history from the values
        chat_history = state_snapshot.values.get("messages", [])

        if not chat_history:
            logger.info(f"Empty chat history for thread {thread_id}")
            return []

        # Format into a more user-friendly structure
        formatted_history = []

        # Process each message in the chat history
        for message in chat_history:
            # Skip any items that don't have the expected structure
            if not hasattr(message, 'content'):
                continue

            if isinstance(message, HumanMessage):
                formatted_history.append({
                    "role": "human",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                formatted_history.append({
                    "role": "ai",
                    "content": message.content
                })
            elif hasattr(message, 'type') and message.type in ["human", "ai"]:
                formatted_history.append({
                    "role": message.type,
                    "content": message.content
                })
            elif hasattr(message, '__class__') and hasattr(message.__class__, '__name__'):
                # Fallback: Determine the role based on message class name
                role = "human" if "Human" in message.__class__.__name__ else "ai"
                formatted_history.append({
                    "role": role,
                    "content": message.content
                })

        logger.info(f"Retrieved {len(formatted_history)} messages for thread {thread_id}")
        return formatted_history

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Error retrieving chat history: {str(e)}")
        logger.error(f"Stack trace: {stack_trace}")
        return []
