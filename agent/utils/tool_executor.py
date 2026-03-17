"""Tool execution utilities."""

from typing import Any, Dict, List

# Global variable to store conversation messages
_CONVERSATION_MESSAGES: List[Dict[str, str]] = []


def set_conversation_messages(messages: List[Dict[str, str]]) -> None:
    """Set the global conversation messages."""
    global _CONVERSATION_MESSAGES
    _CONVERSATION_MESSAGES = messages


def get_conversation_messages() -> List[Dict[str, str]]:
    """Get the global conversation messages (filtered to remove tool roles)."""
    global _CONVERSATION_MESSAGES
    return [msg for msg in _CONVERSATION_MESSAGES if msg.get('role') != 'tool']


def execute_function_call(module, func_name: str, params: Dict[str, Any]) -> Any:
    """Execute a function from a module with given parameters.
    
    Args:
        module: Module containing the function
        func_name: Name of the function to execute
        params: Parameters to pass to the function
    """
    if hasattr(module, func_name):
        func = getattr(module, func_name)
        try:
            result = func(**params)
            return result
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"
    else:
        return f"Function {func_name} not found"