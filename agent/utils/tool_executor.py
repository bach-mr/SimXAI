"""Tool execution utilities."""

from typing import Any, Dict


def execute_function_call(module, func_name: str, params: Dict[str, Any]) -> Any:
    """Execute a function from a module with given parameters."""
    if hasattr(module, func_name):
        func = getattr(module, func_name)
        try:
            result = func(**params)
            return result
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"
    else:
        return f"Function {func_name} not found"