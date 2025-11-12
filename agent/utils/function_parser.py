"""Utilities for parsing and extracting function schemas."""

import inspect
import re
from typing import List, Dict, Any


def get_function_schema(func) -> Dict[str, Any]:
    """Extract function schema from function signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse parameters
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        param_type = "string"  # Default type
        param_desc = ""
        
        # Try to infer type from annotation
        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation
            if annotation == str:
                param_type = "string"
            elif annotation == int:
                param_type = "integer"
            elif annotation == float:
                param_type = "number"
            elif annotation == bool:
                param_type = "boolean"
            elif annotation == dict:
                param_type = "dict"
            elif annotation == list:
                param_type = "array"
        
        # Extract description from docstring (simple parsing)
        if f"{param_name}:" in doc:
            desc_start = doc.find(f"{param_name}:")
            desc_end = doc.find("\n", desc_start)
            if desc_end != -1:
                param_desc = doc[desc_start:desc_end].split(":", 1)[1].strip()
        
        properties[param_name] = {
            "type": param_type,
            "description": param_desc
        }
        
        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "name": func.__name__,
        "description": doc.split("\n")[0] if doc else "",
        "arguments": {
            "type": "dict",
            "properties": properties,
            "required": required
        }
    }


def get_available_tools(module) -> List[Dict[str, Any]]:
    """Get all available tools from a module."""
    tools = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and not name.startswith('_'):
            tools.append(get_function_schema(obj))
    return tools


def parse_function_calls(response: str) -> List[Dict[str, Any]]:
    """Parse function calls from model response."""
    # Pattern: [func_name(param1=value1, param2=value2), ...]
    pattern = r'(\w+)\((.*?)\)'
    matches = re.findall(pattern, response)
    
    calls = []
    for func_name, params_str in matches:
        params = {}
        if params_str.strip():
            # Parse parameters - handle both quoted and unquoted values
            param_pairs = re.findall(r"(\w+)=(?:'([^']*)'|\"([^\"]*)\"|([^,\)]+))", params_str)
            for match in param_pairs:
                param_name = match[0]
                # Check which group captured the value
                param_value = match[1] or match[2] or match[3]
                # Clean up the value
                param_value = param_value.strip().strip('"\'')
                # Try to convert to appropriate type
                try:
                    if param_value.isdigit():
                        param_value = int(param_value)
                    elif param_value.replace('.', '').isdigit():
                        param_value = float(param_value)
                except:
                    pass
                params[param_name] = param_value
        
        calls.append({
            'function': func_name,
            'parameters': params
        })
    
    return calls