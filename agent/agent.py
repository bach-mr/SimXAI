"""Main agent logic for processing conversations."""

from typing import List, Dict, Any
from model_wrapper import ToolACEModel
from utils.function_parser import parse_function_calls
from utils.tool_executor import execute_function_call


class ToolACEAgent:
    """Agent for processing conversations and executing tool calls."""
    
    def __init__(self, model: ToolACEModel, tools_module, system_prompt: str, tools: List[Dict]):
        """Initialize the agent."""
        self.model = model
        self.tools_module = tools_module
        self.system_prompt = system_prompt
        self.tools = tools
    
    def process_user_message(self, messages: List[Dict[str, str]], user_message: str) -> tuple:
        """Process a user message and execute tool calls if needed."""
        # Add user message
        messages.append({'role': 'user', 'content': user_message})
        
        # Generate response
        response = self.model.generate_response(messages)
        
        # Parse function calls
        function_calls = parse_function_calls(response)
        
        if not function_calls:
            return response, None, messages
        
        # Execute function calls
        results = []
        tool_names = []
        for call in function_calls:
            tool_names.append(call['function'])
            result = execute_function_call(
                self.tools_module, 
                call['function'], 
                call['parameters']
            )
            results.append(f"{call['function']}: {result}")
        
        # Add tool results to messages
        tool_results = "\n".join(results)
        messages.append({"role": "tool", "name": function_calls, "content": tool_results})
        
        # Generate final response with tool results
        final_response = self.model.generate_response(messages)
        messages.append({'role': 'assistant', 'content': final_response})
        
        return final_response, tool_names, messages
    
    def create_initial_messages(self) -> List[Dict[str, str]]:
        """Create initial messages with system prompt."""
        return [
            {'role': 'system', 'content': self.system_prompt.format(functions=self.tools)}
        ]