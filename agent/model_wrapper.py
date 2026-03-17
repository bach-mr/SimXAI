"""Wrapper for ToolACE model initialization and inference."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any


class ToolACEModel:
    """Wrapper class for ToolACE model."""
    
    def __init__(self, model_name: str):
        """Initialize the model and tokenizer."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto'
        )
    
    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        """Generate a response from the model."""
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            num_return_sequences=1, 
            eos_token_id=self.tokenizer.eos_token_id
        )
        decoded_inputs = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        )
        
        return response