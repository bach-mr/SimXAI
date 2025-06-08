import torch
from typing import Optional

class TextGenerator:
    def __init__(self, model, tokenizer, device):
        """
        Initialize the text generator.
        
        Args:
            model: The loaded model
            tokenizer: The loaded tokenizer
            device: The device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def format_prompt(self, user_input: str) -> str:
        """
        Format the input prompt as it was during training.
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: Formatted prompt
        """
        return f"""<|begin_of_text|>User: {user_input} <|end_of_text|>
Assistant: """
    
    def generate_response(
        self,
        user_input: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate a response for the given input.
        
        Args:
            user_input (str): The user's input text
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated response
        """
        # Format and tokenize input
        prompt = self.format_prompt(user_input)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
        
        # Decode and extract response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = generated_text.split("Assistant: ")[-1].strip()
        
        return response 