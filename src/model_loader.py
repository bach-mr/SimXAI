import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ModelLoader:
    def __init__(self, base_model_name: str, model_path: str, use_lora: bool = False):
        """
        Initialize the model loader.
        
        Args:
            base_model_name (str): Hugging Face model id (e.g., "meta-llama/Meta-Llama-3.3-8B-Instruct")
            model_path (str): Path to LoRA adapter or local model (if use_lora=True)
            use_lora (bool): Whether to apply LoRA adapters from model_path
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.model_path = model_path
        self.use_lora = use_lora
        
    def load_model_and_tokenizer(self):
        """
        Load the base Llama model from Hugging Face, optionally merge LoRA adapters, and return tokenizer.
        
        Returns:
            tuple: (model, tokenizer)
        """
        # Load tokenizer from the base model on Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Optionally load LoRA adapters
        if self.use_lora:
            model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            model = base_model
        
        model.eval()
        return model, tokenizer