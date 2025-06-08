import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class BaseModelLoader:
    def __init__(self, base_model_name: str, model_path: str = None):
        """
        Initialize the model loader.
        
        Args:
            base_model_name (str): Name of the base model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            model_path (str, optional): Path to the fine-tuned model weights. If None, only base model is loaded.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.model_path = model_path
        
    def load_base_model(self):
        """
        Load the base model and tokenizer.
        
        Returns:
            tuple: (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path if self.model_path else self.base_model_name
        )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        return model, tokenizer
    
    def load_model_with_lora(self):
        """
        Load the base model with LoRA weights and tokenizer.
        
        Returns:
            tuple: (model, tokenizer)
        """
        if not self.model_path:
            raise ValueError("model_path must be provided to load LoRA weights")
            
        model, tokenizer = self.load_base_model()
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, self.model_path)
        model.eval()
        
        return model, tokenizer 