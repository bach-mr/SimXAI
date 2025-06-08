import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ModelLoader:
    def __init__(self, base_model_name: str, model_path: str):
        """
        Initialize the model loader.
        
        Args:
            base_model_name (str): Name of the base model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            model_path (str): Path to the fine-tuned model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.model_path = model_path
        
    def load_model_and_tokenizer(self):
        """
        Load the base model, LoRA weights, and tokenizer.
        
        Returns:
            tuple: (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, self.model_path)
        model.eval()
        
        return model, tokenizer 