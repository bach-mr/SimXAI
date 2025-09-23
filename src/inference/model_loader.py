from model.base_model_loader import BaseModelLoader

class InferenceModelLoader(BaseModelLoader):
    def __init__(self, base_model_name: str, model_path: str):
        """
        Initialize the inference model loader.
        
        Args:
            base_model_name (str): Name of the base model
            model_path (str): Path to the fine-tuned model weights
        """
        super().__init__(base_model_name, model_path)
        
    def load_model_and_tokenizer(self):
        """
        Load the model with LoRA weights and tokenizer for inference.
        
        Returns:
            tuple: (model, tokenizer)
        """
        return self.load_model_with_lora()