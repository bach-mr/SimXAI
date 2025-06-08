from src.model.base_model_loader import BaseModelLoader

class TrainingModelLoader(BaseModelLoader):
    def __init__(self, base_model_name: str, model_path: str = None):
        """
        Initialize the training model loader.
        
        Args:
            base_model_name (str): Name of the base model
            model_path (str, optional): Path to the fine-tuned model weights. If None, starts with base model.
        """
        super().__init__(base_model_name, model_path)
        
    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer for training.
        If model_path is provided, loads with LoRA weights, otherwise loads base model.
        
        Returns:
            tuple: (model, tokenizer)
        """
        if self.model_path:
            return self.load_model_with_lora()
        return self.load_base_model() 