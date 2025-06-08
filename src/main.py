import os
os.environ["WANDB_DISABLED"] = "true"

from data.dataset import (
    load_data,
    prepare_dataset,
    tokenize_dataset,
    get_train_eval_datasets
)
from model.config import (
    get_device,
    load_model_and_tokenizer,
    setup_lora
)
from training.trainer import (
    get_training_args,
    setup_trainer,
    train_and_save
)
from model_loader import ModelLoader
from inference import TextGenerator
from config import ModelConfig, InferenceConfig

def main():
    # Load configuration
    model_config = ModelConfig()
    inference_config = InferenceConfig()
    
    # Initialize model loader
    loader = ModelLoader(
        base_model_name=model_config.base_model_name,
        model_path=model_config.model_path
    )
    
    # Load model and tokenizer
    model, tokenizer = loader.load_model_and_tokenizer()
    
    # Initialize text generator
    generator = TextGenerator(model, tokenizer, loader.device)
    
    # Example inputs
    example_inputs = [
        "What are the most important features for this prediction?",
        "Why do you predict this sample?",
        "Explain the prediction using SHAP",
        "What features matter most?"
    ]
    
    # Generate responses
    for user_input in example_inputs:
        response = generator.generate_response(
            user_input,
            max_new_tokens=inference_config.max_new_tokens,
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            do_sample=inference_config.do_sample
        )
        print(f"\nUser: {user_input}")
        print(f"Model Response: {response}")

if __name__ == "__main__":
    main() 