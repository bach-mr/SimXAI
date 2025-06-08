from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for model loading."""
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_path: str = "./llama_finetuned/final"

@dataclass
class InferenceConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    max_length: int = 128 