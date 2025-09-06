import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import os
import argparse
os.environ["WANDB_DISABLED"] = "true"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune LLaMA model with optional LoRA.")
parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning.")
args = parser.parse_args()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
with open("data/attribute.json", "r") as f:
    attribute_data = json.load(f)
with open("data/cfe.json", "r") as f:
    cfe_data = json.load(f)
data = attribute_data + cfe_data

# Prepare the dataset for fine-tuning
def format_example(example):
    prompt = f"""<|begin_of_text|>User: {example['user']} <|end_of_text|>
Assistant: {example['parsed']} <|end_of_text|>"""
    return {"text": prompt}

# Convert to Hugging Face Dataset
formatted_data = [format_example(item) for item in data]
dataset = Dataset.from_list(formatted_data)

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize the dataset and prepare labels
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    # Create labels by copying input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split dataset into train and eval
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Configure LoRA for efficient fine-tuning
if not args.no_lora:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("Using LoRA for fine-tuning.")
else:
    print("Fine-tuning without LoRA.")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    bf16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./llama_finetuned_32/final")
tokenizer.save_pretrained("./llama_finetuned_32/final")