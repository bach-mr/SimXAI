import json
from datasets import Dataset

def load_data(attribute_path="data/attribute.json", cfe_path="data/cfe.json"):
    """Load and combine attribute and CFE data."""
    with open(attribute_path, "r") as f:
        attribute_data = json.load(f)
    with open(cfe_path, "r") as f:
        cfe_data = json.load(f)
    return attribute_data + cfe_data

def format_example(example):
    """Format a single example into the required prompt format."""
    prompt = f"""<|begin_of_text|>User: {example['user']} <|end_of_text|>
Assistant: {example['parsed']} <|end_of_text|>"""
    return {"text": prompt}

def prepare_dataset(data):
    """Convert raw data into a Hugging Face Dataset."""
    formatted_data = [format_example(item) for item in data]
    return Dataset.from_list(formatted_data)

def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Tokenize the dataset and prepare labels."""
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def get_train_eval_datasets(tokenized_dataset, test_size=0.1):
    """Split dataset into train and evaluation sets."""
    return tokenized_dataset.train_test_split(test_size=test_size) 