from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.XAI.SHAP import SHAP

model_name = "textattack/bert-base-uncased-imdb"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

SHAP(model, tokenizer, "I love sci-fi and am willing to put up with a lot.")