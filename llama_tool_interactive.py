import inspect
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools.tools_list_test import *
import importlib


# def get_prediction_from_classifier(instance: str):
#     """
#     Gets a prediction from the given classifier for a given instance.

#     Args:
#         instance: The instance to get a prediction for.
#     """
#     return "positive"
# def get_important_word(instance: str):
#     """
#     It cannot replace counterfactual explanation.
#     It CANNOT answer why the model made this current prediction instead of the other prediction

#     Args:
#         instance: The instance to get the important word for.
#     """
#     return "movie"  # dummy important word
# def get_counterfactual_explanation(instance: str, prediction: str = None):
#     """
#     Gets a counterfactual for a given instance to explain the model's prediction.
#     It can answer what would need to change in the instance to get a different prediction.
#     It can also answer why the model made this current prediction instead of the other prediction

#     Args:
#         instance: The instance to get the counterfactual for.
#         prediction: The current prediction of the instance (optional).
#     """
#     return "The movie was terrible and boring."  # dummy counterfactual
# def get_scope_of_instance(instance: str):
#     """
#     Gets the scope of the instance to explain the model's prediction.
#     It can answer what is the scope of change in the instance to get the same prediction.

#     Args:
#         instance: The instance to get the scope for.
#     """
#     return "The movie was perfect!"  # dummy scope
# def get_model_information(field: str):
#     """
#     Gets information about the model.

#     Args:
#         field: The field to get information for. Can be "architecture", "training_data", "training_procedure", or "evaluation".
#     """
#     info = {
#         "architecture": "Transformer-based model with 12 layers and 768 hidden units.",
#         "training_data": "Trained on a diverse dataset of text from the internet, including books, articles, and websites.",
#         "training_procedure": "Trained using supervised learning with cross-entropy loss and Adam optimizer.",
#         "evaluation": "Evaluated on a held-out test set with an accuracy of 85%.",
#     }
#     return info.get(field, "Unknown field")
# def get_data_information(field: str):
#     """
#     Gets information about the data.

#     Args:
#         field: The field to get information for. Can be "source", "collection_method", "preprocessing", or "statistics".
#     """
#     info = {
#         "source": "Data collected from various online sources, including social media, news articles, and forums.",
#         "collection_method": "Data was collected using web scraping and API access.",
#         "preprocessing": "Data was cleaned and preprocessed to remove duplicates, irrelevant content, and noise.",
#         "statistics": "The dataset contains 1 million instances with an average length of 100 words per instance.",
#     }
#     return info.get(field, "Unknown field")

# tools = [get_prediction_from_classifier, get_important_word, get_counterfactual_explanation, get_scope_of_instance]

# tools = [
#     get_prediction_from_classifier,
#     get_important_word,
#     get_counterfactual_explanation,
#     get_scope_of_instance,
#     get_model_information,
#     get_data_information,
# ]
_tools_mod = importlib.import_module("tools.tools_list")
tools = [
    fn
    for _, fn in inspect.getmembers(_tools_mod, inspect.isfunction)
    if fn.__module__== _tools_mod.__name__ and not fn.__name__.startswith("_")
]
# tools = [get_classifier_metadata, get_classifier_performance, get_prediction_from_classifier, get_important_features, get_counterfactuals, get_multiple_counterfactuals, get_feature_attribution]
# tools = [get_prediction_from_classifier, get_important_words, get_counterfactual_explanation]
def build_tool_spec(fn):
    sig = inspect.signature(fn)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        properties[name] = {"type": "string", "title": name}
        if param.default is inspect._empty:
            required.append(name)
    return {
        "name": fn.__name__,
        "description": fn.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

FUNCTIONS_BY_NAME = {fn.__name__: fn for fn in tools}
TOOL_SPECS = [build_tool_spec(fn) for fn in tools]
# print("TOOL_SPECS:", TOOL_SPECS)
checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
)

chat = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant with access to XAI tools to explain a classifier. Use them to answer user questions about the classifier, system, model and data. If you don't know the answer, please call function 'unknown'."
    }
]
print("Type an empty line to exit.")
while True:
    user_input = input("User: ").strip()
    if not user_input:
        break
    chat.append({"role": "user", "content": user_input})

    while True:
        tool_prompt = tokenizer.apply_chat_template(
            chat,
            tools=tools,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(model.device)
        # print tool prompt as decoded text
        input_ids = tool_prompt["input_ids"][0].cpu().tolist()
        raw_prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
        clean_prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        # print("Tool prompt (raw):\n" + raw_prompt)
        # print("Tool prompt (clean):\n" + clean_prompt)
        out = model.generate(**tool_prompt, max_new_tokens=256)
        generated_text = out[0, tool_prompt["input_ids"].shape[1]:]
        assistant_text = tokenizer.decode(generated_text, skip_special_tokens=False).strip()
        assistant_clean = tokenizer.decode(generated_text, skip_special_tokens=True).strip()
        # chat.append({"role": "assistant", "content": assistant_text})

        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_text, re.DOTALL)
        if not tool_call_match:
            print("No tool call found.")
            print(f"Assistant: {assistant_clean}")
            break

        try:
            payload = json.loads(tool_call_match.group(1))
        except json.JSONDecodeError:
            print("Assistant (tool call parse error):", assistant_text)
            break

        tool_name = payload.get("name")
        arguments = payload.get("arguments", {})
        tool_call = {"name": tool_name, "arguments": arguments}
        chat.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
        if tool_name not in FUNCTIONS_BY_NAME:
            # print("2")
            print(f"Assistant (unknown tool {tool_name}):", assistant_text)
            break

        result = FUNCTIONS_BY_NAME[tool_name](**arguments)
        tool_response = json.dumps({"result": result})
        chat.append({"role": "tool", "name": tool_name, "content": tool_response})
        print(f"{tool_name} -> {tool_response}")
