import argparse
from os import listdir
from os.path import isfile, join

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_json


def demonstration_selection(train_embeddings, test_embedding, training_set_texts, training_set_labels, num_shot=5):
    cosine_scores = util.cos_sim(test_embedding, train_embeddings)
    _, indices = torch.sort(cosine_scores[0], descending=True)

    demonstration_lists = []
    for item in indices[:num_shot]:
        demonstration_lists.append({
            "user_question": training_set_texts[item],
            "label": training_set_labels[item]
        })

    return demonstration_lists


def get_prompt_template_by_method_name(method, user_question, demonstration_lists, task):
    prompt_template = ""

    if task == "intention":
        prompt_template += (f"You will be given a user's question related to an AI/ML model or system."
                            f"Your task is to accurately predict the primary intention behind this question."
                            f"This is crucial for understanding user needs and improving AI interactions."
                            f"The possible intentions are categorized into the following labels."
                            f"Please select the single best label that represents the user's intention:\n"
                            f"1. adversarial: User is asking about or providing an example intended to mislead or fool the model."
                            f"2. counterfactual: User is asking about or exploring changes to input features and their impact on the model's prediction."
                            f"3. edit_prediction: User wants to modify or correct the prediction for a specific instance."
                            f"4. importance: User is asking about which features are most influential in the model's predictions."
                            f"5. predict: User simply wants to obtain a prediction from the model for a given input."
                            f"6. rationalize: User is asking for an explanation or justification for the model's prediction."
                            f"Below are some examples:\n"
                            )
        for d in demonstration_lists:
            prompt_template += f"question: {d['user_question']} answer: {d['label']}\n"
        prompt_template += f"question: {user_question} answer:"
    else:
        if method == "default":
            prompt_template += (f"You will be given a user question related to explainability. "
                                f"Your task is to identify and extract the custom input from this question. "
                                f"The custom input refers to the specific information provided by the user that is necessary to fulfill their request. "
                                f"Extracting this input is crucial for processing user questions and taking appropriate actions. "
                                f"Please return only the custom input as a text string. If no custom input is clearly present, return an empty string."
                                f"Below are some examples:\n")
            for d in demonstration_lists:
                prompt_template += f"[user_question] {d['user_question']} [custom_input] {d['label']}\n"
            prompt_template += f"[user_question] {user_question} [custom_input] "
        elif method == "GPT_NER":
            prompt_template += (
                # Prompt reference GPT-NER: https://arxiv.org/abs/2304.10428
                f"You are an excellent linguist. You will be given a user question related to explainability. "  # new prompt from GPT-NER paper
                # add the following 3 sentences to keep the consistency with the default prompt
                f"The task is to label the custom input in the given user question. "
                f"The custom input refers to the specific information provided by the user that is necessary to fulfill their request. "
                f"Extracting this input is crucial for processing user questions and taking appropriate actions. "
                # new prompt from GPT-NER paper
                f"Use special tokens @@## to mark the extracted phrase in your response.\n"
                f"Please return a text string with the custom input marked with @@##. If no custom input is clearly present, return an empty string."
                f"Below are some examples:\n"
            )

            for d in demonstration_lists:
                target_output = f"@@{d['label']}##"
                prompt_template += (
                    # GPT-NER specically use Input-Output format
                    f"Input: {d['user_question']}\n"
                    f"Output: {d['user_question'].replace(d['label'], target_output)}\n\n"
                )
            prompt_template += f"Input: {user_question}\nOutput: "

        elif method == "TANL":
            prompt_template += (
                # Prompt reference TANL: https://openreview.net/forum?id=US-TP-xnXI
                # But TANL paper does not mention the Prompt
                # add the following 4 sentences to keep the consistency with the default prompt
                f"You will be given a user question related to explainability. "
                f"Your task is to identify and extract the custom input from this question. "
                f"The custom input refers to the specific information provided by the user that is necessary to fulfill their request. "
                f"Extracting this input is crucial for processing user questions and taking appropriate actions. "
                # new prompt from TANL paper
                f"Use the format `[ extracted_text | custom_input ]` to annotate the custom input in the output.\n"
                f"Please return a text string with the custom input marked with [ extracted_text | custom_input ]. If no custom input is clearly present, return an empty string.\n"
                f"Below are some examples:\n"
            )

            for d in demonstration_lists:
                target_output = f"[ {d['label']} | custom_input ]"
                prompt_template += (
                    # TANL specically use Input-Output format
                    f"Input: {d['user_question']}\n"
                    f"Output: {d['user_question'].replace(d['label'], target_output)}\n\n"
                )

            prompt_template += f"Input: {user_question}\nOutput: "

        elif method == "GoLLIE":
            prompt_template += (
                # Prompt reference GoLLIE: https://openreview.net/forum?id=Y3wpuxd7
                # add the following 4 sentences to keep the consistency with the default prompt
                f"You will be given a user question related to explainability. "
                f"Your task is to identify and extract the custom input from this question. "
                f"Please return a list of custom input.If no custom input is clearly present, return an empty list.\n"
                f"Below is the schema for the custom input annotation:\n"
                # new prompt from GoLLIE paper
                f"@dataclass\n"
                f"class CustomInput(Entity):\n"
                f'    """The custom input refers to the specific information provided by the user that is necessary to fulfill their request.'
                f'       Extracting this input is crucial for processing user questions and taking appropriate actions.'
                f'      """\n'
                f"    span: str  # Example: 'Toilet Water Swirls in Different Directions Depending on Hemisphere'\n\n"

                f"## Examples:\n"
            )

            for d in demonstration_lists:
                prompt_template += (
                    f"text = \"{d['user_question']}\"\n"
                    f"result = [\n"
                    f"    CustomInput(span=\"{d['label']}\")\n"
                    f"]\n\n"
                )

            prompt_template += (
                f"text = \"{user_question}\"\n"
                f"result = "
            )

        else:
            raise NotImplementedError(f"{method} is not supported/implemented!")

    return prompt_template


def intent_recognition_and_extraction(model, tokenizer, method, task, instance):
    s_name = "sentence-transformers/all-MiniLM-L12-v2"

    s_model = SentenceTransformer(s_name).cuda()

    # get training set embeddings
    path_prefix = f"../../data/extraction/"
    json_files = [f for f in listdir(f"{path_prefix}") if
                  isfile(join(f"{path_prefix}", f)) and f.endswith(".json")]

    training_set = []
    if task == "intention":
        field = "operation_name"
        content = "You are an excellent assistant for intent recognition."
        max_new_tokens = 16
    else:
        field = "custom_input"
        content = "You are an excellent assistant for custom input extraction."
        max_new_tokens = 64

    for json_file in json_files:
        training_set += load_json(path_prefix + json_file)

    # convert the dataset to sentence embeddings
    training_set_texts = [d["user_question"] for d in training_set]
    training_set_labels = [d[field] for d in training_set]

    train_embeddings = s_model.encode(training_set_texts, convert_to_tensor=True).cuda()

    test_embedding = s_model.encode(instance, convert_to_tensor=True).cuda()

    # demonstration selection based on cosine similarity
    demonstration_lists = demonstration_selection(train_embeddings, test_embedding, training_set_texts,
                                                  training_set_labels)
    prompt_template = get_prompt_template_by_method_name(method, instance, demonstration_lists,
                                                         task)
    print("[Prompt]: ", prompt_template)

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": prompt_template}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_family",
        default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        choices=["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501",
                 "Qwen/Qwen2.5-72B-Instruct"],
        help="Identify which LLM family to evaluate",
    )

    parser.add_argument(
        "--task",
        default="intention",
        choices=["intention", "extraction"],
        help="Identify which task to evaluate",
    )

    parser.add_argument(
        "--method",
        default="default",
        choices=["default", "GPT_NER", "TANL", "GoLLIE"],
        help="Identify which dataset to evaluate",
    )

    args = parser.parse_args()

    task = args.task
    model_name = args.model_family
    method = args.method

    # For test
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        revision="main"
    )

    instance = "How does the counterfactuals look like for 'the movie is nice'?"

    intent_recognition_and_extraction(model, tokenizer, method, task, instance)