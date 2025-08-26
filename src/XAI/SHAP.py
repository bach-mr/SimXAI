import numpy as np
import torch
from ferret import (
    Benchmark,
    SHAPExplainer,
)


device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


def prediction(model, tokenizer, instance):
    encoding = tokenizer.encode_plus(instance, max_length=512, return_tensors='pt',
                                     truncation=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    input_model = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'token_type_ids': token_type_ids,  # Add token_type_ids
    }
    output_model = model(**input_model)[0]

    # Get logit
    output_model = np.argmax(output_model.cpu().detach().numpy())

    return output_model


def select_topk_words(topk, explanation):
    rankings = list(np.argsort(explanation.scores))
    topk_words = []

    for j in rankings[-topk:]:
        topk_word = explanation.tokens[j]

        # Post process tokenized subwords
        if topk_word.startswith("##"):
            topk_word = explanation.tokens[j - 1] + topk_word.replace("##", "")

    # guarantee top k words
    if len(topk_words) < topk:
        for j in rankings[:-topk][::-1]:
            if len(topk_words) < topk:
                topk_word = explanation.tokens[j]
                if topk_word.startswith("##"):
                    topk_word = explanation.tokens[j - 1] + topk_word.replace("##", "")
                topk_words.append(topk_word)
            else:
                break
    return topk_words


def get_topk_words_scores(target, bench, instance, topk=5):
    explanations = bench.explain(instance, target=target)
    evaluations = bench.evaluate_explanations(explanations, target=target)

    temp = {}

    for ev, ex in zip(evaluations, explanations):
        topk_words = select_topk_words(topk, ex)

        score_keys = [scorer.name for scorer in ev.evaluation_scores]
        score_values = [str(scorer.score) for scorer in ev.evaluation_scores]
        scores = dict(zip(score_keys, score_values))

        temp[f"{ex.explainer}"] = {
            "topk_words": topk_words,
            "scores": scores
        }
    return temp, explanations


def SHAP(model, tokenizer, instance, topk=5):
    s = SHAPExplainer(model, tokenizer)

    bench = Benchmark(model, tokenizer, explainers=[s])

    predicted_label = prediction(model, tokenizer, instance)

    result, explanation = get_topk_words_scores(predicted_label, bench, instance, topk=topk)
    print(result)
    print(explanation)