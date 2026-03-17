import csv
import os
import random
import re
from typing import Any, Dict, List, Optional

# Import global conversation messages
from utils.tool_executor import get_conversation_messages


class SentimentRuleModel:
    """LLM-backed sentiment model for IMDb-style reviews.

    Input:
      - raw text review

    Output labels:
      - "positive"
      - "negative"
      - "neutral"
    """

    _MODEL = None
    _TOKENIZER = None
    _TORCH = None
    _MODEL_NAME = os.environ.get("SENTIMENT_LLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    _DEVICE_MAP = os.environ.get("SENTIMENT_LLM_DEVICE_MAP", "cuda:0")
    _DTYPE = os.environ.get("SENTIMENT_LLM_DTYPE", "bfloat16")

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "dataset",
            "imdb_test.csv",
        )
        self._rng = random.Random(2025)
        self._call_idx = 0
        self._last_text: Optional[str] = None
        self._last_label: Optional[str] = None

    def _choose(self, variants: List[str], salt: str = "") -> str:
        self._call_idx += 1
        offset = (hash(salt) ^ self._call_idx) & 0x7FFFFFFF
        return variants[offset % len(variants)]

    @classmethod
    def _ensure_llm(cls) -> None:
        if cls._MODEL is not None and cls._TOKENIZER is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "LLM sentiment requires `torch` and `transformers` installed, and access to the model on Hugging Face."
            ) from e

        dtype = None
        if cls._DTYPE == "auto":
            dtype = None
        elif cls._DTYPE == "float16":
            dtype = torch.float16
        elif cls._DTYPE == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError("SENTIMENT_LLM_DTYPE must be one of: auto|float16|bfloat16")

        tokenizer = AutoTokenizer.from_pretrained(cls._MODEL_NAME, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cls._MODEL_NAME,
            device_map=cls._DEVICE_MAP,
            torch_dtype="bfloat16",
        )
        model.eval()

        cls._TORCH = torch
        cls._TOKENIZER = tokenizer
        cls._MODEL = model

    @classmethod
    def _chat_generate(cls, system: str, user: str, max_new_tokens: int = 64) -> str:
        cls._ensure_llm()
        torch = cls._TORCH
        tokenizer = cls._TOKENIZER
        model = cls._MODEL

        # Get context messages from global variable
        context_messages = get_conversation_messages()
        
        # Use context messages if available, otherwise create new conversation
        if context_messages:
            messages = context_messages.copy()
            # Replace the system prompt in context with the provided system prompt
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": system}
            # Append the new user message
            messages.append({"role": "user", "content": user})
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            attention_mask = torch.ones_like(input_ids)
            prompt_len = int(input_ids.shape[-1])
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][prompt_len:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(enc["input_ids"].shape[-1])
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _normalize_sentiment_label(text: str) -> str:
        t = (text or "").strip().lower()
        t = re.split(r"\s+", t)[0] if t else ""
        t = re.sub(r"[^a-z_]+", "_", t)
        if t in {"positive", "negative", "neutral"}:
            return t
        # tolerate variants
        if "pos" in t:
            return "positive"
        if "neg" in t:
            return "negative"
        if "neu" in t:
            return "neutral"
        return "neutral"

    def get_prediction(self, instance: str) -> str:
        text = (instance or "").strip()
        if not text:
            return "Please provide a text review to analyze."

        system = "You are a sentiment classifier."
        user = (
            "Classify the following movie review into exactly one label: positive, negative, or neutral.\n"
            "Output ONLY the label.\n\n"
            f"REVIEW: {text}"
        )
        raw = self._chat_generate(system, user, max_new_tokens=8)
        label = self._normalize_sentiment_label(raw)
        self._last_text = text
        self._last_label = label
        return label

    def get_important_features(self, label: str = None) -> str:
        text = self._last_text
        if not text:
            return "Provide a text review first."


        system = "You explain sentiment predictions briefly."
        user = (
            "Give a short reason  why the review supports the sentiment label. Which words or phrases are important?\n"
            f"REVIEW: {text}"
        )
        reason = self._chat_generate(system, user, max_new_tokens=48)
        reason = re.sub(r"\s+", " ", reason).strip()
        return "Reason: " + reason

    def get_counterfactuals(self, instances: str = None, target_label: str = None, number_of_iterations: int = 1) -> List[str]:
        text = instances if instances is not None else self._last_text
        if not text:
            return ["Provide a text review first."]

        k = max(1, int(number_of_iterations or 1))
        target = (target_label or "").lower().strip() if target_label else ""
        if target not in {"positive", "negative", "neutral", ""}:
            target = ""

        system = "You propose minimal text edits."
        user = (
            "Propose minimal edit instructions to change the review's sentiment.\n"
            f"Target label: {target or 'any different label'}\n"
            f"Return EXACTLY a JSON array of {k} short strings.\n"
            f"REVIEW: {text}"
        )
        raw = self._chat_generate(system, user, max_new_tokens=256)
        ideas: List[str] = []
        try:
            import json as _json

            parsed = _json.loads(raw)
            if isinstance(parsed, list):
                ideas = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            ideas = [ln.strip("- ") for ln in raw.splitlines() if ln.strip()]

        ideas = ideas[:k]
        while len(ideas) < k:
            ideas.append("Add/remove a few sentiment-bearing words to shift polarity.")
        return ideas

    def get_instance_with_same_prediction(self, instance: str = None) -> str:
        text = instance if instance is not None else self._last_text
        if not text:
            return "Provide a text review first."

        label = self.get_prediction(text)
        system = "You suggest neutral edits that preserve sentiment."
        user = (
            f"Suggest one short edit to the review that keeps the sentiment label '{label}'. "
            "Be concrete and avoid adding strong sentiment words.\n"
            f"REVIEW: {text}"
        )
        suggestion = self._chat_generate(system, user, max_new_tokens=256)
        suggestion = re.sub(r"\s+", " ", suggestion).strip()
        return suggestion

    def get_global_explanation(self) -> str:
        system = "You describe model behavior briefly."
        user = (
            "In 1-2 sentences, describe how you determine sentiment labels (positive/negative/neutral) from text. "
            "Keep it general and concise."
        )
        return self._chat_generate(system, user, max_new_tokens=64)

    def get_model_performance(self, metric: str) -> str:
        metric = (metric or "accuracy").lower()
        # LLM evaluation can be slow; keep it bounded.
        stats = self._evaluate_dataset(limit=200)
        value = stats.get(metric)
        if value is None:
            return f"Metric '{metric}' not available. Try: accuracy."

        system = "You report model evaluation concisely."
        user = (
            "Write ONE short sentence reporting the metric value.\n"
            "Do not add extra metrics.\n"
            f"METRIC: {metric}\n"
            f"VALUE: {value:.4f}\n"
            "CONTEXT: IMDb test slice (sampled)."
        )
        msg = self._chat_generate(system, user, max_new_tokens=32)
        msg = re.sub(r"\s+", " ", msg).strip()
        return msg

    def get_data_information(self) -> str:
        stats = self._dataset_stats(limit=2000)
        system = "You describe datasets briefly."
        user = (
            "In 1 sentence, describe the dataset used for evaluation.\n"
            f"FILE: imdb_test.csv\n"
            f"N_SAMPLED: {int(stats['n'])}\n"
            f"POS_FRACTION: {stats['pos_frac']:.4f}"
        )
        msg = self._chat_generate(system, user, max_new_tokens=48)
        return re.sub(r"\s+", " ", msg).strip()

    def get_model_information(self) -> str:
        system = "You describe the model briefly."
        user = (
            "In 1 sentence, describe the sentiment model used in this tool.\n"
            f"MODEL_NAME: {self._MODEL_NAME}"
        )
        msg = self._chat_generate(system, user, max_new_tokens=48)
        return re.sub(r"\s+", " ", msg).strip()

    def get_output_information(self) -> str:
        system = "You explain label semantics."
        user = (
            "Explain the output label space for this sentiment tool in 1 short sentence. "
            "Mention exactly: positive, negative, neutral."
        )
        msg = self._chat_generate(system, user, max_new_tokens=32)
        return re.sub(r"\s+", " ", msg).strip()

    def get_system_information(self) -> str:
        system = "You describe system architecture briefly."
        user = "In 1 sentence, describe how the system uses tools and this sentiment component."
        msg = self._chat_generate(system, user, max_new_tokens=48)
        return re.sub(r"\s+", " ", msg).strip()

    # ---- Mistake utilities ----

    def count_mistakes(self, limit: int = 1000) -> int:
        stats = self._evaluate_dataset(limit=limit)
        return int(stats["mistakes"])

    def sample_mistakes(self, n: int = 3, limit: int = 1000) -> List[Dict[str, Any]]:
        mistakes: List[Dict[str, Any]] = []
        for row in self._iter_dataset(limit=limit):
            gold = "positive" if int(row["label"]) == 1 else "negative"
            pred = self.get_prediction(row["text"])
            # Treat neutral as wrong for IMDb binary gold
            pred_bin = pred if pred in ("positive", "negative") else "neutral"
            if pred_bin != gold:
                mistakes.append({"text": row["text"][:120] + ("..." if len(row["text"]) > 120 else ""), "gold": gold, "pred": pred})
                if len(mistakes) >= max(1, int(n)):
                    break
        return mistakes or [{"note": "No mistakes found in sampled subset."}]

    def _iter_dataset(self, limit: int = 1000):
        if not os.path.exists(self.dataset_path):
            return
        with open(self.dataset_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                yield {"text": row.get("text", ""), "label": row.get("label", "0")}

    def _dataset_stats(self, limit: int = 2000) -> Dict[str, float]:
        n = 0
        pos = 0
        for row in self._iter_dataset(limit=limit) or []:
            n += 1
            pos += 1 if str(row.get("label", "0")).strip() == "1" else 0
        return {"n": n, "pos_frac": (pos / n if n else 0.0)}

    def _evaluate_dataset(self, limit: int = 1000) -> Dict[str, float]:
        total = 0
        correct = 0
        mistakes = 0
        for row in self._iter_dataset(limit=limit) or []:
            total += 1
            gold = "positive" if int(row["label"]) == 1 else "negative"
            pred = self.get_prediction(row["text"])
            if pred == gold:
                correct += 1
            else:
                mistakes += 1
        return {"accuracy": (correct / total if total else 0.0), "mistakes": float(mistakes)}
