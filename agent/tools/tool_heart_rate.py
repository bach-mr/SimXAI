import os
import re
from typing import List, Dict, Any, Optional

# Import global conversation messages
from utils.tool_executor import get_conversation_messages


class HeartRateMonitor:
    """LLM-backed heart rate analyzer that provides health insights."""
    
    _MODEL = None
    _TOKENIZER = None
    _TORCH = None
    _MODEL_NAME = os.environ.get("HEART_RATE_LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    _DEVICE_MAP = os.environ.get("HEART_RATE_LLM_DEVICE_MAP", "cuda:0")
    _DTYPE = os.environ.get("HEART_RATE_LLM_DTYPE", "bfloat16")
    
    def __init__(self):
        """Initialize the heart rate monitor."""
        self._last_bpm: Optional[int] = None
        self._last_status: Optional[str] = None

    def _parse_instance(self, instance: str):
        try:
            bpm = int(str(instance).strip())
            if not 0 <= bpm <= 300:
                return "Please provide a valid heart rate between 0 and 300 BPM."
            return bpm
        except ValueError:
            return "Please provide a single numeric value for heart rate (e.g., '75')."

    @classmethod
    def _ensure_llm(cls) -> None:
        if cls._MODEL is not None and cls._TOKENIZER is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "LLM heart rate monitor requires `torch` and `transformers` installed, and access to the model on Hugging Face."
            ) from e

        dtype = None
        if cls._DTYPE == "auto":
            dtype = None
        elif cls._DTYPE == "float16":
            dtype = torch.float16
        elif cls._DTYPE == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError("HEART_RATE_LLM_DTYPE must be one of: auto|float16|bfloat16")

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
    def _normalize_heart_rate_status(text: str) -> str:
        t = (text or "").strip().lower()
        t = re.split(r"\s+", t)[0] if t else ""
        t = re.sub(r"[^a-z_]+", "_", t)
        if t in {"low", "normal", "high"}:
            return t
        # tolerate variants
        if "low" in t or "brady" in t:
            return "low"
        if "high" in t or "tachy" in t:
            return "high"
        if "norm" in t:
            return "normal"
        return "normal"

    def get_prediction(self, instance) -> str:
        """
        Analyze heart rate status using LLM.

        Args:
            instance: Heart rate in BPM (string or int)

        Returns:
            str: Status ('low', 'normal', or 'high')
        """
        parsed = self._parse_instance(str(instance))
        if isinstance(parsed, str):
            return parsed

        bpm = parsed
        self._last_bpm = bpm
        
        system = "You are a heart rate health analyzer."
        user = (
            "Classify the following resting heart rate into exactly one label: low, normal, or high.\n"
            "Standard guidelines: low (bradycardia) is below 60 BPM, normal is 60-100 BPM, high (tachycardia) is above 100 BPM.\n"
            "Output ONLY the label.\n\n"
            f"HEART RATE: {bpm} BPM"
        )
        raw = self._chat_generate(system, user, max_new_tokens=8)
        status = self._normalize_heart_rate_status(raw)
        self._last_status = status
        return status

    def get_important_features(self, label: str = None) -> str:
        """Get explanation for the heart rate status using LLM."""
        if self._last_bpm is None:
            return "Provide a heart rate measurement first."
        
        bpm = self._last_bpm
        status = label if label else self._last_status
        
        system = "You explain heart rate classifications briefly."
        user = (
            "Give a short reason why this heart rate is classified as the given status. "
            "Which factors or ranges are important?\n"
            f"HEART RATE: {bpm} BPM\n"
            f"STATUS: {status}"
        )
        reason = self._chat_generate(system, user, max_new_tokens=48)
        reason = re.sub(r"\s+", " ", reason).strip()
        return "Reason: " + reason

    def get_counterfactuals(self, instance=None, target_label: str = None, number_of_iterations: int = 1) -> list:
        """Get counterfactual explanations using LLM."""
        if instance:
            parsed = self._parse_instance(str(instance))
            if isinstance(parsed, str):
                return [parsed]
            bpm = parsed
        else:
            bpm = self._last_bpm
            
        if bpm is None:
            return ["Provide a heart rate measurement first."]

        k = max(1, int(number_of_iterations or 1))
        target = (target_label or "").lower().strip() if target_label else ""
        if target not in {"low", "normal", "high", ""}:
            target = ""

        system = "You propose minimal changes to heart rate measurements."
        user = (
            "Propose specific BPM values or minimal adjustments to change the heart rate status.\n"
            f"Current BPM: {bpm}\n"
            f"Target status: {target or 'any different status'}\n"
            f"Return EXACTLY a JSON array of {k} short strings with specific BPM values.\n"
        )
        raw = self._chat_generate(system, user, max_new_tokens=256)
        suggestions: List[str] = []
        try:
            import json as _json
            parsed = _json.loads(raw)
            if isinstance(parsed, list):
                suggestions = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            suggestions = [ln.strip("- ") for ln in raw.splitlines() if ln.strip()]

        suggestions = suggestions[:k]
        while len(suggestions) < k:
            suggestions.append("Adjust heart rate by 10-15 BPM to shift status category.")
        return suggestions

    def get_global_explanation(self) -> str:
        """Get global explanation of the heart rate logic using LLM."""
        system = "You describe heart rate classification rules briefly."
        user = (
            "In 1-2 sentences, describe how you determine heart rate status (low/normal/high) from BPM values. "
            "Keep it general and concise."
        )
        return self._chat_generate(system, user, max_new_tokens=64)

    def get_instance_with_same_prediction(self, instance=None) -> str:
        """Get another heart rate value that would yield the same status using LLM."""
        if instance:
            bpm = self._parse_instance(str(instance))
            if isinstance(bpm, str):
                return bpm
        else:
            bpm = self._last_bpm
            
        if bpm is None:
            return "Provide a heart rate measurement first."
            
        status = self.get_prediction(str(bpm))
        
        system = "You suggest neutral edits that preserve heart rate status."
        user = (
            f"Suggest one specific BPM value that keeps the heart rate status '{status}'. "
            "Be concrete with a specific number.\n"
            f"CURRENT BPM: {bpm}"
        )
        suggestion = self._chat_generate(system, user, max_new_tokens=64)
        suggestion = re.sub(r"\s+", " ", suggestion).strip()
        return suggestion

    def get_model_performance(self, metric) -> str:
        """Get information about the model's performance using LLM."""
        system = "You describe model performance concisely."
        user = (
            "In 1 sentence, describe the performance characteristics of a heart rate classification model.\n"
            f"REQUESTED METRIC: {metric}"
        )
        msg = self._chat_generate(system, user, max_new_tokens=48)
        return re.sub(r"\s+", " ", msg).strip()

    def get_data_information(self) -> str:
        """Get information about the data used."""
        system = "You describe medical data briefly."
        user = (
            "In 1 sentence, describe the data or guidelines used for heart rate classification."
        )
        msg = self._chat_generate(system, user, max_new_tokens=48)
        return re.sub(r"\s+", " ", msg).strip()

    def get_model_information(self) -> str:
        """Get general information about the model."""
        system = "You describe the model briefly."
        user = (
            "In 1 sentence, describe the heart rate classification model used in this tool.\n"
            f"MODEL_NAME: {self._MODEL_NAME}"
        )
        msg = self._chat_generate(system, user, max_new_tokens=48)
        return re.sub(r"\s+", " ", msg).strip()

    def get_output_information(self) -> str:
        """Get information about the model's output."""
        system = "You explain label semantics."
        user = (
            "Explain the output label space for this heart rate tool in 1 short sentence. "
            "Mention exactly: low, normal, high."
        )
        msg = self._chat_generate(system, user, max_new_tokens=32)
        return re.sub(r"\s+", " ", msg).strip()
