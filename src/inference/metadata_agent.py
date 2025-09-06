from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Will error at runtime if YAML is missing


@dataclass
class MetadataPaths:
    repo_root: Path

    @property
    def dataset_yaml(self) -> Path:
        return self.repo_root / "metadata" / "dataset" / "metadata.yaml"

    @property
    def dataset_json(self) -> Path:
        return self.repo_root / "metadata" / "dataset" / "metadata.json"

    @property
    def model_yaml(self) -> Path:
        return self.repo_root / "metadata" / "model" / "metadata.yaml"

    @property
    def model_json(self) -> Path:
        return self.repo_root / "metadata" / "model" / "metadata.json"


class MetadataStore:
    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.paths = MetadataPaths(repo_root=repo_root or Path(__file__).resolve().parents[2])
        self.dataset: Dict[str, Any] = {}
        self.model: Dict[str, Any] = {}

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML files. Install with: pip install pyyaml")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        # Load dataset metadata
        if self.paths.dataset_yaml.exists():
            self.dataset = self._load_yaml(self.paths.dataset_yaml)
        elif self.paths.dataset_json.exists():
            self.dataset = self._load_json(self.paths.dataset_json)
        else:
            self.dataset = {}

        # Load model metadata
        if self.paths.model_yaml.exists():
            self.model = self._load_yaml(self.paths.model_yaml)
        elif self.paths.model_json.exists():
            self.model = self._load_json(self.paths.model_json)
        else:
            self.model = {}

    # --------- Convenience getters ---------
    def get_model_accuracy(self) -> Optional[float]:
        try:
            perf = self.model["model"]["performance"]["benchmarks"][0]["metrics"]["accuracy"]
            if isinstance(perf, (int, float)):
                return float(perf)
            # if stored as string like "94.6%" or "0.946"
            if isinstance(perf, str):
                m = re.match(r"^(\d+(?:\.\d+)?)%$", perf.strip())
                if m:
                    return float(m.group(1)) / 100.0
                return float(perf)
        except Exception:
            return None
        return None

    def get_precision_recall_f1(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {"precision": None, "recall": None, "f1": None}
        try:
            metrics = self.model["model"]["performance"]["benchmarks"][0]["metrics"]
            for k, alias in [("precision_macro", "precision"), ("recall_macro", "recall"), ("f1_macro", "f1")]:
                v = metrics.get(k)
                if isinstance(v, (int, float)):
                    out[alias] = float(v)
                elif isinstance(v, str):
                    m = re.match(r"^(\d+(?:\.\d+)?)%$", v.strip())
                    out[alias] = float(m.group(1)) / 100.0 if m else float(v)
        except Exception:
            pass
        return out

    def get_dataset_sizes(self) -> Dict[str, Optional[int]]:
        sizes: Dict[str, Optional[int]] = {"train": None, "test": None, "total": None}
        try:
            sizes["total"] = int(self.dataset["dataset"]["composition"]["num_instances"])  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            splits = self.dataset["dataset"]["evaluation"]["recommended_splits"]
            sizes["train"] = int(splits.get("train")) if "train" in splits else None
            sizes["test"] = int(splits.get("test")) if "test" in splits else None
        except Exception:
            pass
        return sizes

    def get_labels(self) -> Optional[list]:
        try:
            return list(self.dataset["dataset"]["composition"]["labels"])  # type: ignore[list-item]
        except Exception:
            return None

    def get_model_name(self) -> Optional[str]:
        try:
            return str(self.model["model"]["name"])  # type: ignore[return-value]
        except Exception:
            return None


class MetadataQAAgent:
    def __init__(self, store: Optional[MetadataStore] = None) -> None:
        self.store = store or MetadataStore()
        self.store.load()

    @staticmethod
    def _fmt_pct(x: Optional[float]) -> Optional[str]:
        if x is None:
            return None
        return f"{x*100:.1f}%"

    def answer(self, question: str) -> str:
        q = question.lower().strip()

        # Accuracy and metrics
        if any(k in q for k in ["accuracy", "accurate", "acc"]):
            acc = self.store.get_model_accuracy()
            if acc is not None:
                return f"Model accuracy on CIFAR-10: {self._fmt_pct(acc)}."
            return "Accuracy not available in metadata."

        if any(k in q for k in ["precision", "recall", "f1"]):
            m = self.store.get_precision_recall_f1()
            parts = []
            if m.get("precision") is not None:
                parts.append(f"precision {self._fmt_pct(m['precision'])}")
            if m.get("recall") is not None:
                parts.append(f"recall {self._fmt_pct(m['recall'])}")
            if m.get("f1") is not None:
                parts.append(f"F1 {self._fmt_pct(m['f1'])}")
            return ("; ".join(parts) + ".") if parts else "No precision/recall/F1 in metadata."

        # Dataset sizes
        if any(k in q for k in ["how many", "num", "number", "count", "size"]) and any(k in q for k in ["train", "training", "instances", "samples", "images", "examples"]):
            sizes = self.store.get_dataset_sizes()
            train = sizes.get("train")
            total = sizes.get("total")
            if train is not None and total is not None:
                return f"Training instances: {train} (total dataset: {total})."
            if train is not None:
                return f"Training instances: {train}."
            if total is not None:
                return f"Total instances: {total}."
            return "Dataset sizes not available in metadata."

        if any(k in q for k in ["test set", "test split", "test images", "test instances"]) or ("test" in q and any(k in q for k in ["how many", "num", "number", "count"])):
            sizes = self.store.get_dataset_sizes()
            test = sizes.get("test")
            return f"Test instances: {test}." if test is not None else "Test split not available in metadata."

        # Labels/classes
        if any(k in q for k in ["classes", "labels", "categories"]):
            labels = self.store.get_labels()
            if labels:
                return f"Classes ({len(labels)}): {', '.join(labels)}."
            return "Labels not available in metadata."

        # Model name/architecture
        if any(k in q for k in ["model name", "what model", "architecture", "backbone"]):
            name = self.store.get_model_name()
            if name:
                return f"Model: {name}."
            return "Model name not available in metadata."

        # Generic fallback
        return "I can answer questions about accuracy, precision/recall/F1, dataset sizes (train/test/total), and labels from the metadata. Try asking: 'How accurate is the model?' or 'How many training instances were used?'"


class MetadataLLMAgent:
    """LLM-powered agent that answers questions grounded in repo metadata.

    It injects the model and dataset metadata as context and uses the
    tokenizer's chat template when available (e.g., Llama 3.3 Instruct).
    """

    def __init__(self, model, tokenizer, device, store: Optional[MetadataStore] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.store = store or MetadataStore()
        self.store.load()

    def _context(self) -> str:
        # Provide compact, but sufficiently complete context. Fallback to empty dicts
        model_meta = self.store.model or {}
        data_meta = self.store.dataset or {}
        try:
            model_str = json.dumps(model_meta, ensure_ascii=False)
        except Exception:
            model_str = str(model_meta)
        try:
            data_str = json.dumps(data_meta, ensure_ascii=False)
        except Exception:
            data_str = str(data_meta)
        return (
            "You must answer ONLY using the metadata below. "
            "If the answer is not present, reply 'Not available in metadata.'\n\n"
            f"Model metadata JSON:\n{model_str}\n\n"
            f"Dataset metadata JSON:\n{data_str}\n"
        )

    def answer(self, question: str, max_new_tokens: int = 128, temperature: float = 0.2, top_p: float = 0.9, do_sample: bool = False) -> str:
        system_prompt = (
            "You are a concise assistant that answers questions about a model and dataset. "
            "Use only the provided metadata. If unknown, say 'Not available in metadata.'"
        )
        context = self._context()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback simple prompt
            prompt = (
                f"<|begin_of_text|>System: {system_prompt}\n{context}\n<|end_of_text|>\n"
                f"User: {question} <|end_of_text|>\nAssistant: "
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        import torch  # local import to avoid import errors if unused
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Try to extract assistant segment if using simple template
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()
        return text


def demo_answers() -> None:
    agent = MetadataQAAgent()
    queries = [
        "How accurate is the model?",
        "How many training instances were used?",
        "What are the classes?",
        "What is the recall?",
    ]
    for q in queries:
        print(f"Q: {q}\nA: {agent.answer(q)}\n")


if __name__ == "__main__":  # Simple REPL for manual runs
    agent = MetadataQAAgent()
    print("Metadata QA Agent. Ask about model/dataset (type 'exit' to quit).")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break
        if q.lower() in {"exit", "quit", ":q"}:
            break
        print("Agent:", agent.answer(q))
