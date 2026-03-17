"""LLM-as-a-judge: check whether `rule_answer` is preserved in `agent_answer`.

This script reads the CSV produced by `agent/run_custom_dialogues_to_csv.py` and
adds 1 column:
- judge_label: preserved | not_preserved

Supports chat-style causal LMs (e.g. Llama 3 Instruct) and seq2seq LMs (e.g. Flan-T5).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class JudgeResult:
    label: str
    score: float
    rationale: str


_ALLOWED = {"preserved", "not_preserved"}


def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s in _ALLOWED:
        return s
    if s in {"notpreserved", "not-preserved", "not_preserve", "notpreserve"}:
        return "not_preserved"
    if s in {"preserve", "preserved_answer", "preserved_meaning", "yes"}:
        return "preserved"
    if s in {"no", "contradiction"}:
        return "not_preserved"
    return ""


def _build_prompt(rule_answer: str, agent_answer: str) -> str:
    return (
        "You are a strict evaluator. Determine whether the AGENT_ANSWER preserves the key information in the RULE_ANSWER.\n"
        "Preserved means: the agent conveys the same final decision/value(s) and does not contradict.\n"
        "\n"
        "Output EXACTLY ONE token and nothing else:\n"
        "preserved\n"
        "or\n"
        "not_preserved\n"
        "\n"
        "RULE_ANSWER: "
        + json.dumps(str(rule_answer))
        + "\n"
        "AGENT_ANSWER: "
        + json.dumps(str(agent_answer))
        + "\n"
    )


def _extract_json(text: str) -> Optional[Dict[str, object]]:
    if not text:
        return None
    # Try to find the first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _parse_judge_output(text: str) -> str:
    if not text:
        return ""
    # Prefer a clean single-token output.
    first = (text or "").strip().split()[0] if (text or "").strip() else ""
    label = _normalize_label(first)
    if label:
        return label

    # If the model printed extra text, search for either label anywhere.
    lowered = (text or "").lower()
    if "not_preserved" in lowered or "not preserved" in lowered:
        return "not_preserved"
    if re.search(r"\bpreserved\b", lowered):
        return "preserved"

    # JSON fallback (if any)
    data = _extract_json(text)
    if isinstance(data, dict):
        label = _normalize_label(str(data.get("label", "")))
        if label:
            return label
    return ""


class _HFJudge:
    def __init__(self, model_name: str, device_map: str, torch_dtype: str) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Missing dependencies. Install at least `torch` + `transformers` (see `requirements-judge.txt`)."
            ) from e

        self._torch = torch
        self.model_name = model_name

        cfg = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = None
        if torch_dtype == "auto":
            dtype = None
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError("--torch-dtype must be one of: auto|float16|bfloat16")

        is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))
        if is_encoder_decoder:
            self.kind = "seq2seq"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
            )
        else:
            self.kind = "causal"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
            )

        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        torch = self._torch

        if self.kind == "seq2seq":
            prompt = system_prompt + "\n\n" + user_prompt
            enc = self.tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            prompt_len = int(input_ids.shape[-1])
            attention_mask = self._torch.ones_like(input_ids)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = out[0][prompt_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(enc["input_ids"].shape[-1])
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _build_retry_prompt(rule_answer: str, agent_answer: str) -> str:
    return (
        "Decide if AGENT_ANSWER preserves the intent/information in RULE_ANSWER.\n"
        "Answer with exactly ONE token and nothing else: preserved OR not_preserved\n"
        "RULE_ANSWER: "
        + json.dumps(str(rule_answer))
        + "\n"
        "AGENT_ANSWER: "
        + json.dumps(str(agent_answer))
        + "\n"
    )


def run(
    input_csv: str,
    output_csv: str,
    model: str,
    limit: int,
    max_new_tokens: int,
    debug: bool,
    device_map: str,
    torch_dtype: str,
) -> None:
    system_prompt = (
        "You are an impartial judge. Output exactly one token: preserved or not_preserved."
    )
    judge = _HFJudge(model_name=model, device_map=device_map, torch_dtype=torch_dtype)

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    out_fields = list(fieldnames)
    if "judge_label" not in out_fields:
        out_fields.append("judge_label")

    n = len(rows) if limit <= 0 else min(len(rows), limit)

    for i in range(n):
        if rows[i].get("judge_label", "") in _ALLOWED:
            continue  # already judged
        rule_answer = rows[i].get("tool_responses", "")
        agent_answer = rows[i].get("agent_response", "")
        prompt = _build_prompt(rule_answer, agent_answer)
        raw = judge.generate(system_prompt=system_prompt, user_prompt=prompt, max_new_tokens=max_new_tokens)
        label = _parse_judge_output(raw)
        if not label:
            retry_prompt = _build_retry_prompt(rule_answer, agent_answer)
            raw2 = judge.generate(system_prompt=system_prompt, user_prompt=retry_prompt, max_new_tokens=max_new_tokens)
            label2 = _parse_judge_output(raw2)
            if label2:
                label, raw = label2, raw2
        if not label:
            label = "not_preserved"
        rows[i]["judge_label"] = label
        if debug:
            # Debug mode can still include raw, but keep primary judge output label-only.
            if "judge_raw" not in out_fields:
                out_fields.append("judge_raw")
            rows[i]["judge_raw"] = raw

    # for remaining rows (if limited), leave blank
    for i in range(n, len(rows)):
        rows[i].setdefault("judge_label", "not_preserved")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="/home/pikachu/projects/XAgent-LLMs/evaluation/HR_new_llm_judge_context_2.csv",
    )
    p.add_argument(
        "--output",
        default="/home/pikachu/projects/XAgent-LLMs/evaluation/HR_new_llm_judge_context_3.csv",
    )
    p.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Judge model (recommended: meta-llama/Meta-Llama-3.1-8B-Instruct).",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--debug", action="store_true", help="Add a `judge_raw` column with model output (label output remains label-only).")
    p.add_argument(
        "--device-map",
        default="cuda:0",
        help="HF device map for model loading (e.g. auto, cuda:0, cpu).",
    )
    p.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Model dtype: auto|float16|bfloat16",
    )
    args = p.parse_args()

    run(
        input_csv=args.input,
        output_csv=args.output,
        model=args.model,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        debug=args.debug,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
