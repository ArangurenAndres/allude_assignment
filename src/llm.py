# src/llm.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class LLMConfig:
    model: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    temperature: float = 0.2
    max_output_tokens: int = 180
    host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def llm_available() -> bool:
    """
    True if the local Ollama server is reachable.
    """
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def synthesize_with_llm(
    question: str,
    tool_output: str,
    *,
    config: Optional[LLMConfig] = None,
) -> str:
    """
    Grounded synthesis using local Ollama.

    - tool_output is the source of truth.
    - Never crash the app: on any error, return tool_output.
    """
    config = config or LLMConfig()
    host = config.host.rstrip("/")

    # If Ollama isn't available, just return deterministic output
    if not llm_available():
        return tool_output

    prompt = (
        "You are a maintenance analytics assistant.\n"
        "You MUST answer using ONLY the TOOL OUTPUT.\n"
        "Do NOT invent details.\n"
        "Do NOT change numbers.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"TOOL OUTPUT (source of truth):\n{tool_output}\n\n"
        "Write a concise answer in 1 sentence. "
        "If TOOL OUTPUT is a number, restate it clearly.\n"
    )

    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            # Ollama uses num_predict for max tokens
            "num_predict": config.max_output_tokens,
        },
    }

    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        return text if text else tool_output
    except Exception as e:
        print(f"[Ollama disabled: {e}]")
        return tool_output
