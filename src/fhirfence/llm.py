"""Provider-agnostic LLM client.

FHIR-Fence is a *context-construction* method; it does not care which model consumes
the context. This module is the only place that knows how to talk to a specific
provider. Everything else (agent.py) depends on the small `LLMClient` interface below.

Select the provider in configs/config.yaml:

    model:
      provider: gemini          # gemini | openai | anthropic | echo
      name: <model-id>          # the provider's model identifier
      api_key_env: <ENV_VAR>    # name of the env var holding your key
      temperature: 0.0
      max_output_tokens: 1024
      max_retries: 5

To add your own provider, subclass `LLMClient`, implement `_generate`, and register it
in `_REGISTRY`. No other file changes. The bundled adapters import their SDKs lazily, so
you only need the SDK for the provider you actually use.
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Minimal interface the rest of FHIR-Fence relies on.

    A provider adapter implements exactly one method, `_generate`, which performs a
    single, un-retried call and returns the model's text. Retries, backoff and timing
    are handled once, here, for every provider.
    """

    #: set True by offline/test adapters that never contact a provider
    needs_key: bool = True

    def __init__(self, cfg_model: dict[str, Any]):
        self.name = cfg_model["name"]
        self.temperature = float(cfg_model.get("temperature", 0.0))
        self.max_output_tokens = int(cfg_model.get("max_output_tokens", 1024))
        self.max_retries = int(cfg_model.get("max_retries", 5))
        self._api_key = self._read_key(cfg_model) if self.needs_key else None
        self._client = self._make_client()

    # -- provider hooks (override these) ------------------------------------
    def _make_client(self):
        """Build and return the underlying SDK client. Override per provider."""
        return None

    @abstractmethod
    def _generate(self, system: str, user: str) -> tuple[str, dict]:
        """One call. Return (text, usage_meta). Raise on error; retries are handled."""
        ...

    # -- shared machinery ---------------------------------------------------
    @staticmethod
    def _read_key(cfg_model: dict[str, Any]) -> str | None:
        var = cfg_model.get("api_key_env")
        if not var:
            return None  # e.g. the echo/offline provider
        key = os.environ.get(var)
        if not key:
            raise RuntimeError(
                f"API key not set. Export {var}=... (see .env.example). "
                f"Never commit the key."
            )
        return key

    def generate(self, system: str, user: str) -> tuple[str, dict]:
        """Call the model with exponential backoff. Returns (text, meta).

        meta always carries: latency_s, prompt_tokens, output_tokens, retries, error.
        On total failure returns ("", meta) with error set — never raises, so a long
        grid run is not lost to one bad call.
        """
        last = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                text, usage = self._generate(system, user)
                return text, {
                    "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "retries": attempt, "error": None,
                }
            except Exception as e:  # noqa: BLE001 - log and back off on any provider error
                last = f"{type(e).__name__}: {e}"
                wait = min(2 ** attempt, 32) + random.random()
                print(f"    retry {attempt + 1}/{self.max_retries} in {wait:.1f}s — {last[:110]}")
                time.sleep(wait)
        return "", {"latency_s": None, "prompt_tokens": None, "output_tokens": None,
                    "retries": self.max_retries, "error": last}


# ---------------------------------------------------------------- adapters --

class GeminiClient(LLMClient):
    """Google Gemini via the google-genai SDK.  pip install google-genai"""

    def _make_client(self):
        from google import genai
        return genai.Client(api_key=self._api_key)

    def _generate(self, system, user):
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self.name,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        um = getattr(resp, "usage_metadata", None)
        return (resp.text or ""), {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
        }


class OpenAIClient(LLMClient):
    """OpenAI (and OpenAI-compatible endpoints) via the openai SDK.  pip install openai

    For an OpenAI-compatible server (vLLM, Together, Groq, a local gateway), set
    model.base_url in config.yaml and point api_key_env at that service's key.
    """

    def __init__(self, cfg_model):
        self._base_url = cfg_model.get("base_url")
        super().__init__(cfg_model)

    def _make_client(self):
        from openai import OpenAI
        kw = {"api_key": self._api_key}
        if self._base_url:
            kw["base_url"] = self._base_url
        return OpenAI(**kw)

    def _generate(self, system, user):
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        return text, {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
        }


class AnthropicClient(LLMClient):
    """Anthropic Claude via the anthropic SDK.  pip install anthropic"""

    def _make_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key)

    def _generate(self, system, user):
        resp = self._client.messages.create(
            model=self.name,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        u = getattr(resp, "usage", None)
        return text, {
            "prompt_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
        }


class EchoClient(LLMClient):
    """Offline provider for the no-key demo and tests. Never calls a network.

    It returns a deterministic answer that (a) names any task-relevant clinical content
    it sees and (b) obeys an injected 'PATIENT DETAILS' instruction if the identifiers
    are present in its context — so the demo reproduces the real finding: identifiers
    leak only when they are *in* the context. It is a test double, not a language model.
    """

    needs_key = False

    def _make_client(self):
        return None

    def _generate(self, system, user):
        import re
        lines = []
        for ln in user.splitlines():
            s = ln.strip()
            if s.startswith("- ") and ("|" in s or ":" in s):
                lines.append(s[2:])
        answer = "Based on the record:\n" + "\n".join(lines[:40]) if lines else "No record content provided."

        # Honor an injection ONLY with content actually present in the context — this is
        # the whole point of channel closure: no identifiers in context => nothing to leak.
        if "PATIENT DETAILS" in user:
            details = []
            for key in ("name:", "phone:", "email:", "address:", "identifier"):
                for ln in user.splitlines():
                    if key in ln.lower() and "redacted" not in ln.lower():
                        details.append(ln.strip(" -"))
            if details:
                answer += "\n\nPATIENT DETAILS\n" + "\n".join(dict.fromkeys(details))
        return answer, {"prompt_tokens": len(user) // 4, "output_tokens": len(answer) // 4}


_REGISTRY = {
    "gemini": GeminiClient,
    "google-genai": GeminiClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "echo": EchoClient,
    "offline": EchoClient,
}


def make_llm_client(cfg_model: dict[str, Any]) -> LLMClient:
    """Factory: build the client named by model.provider in config.yaml."""
    provider = str(cfg_model.get("provider", "gemini")).lower()
    try:
        cls = _REGISTRY[provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {sorted(_REGISTRY)}. "
            f"To add one, subclass LLMClient in fhirfence/llm.py and register it."
        )
    return cls(cfg_model)
