"""Proposers — decide the next model design in an automated search.

A *proposer* is the "brain" of the auto-run loop: given the leaderboard so far
and the search space, it returns the next design to try. It is deliberately
LLM-agnostic — any callable with the signature

    propose(board, space, context) -> dict   # a partial model_design

works. Two are provided:

* :func:`propose_random` — samples the search space uniformly. No dependencies;
  the durable fallback that proves the loop end-to-end (mirrors
  ``agent/search_example.py``).
* :class:`ClaudeProposer` — asks Claude to read what has worked and propose the
  next design. Requires the ``anthropic`` SDK and an API key; falls back to
  random automatically if either is missing, so the loop never stalls.

Returned designs are always clamped back into the advertised ranges before use,
so a proposer can never submit an out-of-bounds architecture.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from src.agent.client import SEARCH_SPACE

# A proposer maps (leaderboard, search space, context) -> a partial design dict.
Proposer = Callable[[List[dict], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

# Model Claude uses to propose designs. Opus is the most capable; a sweep is
# cheap relative to a GPU training run, so default to it.
DEFAULT_MODEL = os.environ.get("FLARE_AGENT_MODEL", "claude-opus-4-8")

# Default base URL for a local OpenAI-compatible LLM server (llama.cpp's
# ``llama-server`` runs here by default; LM Studio uses :1234, Ollama :11434).
LOCAL_DEFAULT_URL = os.environ.get("FLARE_LOCAL_LLM_URL", "http://localhost:8080/v1")


# ----------------------------------------------------------------------
# Clamping — keep any proposal inside the advertised search space.
# ----------------------------------------------------------------------
def clamp_design(design: Dict[str, Any], space: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce ``design`` so every field respects ``space`` (choices / ranges)."""
    out: Dict[str, Any] = {}
    for field, spec in space.items():
        if field not in design or design[field] is None:
            continue
        val = design[field]
        try:
            if "choices" in spec:
                choices = spec["choices"]
                # snap to the nearest advertised choice
                val = min(choices, key=lambda c: abs(c - val))
            else:
                lo, hi = spec["range"]
                val = max(lo, min(hi, val))
                val = int(round(val)) if spec.get("type") == "int" else round(float(val), 5)
        except (TypeError, ValueError):
            continue
        out[field] = val
    return out


# ----------------------------------------------------------------------
# Random proposer (fallback / reference).
# ----------------------------------------------------------------------
def propose_random(board: List[dict], space: Dict[str, Any],
                   context: Dict[str, Any]) -> Dict[str, Any]:
    """Sample one design uniformly from ``space`` (ignores the leaderboard)."""
    rng: random.Random = context.get("rng") or random
    design: Dict[str, Any] = {}
    for field, spec in space.items():
        if "choices" in spec:
            design[field] = rng.choice(spec["choices"])
        else:
            lo, hi = spec["range"]
            design[field] = (rng.randint(lo, hi) if spec.get("type") == "int"
                             else round(rng.uniform(lo, hi), 4))
    return design


# ----------------------------------------------------------------------
# Claude proposer.
# ----------------------------------------------------------------------
def _agent_md() -> str:
    """The human-facing control surface (agent/AGENT.md), used as the system
    prompt so the operator can steer the search by editing that file."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "agent", "AGENT.md")
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return "You are an automated search over neural-network surrogate models."


def _design_schema(space: Dict[str, Any]) -> Dict[str, Any]:
    """A json_schema constraining Claude's output to the search-space fields.

    Numeric bounds aren't expressible as hard schema constraints (structured
    outputs strip them), so they're described here and enforced by
    :func:`clamp_design` afterwards.
    """
    props = {}
    for field, spec in space.items():
        if "choices" in spec:
            desc = f"one of {spec['choices']}"
            typ = "integer"
        else:
            lo, hi = spec["range"]
            desc = f"between {lo} and {hi}"
            typ = "integer" if spec.get("type") == "int" else "number"
        if spec.get("note"):
            desc += f" ({spec['note']})"
        props[field] = {"type": typ, "description": desc}
    return {
        "type": "object",
        "properties": props,
        "required": list(space.keys()),
        "additionalProperties": False,
    }


def _board_summary(board: List[dict], limit: int = 12) -> str:
    """Compact, token-cheap view of the best designs so far."""
    if not board:
        return "No completed runs yet — this is the first proposal."
    lines = []
    for rank, rec in enumerate(board[:limit], 1):
        design = (rec.get("spec", {}) or {}).get("model_design") or {}
        knobs = {k: design.get(k) for k in SEARCH_SPACE if design.get(k) is not None}
        match = rec.get("metrics", {}).get("mean_match")
        lines.append(f"{rank}. match={match:.4f}  {json.dumps(knobs, separators=(',', ':'))}"
                     if match is not None else f"{rank}. {json.dumps(knobs)}")
    return "\n".join(lines)


def _user_prompt(board: List[dict], space: Dict[str, Any], context: Dict[str, Any],
                 require_json_note: bool = False) -> str:
    """The per-turn user message shared by every LLM proposer."""
    base = context.get("base_design") or {}
    objective = context.get("objective", "maximize evaluation mean_match")
    prompt = (
        f"Objective: {objective}.\n\n"
        f"Search space (fields you may set, with ranges):\n"
        f"{json.dumps(space, indent=2)}\n\n"
        f"Leaderboard so far (best first):\n{_board_summary(board)}\n\n"
        f"Baseline design for reference:\n{json.dumps({k: base.get(k) for k in space}, indent=2)}\n\n"
        "Propose the single next design to try. Reason from what has worked: "
        "change one region at a time (amp net, phase net, Fourier features) so "
        "improvements are attributable, and explore rather than repeat past runs. "
    )
    if require_json_note:
        prompt += (
            "Respond with ONLY a single JSON object mapping every field name to a "
            "number — no prose, no markdown, no code fences. Fields: "
            f"{', '.join(space)}."
        )
    else:
        prompt += "Return every field in the required schema."
    return prompt


class ClaudeProposer:
    """Propose the next design with Claude, reading the leaderboard each turn."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        import anthropic  # raises ImportError if the SDK isn't installed

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.system = _agent_md()

    def __call__(self, board: List[dict], space: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, Any]:
        user = _user_prompt(board, space, context)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=self.system,
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": _design_schema(space)}},
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return clamp_design(json.loads(text), space)


# ----------------------------------------------------------------------
# Local proposer — any OpenAI-compatible server (llama.cpp / LM Studio / Ollama)
# serving a GGUF model on localhost.
# ----------------------------------------------------------------------
def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort parse of a JSON object from a chat completion.

    Local models often wrap output in prose or ```json fences, so pull the first
    balanced ``{...}`` rather than trusting the whole string to be clean JSON.
    """
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


# Fallback key llama.cpp's own docs use when its server has no real API-key
# check but a client (or a proxy in front of it) still insists an Authorization
# header be present. Sent only when the caller supplies no real key.
_PLACEHOLDER_KEY = os.environ.get("FLARE_LOCAL_LLM_API_KEY", "sk-no-key-required")


def _local_headers(api_key: Optional[str]) -> Dict[str, str]:
    """Auth headers for a local OpenAI-compatible server.

    Always sends *something* — several local servers (and proxies in front of
    them, e.g. LiteLLM/vLLM/text-generation-webui) 401 with "missing
    Authorization header" if the header is absent at all, even when they don't
    validate the value. Real deployments that enforce a key should pass one in.
    """
    key = api_key or _PLACEHOLDER_KEY
    return {"Authorization": f"Bearer {key}", "X-API-Key": key}


def list_local_models(base_url: str = LOCAL_DEFAULT_URL, api_key: Optional[str] = None,
                      timeout: float = 5.0) -> List[str]:
    """Model ids the local server exposes (its ``GET /models`` list).

    For llama.cpp/LM Studio this is the loaded GGUF; for Ollama it's every pulled
    model. Raises on connection failure so callers can surface "server not up".
    """
    r = requests.get(f"{base_url.rstrip('/')}/models", headers=_local_headers(api_key),
                     timeout=timeout)
    r.raise_for_status()
    data = r.json().get("data", [])
    return [m.get("id") for m in data if m.get("id")]


def local_available(base_url: str = LOCAL_DEFAULT_URL, api_key: Optional[str] = None) -> bool:
    """True if a local OpenAI-compatible server answers at ``base_url``."""
    try:
        list_local_models(base_url, api_key=api_key)
        return True
    except Exception:  # noqa: BLE001 — server down / wrong URL / bad key
        return False


class LocalProposer:
    """Propose designs via a local OpenAI-compatible chat-completions endpoint."""

    def __init__(self, base_url: str = LOCAL_DEFAULT_URL, model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 180.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        # Default to whatever model the server currently has loaded.
        self.model = model or (list_local_models(base_url, api_key=api_key) or ["local"])[0]
        self.timeout = timeout
        self.system = _agent_md()

    def __call__(self, board: List[dict], space: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, Any]:
        user = _user_prompt(board, space, context, require_json_note=True)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system},
                         {"role": "user", "content": user}],
            "temperature": 0.7,
            "max_tokens": 1024,
            # Widely supported by local servers; ignored gracefully by ones that
            # don't, in which case the prompt's "ONLY JSON" instruction carries it.
            "response_format": {"type": "json_object"},
        }
        r = requests.post(f"{self.base}/chat/completions", json=body,
                          headers=_local_headers(self.api_key), timeout=self.timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return clamp_design(_extract_json(text), space)


# ----------------------------------------------------------------------
# Factory.
# ----------------------------------------------------------------------
def claude_available() -> bool:
    """True if the Claude proposer can be constructed (SDK + credentials)."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    # An API key env var, or an `ant auth login` profile, both satisfy the SDK.
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.isdir(os.path.expanduser("~/.config/anthropic")))


def make_proposer(name: str, base_url: Optional[str] = None,
                  model: Optional[str] = None,
                  api_key: Optional[str] = None) -> Tuple[Proposer, str]:
    """Return ``(proposer, resolved_name)``.

    ``"claude"`` and ``"local"`` fall back to ``"random"`` when unavailable
    (missing SDK/credentials, or an unreachable local server) so an auto-run
    always has a working brain and degrades *before* the loop starts.
    """
    if name == "random":
        return propose_random, "random"
    if name == "claude":
        # The anthropic client doesn't fail at construction when credentials are
        # missing — it errors at request time — so gate on availability up front
        # to degrade to random *before* the loop starts rather than mid-run.
        if not claude_available():
            return propose_random, "random"
        try:
            return ClaudeProposer(), "claude"
        except Exception:  # noqa: BLE001 — missing SDK/key -> degrade gracefully
            return propose_random, "random"
    if name == "local":
        url = base_url or LOCAL_DEFAULT_URL
        if not local_available(url, api_key=api_key):
            return propose_random, "random"
        try:
            return LocalProposer(base_url=url, model=model, api_key=api_key), "local"
        except Exception:  # noqa: BLE001 — server hiccup -> degrade gracefully
            return propose_random, "random"
    raise ValueError(f"unknown proposer: {name!r}")
