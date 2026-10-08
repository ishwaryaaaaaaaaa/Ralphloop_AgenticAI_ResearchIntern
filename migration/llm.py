"""Thin wrapper around the Anthropic API: one place that talks to the model.

`ask`      -> plain text answer.
`ask_json` -> answer validated against a pydantic model, retrying with the
              validation error fed back to the model (a mini Ralph loop).
"""
import json
import re
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

MODEL = "claude-opus-5-5"  # change here to switch every agent's model
MAX_TOKENS = 32000

T = TypeVar("T", bound=BaseModel)

# Reads ANTHROPIC_API_KEY from the environment.
_client = anthropic.Anthropic()


def ask(system: str, user: str, *, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """Send one prompt, return the model's text. Streams to avoid HTTP timeouts on long output."""
    with _client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("answer was cut off at max_tokens; raise MAX_TOKENS")
    # content may also hold thinking blocks; keep only the text
    return "".join(b.text for b in response.content if b.type == "text")


def _extract_json(text: str) -> str:
    """Pull the JSON out of an answer that may be wrapped in ```json fences or prose."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text.strip()


def ask_json(system: str, user: str, schema: type[T], *, retries: int = 2, **kwargs) -> T:
    """Ask for JSON matching `schema`. On a bad answer, show the model its error and retry."""
    system = (
        f"{system}\n\nRespond with a single JSON object only, matching this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    prompt = user
    for attempt in range(retries + 1):
        raw = ask(system, prompt, **kwargs)
        try:
            return schema.model_validate_json(_extract_json(raw))
        except ValidationError as err:
            if attempt == retries:
                raise
            prompt = (
                f"{user}\n\nYour previous answer was invalid:\n{err}\n\n"
                f"Previous answer:\n{raw}\n\nReturn a corrected JSON object."
            )
    raise AssertionError("unreachable")
