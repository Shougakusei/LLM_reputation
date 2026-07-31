"""Tolerant extraction of a JSON object from the LLM response text.

Used by the agent (TALK/DECIDE/PREDICT/REFLECT phases) and the LLM judge: models
often wrap JSON in prose or code fences, so we try several candidates.
"""

from __future__ import annotations

import json
import re


def extract_json_obj(text: str) -> dict | None:
    """Extract the first JSON object from the text: raw / in a ```-fence / by braces.

    Args:
        text: Full text of the model's response.

    Returns:
        A dict if some candidate parsed as a JSON object, otherwise None.
    """
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    block = _first_brace_block(text)
    if block:
        candidates.append(block)
    for candidate in candidates:
        try:
            # strict=False: allow raw control characters (newlines) inside strings —
            # models write multi-line notes without escaping them.
            obj = json.loads(candidate, strict=False)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _first_brace_block(text: str) -> str | None:
    # Naive balanced-brace scan: good enough for prose-wrapped JSON; does not account
    # for braces inside string values (rare in our outputs).
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
