"""Sequential OHCA classifier for Qwen served through an OpenAI-compatible API."""

from __future__ import annotations

import re
from typing import Dict

from .config import (
    LABEL_NOT_OHCA,
    LABEL_OHCA,
    LABEL_TRANSFER,
    MAX_TOKENS,
    SEED,
    TEMPERATURE,
)
from .classifier import STEPS
from .openai_client import OpenAIChatClient
from .parsing import parse_yes_no


RESIDENTIAL_ORIGIN_RE = re.compile(
    r"\b(nursing\s*home|snf|skilled\s*nursing|long[-\s]?term\s*care|\bltc\b|"
    r"assisted\s*living|\balf\b|group\s*home|residential\s*(?:care|facility)|"
    r"rehab(?:ilitation)?\s*facility|extended\s*care|care\s*(?:home|facility)|"
    r"convalescent|memory\s*care|board\s*and\s*care|at\s*home|residence|"
    r"patient'?s?\s*home)\b",
    re.I,
)

ACUTE_CARE_TRANSFER_RE = re.compile(
    r"\b(outside\s*hospital|\bosh\b|another\s*(?:hospital|ed|emergency|facility\s*ed)|"
    r"transferred\s*from\s*(?:[a-z .]{0,30})?(?:hospital|medical\s*center|\bmc\b|\bed\b)|"
    r"transfer\s*from\s*(?:an?\s*)?(?:outside|other)\s*(?:hospital|facility)|"
    r"referring\s*hospital|sending\s*(?:hospital|facility)|"
    r"initially\s*(?:seen|presented|taken)\s*(?:to|at)\s*(?:an?\s*)?"
    r"(?:outside|other|another))\b",
    re.I,
)


def residential_origin_rescue(note: str, rationale: str) -> tuple[bool, str]:
    """Identify residential-origin cases that are not acute-care transfers."""

    haystack = f"{note}\n{rationale}"
    if ACUTE_CARE_TRANSFER_RE.search(haystack):
        return False, "acute-care transfer cue present"
    match = RESIDENTIAL_ORIGIN_RE.search(haystack)
    if match:
        origin = match.group(0).strip()
        return True, f"residential origin ({origin}) is not an acute-care transfer"
    return False, ""


def classify_note_qwen(
    note_text: str,
    client: OpenAIChatClient,
    *,
    note_char_cap: int = 3000,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    seed: int | None = SEED,
    rescue_residential_origin: bool = True,
    debug_raw: bool = False,
    row_id: object | None = None,
) -> Dict:
    """Run the four-step OHCA chain against a Qwen/OpenAI-compatible endpoint.

    This intentionally uses the sequential prompt chain, not the faster
    single-call JSON prompt, because the first arrest-existence gate is the
    most important accuracy guardrail for this phenotype.
    """

    result = {
        "llm_label": LABEL_NOT_OHCA,
        "llm_confidence": 0.0,
        "llm_rationale": "",
        "steps_passed": 0,
        "step_responses": {},
        "llm_step_detail": "",
        "residential_rescue": "",
    }
    rationale_parts = []

    for step in STEPS:
        step_key, prompt_template, step_name, fail_label = step[:4]
        invert = bool(step[4]) if len(step) > 4 and step[4] is True else False
        prompt = prompt_template.format(note=str(note_text)[:note_char_cap])

        try:
            raw_response = client.complete(
                prompt,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
        except RuntimeError as exc:
            result["llm_label"] = "Error"
            result["llm_rationale"] = str(exc)
            return result

        is_yes, rationale = parse_yes_no(raw_response)
        step_passed = (not is_yes) if invert else is_yes
        answer = "YES" if is_yes else "NO"
        result["step_responses"][step_key] = {
            "answer": answer,
            "rationale": rationale,
            "raw": (raw_response or "")[:500],
        }
        rationale_parts.append(f"[{step_name}] {rationale}")

        if debug_raw:
            print(
                f"\n  --- row {row_id} | {step_name} --- parsed={answer} "
                f"({'passed' if step_passed else 'FAILED'})"
            )
            print(f"      RAW: {(raw_response or '')[:600]!r}")

        if not step_passed:
            result["llm_label"] = fail_label
            result["llm_confidence"] = result["steps_passed"] / len(STEPS)
            result["llm_rationale"] = "; ".join(rationale_parts)

            if (
                rescue_residential_origin
                and fail_label == LABEL_TRANSFER
            ):
                rescue, reason = residential_origin_rescue(note_text, rationale)
                result["residential_rescue"] = reason
                if rescue:
                    result["llm_label"] = LABEL_OHCA
                    result["steps_passed"] += 1
                    result["llm_confidence"] = 1.0
                    result["llm_rationale"] += f"; [RESCUED] {reason}"
            result["llm_step_detail"] = _step_detail(result["step_responses"])
            return result

        result["steps_passed"] += 1

    result["llm_label"] = LABEL_OHCA
    result["llm_confidence"] = 1.0
    result["llm_rationale"] = "; ".join(rationale_parts)
    result["llm_step_detail"] = _step_detail(result["step_responses"])
    return result


def _step_detail(step_responses: dict) -> str:
    return "; ".join(
        f"{key}={value['answer']}"
        for key, value in step_responses.items()
        if isinstance(value, dict) and "answer" in value
    )
