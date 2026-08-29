"""
llm_explainer.py

Turns structured decision context (current state, reward decomposition,
top SHAP features, uncertainty level) into a natural-language explanation
via a single, well-prompted LLM call -- deliberately NOT a multi-agent
system. See docs/future_work.md for why multi-agent orchestration was
scoped out.

Two modes:
  - LLM mode (default if ANTHROPIC_API_KEY is set): calls the Anthropic API.
  - Template mode (automatic fallback, and used in tests/CI so nothing here
    requires network access or a secret to be testable): a deterministic,
    rule-based sentence builder using the same structured context. Less
    fluent, but produces a real explanation grounded in the same numbers.

For the live demo, set ANTHROPIC_API_KEY in your environment (get one at
console.anthropic.com) to use the LLM mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DecisionContext:
    """Everything the explanation is grounded in -- kept as one object so
    both the LLM prompt and the template fallback draw from exactly the
    same numbers (no separate, potentially-drifting explanation logic)."""

    T_in: float
    T_out: float
    occupancy: bool
    price: float
    action: float           # -1..1, HVAC power fraction (- = cooling, + = heating)
    reward_components: dict  # {"cost": ..., "comfort": ..., "carbon": ..., "peak": ...}
    dominant_factor: str
    uncertainty_std: float
    uncertainty_threshold: float
    top_shap_features: list  # [(feature_name, shap_value), ...]
    used_fallback_control: bool = False


def _action_description(action: float) -> str:
    if action > 0.15:
        return f"heating at {abs(action)*100:.0f}% power"
    elif action < -0.15:
        return f"cooling at {abs(action)*100:.0f}% power"
    else:
        return "holding (near-zero HVAC power)"


def _template_explanation(ctx: DecisionContext) -> str:
    """Deterministic, no-LLM-required fallback. Grounded in the exact same
    DecisionContext an LLM call would use."""
    parts = []

    parts.append(
        f"Indoor temperature is {ctx.T_in:.1f}°C (outdoor {ctx.T_out:.1f}°C), "
        f"and the agent is {_action_description(ctx.action)}."
    )

    if ctx.used_fallback_control:
        parts.append(
            f"The twin's confidence in its own prediction was low (uncertainty "
            f"{ctx.uncertainty_std:.3f}°C, above the {ctx.uncertainty_threshold:.3f}°C threshold), "
            f"so control deferred to the conservative rule-based thermostat rather than "
            f"trusting the learned policy."
        )
    else:
        parts.append(
            f"Prediction confidence was within normal range (uncertainty {ctx.uncertainty_std:.3f}°C), "
            f"so the trained policy's action was used directly."
        )

    dominant_map = {
        "cost": "electricity cost",
        "comfort": "occupant comfort",
        "carbon": "carbon emissions",
        "peak": "peak demand",
    }
    parts.append(
        f"The main factor driving this decision was {dominant_map.get(ctx.dominant_factor, ctx.dominant_factor)}"
        + (f" (price is currently ${ctx.price:.2f}/kWh, occupancy is {'on' if ctx.occupancy else 'off'})." if ctx.dominant_factor == "cost" else ".")
    )

    if ctx.top_shap_features:
        feat_str = ", ".join(f"{name} ({val:+.3f})" for name, val in ctx.top_shap_features[:3])
        parts.append(f"The twin's temperature correction was most influenced by: {feat_str}.")

    return " ".join(parts)


def _build_llm_prompt(ctx: DecisionContext) -> str:
    reward_str = ", ".join(f"{k}: {v:.3f}" for k, v in ctx.reward_components.items())
    shap_str = ", ".join(f"{name}: {val:+.4f}" for name, val in ctx.top_shap_features[:3]) if ctx.top_shap_features else "n/a"

    return f"""You are the explanation module of CoolTwin, an HVAC control system for a building zone.
Given the structured decision context below, write ONE short paragraph (2-4 sentences) in plain
language explaining this control decision to a non-technical building operator. Be concrete and
reference the actual numbers. Do not invent information not present below.

Current state:
- Indoor temperature: {ctx.T_in:.1f} C
- Outdoor temperature: {ctx.T_out:.1f} C
- Occupied: {ctx.occupancy}
- Electricity price: ${ctx.price:.2f}/kWh

Action taken: {_action_description(ctx.action)}

Reward component breakdown (weighted penalty contributions, higher = more of a driver): {reward_str}
Dominant factor: {ctx.dominant_factor}

Digital twin prediction uncertainty: {ctx.uncertainty_std:.3f} C (threshold for falling back to
rule-based control: {ctx.uncertainty_threshold:.3f} C). Fallback control used: {ctx.used_fallback_control}

Top contributing features to the twin's temperature correction (SHAP values): {shap_str}
"""


def generate_explanation(ctx: DecisionContext, use_llm: bool | None = None, model: str = "claude-sonnet-4-6") -> str:
    """Generates a natural-language explanation of one control decision.

    use_llm: True forces LLM mode (raises if no API key), False forces
    template mode, None (default) auto-detects based on ANTHROPIC_API_KEY.
    """
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not use_llm:
        return _template_explanation(ctx)

    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = _build_llm_prompt(ctx)
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# --- canned natural-language Q&A over a logged decision history ---
# A handful of keyword-routed questions, per the Phase 5 scope decision to
# avoid building a full RAG/vector-DB stack for a small, fixed question set.

def answer_question(question: str, history: list[DecisionContext], use_llm: bool | None = None) -> str:
    q = question.lower()

    if not history:
        return "No decision history available yet."

    if "why" in q and ("hot" in q or "warm" in q or "cold" in q or "cool" in q):
        return generate_explanation(history[-1], use_llm=use_llm)

    if "predict" in q or "tomorrow" in q:
        avg_cost = sum(h.reward_components["cost"] for h in history) / len(history)
        return (
            f"Based on the last {len(history)} logged steps, average per-step cost penalty was "
            f"{avg_cost:.3f}. A full forecast requires running the trained policy forward through "
            f"the twin with a weather forecast as input -- not yet wired into this Q&A layer "
            f"(see docs/future_work.md)."
        )

    if "uncertain" in q or "confiden" in q:  # matches "confidence" and "confident"
        last = history[-1]
        status = "low confidence, using fallback control" if last.used_fallback_control else "normal confidence"
        return (
            f"Current predictive uncertainty is {last.uncertainty_std:.3f}°C "
            f"(fallback threshold {last.uncertainty_threshold:.3f}°C) -- {status}."
        )

    if "summary" in q or "week" in q:
        dominant_counts = {}
        for h in history:
            dominant_counts[h.dominant_factor] = dominant_counts.get(h.dominant_factor, 0) + 1
        top_factor = max(dominant_counts, key=dominant_counts.get)
        fallback_rate = sum(h.used_fallback_control for h in history) / len(history)
        return (
            f"Summary over {len(history)} logged steps: the dominant reward factor was "
            f"'{top_factor}' in {dominant_counts[top_factor]}/{len(history)} steps. "
            f"Fallback (uncertainty-gated) control was used {fallback_rate*100:.1f}% of the time."
        )

    # default: explain the most recent decision
    return generate_explanation(history[-1], use_llm=use_llm)
