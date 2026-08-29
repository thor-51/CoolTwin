"""
app.py

CoolTwin's Phase 7 deliverable: an interactive Streamlit dashboard tying
together every prior phase into one demo-able artifact.

Run with:
    cd CoolTwin
    streamlit run dashboard/app.py

Tabs:
  1. Digital Twin      -- physics-only vs hybrid-corrected vs ground truth
  2. Control & Reward  -- RL policy's decisions + reward decomposition over
                          an episode
  3. Uncertainty       -- predictive uncertainty over the episode + when the
                          safety layer fell back to rule-based control
  4. Pareto Explorer   -- fast preview sweep across reward weightings
                          (see caveat in the UI: not the final numbers)
  5. Ask CoolTwin      -- natural-language Q&A over the logged episode
                          (LLM if ANTHROPIC_API_KEY is set, else template)

All heavy computation (model training, episode simulation) is wrapped in
st.cache_resource / st.cache_data so it only runs once per session, not on
every UI interaction.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from rl.reward import RewardWeights, PARETO_WEIGHT_SET
from dashboard.data import (
    train_demo_residual_model,
    train_demo_ppo,
    get_twin_fidelity_data,
    run_control_episode,
    get_pareto_preview,
)
from explainability.llm_explainer import DecisionContext, answer_question


st.set_page_config(page_title="CoolTwin Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Cached model training / simulation -- runs once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training residual model (demo budget, ~5s)...")
def _residual_model():
    return train_demo_residual_model()


@st.cache_resource(show_spinner="Training PPO policy (demo budget, ~5s)...")
def _ppo_model(weights_key: str):
    weights = PARETO_WEIGHT_SET.get(weights_key, RewardWeights())
    return train_demo_ppo(weights=weights, total_timesteps=8_000)


@st.cache_data(show_spinner="Simulating control episode...")
def _control_logs(weights_key: str, episode_hours: int, seed: int):
    weights = PARETO_WEIGHT_SET.get(weights_key, RewardWeights())
    model = _ppo_model(weights_key)
    logs = run_control_episode(model, _residual_model(), weights, episode_hours=episode_hours, seed=seed)
    return logs


@st.cache_data(show_spinner="Running Pareto preview sweep (fast/demo budget)...")
def _pareto_preview():
    return get_pareto_preview()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("CoolTwin")
st.sidebar.caption("Explainable, uncertainty-aware digital twin for HVAC scheduling")

weights_key = st.sidebar.selectbox(
    "Reward weighting",
    options=list(PARETO_WEIGHT_SET.keys()),
    index=list(PARETO_WEIGHT_SET.keys()).index("balanced"),
    help="Selects a fixed named weighting from rl/reward.py's PARETO_WEIGHT_SET "
         "(cost/comfort/carbon/peak), rather than a continuous slider -- keeps "
         "the demo honest about only having trained a policy for these 5 points, "
         "not a continuum.",
)
episode_hours = st.sidebar.slider("Episode length (hours)", min_value=24, max_value=96, value=48, step=24)
seed = st.sidebar.number_input("Episode seed", value=7, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ All models on this page are trained with a **short, demo-speed budget** "
    "for interactivity. Final report numbers come from the longer offline runs "
    "in `notebooks/02`, `notebooks/03`, and `notebooks/05`."
)

logs = _control_logs(weights_key, episode_hours, seed)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_twin, tab_control, tab_uncertainty, tab_pareto, tab_chat = st.tabs(
    ["🏢 Digital Twin", "🎛️ Control & Reward", "📊 Uncertainty", "⚖️ Pareto Explorer", "💬 Ask CoolTwin"]
)

with tab_twin:
    st.subheader("Physics-only vs hybrid-corrected twin prediction")
    fidelity = get_twin_fidelity_data(_residual_model(), n_steps=96, seed=42)

    df = pd.DataFrame({
        "Ground truth": fidelity.T_in_true,
        "Physics-only (RC network)": fidelity.T_in_physics,
        "Hybrid (RC + residual LSTM)": fidelity.T_in_hybrid,
    })
    st.line_chart(df)

    c1, c2 = st.columns(2)
    c1.metric("Physics-only RMSE", f"{fidelity.rmse_physics:.3f} °C")
    c2.metric("Hybrid RMSE", f"{fidelity.rmse_hybrid:.3f} °C",
              delta=f"{fidelity.rmse_hybrid - fidelity.rmse_physics:+.3f} °C", delta_color="inverse")
    st.caption(
        "Ground truth is a synthetic building with a nonlinear solar-gain effect the "
        "linear RC network can't represent (see twin/data_gen.py) -- this is what the "
        "residual model is correcting for. See README for the full-budget Phase 2 result."
    )

with tab_control:
    st.subheader(f"RL policy decisions over a {episode_hours}h episode ({weights_key} weighting)")

    df_ctrl = pd.DataFrame([{
        "step": l.t, "T_in": l.T_in, "T_out": l.T_out, "action": l.action,
        **l.cost, "dominant": l.dominant_factor,
    } for l in logs])

    st.line_chart(df_ctrl.set_index("step")[["T_in", "T_out"]])
    st.caption("Indoor vs outdoor temperature over the episode.")

    st.markdown("**Reward decomposition (weighted penalty contribution per step)**")
    st.bar_chart(df_ctrl.set_index("step")[["cost", "comfort", "carbon", "peak"]])

    dominant_counts = df_ctrl["dominant"].value_counts()
    st.markdown("**Dominant reward factor across the episode**")
    st.bar_chart(dominant_counts)

with tab_uncertainty:
    st.subheader("Predictive uncertainty and the safety-layer fallback")

    df_unc = pd.DataFrame([{"step": l.t, "uncertainty_std": l.uncertainty_std, "used_fallback": l.used_fallback} for l in logs])
    st.line_chart(df_unc.set_index("step")[["uncertainty_std"]])

    fallback_rate = df_unc["used_fallback"].mean()
    st.metric("Fallback control triggered", f"{fallback_rate*100:.1f}% of steps")
    st.caption(
        "When the residual model's predictive uncertainty exceeds the threshold, control "
        "defers to the rule-based thermostat instead of trusting the RL policy "
        "(uncertainty/safety_layer.py). See the Phase 4 result for the full calibration "
        "story and out-of-distribution sensitivity comparison between MC Dropout and "
        "Deep Ensembles."
    )

with tab_pareto:
    st.subheader("Pareto front preview (fast/demo training budget)")
    st.warning(
        "This is a FAST preview (short training budget) for interactivity, not the "
        "final Pareto front. See `results/pareto_front.png` and `rl/pareto.py` for the "
        "longer, more representative sweep."
    )
    if st.button("Run Pareto preview sweep (trains 5 small policies, ~30-60s)"):
        pareto_results = _pareto_preview()
        df_pareto = pd.DataFrame([
            {"weighting": name, "energy_kWh": m["total_energy_kwh"], "discomfort": m["total_discomfort"]}
            for name, m in pareto_results.items()
        ])
        st.scatter_chart(df_pareto, x="energy_kWh", y="discomfort", color="weighting")
        st.dataframe(df_pareto, width="stretch")

with tab_chat:
    st.subheader("Ask CoolTwin about this episode")
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_llm:
        st.success("ANTHROPIC_API_KEY detected -- using the real LLM.")
    else:
        st.info("No ANTHROPIC_API_KEY set -- using the grounded template explainer. "
                "Set the env var and restart to use the real LLM.")

    contexts = [
        DecisionContext(
            T_in=l.T_in, T_out=l.T_out, occupancy=bool(l.occupancy), price=l.price,
            action=l.action, reward_components=l.cost, dominant_factor=l.dominant_factor,
            uncertainty_std=l.uncertainty_std, uncertainty_threshold=0.15,
            top_shap_features=[], used_fallback_control=l.used_fallback,
        )
        for l in logs
    ]

    example_qs = ["Why is the room warm?", "How confident are you right now?", "Give me a weekly summary"]
    cols = st.columns(len(example_qs))
    clicked = None
    for col, q in zip(cols, example_qs):
        if col.button(q):
            clicked = q

    user_q = st.text_input("Or ask your own question:")
    question = clicked or user_q

    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question, contexts, use_llm=use_llm)
        st.markdown(f"**Q:** {question}")
        st.markdown(f"**A:** {answer}")
