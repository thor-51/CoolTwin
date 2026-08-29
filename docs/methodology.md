# CoolTwin — Methodology

This doc explains *how* each phase was built and why, at a level someone reviewing
the code (or judging the pitch) can follow without reading every source file.
For *what* was built and the results, see the main [README](../README.md); for
the system diagram, see [architecture.md](architecture.md).

## 1. Hybrid digital twin (Phase 2)

**Physics backbone.** The zone is modeled as a 3R2C thermal network (three
resistances, two capacitances) — `T_out` (input) → `T_wall` (state) → `T_in`
(state), with HVAC power and internal gains injected directly at the indoor-air
node. This is a standard grey-box building model: enough physical structure to
be interpretable and stable outside the training distribution, but far cheaper
to simulate than a full EnergyPlus model. See `twin/rc_model.py`.

**Parameter estimation.** Rather than hand-picking R/C values, `fit_rc_params`
fits them to ground-truth zone temperature via nonlinear least-squares
(`scipy.optimize.least_squares`): simulate the 3R2C ODE forward under a
candidate parameter vector, compare against observed `T_in`, and let the
optimizer minimize the residual. This is treated as a real methodological
contribution, not a formality — it's what turns "a plausible thermal model"
into "a thermal model fit to this specific zone's behavior."

**Residual correction.** The RC model alone can't capture nonlinear effects
(solar gain, occupancy-driven internal loads, sensor noise) it wasn't given as
explicit inputs. A small LSTM (`twin/residual_lstm.py`) is trained on a sliding
window of exogenous features (`T_out`, hour-of-day sin/cos, normalized HVAC
power, normalized internal gain, and the RC model's own prediction) to predict
the *residual* — `T_in_ground_truth - T_in_physics` — rather than the absolute
temperature. Two baselines are trained alongside it for an honest comparison:
physics-only (no correction) and a pure-ML `DirectLSTM` with the same
architecture but no physics prior at all, so the RMSE comparison isolates the
effect of the physics prior rather than model capacity.

A shared `build_feature_vector()` helper is used both offline (training-data
construction) and online (`twin/env.py`'s per-step correction) so the two
feature pipelines can't silently drift apart — a real bug class in grey-box
systems that this sidesteps by construction.

## 2. RL agent and reward design (Phase 3)

**Environment.** `CoolTwinEnv` (`twin/env.py`) wraps the hybrid twin as a
standard Gymnasium environment: continuous action space (HVAC setpoint as a
fraction of max power, -1 to 1), observation = zone temp, outdoor temp,
occupancy, price signal, and time features. Continuous was chosen over
discrete actions as the more defensible, standard choice for this class of
control problem.

**Reward.** `R = -(w_cost·cost + w_comfort·discomfort + w_carbon·carbon_kg +
w_peak·peak_frac)`, implemented in `rl/reward.py` as a small, independently
testable module rather than inlined into the environment's `step()`. The four
terms map directly to the four objectives named in the project abstract — no
fifth term was added to avoid diluting the story.

**Multi-objective claim.** A fixed weighted sum by itself doesn't yet
demonstrate multi-objective optimization — it's one point in trade-off space.
`PARETO_WEIGHT_SET` defines five named weightings (cost-, comfort-, carbon-,
and peak-focused, plus balanced) that are each used to train a separate policy
in `rl/pareto.py`, and the resulting (energy, discomfort) pairs are plotted as
the Pareto front. This is what turns "we have weights in the reward" into "we
can show you the actual trade-off curve."

**Algorithms.** PPO and SAC (via `stable-baselines3`, not reimplemented) give
a legitimate on-policy vs. off-policy comparison without the cost or fragility
of reimplementing something more exotic (DreamerV3, MuZero, Decision
Transformer) — see `docs/future_work.md` for why those were scoped out.

**Training budget.** All reported numbers use 50k timesteps (PPO/SAC head-to-
head) and 200k timesteps per weighting (Pareto sweep) — enough to produce a
real, separated trade-off curve on CPU in well under an hour total, rather
than a CI-fast but uninformative budget. See the README's Phase 3 note for the
before/after difference this made.

## 3. Uncertainty quantification (Phase 4)

Two independent methods are trained on the residual model, deliberately kept
to two rather than adding Bayesian NNs, evidential deep learning, or conformal
prediction (see `future_work.md`):

- **MC Dropout** — keep dropout active at inference, run N stochastic forward
  passes, treat the sample variance as predictive uncertainty. Cheap: no
  extra training cost beyond the base model.
- **Deep Ensembles** — 3–5 independently trained residual models with
  different seeds; disagreement across the ensemble is the uncertainty
  signal. More expensive to train, but a genuinely independent estimate to
  compare against MC Dropout.

**Calibration.** Both methods were meaningfully overconfident out of the box
(measured via reliability diagrams and Expected Calibration Error — see
`uncertainty/calibration.py`). Rather than treating that as a dead end, a
standard post-hoc fix (`fit_variance_scale`) rescales predictive variance by a
single scalar fit on a validation split, evaluated on held-out test data —
this is the honest way to report calibration: show the raw number, apply a
principled fix, report the corrected number too.

**Safety layer.** When predictive variance exceeds a threshold, control
defers to the rule-based thermostat instead of trusting the RL policy
(`uncertainty/safety_layer.py`). The method used for this (Deep Ensembles) was
chosen specifically because it showed higher sensitivity to distribution shift
on an out-of-distribution "heatwave" test episode — the property that
actually matters for a safety trigger, not just in-distribution calibration
quality.

## 4. Explainability (Phase 5)

Three layers, cheapest-and-most-defensible first:

1. **Reward decomposition** — at each decision, the contribution of each
   reward term (cost/comfort/carbon/peak) to the current step is already
   computed as part of `compute_reward`; `explainability/reward_decomposition.py`
   just exposes it as a labeled breakdown with a "dominant factor" call.
2. **SHAP** — feature attribution on both the residual model (why did the
   twin predict this temperature) and the RL policy (why did the agent take
   this action), via `explainability/shap_explain.py`.
3. **LLM explanation layer** — a single well-prompted call
   (`explainability/llm_explainer.py`) that takes structured context (state,
   reward decomposition, SHAP top features, uncertainty level) and produces a
   natural-language explanation. Deliberately *not* a multi-agent system —
   one call over structured data, matching the roadmap's explicit warning
   against scope creep here. A deterministic template-based fallback covers
   the no-API-key case, so the same underlying numbers are explainable
   whether or not `ANTHROPIC_API_KEY` is set — this also keeps CI and the
   dashboard usable offline.

## 5. Evaluation protocol (Phase 6)

All controllers — random, rule-based thermostat, PID, single-objective
(cost-only) PPO, multi-objective PPO, multi-objective SAC — are evaluated on
**the same held-out week-long episode** (fixed seed), using the same
accounting for every metric regardless of what each controller was trained
on. This matters: a controller shouldn't get to be scored on an easier episode
than the ones it's being compared against.

Metrics reported: energy (kWh), comfort in actual °C-hours (converted from the
internal per-step accumulator via `compute_comfort_hours`, since the raw
step-summed unit depends on the control interval and isn't comparable across
configurations), peak demand and its % change vs. the rule-based baseline,
carbon emissions, and per-decision inference latency
(`benchmark_inference_latency`) — cheap to measure, and it shows the policy is
actually fast enough to run on a real controller, not just accurate in
simulation.

## 6. Dashboard (Phase 7)

The dashboard (`dashboard/app.py`, logic factored into `dashboard/data.py` so
it's unit-testable without Streamlit) intentionally trains everything at a
short, interactive-speed budget (seconds, not minutes) and labels this
explicitly in the UI — the numbers shown there are for interactivity and
demo-feel, not the report's final numbers, which come from the longer offline
runs in `notebooks/02`, `03`, and `05`. This separation was a deliberate
choice: a dashboard that silently mixes demo-quality and report-quality
numbers is a credibility risk in front of judges who might ask "wait, why do
these numbers not match the README?"

## What's still open

See [`future_work.md`](future_work.md) for scope deliberately left out, and
the README's Phase 1 status note for the one still-open item from the
original plan: Sinergym (EnergyPlus-backed) was not wired in — the twin uses
a custom, lighter RC-only Gym environment instead, documented in
[`sinergym_setup.md`](sinergym_setup.md). This is a real trade-off, not an
oversight, and validating against Sinergym is the next planned step after the
current 7 phases are polished.
