# CoolTwin — Pitch Prep (Phase 9)

Working doc for the Schneider Electric Co-Creation Challenge pitch. Every number
below is pulled from `results/` and the README as of Phase 8 — update this file
if you re-run anything with a larger budget before the final pitch.

## Story arc (~5 slides worth)

1. **Problem.** Building HVAC control today is either a fixed-schedule thermostat
   or a single-objective optimizer — both ignore the real trade-off between cost,
   comfort, carbon, and peak demand, and neither can explain *why* it did what it
   did or *when it isn't sure*.
2. **Approach.** A hybrid digital twin (physics + learned correction) feeds a
   multi-objective RL agent, wrapped with uncertainty quantification (so it knows
   when to defer to a safe fallback) and explainability (so a building operator
   can ask why).
3. **Results.** Four concrete numbers, in order of how well they're understood
   by a non-ML audience:
   - Hybrid twin beats physics-only and pure-ML on prediction accuracy.
   - The Pareto front — a real, separated trade-off curve, not a flat line.
   - SAC beats rule-based/PID/random on cost, comfort, *and* carbon simultaneously.
   - The single-objective-vs-multi-objective comparison, which is the literal
     empirical proof of the problem statement in step 1.
4. **Why Schneider should care.** Maps directly onto EcoStruxure Building /
   demand-response products — say this explicitly, don't make them infer it.
5. **What's next, honestly.** Point at `docs/future_work.md` and the still-open
   Sinergym validation step. Judges respond well to "here's what we deliberately
   didn't build and why," not to pretending the scope is finished.

## The numbers to have memorized

| Claim | Number | Source |
|---|---|---|
| Hybrid twin beats both baselines | Hybrid RMSE 0.18°C vs physics-only 0.42°C vs pure-ML 0.21°C | `notebooks/01_train_residual_lstm.py` |
| Pareto front is real | Energy ranges 129–248 kWh across 5 weightings; comfort-focused trades ~2x the energy of the leanest weighting for the lowest discomfort | `results/pareto_front.png` |
| SAC beats all 3 baselines simultaneously | 205 kWh energy / 92.0°C-hr comfort / 92.3 kg carbon vs. rule-based's 210.5 kWh / 95.5°C-hr / 94.7 kg | `results/final_evaluation_table.md` |
| Single- vs multi-objective trade-off (headline finding) | Cost-only PPO cuts energy 92% and peak 84% vs rule-based — but comfort violation is ~2.7x worse (255.7°C-hr vs 86–92°C-hr) | `results/final_evaluation_table.md` |
| Uncertainty is honestly calibrated | Raw ECE 0.33 (MC Dropout) / 0.45 (Deep Ensemble) → calibrated to 0.009 / 0.029 via post-hoc variance scaling | `results/calibration_reliability_diagram.png` |
| Safety layer actually responds to risk | Fallback triggers on 21% of steps in-distribution vs 59% during an out-of-distribution heatwave episode | `notebooks/03_uncertainty_quantification.py` |
| Inference is fast enough for a real controller | Sub-millisecond to ~0.26ms per decision across all controllers | `results/final_evaluation_table.md` |

**Do not round these up.** If asked for the exact number, give the exact number —
these are strong enough as-is, and getting caught rounding favorably is worse
than the number itself.

## Prepared answers for hard questions

**"Why RC network and not a full EnergyPlus digital twin at inference time?"**
Speed and deployability. A 3R2C ODE solve is orders of magnitude cheaper than an
EnergyPlus timestep, which matters if this runs on a real building controller
rather than a cloud GPU. The residual LSTM recovers most of what pure physics
misses, and we show that quantitatively (0.18°C vs 0.42°C RMSE) rather than
asserting it.

**"Did you validate against real building physics, or only your own synthetic
RC world?"** *(the honest weak point — don't dodge it)* Not yet at full fidelity
— we built and validated against a custom lightweight RC-only Gym environment
rather than wiring in Sinergym/EnergyPlus, documented in
`docs/sinergym_setup.md`. That was a deliberate scope trade-off to keep the
pipeline fast and CI-friendly during the build. It's the next concrete step:
validate the twin's predictions against Sinergym's EnergyPlus backend before
calling the physics fidelity claim complete.

**"How do you know your RL agent is safe to deploy?"**
It doesn't get unconditional trust — the uncertainty-gated safety layer defers
to rule-based control when the residual model's predictive variance is high,
and we show that trigger rate is sensitive to actual risk: 21% in-distribution
vs 59% on an out-of-distribution heatwave test. That's the concrete demo
moment — show it getting cautious live.

**"How would this scale to a real multi-zone building?"**
Single-zone was the right scope for validating the core research contribution
(hybrid twin + multi-objective RL + uncertainty + explainability working
together). Multi-zone is an engineering scale-up, not a new idea — hierarchical
RL across zones and federated learning across sites without sharing raw data
are the natural extensions, in `docs/future_work.md`.

**"Why PPO and SAC and not something more advanced (DreamerV3, MuZero,
Decision Transformer)?"**
Those are disproportionate effort for this problem size and harder to defend
under questioning than "we compared an on-policy and an off-policy algorithm
and reported both honestly." The PPO/SAC comparison already demonstrates the
multi-objective framework works; more exotic algorithms are worth exploring
once the core result is validated, not before.

**"Why only one LLM call instead of an agentic system?"**
Explainability doesn't require multiple agents — reward decomposition, SHAP
attribution, and uncertainty are all already computed as structured data; one
well-prompted call turns that into natural language. A multi-agent system adds
failure surface in a live demo without adding explanatory power we don't
already have.

**"What happens if the LLM API is down during the demo?"**
Nothing — `explainability/llm_explainer.py` has a deterministic template
fallback that uses the exact same underlying numbers, and the dashboard uses it
automatically when `ANTHROPIC_API_KEY` isn't set. Worth demonstrating this
explicitly rather than hoping it doesn't come up: turn off the API key live and
show the answer doesn't change in substance.

## Demo script (dry run this exact sequence)

1. Open the dashboard, **Digital Twin tab** — show hybrid tracking ground truth
   tighter than physics-only.
2. **Control & Reward tab** — pick the balanced weighting, walk through reward
   decomposition on a couple of steps.
3. **Uncertainty tab** — this is the best live moment. Point out a step where
   the fallback triggered and explain why.
4. **Pareto Explorer tab** — click through 2–3 weightings, make the trade-off
   visible without narrating every axis.
5. **Ask CoolTwin tab** — ask a canned question live, then ask something
   off-script to show it's not just a scripted response.
6. Close on the `results/final_evaluation_table.md` headline number (2.7x
   comfort trade-off) — this is the one number that ties directly back to the
   problem statement from slide 1.

Have the **3–5 minute recorded demo video** ready as the WiFi fallback per the
original roadmap — don't build it the night before.

## Rehearsal checklist

- [ ] Full run-through, timed, at least 3 times before the pitch
- [ ] Someone plays devil's advocate and asks the hard questions above out of order
- [ ] Confirm the dashboard runs on the actual pitch-day laptop, not just your dev machine
- [ ] Recorded fallback video is current (re-record if any numbers changed since last recording)
- [ ] Everyone on the team can answer the Sinergym question without looking surprised — it's the one gap a technical judge is most likely to probe
