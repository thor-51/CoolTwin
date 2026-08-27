# CoolTwin — Realistic Implementation Roadmap
### For Schneider Electric Co-Creation Challenge 2026

---

## 0. Ground rules before you write a line of code

**Read this first, seriously.** You were shortlisted from 300+ teams on your *abstract* — a well-scoped, honest research idea. Do not let the "build it like DeepMind" mega-prompt derail you. That document is a scope-creep generator, not a plan. Judges at a corporate co-creation challenge want:

1. A **working prototype** they can watch run
2. **Honest, defensible technical depth** in the 4–5 things you actually built
3. A clear story: *problem → approach → result → why it matters for Schneider's business*
4. Evidence you understand the trade-offs, not just that you can list 40 acronyms

A team that deeply nails the hybrid twin + RL + uncertainty + explainability loop will beat a team that shallowly bolts on Kafka, Kubernetes, and a knowledge graph nobody queries live. **Depth over breadth, always.**

### Team split (3–4 people)
| Role | Owns |
|---|---|
| **Person A — Digital Twin / Physics** | RC thermal model, parameter estimation, hybrid residual model, simulation environment integration |
| **Person B — RL / Control** | Environment wrapper, reward function, PPO/SAC training, multi-objective weighting, evaluation vs baselines |
| **Person C — Uncertainty + XAI** | MC Dropout/ensembles, calibration, SHAP/reward decomposition, LLM explanation layer |
| **Person D (if 4th) — Systems/Frontend** | Dashboard, GitHub repo hygiene, API layer, demo packaging, docs |

If only 3, Person D's work gets split across A/B/C in the final phase — it's the smallest-effort track.

---

## Phase 1 — Foundations (Week 1–2)
**Goal:** Everyone can run a baseline simulation and agree on scope.

1. Set up the repo (structure below) with `environment.yml` / `requirements.txt`, pre-commit hooks, and a `docs/` folder from day one — habits are cheaper to build early.
2. Install and run **Sinergym** (EnergyPlus + Gymnasium wrapper) — it's the more research-credible choice over CityLearn for a single-zone HVAC problem since it gives you real building physics (EnergyPlus backend) rather than a pre-computed dataset. Get one default scenario running end-to-end with a random-action baseline.
3. Get **weather + occupancy data** flowing (Sinergym ships this; don't build your own ingestion pipeline — that's wasted effort for the demo).
4. Write down, as a team, the **exact scope decision**: single building zone, single HVAC unit, discrete or continuous action space (pick continuous — more defensible), and your 4 reward terms (energy, comfort, carbon, peak demand — matches your abstract, don't add more).
5. Deliverable: `notebooks/00_baseline_random_agent.ipynb` running in CI, logged to GitHub.

---

## Phase 2 — Hybrid Digital Twin (Week 2–4)
**Owner: Person A**

1. **RC thermal network**: model the zone as a 2R2C or 3R2C network (wall, air, thermal mass nodes). This is the "physics" backbone — implement it as a small differential-equation solver (scipy `odeint` or a simple Euler integrator is fine and easier to defend than something exotic).
2. **Parameter estimation**: fit R/C values against Sinergym's ground-truth zone temperature using least-squares or a simple gradient-based fit. This step alone is a legitimate research contribution — document it well.
3. **Residual correction model**: train an LSTM (or even a small GRU/MLP with lag features — simpler to defend) to predict the *error* between RC-model prediction and ground truth. This is your hybrid grey-box model.
4. **Justify grey-box over pure PINN/Neural-ODE** in your report: physically interpretable, cheaper to train, easier to explain to a non-ML audience (which matters — Schneider judges include building engineers, not just AI researchers). Mention PINNs/Neural ODEs as "considered alternatives" in 2–3 sentences — don't build them unless Phase 2 finishes early.
5. Deliverable: `twin/rc_model.py`, `twin/residual_lstm.py`, a notebook showing hybrid model beats pure-physics and pure-ML baselines on held-out temperature prediction (RMSE table).

---

## Phase 3 — RL Agent (Week 3–6, overlaps Phase 2)
**Owner: Person B**

1. Wrap the hybrid twin as a Gymnasium environment (`CoolTwinEnv`) — state = zone temp, outdoor temp, occupancy, price signal, time features; action = HVAC setpoint/power.
2. Implement the **multi-term reward function** exactly as in your abstract:
   `R = -w1*cost - w2*discomfort - w3*carbon - w4*peak_penalty`
   Keep it to these 4 terms. Document the math clearly (this is a place judges will ask "why these weights?").
3. Train **PPO** first (stable-baselines3 — don't reimplement from scratch, that's wasted time with no marginal credibility gain). Get it beating a rule-based thermostat baseline.
4. Add **SAC** as a second algorithm for comparison — this gives you a legitimate "we compared algorithms" section without needing DreamerV3/MuZero/Decision Transformer, which are disproportionate effort for this problem size and hard to defend if asked "why didn't you just use PPO."
5. **Pareto front**: train with 4–5 different reward weightings, plot cost vs. comfort trade-off curve. This single plot is one of your strongest visuals — it directly demonstrates "multi-objective optimization" from your abstract.
6. Deliverable: `rl/train_ppo.py`, `rl/train_sac.py`, `results/pareto_front.png`, comparison table vs rule-based/PID baseline.

---

## Phase 4 — Uncertainty Quantification (Week 5–7)
**Owner: Person C**

1. Implement **MC Dropout** on the residual LSTM (cheap, fast, defensible) — run N stochastic forward passes, report predictive variance.
2. Implement **Deep Ensembles** (3–5 independently trained residual models) as your second method — this gives you a genuine comparison, matching your abstract's "MC dropout or deep ensembles."
3. **Calibration**: build a reliability diagram (predicted confidence vs. observed accuracy) — this is a strong, cheap-to-build visual that signals rigor.
4. **Decision-time use of uncertainty**: when predictive variance is high, have the agent fall back to a conservative rule-based action (this is your "safety layer" — simple to implement, very effective as a demo moment: "watch it get cautious when it's unsure").
5. Skip Bayesian Neural Nets, Evidential Deep Learning, Conformal Prediction, and Distributional RL unless Phase 4 finishes with 2+ weeks to spare — mention them as future work.

---

## Phase 5 — Explainability (Week 6–8)
**Owner: Person C**

1. **Reward decomposition**: at each decision, show the contribution of each reward term (energy/comfort/carbon/peak) to the action taken — this is cheap (you already compute it) and very demo-friendly.
2. **SHAP** on the residual model and/or the RL policy — feature attribution for "why did the twin predict this temperature" / "why did the agent take this action."
3. **LLM explanation layer** (this is your one "wow" AI-integration feature, keep it lean):
   - Use a single LLM call (via API) that takes structured context — current state, reward decomposition, SHAP top features, uncertainty level — and turns it into a natural-language explanation: *"HVAC increased cooling because occupancy rose and electricity price is currently low, with high confidence (low residual uncertainty)."*
   - This is NOT an agentic multi-agent system. It's one well-prompted LLM call over structured data. Don't build "10 collaborating agents" — that's the single biggest scope trap in the mega-prompt document and the hardest thing to fake convincingly in a live demo.
   - Optionally support 3–4 canned natural-language questions ("Why is the room warm?", "Predict tomorrow's cost") mapped to structured queries over your logged decision history — this gives you a working "chat with the twin" demo without needing RAG/vector DB/knowledge graph infrastructure.

---

## Phase 6 — Evaluation (Week 8–9)
**Owner: Person B, supported by all**

Compare your final policy against:
- Rule-based / fixed-schedule thermostat
- PID controller (simple to implement, standard baseline)
- Single-objective PPO (cost-only reward)

On these metrics: energy consumption, comfort violation (°C-hours out of band), peak demand reduction, carbon emissions (using a static or time-varying grid carbon intensity signal — ElectricityMap or a simple diurnal proxy is fine, don't build a full carbon market model). Also report calibration error and inference latency — cheap to compute, shows engineering maturity.

This table + the Pareto front plot are your two most important results for the pitch.

---

## Phase 7 — Dashboard & Demo Packaging (Week 8–10)
**Owner: Person D / shared**

1. Build a **Streamlit dashboard** (not a full React/Next.js/D3/Three.js stack — that's 3+ weeks of frontend work for marginal judging benefit vs. a clean Streamlit app that live-updates). Show:
   - Live twin state (actual vs. RC-predicted vs. hybrid-corrected temperature)
   - Current action + reward decomposition bar chart
   - Uncertainty band on predictions
   - Pareto front explorer (slider over reward weights)
   - The LLM chat box for natural-language questions
2. If you have real spare time and a 4th person free, a **Next.js + Tailwind** version is a nice upgrade — but only after the Streamlit version works end-to-end. Never let frontend polish block the pipeline being demoable.
3. Record a 3–5 minute demo video as a fallback in case live demo has issues (WiFi at pitch venues is not to be trusted).

---

## Phase 8 — GitHub Repo & Documentation (ongoing, finalize Week 9–10)

```
CoolTwin/
├── README.md                  # problem, architecture diagram, results, how to run
├── environment.yml
├── twin/
│   ├── rc_model.py
│   ├── residual_lstm.py
│   └── hybrid_twin.py
├── rl/
│   ├── env.py                 # Gymnasium wrapper
│   ├── reward.py
│   ├── train_ppo.py
│   ├── train_sac.py
│   └── pareto.py
├── uncertainty/
│   ├── mc_dropout.py
│   ├── ensembles.py
│   └── calibration.py
├── explainability/
│   ├── reward_decomposition.py
│   ├── shap_explain.py
│   └── llm_explainer.py
├── dashboard/
│   └── app.py                 # Streamlit
├── evaluation/
│   ├── baselines.py
│   └── metrics.py
├── notebooks/
├── results/                   # plots, tables, saved metrics
├── docs/
│   ├── architecture.md        # Mermaid diagrams
│   ├── methodology.md
│   └── future_work.md         # everything you scoped OUT — knowledge graph, multi-agent, K8s, etc.
└── tests/
```

Put the "impressive but unbuilt" ideas from the mega-prompt (knowledge graph, multi-agent orchestration, federated learning, K8s deployment, Kafka event mesh) into `docs/future_work.md` as a thoughtful roadmap — this actually reads *well* to judges: it shows you understand the full industrial picture and made deliberate scoping choices, rather than either ignoring it or half-building it.

---

## Phase 9 — Pitch Prep (final week)

1. **Story arc**: problem (black-box, single-objective HVAC control) → your framework (hybrid twin + multi-objective RL + uncertainty + explainability) → results (energy/comfort/carbon numbers, Pareto front) → why Schneider should care (this maps directly onto EcoStruxure Building / demand-response products — say so explicitly).
2. Prepare answers for the obvious hard questions:
   - "Why RC network and not a full EnergyPlus digital twin at inference time?" → speed, deployability on edge/low-compute building controllers.
   - "How do you know your RL agent is safe to deploy?" → uncertainty-gated fallback to rule-based control (Phase 4).
   - "How would this scale to a real multi-zone building?" → point to `docs/future_work.md`, discuss hierarchical RL and federated learning as the natural extension, be honest about what's not yet built.
3. Rehearse the live demo start-to-finish at least 3 times before the pitch.

---

## What to explicitly cut (put in future_work.md, don't build)

- Kubernetes/Kafka/full microservices mesh — a monolith + Streamlit is fine for a prototype
- Neo4j knowledge graph — not needed unless you have a genuinely multi-entity query use case
- Multi-agent orchestration (10 collaborating agents) — one well-designed LLM explanation call does the job
- DreamerV3/MuZero/Decision Transformer — PPO + SAC comparison is sufficient and far more defensible
- Full AWS production deployment — local/Colab/single-cloud-VM training is fine; mention cloud architecture as a design doc, not a live deployment
- RAG over building manuals/ASHRAE standards — interesting but orthogonal to your core contribution; only add if everything else is done early

---

## Suggested week-by-week timeline (compress/expand based on your actual deadline)

| Week | Milestone |
|---|---|
| 1–2 | Repo setup, Sinergym running, scope locked |
| 2–4 | RC model + parameter estimation + residual LSTM (hybrid twin working) |
| 3–6 | Gym wrapper, reward function, PPO training beating rule-based baseline |
| 5–7 | SAC comparison, Pareto front across reward weightings |
| 5–7 | MC Dropout + Ensembles + calibration diagram |
| 6–8 | Reward decomposition, SHAP, LLM explanation layer |
| 8–9 | Full baseline comparison table, final metrics |
| 8–10 | Dashboard, GitHub polish, docs, demo video |
| Final week | Pitch deck, rehearsal |

Tell me your actual deadline and I can convert this into exact dated milestones, or we can start right now on Phase 1 — I can help you scaffold the repo, write the RC thermal model, or set up the Sinergym environment first.
