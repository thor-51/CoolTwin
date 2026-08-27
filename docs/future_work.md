# Future Work / Deliberately Out of Scope

These are real, valuable directions that a production version of CoolTwin would need —
we scoped them out of the prototype to focus effort on the core hybrid-twin +
multi-objective-RL + uncertainty + explainability loop, and to keep every claim in the
demo backed by working code.

| Direction | What it would add | Why deferred |
|---|---|---|
| Multi-zone / multi-building | Hierarchical RL across zones/buildings, federated learning across sites without sharing raw data | Single-zone already covers the full research contribution; multi-zone is an engineering scale-up, not a new idea |
| Knowledge graph (Neo4j) | Structured representation of rooms/equipment/maintenance history for complex relational queries | Our query surface (a handful of canned questions) doesn't yet need graph traversal; a KG becomes valuable once the entity count is large |
| Multi-agent orchestration | Separate Weather/Pricing/Fault-Detection/Maintenance agents collaborating | A single well-prompted LLM call over structured context achieves the same explainability outcome with far less failure surface for a live demo |
| RAG over building manuals / ASHRAE standards | Grounded answers to compliance/maintenance questions | Orthogonal to the core control + explainability contribution; valuable as a follow-on |
| Full cloud deployment (K8s, Kafka, multi-service mesh) | Production scalability, real-time event streaming | Prototype validates the approach; infra should be designed for the actual deployment target (e.g. building management system integration) rather than guessed at generically |
| DreamerV3 / MuZero / Decision Transformer | Potentially higher sample efficiency or long-horizon planning | PPO/SAC comparison already demonstrates the multi-objective framework works; these are worth exploring once the core result is validated |
| Bayesian Neural Nets / Evidential DL / Conformal Prediction | Alternative uncertainty quantification methods | MC Dropout + Deep Ensembles already give two independent, well-validated UQ methods with calibration; more methods add comparison value but not new capability |
| Grid/demand-response market integration | Participating in real electricity markets, EV charging, battery scheduling | Natural extension once single-building control is validated |

The design in `docs/architecture.md` and the reward function in `rl/reward.py` are built
so most of these can be added without re-architecting the core twin/agent — e.g. the
reward function already has explicit terms that would extend naturally to grid-interaction
signals.
