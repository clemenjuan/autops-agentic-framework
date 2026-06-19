# Flamingo-Lite Multi-Satellite MVP

This is the first multi-satellite increment after the EventSat LLM closeout. Its
purpose is not to build the final Vyoma/Flamingo model in one jump; it is to make
the Organisation axis measurable with a small SSA scheduling scenario.

## Scope

Start with a lightweight SSA environment at `N = 3`, then scale the same scenario
to `N = 6` and `N = 12`. Keep representation and operational paradigm fixed for
the first sweep so the experiment isolates organization topology.

Recommended first sweep:

- Scenario: `flamingo`
- Mission domain: SSA target tracking / revisit scheduling
- Representation: `symb`
- Operational paradigm: `ag`
- Behaviour: `hand_designed`
- Seeds: paired across all organization and scale runs

This avoids waiting on RL and avoids mixing the organization question with the
representation question.

## Five Literature Configs

These are the five organization configs to preserve from the literature plan.
The filename token is short; the YAML `agent_organization` value uses the runner's
current internal name.

| Planned config ID | YAML organization | Status | Literature role |
|---|---|---|---|
| `flamingo_sas_ag_symb` | `sas` | runnable | one global agent controls the constellation |
| `flamingo_cmas_ag_symb` | `centralized_mas` | runnable | mission manager / orchestrator with local satellite agents |
| `flamingo_imas_ag_symb` | `independent_mas` | runnable | local satellite agents, no inter-agent communication |
| `flamingo_dmas_ag_symb` | `decentralized_mas` | deferred | peer-to-peer coordination with consensus |
| `flamingo_hmas_ag_symb` | `hybrid_mas` | deferred | clustered or heterogeneous hierarchy plus peer/local behavior |

Decentralized MAS should be part of the first organization sweep, not a later
optional add-on. The first DMAS implementation can use all-to-all consensus for
`N = 3`; topology ablations such as ring, mesh, or visibility-limited links can
come after the five-config baseline is runnable.

### How the organisation axis is made measurable (and a CMAS caveat)

The symbolic representation `rule_based_flamingo` plans only the satellites
present in *its* observation. The organisation therefore controls the outcome
through **what each agent sees** and **how the actions are merged**:

- `sas` — one agent sees the whole constellation and produces a globally
  deconflicted assignment.
- `independent_mas` (IMAS) — each agent sees only its own satellite (and only
  that satellite's visible tasks) and acts alone; `collect_actions` merges the
  per-satellite actions **without deconfliction**, so independent agents collide
  on the same high-priority RSO and the environment counts duplicates.

For the axis to actually separate these, the scenario must contain contention:
`configs/scenarios/flamingo.yaml` sets `satellite_phase_shift: 0` so the whole
constellation sees the **same** RSO windows at the same time and must compete for
the top-priority targets. Under that scenario SAS keeps duplicates at zero while
IMAS wastes a large fraction of its observations (and loses utility/coverage).

**CMAS honesty note.** `centralized_mas` currently gives its mission manager and
every local agent the *full* observation and, via `rule_based_flamingo`, each
produces the same global greedy assignment; `collect_actions` then returns one
local agent's (already global) plan. So with this representation **CMAS is
expected to match SAS exactly on mission metrics** — the manager directive is
threaded as a message but the symbolic core does not read it. CMAS only diverges
from SAS once the representation consumes the manager directive or the manager
hands out non-overlapping target *partitions* to the locals; until then CMAS
differs from SAS only in coordination cost (it runs `N + 1` agent loops), not in
mission outcome. The measurable contrast in this first sweep is therefore
**{SAS, CMAS} vs IMAS**, not SAS vs CMAS.

## Scenario MVP

The environment should implement only the minimum dynamics needed to compare
organization:

- A catalog of RSOs with priority weights and time-varying visibility windows.
- Per-satellite observation capacity: one target per step or per slot.
- Simple pointing/FOV feasibility and optional slew/handoff penalty.
- Per-satellite power/data budgets at a coarse level.
- Optional ISL availability/bandwidth for DMAS/HMAS communication cost.
- A global action dictionary keyed by `satellite_id`, matching
  `SatelliteEnvironment.step(actions)`.

The MVP does not need full EventSat subsystem fidelity. It needs a faithful
coordination bottleneck: multiple agents must sometimes compete for the same RSO,
miss high-priority opportunities, or duplicate observations unless the topology
coordinates them.

## Metrics

Keep the EventSat metric registry where possible, but add/derive SSA-specific
readouts:

- Mission utility: weighted target coverage / successful observations.
- Mean and peak revisit time by RSO priority class.
- Duplicate observation rate: wasted simultaneous observations of the same RSO.
- Handoff success rate between satellites.
- Coordination cost: messages, consensus rounds, or manager directives.
- Scale efficiency M-10: `(U(N) / N) / U(1)` for `N in {3, 6, 12}`, with an
  `N = 1` Flamingo anchor run used for normalization.

## Implementation Gates

1. Add `src/environment/scenarios/flamingo.py` and
   `configs/scenarios/flamingo.yaml`.
2. Teach `ExperimentRunner._create_environment()` and the metrics factory to
   instantiate Flamingo.
3. Implement `independent_mas`, `decentralized_mas`, and `hybrid_mas` for
   `N >= 3`; keep the existing early guard for EventSat/N=1.
4. Add the five planned YAML configs once they are runnable.
5. Run a smoke matrix at `N = 3`, then scale to `N = 6` and `N = 12`.

## Paper Bridge

EventSat closes the representation/paradigm benchmark on a real single-satellite
mission. Flamingo-lite opens the organization component under a controlled
multi-satellite SSA task. That gives a clean two-stage story: first validate the
architectures on a high-fidelity mission, then test whether the organization
dimension scales or collapses under coordination overhead.

