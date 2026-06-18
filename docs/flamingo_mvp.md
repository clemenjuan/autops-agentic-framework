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

| Planned config ID | YAML organization | Literature role |
|---|---|---|
| `flamingo_sas_ag_symb` | `sas` | one global agent controls the constellation |
| `flamingo_cmas_ag_symb` | `centralized_mas` | mission manager / orchestrator with local satellite agents |
| `flamingo_imas_ag_symb` | `independent_mas` | local satellite agents, no inter-agent communication |
| `flamingo_dmas_ag_symb` | `decentralized_mas` | peer-to-peer coordination with consensus |
| `flamingo_hmas_ag_symb` | `hybrid_mas` | clustered or heterogeneous hierarchy plus peer/local behavior |

Decentralized MAS should be part of the first organization sweep, not a later
optional add-on. The first DMAS implementation can use all-to-all consensus for
`N = 3`; topology ablations such as ring, mesh, or visibility-limited links can
come after the five-config baseline is runnable.

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

