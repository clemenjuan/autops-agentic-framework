# Scripts

Utility scripts that sit outside the main `autops` CLI.

## Canonical Experiment Helpers

- `generate_experiment_configs.py`: regenerate the committed EventSat experiment matrix.
- `generate_ssa_configs.py`: regenerate the committed SSA AO config slice.
- `smoke_llm.py`: run mocked LLM smoke checks without a live endpoint.
- `export_eventsat_world_model_traces.py`: export EventSat telemetry for world-model datasets.

## Results And Boards

- `refresh_board.py`: one-shot refresh of extracts and generated boards.
- `build_results_board.py`: EventSat board builder.
- `build_results_inspector.py`: standalone result inspector.
- `build_index.py`: board landing page.
- `extract_telemetry.py`: compact telemetry for board drill-downs.
- `recompute_metrics.py`: recompute metrics from DEBUG decision traces.

Do not run board refreshes in loops. Generated files belong under ignored `data/figures/`.
