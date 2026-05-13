# tools/halo — internal pointer

User-facing workflow (data collection → fit → use): [`README.md`](README.md).

Design + decisions: [`ms_dev/halo_dev/prediction_model.md`](../../ms_dev/halo_dev/prediction_model.md)
and `ms_dev/halo_dev/CLAUDE.md` §20.

Module layout under `tools/halo/`:
- `fit_halo_cost_model.py` — OLS fit JSONL → `halo_step_v1` JSON.

Tests in `test/registered/halo/`:
- `test_halo_step_cost_model.py`
- `test_halo_cost_model_sampler.py`
- `test_fit_halo_cost_model.py`
