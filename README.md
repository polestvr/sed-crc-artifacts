# sed-crc-artifacts

Code and cached artifacts for the paper:

> G. Mkrtchian, "Detection with Guarantees: Distribution-Free Control of
> Non-Monotone Event-Level Miss Rates in Polyphonic Sound Event Detection."

Every table in the paper regenerates from the cached per-clip count tensors in
this repository by deterministic array arithmetic; no audio, model inference,
or GPU is required.

## Contents

- `scripts/` — calibration routes, experiment drivers, analyses, and table
  generation. `scripts/sed_crc/` is the shared library: cache I/O (`gt.py`),
  per-clip loss statistics via `sed_scores_eval` (`stats.py`), calibration
  routes (`routes.py`), split/gate/efficiency machinery (`evalx.py`).
- `results/` — one JSON per experiment; the numbers behind every table.
- `cache/` — cached per-clip score curves (`*_scores.pkl`), per-clip count
  tensors (`stats_*.npz`), clip durations and metadata for the primary
  training run, plus the tuned cSEBB parameters and the reproduction-gate
  record.
- `cache_seed2/`–`cache_seed4/` — the same for the three additional
  independent training runs (used by the floor-stability and route-ranking
  replications).

## Environment

Python 3.10+ with `numpy`, `scipy`, `pandas`, and `sed_scores_eval==0.0.4`
(pinned: its matching implementations enter the loss definition).

```bash
pip install numpy scipy pandas sed_scores_eval==0.0.4
```

## Regenerating the paper's tables

```bash
python scripts/make_tables.py
```

renders `results/*.json` into LaTeX fragments under `tables_out/` and re-runs
nothing. To recompute a `results/` entry from the cached tensors, e.g. the
headline operating point:

```bash
python scripts/run_experiment.py --exp-id demo --variant csebbmax \
    --matching intersect50 --alpha 0.2 --route ltt_split_clipmean --grouping marginal
```

Split seeds are deterministic: split `s` of the 100 calibration/evaluation
splits uses `numpy.random.default_rng(seed_base + s)` over the clip
permutation (`seed_base` 0 for protocol runs, 500 for verification runs), and
route-internal randomness uses `default_rng(10000 + s)`; see
`scripts/sed_crc/evalx.py`. Analyses: `instrument_study.py` (10,000-split gate
study), feasibility floors (`feasibility_probe.py`, `floors_extra.py`,
`floors_valpool.py`, `seed_floors.py`), stress tests (`stress_subgroups.py`,
`stress_crossdecode.py`, `stress_crosscorpus.py`, `loo_environment.py`),
confirmations (`holdout_confirm.py`, `replicate_seeds.py`,
`group_conditional.py`), interval estimates (`paired_diffs.py`, `misc_cis.py`,
`crcc_meanrisk.py`, `pooledcap_analysis.py`).

Scripts read the primary cache from `./cache` by default; point
`SED_CRC_CACHE` at `cache_seed2`/`cache_seed3`/`cache_seed4` for the
replication runs.

## Rebuilding the caches from audio (optional)

The score curves were produced by `extract_scores.py` (and
`reproduce_gate.py` / `rescore_topk.py` for the cSEBB variants) from a frozen
RealDESED baseline checkpoint, and the count tensors by `build_stats.py`.
Rebuilding them requires the RealDESED baseline code and a trained checkpoint
(checkpoints are not part of this release; the cached curves make them
unnecessary for reproducing the paper). Data:

- RealDESED: https://zenodo.org/records/20056072
- DESED public evaluation set: https://zenodo.org/records/3588172

## License

MIT — see `LICENSE`.
