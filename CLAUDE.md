# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Poker44 is a **Bittensor subnet (netuid 126)** for detecting bots in online poker. Validators query miners with chunks of poker hands, score the returned bot-risk predictions, and set weights on-chain. The repo contains the reference subnet code (`poker44/`, `neurons/`) plus a *separate* private ML solution under `my_solution/` (see below).

There are two distinct bodies of work here — keep them straight:

- **Subnet reference code** (`poker44/`, `neurons/`, `tests/`, `scripts/`, `docs/`) — the validator/miner protocol shipped to the network. The shipped miner is a deliberately simple heuristic baseline.
- **`my_solution/`** — the user's own competitive miner ML pipeline (feature extraction + XGBoost). This is the actual work product; the subnet code is mostly the fixed contract it must satisfy.

## Commands

```bash
pip install -e .                       # install package + deps (Python >=3.10)
python -m pytest tests/ -q             # run the full test suite (also works: python -m unittest discover -s tests)
python -m pytest tests/test_burn_weights.py -q          # single test file
python -m pytest tests/test_burn_weights.py::BurnWeightTests::test_podium_burn_assigns_uid_zero_and_top_three   # single test
```

Tests are stdlib `unittest.TestCase` classes; pytest is only the runner. There is no lint config.

Validators and miners run under **PM2** via the shell launchers (not invoked directly in production):

```bash
./scripts/validator/run/run_vali.sh    # starts neurons/validator.py under pm2 as poker44_validator
./scripts/miner/run/run_miner.sh       # starts neurons/miner.py under pm2 as poker44_miner
```

Both scripts are driven entirely by environment variables (wallet name/hotkey, netuid, `POKER44_*` knobs). Read the top of each script for the full list and defaults.

## Architecture (subnet code)

The validator/miner neurons subclass Bittensor base scaffolds in `poker44/base/`. Entry points are `neurons/validator.py` and `neurons/miner.py`.

**Validator cycle** — `poker44/validator/forward.py::forward` is the heart of the system, called repeatedly by the base run loop:

1. `ProviderRuntimeDatasetProvider.fetch_hand_batch()` (`runtime_provider.py`) pulls a labeled eval snapshot from the central platform API (`/internal/eval/current`). `provider_runtime` is the **only** supported runtime mode — legacy local runtime was removed and the validator raises if `POKER44_RUNTIME_MODE != provider_runtime`.
2. Each `LabeledHandBatch` becomes one **chunk** (a list of hand dicts). Labels: `is_human=False` → label `1` (bot), `is_human=True` → label `0`.
3. Hands are sanitized through `payload_view.prepare_hand_for_miner` before sending — this **strips leakage keys** (`label`, `is_bot`, `bot_family_id`, etc.) and normalizes amounts to bucketed BB units. Miners never see labels or true blinds.
4. Miners are queried via `DetectionSynapse(chunks=...)` (`synapse.py`). The contract: **one `risk_score` per chunk**; a response whose score count ≠ chunk count is discarded entirely.
5. Per-UID predictions/labels accumulate in buffers; rewards are computed over a window equal to the current chunk count.
6. `_select_weight_targets` turns rewards into on-chain weights, then `update_scores` / base `set_weights` publishes them.

**Scoring** — `poker44/score/scoring.py::reward`:

```
base_score = 0.75 * AP + 0.25 * bot_recall   # bot_recall = best recall while human FPR <= 0.05
reward     = base_score * human_safety_penalty   # human_safety_penalty currently constant 1.0
```

⚠️ **The reward formula in `my_solution/POKER44_CONTEXT.md` is out of date.** That doc describes `0.65*AP + 0.35*recall` with `human_safety_penalty=(1-fpr)^2` and a hard FPR cliff at 0.10. The actual shipped code uses `0.75/0.25`, a flat `human_safety_penalty=1.0`, and an FPR ceiling of `0.05` inside `bot_recall`. If reasoning about reward, trust `scoring.py`, not the context doc — but flag the discrepancy to the user rather than silently assuming either is canonical, since the network validator may differ from this repo snapshot.

**Weight distribution** (`poker44/validator/constants.py`): currently `WINNER_TAKE_ALL=False` → **podium mode** (top 3 get 50/30/20). `BURN_EMISSIONS=True` but `BURN_FRACTION=0.00`, so nothing is burned to UID 0 right now. These flags are tuned frequently — check current values, don't assume.

**Integrity / observability lanes** run alongside scoring and persist JSON registries under the neuron state dir: model manifests (`model_manifest.py` / `integrity.py`), compliance, suspicion, served-chunk fingerprints, and an encrypted audit lane (`audit.py`). The validator also POSTs signed runtime/network/competition-score snapshots to `api.poker44.net/internal/...` (signing via `utils/runtime_info.build_signed_runtime_request`).

**Competition epochs** are computed in `runtime_provider._current_competition_epoch` (hardcoded anchor dates, 120h epochs over 24h eval windows). This is reference/observability metadata; actual settlement weights come from the backend.

**Data model**: `poker44/core/models.py` defines `HandHistory`, `LabeledHandBatch`, `ActionEvent`, etc. with `from_payload`/`to_payload`. Hands flow as plain dicts end-to-end in the forward loop; the dataclasses are mostly for typed access.

## `my_solution/` — the competitive miner pipeline

This is the user's actual ML work (the subnet miner that ships is just a heuristic). Read `my_solution/POKER44_CONTEXT.md` first — it captures hard-won findings and traps. Key points (notes are in Polish):

- **Pipeline**: pull labeled "API" data → extract behavioral features (`create_features3/4/5.py`, three feature versions combined per chunk) → train XGBoost (`train_12models.py`, a 12-variant grid) → save `.joblib` artifact → miner `score_chunk` loads it for inference.
- **Two data distributions**: labeled training data ("API", in `api_data/`) vs unlabeled production ("PRD", in `prd_data/`) suffer **covariate shift**. Mitigation: per-set `QuantileTransformer(output_distribution='normal')` — `qt_train` fit on train, `qt_prd` fit once on a large PRD sample then **frozen** (never refit at inference). Impute missing with **train median**, never `-1`.
- **Signal lives in temporal/higher-order features** (autocorrelation, time-variability like `fold_cv_time`, `vpip_lag1_autocorr`), not marginal distributions (sizing/VPIP/fold-rate are ~AUC 0.5). Validate any feature with a **hand-order shuffle test**: real temporal signal collapses to AUC ~0.5 after shuffle.
- **Documented traps** (from the context doc): reward 1.0 / std 0.0 means overfit (`fit_on='train+test'`), not success; high test F1 ≠ leaderboard gain (distribution mismatch); evaluate reward **per-day batch**, not over the whole test at once, because FPR variance matters; pseudo-labeling under shift is risky.

The `.ipynb` notebooks and top-level CSVs (`api_data.csv`, `prd.csv`, `auc_.csv`) are exploratory data-prep artifacts, untracked in git.

## Conventions & gotchas

- **Configuration is environment-variable driven**, not config files. Validator behavior is controlled by `POKER44_*` vars read with `os.getenv` (defaults inline). Search for `POKER44_` to find a knob.
- **Never expose label/identity fields to miners** — any new field added to hand payloads must be considered against `payload_view._LEAKAGE_KEYS` and the sanitization allow-listing in `prepare_hand_for_miner`.
- Version strings live in `poker44/__init__.py` (`__version__`, `VALIDATOR_DEPLOY_VERSION`); `setup.py` parses `__version__` from it.
- Many recent commits tune validator eval workload/timeouts/burn — these values are volatile, so verify current constants rather than relying on memory or this doc.
