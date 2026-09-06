# Episode QC — Problem 1 (Rescue a Failing Robot-Learning Dataset)

An episode-analysis tool that assigns every episode in a manipulation dataset a
PASS / REVIEW / FAIL recommendation, with quantified evidence, by checking calibration validity,
cross-stream synchronization, frame continuity, hand-pose trajectory plausibility, and episode
metadata — all scored against thresholds computed from the corpus being analyzed, not hardcoded.

Read `docs/technical_note.md` for the dataset choice, what's missing, and the fault taxonomy.

**Problem 3** (mandatory reasoning answer) is in `docs/problem3_reasoning.md`; the plain-language
customer summary is `docs/five_line_summary.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .          # installs episode_qc from src/, plus numpy/scipy/pandas/pyyaml
pip install pytest tabulate   # only needed for `pytest` and data_gen/validate.py's markdown table
```

Real calibration data (already fetched into `data/real/calibration/` in this submission) can be
re-fetched with:

```bash
curl -sL -o /tmp/calibration.tar.gz \
  "https://drive.google.com/uc?export=download&id=1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ"
tar xzf /tmp/calibration.tar.gz -C data/real/
```

(This is DexYCB's own small, separate, ungated `calibration.tar.gz` — see `docs/technical_note.md`
for why the full 12–119GB dataset was not pulled in.)

## Run the analysis tool

```bash
python -m episode_qc.cli \
  --calibration data/real/calibration \
  --episodes data/synthetic/holdout \
  --out outputs/holdout
```

This is the actual, unmodified command used for both batches in this submission — just point
`--episodes` at any directory of episode JSON files matching the schema (documented in
`src/episode_qc/io.py`'s `load_episode`) and `--calibration` at a real DexYCB-style calibration
directory. No episode IDs or per-clip thresholds are hardcoded anywhere in the tool: every
threshold is computed from the corpus passed in at run time (see `scoring.py`'s `RobustStats` and
`runner.py`'s two-pass collect-then-evaluate design).

Output: `outputs/<name>/summary.md` (+ `summary.csv`) for the corpus-level PASS/REVIEW/FAIL table,
and `outputs/<name>/episodes/<episode_id>.json` for the full per-episode evidence.

## Regenerate the data from scratch

```bash
python data_gen/generate_episodes.py --out data/synthetic/build   --n 40 --seed 1   --batch build
python data_gen/generate_episodes.py --out data/synthetic/holdout --n 24 --seed 999 --batch holdout

python data_gen/inject_faults.py --episodes data/synthetic/build   --fixture data/fixtures/build_ground_truth.json   --seed 2
python data_gen/inject_faults.py --episodes data/synthetic/holdout --fixture data/fixtures/holdout_ground_truth.json --seed 777
```

## Validate (compare the tool's output against the injected-fault ground truth)

```bash
python -m episode_qc.cli --calibration data/real/calibration --episodes data/synthetic/build   --out outputs/build
python -m episode_qc.cli --calibration data/real/calibration --episodes data/synthetic/holdout --out outputs/holdout

python data_gen/validate.py --reports outputs/build   --fixture data/fixtures/build_ground_truth.json   --out outputs/build/validation_report.md
python data_gen/validate.py --reports outputs/holdout --fixture data/fixtures/holdout_ground_truth.json --out outputs/holdout/validation_report.md
```

`data_gen/validate.py` is validation-only tooling — the analysis tool (`episode_qc`) never reads
the ground-truth fixtures it checks against.

**Current results** (see `outputs/{build,holdout}/validation_report.md` for full detail):

| | build (n=40, tuned here) | holdout (n=24, never tuned against) |
|---|---|---|
| Healthy-episode false-flag rate | 0.0% | 8.3% (1 episode — a legitimately fast-but-plausible reach segment flagged REVIEW, not FAIL; see `outputs/holdout/validation_report.md`) |
| Faulted-episode recall (any flag) | 100% | 100% |
| Faulted-episode recall (correct specific check) | 100% | 100% |

## Tests

```bash
python -m pytest -q
```

`tests/test_geometry.py` and `tests/test_scoring.py` are unit tests against hand-built fixtures;
`tests/test_end_to_end.py` regenerates a tiny fresh corpus (using the real calibration data) and
confirms every one of the 6 injected fault types is caught by its expected check module.

## Repository layout

This is the tracked contents of the git repository — i.e. everything **except** `.venv/`,
`__pycache__/`/`.pytest_cache/`/`*.egg-info/` (local environment/build artifacts) and
`data/synthetic/` (seeded, regenerable episode inputs — see "Regenerate the data from scratch"
above; excluded to keep the repo small, not because it's optional to reproduce). See `.gitignore`.

```
README.md                  this file — setup, run instructions, layout
AI_USAGE.md                required disclosure of generative-AI assistance (both problems)
pyproject.toml, requirements.txt, pytest.ini   packaging/dependency/test config
.gitignore

data/real/calibration/     real DexYCB calibration (intrinsics, extrinsics, MANO shapes) — see technical_note.md
data/fixtures/*_ground_truth.json   injected-fault ground truth (validation-only, not read by episode_qc)

src/episode_qc/            the analysis tool
  io.py                    load calibration + episode schema
  geometry.py              rotation validity, projection, multi-view triangulation
  runner.py                two-pass corpus runner (collect metrics -> pool -> evaluate)
  scoring.py                robust stats, severity bands, PASS/REVIEW/FAIL rollup
  checks/                  5 check modules (calibration, sync, continuity, trajectory, metadata)
  report.py                per-episode JSON + corpus summary.csv/md
  cli.py                   `python -m episode_qc.cli`

data_gen/                  generator + fault injector + validator (not shipped as part of the tool)
tests/                     pytest suite
outputs/                   generated reports + validation results for build/holdout — the
                           assignment-required "generated metrics/artifacts" and per-episode
                           health assessments
docs/
  technical_note.md        dataset choice, what's missing, fault taxonomy (assignment-required)
  problem3_reasoning.md    Problem 3: the mandatory reasoning answer (hypotheses, experiments, decision)
  five_line_summary.md     the required 5-line plain-language customer/lead summary
  problem_choice_note.md   the required <=150-word note
```

## How this maps to the submission requirements

The assignment's "Submission" section lists eight required deliverables. Every one has exactly one
tracked home in this repo:

| Assignment requires | Where it is |
|---|---|
| Source code and run instructions | `src/episode_qc/`, `data_gen/` (code) + this `README.md` (instructions) |
| Generated metrics/artifacts | `outputs/{build,holdout}/summary.{md,csv}`, `outputs/{build,holdout}/validation_report.{md,csv}`, `outputs/{build,holdout}/episodes/*.json` (the per-episode health assessments Problem 1 itself requires) |
| Technical note incl. dataset choice, assumptions, known limitations | `docs/technical_note.md` |
| Problem-choice note (≤150 words) | `docs/problem_choice_note.md` |
| 5-line customer/lead summary | `docs/five_line_summary.md` |
| `AI_USAGE.md` | `AI_USAGE.md` |
| Problem 3 (mandatory reasoning) | `docs/problem3_reasoning.md` |
| Faults deliberately injected + how generated | documented in `docs/technical_note.md`, implemented in `data_gen/inject_faults.py` |
