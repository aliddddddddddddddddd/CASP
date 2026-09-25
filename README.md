# CASP: Constitutional AI-Aligned Structured Prompting

Code, data and raw model outputs for the paper
**"Structural Prompt Engineering in Constitutional AI Models: A Technical Analysis of Claude's Techniques."**

CASP is a four-component prompting procedure for Claude: (R) role and context framing,
(X) XML input structuring, (C) explicit step-by-step reasoning and (O) output control,
with prompt caching for reused contexts. The paper specifies it as Algorithm 1 and evaluates it
with a leave-one-out ablation study. This repository contains everything needed to inspect the
reported run, regenerate every table and figure in Section 5 of the paper, and repeat or extend
the experiment.

## The reported run at a glance

| Setting | Value |
|---|---|
| Model | Claude Sonnet 4.6 (`claude-sonnet-4-6`) |
| Output control | Structured outputs (`output_config.format`); prefill is rejected by this model (HTTP 400) |
| Tasks | T1 entity extraction (CoNLL-2003), T1i extraction with an embedded conflicting instruction, T2 arithmetic reasoning (GSM8K) |
| Items | First 50 items of each task file (T1 items contain 148 gold entities) |
| Conditions | T1/T1i: B0, FULL, −R, −X, −O. T2: B0, FULL, −R, −X, −C, −O |
| Repetitions | 1 per item and condition |
| Caching experiment (E3) | First 30 T1 items, each run once with and once without a cacheable ~5,000-token context |
| Total calls | 50 × (5 + 5 + 6) = 800, plus 60 for E3 = **860** |
| Sampling settings | Temperature not set (API default 1.0); `max_tokens` = 1000; `thinking` parameter not sent |
| Date | 22 September 2026, 16:06-21:56 UTC |

The calls were sent to the Anthropic Messages API from **`casp_runner.html`**, a Claude.ai
artifact that runs under a Claude.ai subscription rather than a separate API key. Its prompt
builder is a line-by-line port of `casp/prompts.py`, and it embeds the same items as `data/`.

## Repository layout

```
casp_runner.html          In-browser runner used for the reported run (Claude.ai artifact)
score_runner_results.py   Scores the runner's export so analyze.py can read it
run_experiment.py         Equivalent command-line runner (Anthropic / OpenAI / Gemini APIs)
analyze.py                Tables (CSV, Markdown) and figures (PNG, 300 dpi)
models.json               Model configurations and prices used for cost figures
casp/
  prompts.py              Algorithm 1: prompt construction for every condition
  scoring.py              Automatic, gold-based scoring (no human rating)
  backends.py             API clients (Anthropic, OpenAI, Gemini, mock)
  datasets.py             Deterministic construction of the evaluation sets (seed 2026)
data/
  extraction.jsonl        T1 items (100 sampled; the first 50 were used)
  extraction_injected.jsonl  T1i items
  gsm8k.jsonl             T2 items
  caching_context.txt     Fixed guideline + 60 demonstrations used in E3
results/
  casp_results.jsonl      Raw export of the reported run (860 calls + metadata line)
```

## Reproduce the paper's tables and figures (no API access needed)

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python score_runner_results.py        # results/casp_results.jsonl -> results/raw_claude-sonnet-4-6.jsonl
python analyze.py --results results   # writes results/analysis/
python prose_parse.py                 # post-hoc parse of B0 prose answers (Section 5.1)
```

`results/analysis/` then contains `summary.csv` (Table 6), `stats.csv` (Table 7),
`caching.csv` (Table 8), `report.md` and the figures `Fig_main_results.png`,
`Fig_format_compliance.png`, `Fig_ablation_delta.png`, `Fig_injection.png` and `Fig_caching.png`
(Figures 7-11). Key values you should obtain: T1 FULL typed F1 = 0.855 vs B0 = 0.000;
T2 accuracy 0.960 for all conditions except −C = 0.740 (Holm p = 0.017, r = 0.846);
E3 cache-hit rate 0.967 (29/30), output agreement 0.933 (28/30), input cost per call
$0.0046 (cached) vs $0.0302 (uncached).

## Repeat the experiment

**Option A - in Claude.ai (as in the paper).** Open `casp_runner.html` as an artifact in a
Claude.ai conversation (for example, upload the file and ask Claude to display it as an
artifact). Run the connection test, keep the default settings, start the run, then download
`casp_results.jsonl` and score it:

```bash
python score_runner_results.py --inp path/to/casp_results.jsonl --out results_new
python analyze.py --results results_new
```

The runner calls `https://api.anthropic.com/v1/messages` without an API key, which works only
inside a Claude.ai artifact; opened as a normal web page it cannot connect.

**Option B - through the API.** Set `ANTHROPIC_API_KEY` and run:

```bash
python run_experiment.py --probe      # one call per condition, checks settings
python run_experiment.py              # defaults match the reported run: n=50, reps=1, E3 on 30 items
python analyze.py --results results_api
```

Use `--out results_api` to keep API results separate from the reported run. The defaults of
`run_experiment.py` and the enabled entry in `models.json` (`claude-sonnet-4-6`, `max_tokens`
1000, no temperature or thinking parameter) mirror the runner. Larger designs are possible,
for example `--n 100 --reps 3`.

**Extending to other models.** `models.json` also contains disabled entries for Claude Haiku 4.5
(prefill output control), GPT and Gemini. Fill in the model name, set `"enabled": true`, and set
`OPENAI_API_KEY` or `GEMINI_API_KEY` as needed. The same items, prompts and scorer are used for
every provider. These models were **not** evaluated in the paper.

## Rebuilding the evaluation sets

The files in `data/` are already included. To regenerate them from the public sources, place
the raw files in `data/raw/`:

- `conll2003_eng_testb.txt` - https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testb
- `conll2003_eng_train.txt` - https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.train
- `gsm8k_test.jsonl` - https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

then run `python -m casp.datasets`. Sampling uses a fixed seed (2026).

## Notes and limitations

- One model, 50 items per task and a single repetition: run-to-run consistency was not measured,
  and small effects cannot be excluded. See Section 5.7 of the paper.
- T1i contains an intentionally embedded instruction; responses that obey it are counted by the
  `injected` metric.
- Model behaviour, API features and prices change over time; all such facts are as of September 2026.

## Citation

If you use this code or data, please cite the paper (full reference to be added on publication).
