#!/usr/bin/env python3
"""Score the raw output of the in-browser runner (casp_runner.html) with the same
automatic scorer used by run_experiment.py, so that analyze.py can process it.

  python score_runner_results.py                      # results/casp_results.jsonl -> results/raw_claude-sonnet-4-6.jsonl
  python score_runner_results.py --inp my_run.jsonl --out results_new

The first line of the runner export is a metadata record ({"meta": ...}); it is copied
to <out>/run_meta.json. Every other line is one API call.
"""
import argparse
import json
import os

from casp import prompts, scoring
from casp.datasets import DATA

TASK_FILES = {"T1": "extraction.jsonl", "T1i": "extraction_injected.jsonl", "T2": "gsm8k.jsonl",
              "E3": "extraction.jsonl"}


def load_items():
    items = {}
    for task, fname in TASK_FILES.items():
        for line in open(os.path.join(DATA, fname), encoding="utf-8"):
            r = json.loads(line)
            items[(task, r["id"])] = r
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join("results", "casp_results.jsonl"))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = load_items()
    by_model, meta = {}, None
    for line in open(a.inp, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "meta" in r:
            meta = r["meta"]
            continue
        if not r.get("error"):
            item = items[(r["task"], r["item_id"])]
            if r["task"] == "T2":
                cond = prompts.TASK_CONDITIONS["T2"][r["condition"]]
                r.update(scoring.score_reasoning(r["text"], item["gold"],
                                                 scoring.expected_format(cond, r["output_mode"])))
            else:
                r.update(scoring.score_extraction(r["text"], item["gold"]))
        by_model.setdefault(r["model"], []).append(r)
    if meta is not None:
        with open(os.path.join(a.out, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
    for model, rows in by_model.items():
        path = os.path.join(a.out, f"raw_{model}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_err = sum(1 for r in rows if r.get("error"))
        print(f"{model}: {len(rows)} rows ({n_err} errors) -> {path}")


if __name__ == "__main__":
    main()
