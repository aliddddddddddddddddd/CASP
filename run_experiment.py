#!/usr/bin/env python3
"""Run the CASP controlled evaluation.

Examples
  python run_experiment.py --probe                      # 1 call per condition: checks API settings
  python run_experiment.py --pilot                      # n=10, reps=1 quick check
  python run_experiment.py                              # full run: n=100, reps=3, enabled models
  python run_experiment.py --models claude-sonnet-5 --tasks T2
  python run_experiment.py --models mock --out results_mock   # pipeline test only

Results are appended to <out>/raw_<model>.jsonl; the run is resumable (finished calls skipped).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time

from casp import backends, prompts, scoring
from casp.datasets import DATA

TASK_FILES = {"T1": "extraction.jsonl", "T1i": "extraction_injected.jsonl", "T2": "gsm8k.jsonl",
              "E3": "extraction.jsonl"}
E3_N = 50


def load(task, n):
    rows = [json.loads(l) for l in open(os.path.join(DATA, TASK_FILES[task]), encoding="utf-8")]
    return rows[: (min(n, E3_N) if task == "E3" else n)]


def key_of(r):
    return (r["model"], r["task"], r["condition"], r["item_id"], r["rep"], r["cache"])


def score(task, cond_name, output_mode, text, item):
    cond = prompts.TASK_CONDITIONS[task][cond_name]
    if task == "T2":
        return scoring.score_reasoning(text, item["gold"], scoring.expected_format(cond, output_mode))
    return scoring.score_extraction(text, item["gold"])


def jobs_for(cfg, tasks, n, reps, done):
    ctx = open(os.path.join(DATA, "caching_context.txt"), encoding="utf-8").read()
    jobs = []
    for task in tasks:
        items = load(task, n)
        if task == "E3":
            for item in items:
                for rep in range(reps):
                    for cache in (True, False):   # interleaved to control for time drift
                        k = (cfg["id"], task, "FULL", item["id"], rep, cache)
                        if k not in done:
                            jobs.append((task, "FULL", item, rep, cache, ctx))
            continue
        for cond_name in prompts.TASK_CONDITIONS[task]:
            for item in items:
                for rep in range(reps):
                    k = (cfg["id"], task, cond_name, item["id"], rep, False)
                    if k not in done:
                        jobs.append((task, cond_name, item, rep, False, None))
    return jobs


def run_one(cfg, job, temperature):
    task, cond_name, item, rep, cache, ctx = job
    p = prompts.build(task, item, cond_name, cfg["output_mode"], cache_context=ctx)
    fn = backends.PROVIDERS[cfg["provider"]]
    kw = dict(use_cache=cache, temperature=temperature,
              max_tokens=cfg.get("max_tokens", 1024))
    if cfg["provider"] == "mock":
        kw["item"] = item
    try:
        res = fn(cfg, p, **kw)
    except Exception as e:  # never let one job kill the run
        res = backends._empty(f"exception: {type(e).__name__}: {e}")
    row = dict(model=cfg["id"], model_string=cfg["model"], provider=cfg["provider"],
               output_mode=cfg["output_mode"], task=task, condition=cond_name,
               item_id=item["id"], rep=rep, cache=cache, temperature=temperature,
               prompt_sha1=hashlib.sha1((p.system + "\x00" + p.cache_prefix + "\x00" + p.user +
                                         "\x00" + p.prefill).encode()).hexdigest()[:12],
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), **res)
    if not res["error"]:
        row.update(score(task, cond_name, cfg["output_mode"], res["text"], item))
    return row, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models.json")
    ap.add_argument("--models", nargs="*", help="model ids (default: all enabled)")
    ap.add_argument("--tasks", nargs="*", default=["T1", "T1i", "T2", "E3"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--pilot", action="store_true", help="n=10, reps=1")
    ap.add_argument("--probe", action="store_true", help="1 call per condition, print outcome")
    a = ap.parse_args()
    if a.pilot:
        a.n, a.reps = 10, 1
    if a.probe:
        a.n, a.reps = 1, 1

    cfgs = json.load(open(a.config))
    if a.models:
        cfgs = [c for c in cfgs if c["id"] in a.models]
    else:
        cfgs = [c for c in cfgs if c.get("enabled", True)]
    if not cfgs:
        sys.exit("No models selected - check models.json / --models")
    os.makedirs(a.out, exist_ok=True)

    for cfg in cfgs:
        path = os.path.join(a.out, f"raw_{cfg['id']}{'_probe' if a.probe else ''}.jsonl")
        done = set()
        if os.path.exists(path) and not a.probe:
            for l in open(path, encoding="utf-8"):
                r = json.loads(l)
                if not r.get("error"):
                    done.add(key_of(r))
        jobs = jobs_for(cfg, a.tasks, a.n, a.reps, done)
        print(f"\n=== {cfg['id']} ({cfg['model']}, output control: {cfg['output_mode']}) - "
              f"{len(jobs)} calls to run, {len(done)} already done")
        lock = threading.Lock()
        n_ok = n_err = 0
        examples = {}
        # E3 must run sequentially so the cache written by one call is read by the next
        e3 = [j for j in jobs if j[0] == "E3"]
        rest = [j for j in jobs if j[0] != "E3"]
        with open(path, "a", encoding="utf-8") as f:
            def handle(row, p):
                nonlocal n_ok, n_err
                with lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                    if row["error"]:
                        n_err += 1
                        if a.probe or n_err <= 5:
                            print(f"  ERROR {row['task']}/{row['condition']}: {row['error'][:300]}")
                    else:
                        n_ok += 1
                        examples.setdefault((row["task"], row["condition"]), (p, row))
                    if (n_ok + n_err) % 100 == 0:
                        print(f"  progress {n_ok + n_err}/{len(jobs)} (errors {n_err})")
            with cf.ThreadPoolExecutor(max_workers=1 if a.probe else a.workers) as ex:
                futs = [ex.submit(run_one, cfg, j, a.temperature) for j in rest]
                for fu in cf.as_completed(futs):
                    handle(*fu.result())
            for j in e3:
                handle(*run_one(cfg, j, a.temperature))
        print(f"  finished: {n_ok} ok, {n_err} errors -> {path}")
        if a.probe:
            for (t, c), (p, row) in sorted(examples.items()):
                print(f"  OK {t:4s} {c:5s} | {row['text'][:90]!r}")
        # save one full prompt + response per condition (for the paper's appendix)
        if not examples:
            continue
        ex_path = os.path.join(a.out, f"examples_{cfg['id']}.md")
        with open(ex_path, "w", encoding="utf-8") as g:
            for (t, c), (p, row) in sorted(examples.items()):
                g.write(f"## {cfg['id']} | {t} | {c}\n\n**System:**\n```\n{p.system or '(none)'}\n```\n")
                if p.cache_prefix:
                    g.write(f"**Cached prefix:** {len(p.cache_prefix)} characters (guideline + demonstrations)\n\n")
                g.write(f"**User:**\n```\n{p.user}\n```\n")
                if p.prefill:
                    g.write(f"**Prefill:** `{p.prefill}`\n\n")
                if p.schema:
                    g.write(f"**Output schema:**\n```json\n{json.dumps(p.schema, indent=1)}\n```\n")
                g.write(f"**Response:**\n```\n{row['text']}\n```\n\n")


if __name__ == "__main__":
    main()
