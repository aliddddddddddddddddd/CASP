#!/usr/bin/env python3
"""Analyse raw results -> tables (CSV + Markdown) and figures (PNG, 300 dpi).

  python analyze.py --results results
Writes <results>/analysis/: summary.csv, stats.csv, caching.csv, consistency.csv,
report.md, and Fig_*.png.

Statistics
  * unit of analysis = item (scores averaged over repetitions first)
  * 95% CI = percentile bootstrap over items (5,000 resamples, seed 7)
  * FULL vs each other condition: paired Wilcoxon signed-rank test (two-sided,
    zero differences dropped), Holm correction within each model x task family,
    effect size = matched-pairs rank-biserial correlation
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY = {"T1": "f1_typed", "T1i": "f1_typed", "T2": "correct", "E3": "f1_typed"}
SECONDARY = {"T1": ["f1_untyped", "strict_parse", "lenient_parse"],
             "T1i": ["f1_untyped", "strict_parse", "injected"],
             "T2": ["correct_strict", "strict_format"]}
TASK_LABEL = {"T1": "T1 Entity extraction (F1)", "T1i": "T1i Extraction under embedded instruction (F1)",
              "T2": "T2 GSM8K reasoning (accuracy)"}
COND_ORDER = ["B0", "FULL", "-R", "-X", "-C", "-O"]
COND_LABEL = {"B0": "Baseline", "FULL": "Full CASP", "-R": "\u2212Role", "-X": "\u2212XML (Markdown)",
              "-C": "\u2212Reasoning", "-O": "\u2212Output ctrl"}
RNG = np.random.default_rng(7)


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcoxon(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 0.0
    try:
        p = stats.wilcoxon(nz, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    r = stats.rankdata(np.abs(nz))
    rbc = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(p), float(rbc)


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps)
    order = np.argsort(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def cost(df, prices):
    pr = df["model"].map(prices)
    return ((df["input_tokens"] * pr.map(lambda p: p["in"]) +
             df["output_tokens"] * pr.map(lambda p: p["out"]) +
             df["cache_write_tokens"] * pr.map(lambda p: p["cw"]) +
             df["cache_read_tokens"] * pr.map(lambda p: p["cr"])) / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--config", default="models.json")
    a = ap.parse_args()
    out = os.path.join(a.results, "analysis"); os.makedirs(out, exist_ok=True)
    rows = []
    for p in glob.glob(os.path.join(a.results, "raw_*.jsonl")):
        if p.endswith("_probe.jsonl"):
            continue
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows)
    n_err = int(df["error"].notna().sum())
    df = df[df["error"].isna()].copy()
    # keep the last successful record per call key
    df = df.drop_duplicates(["model", "task", "condition", "item_id", "rep", "cache"], keep="last")
    cfg = {c["id"]: c for c in json.load(open(a.config))}
    prices = {m: dict(inp=0, **{}) for m in df["model"].unique()}
    prices = {m: {"in": cfg.get(m, {}).get("price_in", 0), "out": cfg.get(m, {}).get("price_out", 0),
                  "cw": cfg.get(m, {}).get("price_cache_write", 0),
                  "cr": cfg.get(m, {}).get("price_cache_read", 0)} for m in df["model"].unique()}
    df["cost_usd"] = cost(df, prices)
    models = sorted(df["model"].unique())

    # ------------------------------------------------------------ summary
    summ, stat_rows, cons_rows = [], [], []
    main_df = df[df["task"] != "E3"]
    for (m, t), g in main_df.groupby(["model", "task"]):
        prim = PRIMARY[t]
        per_item = {}
        for c, gc in g.groupby("condition"):
            it = gc.groupby("item_id")
            s = it[prim].mean()
            per_item[c] = s
            lo, hi = boot_ci(s.values)
            rec = dict(model=m, task=t, condition=c, n_items=len(s), reps=int(it.size().median()),
                       primary=prim, mean=s.mean(), ci_low=lo, ci_high=hi,
                       mean_in_tokens=gc["input_tokens"].mean(), mean_out_tokens=gc["output_tokens"].mean(),
                       mean_latency_s=gc["latency_s"].mean(), cost_per_100_items=gc["cost_usd"].mean() * 100)
            for sec in SECONDARY[t]:
                rec[sec] = it[sec].mean().mean()
            summ.append(rec)
            # consistency: identical parsed answer across all repetitions of an item
            if it.size().min() > 1:
                cons = it["canonical"].nunique().eq(1).mean()
                cons_rows.append(dict(model=m, task=t, condition=c, consistency=cons))
        conds = [c for c in per_item if c != "FULL"]
        ps, recs = [], []
        for c in conds:
            common = per_item["FULL"].index.intersection(per_item[c].index)
            a_, b_ = per_item["FULL"][common], per_item[c][common]
            p, rbc = wilcoxon(a_, b_)
            d = a_ - b_
            lo, hi = boot_ci(d.values)
            recs.append(dict(model=m, task=t, comparison=f"FULL vs {c}", n=len(common),
                             mean_full=a_.mean(), mean_other=b_.mean(), delta=d.mean(),
                             delta_ci_low=lo, delta_ci_high=hi, p_raw=p, rank_biserial=rbc))
            ps.append(p)
        for r_, pa in zip(recs, holm(ps) if ps else []):
            r_["p_holm"] = pa
            r_["significant_0.05"] = bool(pa < 0.05)
        stat_rows += recs
    summ = pd.DataFrame(summ); st = pd.DataFrame(stat_rows); cons = pd.DataFrame(cons_rows)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    st.to_csv(os.path.join(out, "stats.csv"), index=False)
    cons.to_csv(os.path.join(out, "consistency.csv"), index=False)

    # ------------------------------------------------------------ caching (E3)
    cache_rows = []
    e3 = df[df["task"] == "E3"]
    for m, g in e3.groupby("model"):
        on, off = g[g["cache"]], g[~g["cache"]]
        mo = on.groupby("item_id")["f1_typed"].mean(); mf = off.groupby("item_id")["f1_typed"].mean()
        common = mo.index.intersection(mf.index)
        p_f1, _ = wilcoxon(mo[common], mf[common])
        lo_on = on.groupby("item_id")["latency_s"].mean(); lo_off = off.groupby("item_id")["latency_s"].mean()
        p_lat, _ = wilcoxon(lo_on[common], lo_off[common])
        merged = on.merge(off, on=["item_id", "rep"], suffixes=("_on", "_off"))
        agree = (merged["canonical_on"] == merged["canonical_off"]).mean()
        warm = on[on["cache_read_tokens"] > 0]
        cache_rows.append(dict(
            model=m, n_items=len(common), f1_cache_on=mo[common].mean(), f1_cache_off=mf[common].mean(),
            p_f1=p_f1, output_agreement=agree,
            latency_on_s=lo_on[common].mean(), latency_off_s=lo_off[common].mean(), p_latency=p_lat,
            cache_hit_rate=(on["cache_read_tokens"] > 0).mean(),
            input_cost_per_call_on=((on["input_tokens"] * prices[m]["in"] + on["cache_write_tokens"] * prices[m]["cw"]
                                     + on["cache_read_tokens"] * prices[m]["cr"]) / 1e6).mean(),
            input_cost_per_call_off=((off["input_tokens"] * prices[m]["in"] + off["cache_write_tokens"] * prices[m]["cw"]
                                      + off["cache_read_tokens"] * prices[m]["cr"]) / 1e6).mean(),
            warm_latency_on_s=warm["latency_s"].mean() if len(warm) else np.nan))
    cache = pd.DataFrame(cache_rows)
    cache.to_csv(os.path.join(out, "caching.csv"), index=False)

    # ------------------------------------------------------------ figures
    colors = plt.cm.tab10.colors
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

    # Fig A: main results
    tasks = [t for t in ["T1", "T1i", "T2"] if t in summ["task"].unique()]
    if tasks:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            s = summ[summ["task"] == t]
            conds = [c for c in COND_ORDER if c in s["condition"].unique()]
            w = 0.8 / max(1, len(models)); x = np.arange(len(conds))
            for k, m in enumerate(models):
                sm = s[s["model"] == m].set_index("condition").reindex(conds)
                ax.bar(x + k * w - 0.4 + w / 2, sm["mean"], w, color=colors[k % 10], label=m,
                       yerr=[sm["mean"] - sm["ci_low"], sm["ci_high"] - sm["mean"]], capsize=2, error_kw={"lw": 0.8})
            ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=35, ha="right")
            ax.set_ylim(0, 1.05); ax.set_title(TASK_LABEL[t], fontsize=9); ax.grid(axis="y", alpha=0.3)
        axes[0][0].set_ylabel("Mean score (95% bootstrap CI)")
        axes[0][-1].legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "Fig_main_results.png"), dpi=300); plt.close(fig)

    # Fig B: ablation deltas heatmap
    if len(st):
        st2 = st.copy(); st2["cond"] = st2["comparison"].str.replace("FULL vs ", "", regex=False)
        rows_lbl = [f"{m} | {t}" for m in models for t in tasks]
        cols = [c for c in COND_ORDER if c != "FULL" and c in st2["cond"].unique()]
        M = np.full((len(rows_lbl), len(cols)), np.nan); S = np.zeros_like(M, dtype=bool)
        for i, lbl in enumerate(rows_lbl):
            m, t = lbl.split(" | ")
            for j, c in enumerate(cols):
                r = st2[(st2["model"] == m) & (st2["task"] == t) & (st2["cond"] == c)]
                if len(r):
                    M[i, j] = r["delta"].iloc[0]; S[i, j] = r["significant_0.05"].iloc[0]
        lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2.5, 0.45 * len(rows_lbl) + 1.4))
        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.3f}{'*' if S[i, j] else ''}", ha="center", va="center", fontsize=7,
                            color="white" if abs(M[i, j]) > 0.5 * lim else "black",
                            fontweight="bold" if S[i, j] else "normal")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([COND_LABEL[c] for c in cols], rotation=30, ha="right")
        ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl, fontsize=7)
        ax.set_title("Score of Full CASP minus score of each condition\n(positive = component helps; * Holm-adjusted p < 0.05)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
        fig.savefig(os.path.join(out, "Fig_ablation_delta.png"), dpi=300); plt.close(fig)

    # Fig C: format compliance
    fc = []
    for t, col in [("T1", "strict_parse"), ("T2", "strict_format")]:
        if t in tasks:
            fc.append((t, col))
    if fc:
        fig, axes = plt.subplots(1, len(fc), figsize=(4.2 * len(fc), 3.4), squeeze=False)
        for ax, (t, col) in zip(axes[0], fc):
            s = summ[summ["task"] == t]
            conds = [c for c in COND_ORDER if c in s["condition"].unique() and not (t == "T2" and c == "B0")]
            w = 0.8 / max(1, len(models)); x = np.arange(len(conds))
            for k, m in enumerate(models):
                sm = s[s["model"] == m].set_index("condition").reindex(conds)
                ax.bar(x + k * w - 0.4 + w / 2, sm[col], w, color=colors[k % 10], label=m)
            ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=35, ha="right")
            ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
            ax.set_title("T1: directly machine-parseable JSON" if t == "T1" else "T2: answer in requested format", fontsize=9)
        axes[0][0].set_ylabel("Proportion of responses"); axes[0][-1].legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "Fig_format_compliance.png"), dpi=300); plt.close(fig)

    # Fig D: injection
    if "T1i" in tasks:
        s = summ[summ["task"] == "T1i"]
        conds = [c for c in COND_ORDER if c in s["condition"].unique()]
        fig, ax = plt.subplots(figsize=(5, 3.3)); w = 0.8 / max(1, len(models)); x = np.arange(len(conds))
        for k, m in enumerate(models):
            sm = s[s["model"] == m].set_index("condition").reindex(conds)
            ax.bar(x + k * w - 0.4 + w / 2, sm["injected"], w, color=colors[k % 10], label=m)
        ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=35, ha="right")
        ax.set_ylabel("Share of responses containing\nthe payload token"); ax.grid(axis="y", alpha=0.3)
        ax.set_title("T1i: instruction/data separation (lower is better)", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "Fig_injection.png"), dpi=300); plt.close(fig)

    # Fig E: caching
    if len(cache):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2)); x = np.arange(len(cache)); w = 0.38
        for ax, (on, off, lab) in zip(axes, [("f1_cache_on", "f1_cache_off", "Extraction F1"),
                                             ("latency_on_s", "latency_off_s", "Mean latency (s)"),
                                             ("input_cost_per_call_on", "input_cost_per_call_off", "Input cost per call (USD)")]):
            ax.bar(x - w / 2, cache[on], w, label="cache on", color=colors[0])
            ax.bar(x + w / 2, cache[off], w, label="cache off", color=colors[1])
            ax.set_xticks(x); ax.set_xticklabels(cache["model"], fontsize=7); ax.set_title(lab, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=7)
        fig.suptitle("E3: prompt caching - output quality vs. latency and cost", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, "Fig_caching.png"), dpi=300); plt.close(fig)

    # ------------------------------------------------------------ report
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# CASP evaluation report\n\nModels: {', '.join(models)}\n\n"
                f"Successful calls analysed: {len(df)}; failed calls excluded: {n_err}\n\n")
        f.write("## Summary by condition\n\n" + summ.round(4).to_markdown(index=False) + "\n\n")
        f.write("## FULL vs other conditions (paired Wilcoxon, Holm)\n\n" + st.round(4).to_markdown(index=False) + "\n\n")
        if len(cons):
            f.write("## Consistency across repetitions\n\n" + cons.round(4).to_markdown(index=False) + "\n\n")
        if len(cache):
            f.write("## Prompt caching (E3)\n\n" + cache.round(5).to_markdown(index=False) + "\n")
    print(f"Analysed {len(df)} calls ({n_err} failed calls excluded). Output: {out}")
    print(summ[["model", "task", "condition", "mean", "ci_low", "ci_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
