# -*- coding: utf-8 -*-
"""v45 过拟合审计：价格口径 CSCV/PBO、DSR、年度滚动选参与候选族分布。"""
import copy
import itertools
import json
import math
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import run
from run_backtest import exec_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(ROOT, "references", "final_cfg_v44.json")))
FINAL = json.load(open(os.path.join(ROOT, "references", "final_cfg_v45.json")))
CALIBER = os.environ.get("ETF_AUDIT_CALIBER", "price")
OUTFILE = os.path.join(
    OUT,
    "audit_v45_overfit.json" if CALIBER == "price" else f"audit_v45_overfit_{CALIBER}.json",
)


def cand(tag, **kw):
    c = copy.deepcopy(BASE)
    c.update(kw)
    c["version"] = tag
    return c


CANDS = {
    "base_v44": cand("base_v44"),
    "score_tech_confirm": cand("score_tech_confirm", score_mode="tech_confirm"),
    "score_dedup_us": cand("score_dedup_us", score_mode="dedup_us"),
    "score_us_lead": cand("score_us_lead", score_mode="us_lead"),
    "score_cn_lead": cand("score_cn_lead", score_mode="cn_lead"),
    "k05_cnlead": cand("k05_cnlead", lock_defense_keep=0.05, score_mode="cn_lead"),
    "k10_cnlead": cand("k10_cnlead", lock_defense_keep=0.10, score_mode="cn_lead"),
    "k15_cnlead": FINAL,
    "k20_cnlead": cand("k20_cnlead", lock_defense_keep=0.20, score_mode="cn_lead"),
    "k25_cnlead": cand("k25_cnlead", lock_defense_keep=0.25, score_mode="cn_lead"),
    "k30_cnlead": cand("k30_cnlead", lock_defense_keep=0.30, score_mode="cn_lead"),
    "k50_cnlead": cand("k50_cnlead", lock_defense_keep=0.50, score_mode="cn_lead"),
    "k100_cnlead": cand("k100_cnlead", lock_defense_keep=1.00, score_mode="cn_lead"),
    "k25_mom": cand("k25_mom", lock_defense_keep=0.25, cn_seesaw=True,
                    cn_seesaw_mode="momentum", cn_seesaw_thr=0.0, cn_seesaw_frac=0.25),
    "rebound_us_f25": cand("rebound_us_f25", cn_seesaw=True, cn_seesaw_mode="contrarian",
                           cn_seesaw_thr=0.10, cn_seesaw_frac=0.25,
                           cn_seesaw_rebound=True, cn_seesaw_confirm_us=True),
    "rebound_us_f50": cand("rebound_us_f50", cn_seesaw=True, cn_seesaw_mode="contrarian",
                           cn_seesaw_thr=0.10, cn_seesaw_frac=0.50,
                           cn_seesaw_rebound=True, cn_seesaw_confirm_us=True),
    "techrot20_f25": cand("techrot20_f25", tech_rotate=True, tech_rotate_mode="procyclical",
                          tech_rotate_win=20, tech_rotate_frac=0.25),
    "techrot60_f25": cand("techrot60_f25", tech_rotate=True, tech_rotate_mode="procyclical",
                          tech_rotate_win=60, tech_rotate_frac=0.25),
    "k25_inv20": cand("k25_inv20", lock_defense_keep=0.25, tech_rotate=True,
                      tech_rotate_mode="inverse", tech_rotate_win=20, tech_rotate_frac=0.25),
    "k50_inv20": cand("k50_inv20", lock_defense_keep=0.50, tech_rotate=True,
                      tech_rotate_mode="inverse", tech_rotate_win=20, tech_rotate_frac=0.25),
    "k25_cnlead_inv20": cand("k25_cnlead_inv20", lock_defense_keep=0.25, score_mode="cn_lead",
                             tech_rotate=True, tech_rotate_mode="inverse",
                             tech_rotate_win=20, tech_rotate_frac=0.25),
}


def metrics(r):
    r = np.asarray(pd.Series(r).dropna(), dtype=float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / peak - 1.0))
    cagr = float(nav[-1] ** (252.0 / len(r)) - 1.0)
    sd = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r) / sd * math.sqrt(252.0)) if sd > 0 else float("nan")
    return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe,
            "calmar": cagr / abs(mdd) if mdd < 0 else float("nan")}


def cscv_pbo(rets, blocks=16):
    """按候选族做 CSCV；以 Sharpe 选参，返回 PBO 与 OOS 衰减。"""
    n = len(rets)
    edges = np.linspace(0, len(rets.index), blocks + 1, dtype=int)
    sums, sqs, counts = [], [], []
    for i in range(blocks):
        x = rets.iloc[edges[i]:edges[i + 1]].to_numpy()
        sums.append(x.sum(0))
        sqs.append((x * x).sum(0))
        counts.append(len(x))
    sums, sqs, counts = np.asarray(sums), np.asarray(sqs), np.asarray(counts)
    picks = list(itertools.combinations(range(blocks), blocks // 2))
    pbo = 0
    degrad = []
    selected = {}
    for idx in picks:
        oos_idx = [i for i in range(blocks) if i not in idx]
        def sr(which):
            m = sums[which].sum(0) / counts[which].sum()
            v = sqs[which].sum(0) / counts[which].sum() - m * m
            v = np.maximum(v, 1e-18)
            return m / np.sqrt(v) * math.sqrt(252.0)
        sr_is, sr_oos = sr(list(idx)), sr(oos_idx)
        j = int(np.nanargmax(sr_is))
        selected[rets.columns[j]] = selected.get(rets.columns[j], 0) + 1
        rank = float(np.mean(sr_oos <= sr_oos[j]))
        if rank <= 0.5:
            pbo += 1
        degrad.append(float(sr_is[j] - sr_oos[j]))
    return {
        "blocks": blocks,
        "splits": len(picks),
        "pbo": pbo / len(picks),
        "median_is_oos_sharpe_gap": float(np.median(degrad)),
        "selected_counts": dict(sorted(selected.items(), key=lambda kv: -kv[1])[:8]),
        "_n": n,
    }


def deflated_sharpe(r, trial_sharpes, n_trials):
    r = np.asarray(pd.Series(r).dropna(), dtype=float)
    sr = float(np.mean(r) / np.std(r, ddof=1))
    t = len(r)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis() + 3.0)
    var_sr = float(np.var(np.asarray(trial_sharpes) / math.sqrt(252.0), ddof=1))
    gamma = 0.5772156649015329
    nd = NormalDist()
    sr0 = math.sqrt(max(var_sr, 1e-18)) * (
        (1.0 - gamma) * nd.inv_cdf(1.0 - 1.0 / n_trials)
        + gamma * nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    denom = math.sqrt(max(1e-18, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - sr0) * math.sqrt(t - 1.0) / denom
    return {"dsr": nd.cdf(z), "sr0_ann": sr0 * math.sqrt(252.0),
            "sr_ann": sr * math.sqrt(252.0)}


def walk_forward(rets):
    years = range(2018, 2027)
    out = {}
    for y in years:
        train = rets[rets.index < f"{y}-01-01"]
        test = rets[(rets.index >= f"{y}-01-01") & (rets.index < f"{y + 1}-01-01")]
        if len(train) < 500 or test.empty:
            continue
        train_cal = {c: metrics(train[c])["calmar"] for c in rets.columns}
        chosen = max(train_cal, key=train_cal.get)
        r = metrics(test[chosen])
        out[str(y)] = {"chosen": chosen, **r}
    return out


def main():
    P = exec_panel(V.R, "proxy", BASE, CALIBER)
    rows, series = {}, {}
    print(f"== 候选族（{CALIBER}，全历史） ==", flush=True)
    for name, cfg in CANDS.items():
        res, e = run(cfg, V.R, P, V.PROXY_START, None)
        rows[name] = V.pack(e)
        series[name] = res["rets"]
        print(f"{name:20s} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / Cal {e['calmar']:5.3f}", flush=True)
    rets = pd.DataFrame(series).dropna()
    pbo = cscv_pbo(rets)
    trial_sharpes = [metrics(rets[c])["sharpe"] for c in rets.columns]
    corr = rets.corr().to_numpy()
    rho = float((corr.sum() - len(corr)) / (len(corr) * (len(corr) - 1)))
    n_eff = 1.0 + (len(rets.columns) - 1.0) * (1.0 - rho)
    dsr = {str(n): deflated_sharpe(rets["k15_cnlead"], trial_sharpes, n)
           for n in (len(CANDS), max(2, int(round(n_eff))), 100, 500)}
    wf = walk_forward(rets)
    v45 = rows["k15_cnlead"]
    rank_cagr = int(sum(v["cagr"] > v45["cagr"] for v in rows.values()) + 1)
    rank_cal = int(sum(v["calmar"] > v45["calmar"] for v in rows.values()) + 1)
    report = {"caliber": CALIBER, "candidates": rows, "pbo": pbo,
              "avg_pairwise_corr": rho, "effective_trials": n_eff,
              "dsr": dsr, "walk_forward": wf,
              "v45_rank": {"cagr": rank_cagr, "calmar": rank_cal, "n": len(rows)}}
    json.dump(report, open(OUTFILE, "w"), ensure_ascii=False, indent=1)
    print("\n== PBO ==")
    print(json.dumps(pbo, ensure_ascii=False, indent=1))
    print(f"\n候选平均相关 {rho:.3f}；有效独立试验数约 {n_eff:.1f}")
    print(json.dumps(dsr, ensure_ascii=False, indent=1))
    print("\n== 年度滚动选参 ==")
    print(json.dumps(wf, ensure_ascii=False, indent=1))
    print(f"\nv45 在 {len(rows)} 个候选族中：CAGR 第 {rank_cagr}，Calmar 第 {rank_cal}")
    print(f"[ok] {os.path.relpath(OUTFILE, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
