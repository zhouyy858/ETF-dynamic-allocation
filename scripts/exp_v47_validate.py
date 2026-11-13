# -*- coding: utf-8 -*-
"""Full anti-overfit validation for the v47 candidate."""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import numpy as np
import pandas as pd

from data_prep import build_panel, read_table
from engine import evaluate, run_backtest
from strategy import DynamicStrategy
from swaps import pnl_panel, traded_code
from stress_test import synthetic_resonance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = json.load(open(os.path.join(ROOT, "references", "final_cfg_v45.json"), encoding="utf-8"))
CAND = json.load(open(os.path.join(ROOT, "out", "candidate_v47.json"), encoding="utf-8"))
R_PROXY, _ = build_panel("proxy")
R_REAL, _ = build_panel("real")
BOND = read_table("511260_nav.csv")["cum_nav"].pct_change().dropna()
PRICE_FILES = {"159941": "qdii_price_159941.csv", "513500": "qdii_price_513500.csv"}
PRICE_DROP = {"159941": ["2022-07-05"], "513500": ["2022-03-30"]}


def price_rets(code):
    s = read_table(PRICE_FILES[code])["close"].sort_index()
    return s.pct_change().dropna().drop(
        [pd.Timestamp(d) for d in PRICE_DROP[code]], errors="ignore"
    )


def exec_pnl(returns, layer, cfg):
    panel = pnl_panel(returns, layer, cfg).copy()
    for slot in PRICE_FILES:
        if traded_code(slot, cfg) == slot:
            panel[slot] = price_rets(slot).reindex(panel.index).combine_first(panel[slot])
    return panel


def run(cfg, signals, panel, start, end=None, fee=0.0005):
    strategy = DynamicStrategy(signals, cfg=cfg)
    result = run_backtest(
        panel, target_weights_fn=strategy.target_fn(), daily_override_fn=strategy.daily_fn(),
        start=start, end=end, name="DYN", fee=fee, min_delta=cfg.get("min_delta", 0.035),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=BOND, cash_bond_pct=cfg.get("cash_bond_pct", 1.0),
        rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"), strict=True,
    )
    return result, evaluate(result)


def pack(result):
    return {key: float(result[key]) for key in
            ("cagr", "max_dd", "calmar", "sharpe", "turnover", "avg_cash")}


def bootstrap(rets, n=2000, block=20, seed=20261113):
    values = pd.Series(rets).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block))
    starts = rng.integers(0, len(values) - block + 1, size=(n, blocks))
    calmars = []
    mdds = []
    for row in starts:
        sample = np.concatenate([values[s:s + block] for s in row])[:len(values)]
        wealth = np.cumprod(1.0 + sample)
        ann = wealth[-1] ** (252.0 / len(sample)) - 1.0
        mdd = float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))
        calmars.append(ann / abs(mdd) if mdd < 0 else np.nan)
        mdds.append(mdd)
    calmars = np.asarray(calmars)
    mdds = np.asarray(mdds)
    return {
        "cal_p05": float(np.nanquantile(calmars, .05)),
        "cal_p50": float(np.nanquantile(calmars, .50)),
        "cal_lt_1": float(np.nanmean(calmars < 1.0)),
        "mdd_p05": float(np.quantile(mdds, .05)),
        "mdd_p50": float(np.quantile(mdds, .50)),
    }


def main():
    pp_base = exec_pnl(R_PROXY, "proxy", BASE)
    pr_base = exec_pnl(R_REAL, "real", BASE)
    pp_cand = exec_pnl(R_PROXY, "proxy", CAND)
    pr_cand = exec_pnl(R_REAL, "real", CAND)
    report = {"metrics": {}, "costs": {}, "stress": {}, "bootstrap": {}, "gate_sensitivity": {}}

    print("== windows ==", flush=True)
    for name, cfg, pp, pr in (("v45", BASE, pp_base, pr_base), ("v47", CAND, pp_cand, pr_cand)):
        report["metrics"][name] = {}
        for window, start, end in (
            ("proxy", "2014-06-23", None),
            ("train", "2014-06-23", "2021-12-31"),
            ("oos", "2022-01-04", None),
            ("real", "2025-04-23", None),
        ):
            res, result = run(cfg, R_REAL if window == "real" else R_PROXY,
                              pr if window == "real" else pp, start, end)
            report["metrics"][name][window] = pack(result)
            if window == "proxy":
                report["bootstrap"][name] = bootstrap(res["rets"])
            print(f"{name} {window:5} {result['cagr']*100:6.2f}/{result['max_dd']*100:6.2f}/"
                  f"{result['calmar']:5.3f} TO {result['turnover']:6.1f}", flush=True)

    print("== costs ==", flush=True)
    for name, cfg, pp in (("v45", BASE, pp_base), ("v47", CAND, pp_cand)):
        report["costs"][name] = {}
        for bp in (5, 10, 20, 30):
            _, result = run(cfg, R_PROXY, pp, "2014-06-23", None, bp / 10000.0)
            report["costs"][name][f"{bp}bp"] = pack(result)
            print(f"{name} {bp:02d}bp {result['cagr']*100:6.2f}/{result['max_dd']*100:6.2f}/"
                  f"{result['calmar']:5.3f}", flush=True)

    print("== stress ==", flush=True)
    scenarios = (
        ("2015股灾", R_PROXY, "2015-06-15", "2016-02-29"),
        ("2018熊市", R_PROXY, "2018-01-02", "2019-01-03"),
        ("2021-22熊市", R_PROXY, "2021-02-19", "2022-10-31"),
        ("2019-21牛市", R_PROXY, "2019-01-04", "2021-02-18"),
        ("2023-24震荡", R_PROXY, "2023-01-03", "2024-08-30"),
    )
    for label, signals, start, end in scenarios:
        report["stress"][label] = {}
        for name, cfg in (("v45", BASE), ("v47", CAND)):
            panel = exec_pnl(signals, "proxy", cfg)
            _, result = run(cfg, signals, panel, start, end)
            report["stress"][label][name] = pack(result)
            print(f"{label:10} {name} {result['cagr']*100:6.2f}/{result['max_dd']*100:6.2f}/"
                  f"{result['calmar']:5.3f}", flush=True)

    print("== synthetic resonance ==", flush=True)
    for tag, us, cn, idx in (("std", -.30, -.20, -.25), ("hard", -.40, -.30, -.35)):
        synth, _ = synthetic_resonance(pr_cand, ("2024-09-02", "2026-07-31"), us, cn, idx)
        report["stress"][f"synth_{tag}"] = {}
        for name, cfg in (("v45", BASE), ("v47", CAND)):
            _, result = run(cfg, synth, exec_pnl(synth, "real", cfg), "2024-09-02", "2026-07-31")
            report["stress"][f"synth_{tag}"][name] = pack(result)
            print(f"synth_{tag:4} {name} {result['cagr']*100:6.2f}/{result['max_dd']*100:6.2f}/"
                  f"{result['calmar']:5.3f}", flush=True)

    print("== gate sensitivity ==", flush=True)
    for gate in (80, 90, 100, 110, 120):
        cfg = copy.deepcopy(CAND)
        cfg["gate_win"] = gate
        _, result = run(cfg, R_PROXY, pp_cand, "2014-06-23", None)
        report["gate_sensitivity"][gate] = pack(result)
        print(f"display_gate_win={gate} {result['cagr']*100:6.2f}/{result['max_dd']*100:6.2f}/"
              f"{result['calmar']:5.3f}", flush=True)

    with open(os.path.join(ROOT, "out", "exp_v47_validate.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

