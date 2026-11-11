# -*- coding: utf-8 -*-
"""v41 候选的完整纸面验证（2026-11-11，数据截至 2026-11-10）。

验证内容包括：双窗口/OOS、分阶段、费用敏感性、七个压力情景、20 日块
自助法、WFO 训练/测试、自由现金流零暴露。仅用于研究，不执行下单。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from swaps import pnl_panel
from stress_test import synthetic_resonance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
CAND_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REF, "final_cfg_v41.json")
TAG = sys.argv[2] if len(sys.argv) > 2 else "v41"
BASE_FILE = os.path.join(REF, "final_cfg_v40.json")

PROXY_START = "2014-06-23"
REAL_START = "2025-04-23"
OOS_START = "2022-01-04"
PERIODS = {
    "2014H2-2015牛市": ("2014-06-23", "2015-06-12"),
    "2015股灾": ("2015-06-15", "2016-02-29"),
    "2016-2017修复": ("2016-03-01", "2017-12-29"),
    "2018熊市": ("2018-01-02", "2019-01-03"),
    "2019-2021牛市": ("2019-01-04", "2021-02-18"),
    "2021-2022熊市": ("2021-02-19", "2022-10-31"),
    "2023-2024震荡": ("2023-01-03", "2024-08-30"),
    "2024Q4-2026双牛": ("2024-09-02", "2026-11-10"),
}

R, _ = build_panel("proxy")
Rr, _ = build_panel("real")
BOND = rets_from(read_table("511260_nav.csv"), "cum_nav")
SYNTH, S_IDX = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)


def load(path):
    return json.load(open(path))


def run(cfg, Rs, start, end=None, fee=0.0005, a_mkt=None, periods=None):
    layer = "real" if Rs is Rr else "proxy"
    pnl = pnl_panel(Rs, layer, cfg)
    ds = DynamicStrategy(Rs, cfg=cfg, a_mkt_override=a_mkt)
    res = run_backtest(
        pnl,
        target_weights_fn=ds.target_fn(),
        daily_override_fn=ds.daily_fn(),
        start=start,
        end=end,
        name="DYN",
        fee=fee,
        min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=BOND,
        cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True,
    )
    e = evaluate(res, periods=periods)
    return res, e


def pack(e):
    keys = ("cagr", "max_dd", "calmar", "sharpe", "turnover", "avg_cash", "total_ret")
    return {k: float(e[k]) for k in keys}


def bootstrap(r, n=2000, block=20, seed=20261111):
    r = pd.Series(r).dropna().to_numpy(dtype=float)
    if len(r) <= block:
        return {}
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(r) / block))
    starts = rng.integers(0, len(r) - block + 1, size=(n, blocks))
    cal, mdd, cagr = [], [], []
    for row in starts:
        x = np.concatenate([r[s:s + block] for s in row])[:len(r)]
        w = np.cumprod(1.0 + x)
        ann = w[-1] ** (252.0 / len(x)) - 1.0
        peak = np.maximum.accumulate(w)
        d = float(np.min(w / peak - 1.0))
        cal.append(ann / abs(d) if d < 0 else np.nan)
        mdd.append(d)
        cagr.append(ann)
    cal = np.asarray(cal, dtype=float)
    mdd = np.asarray(mdd, dtype=float)
    cagr = np.asarray(cagr, dtype=float)
    return {
        "n": n, "block": block,
        "p_cal_lt_1": float(np.nanmean(cal < 1.0)),
        "p_mdd_lt_10": float(np.mean(mdd < -0.10)),
        "cal_p05": float(np.nanquantile(cal, 0.05)),
        "cal_p50": float(np.nanquantile(cal, 0.50)),
        "cal_p95": float(np.nanquantile(cal, 0.95)),
        "mdd_p05": float(np.quantile(mdd, 0.05)),
        "mdd_p50": float(np.quantile(mdd, 0.50)),
        "mdd_p95": float(np.quantile(mdd, 0.95)),
        "cagr_p05": float(np.quantile(cagr, 0.05)),
        "cagr_p50": float(np.quantile(cagr, 0.50)),
        "cagr_p95": float(np.quantile(cagr, 0.95)),
    }


def main():
    cand = load(CAND_FILE)
    base = load(BASE_FILE)
    report = {"tag": TAG, "candidate": CAND_FILE, "base": BASE_FILE, "metrics": {}, "costs": {}, "stress": {}, "bootstrap": {}, "wfo": {}, "exposure": {}}

    print("== 双窗口 / OOS ==", flush=True)
    for name, cfg in (("base_v40", base), ("candidate", cand)):
        report["metrics"][name] = {}
        for layer, Rs, start, periods in (
            ("proxy", R, PROXY_START, PERIODS),
            ("real", Rr, REAL_START, None),
            ("oos", R, OOS_START, None),
        ):
            res, e = run(cfg, Rs, start, periods=periods)
            report["metrics"][name][layer] = {"base": pack(e), "periods": e.get("periods", {})}
            print(f"  {name:10} {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)
            if layer == "proxy":
                report["bootstrap"][name] = bootstrap(res["rets"])
                weights = res["weights"]
                report["exposure"][name] = {
                    "max_159232": float(weights["159232"].max()),
                    "max_cash": float(weights["cash"].max()),
                    "max_sum_assets": float(weights.iloc[:, :-1].sum(axis=1).max()),
                }

    print("== 费用敏感性（proxy，单边 bp） ==", flush=True)
    for name, cfg in (("base_v40", base), ("candidate", cand)):
        report["costs"][name] = {}
        for bp in (5, 10, 15, 20):
            _, e = run(cfg, R, PROXY_START, fee=bp / 10000.0)
            report["costs"][name][f"{bp}bp"] = pack(e)
            print(f"  {name:10} {bp:02d}bp {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    print("== 七个压力情景 ==", flush=True)
    scenarios = [
        ("2019-2021牛市", R, None, "2019-01-04", "2021-02-18"),
        ("2025-2026牛市", Rr, None, "2025-04-23", "2026-07-31"),
        ("2023-2024震荡", R, None, "2023-01-03", "2024-08-30"),
        ("2015股灾", R, None, "2015-06-15", "2016-02-29"),
        ("2018熊市", R, None, "2018-01-02", "2019-01-03"),
        ("2021-2022熊市", R, None, "2021-02-19", "2022-10-31"),
        ("共振熊市_合成", SYNTH, S_IDX, "2024-09-02", "2026-07-31"),
    ]
    for sn, Rs, am, ps, pe in scenarios:
        report["stress"][sn] = {}
        for name, cfg in (("base_v40", base), ("candidate", cand)):
            _, e = run(cfg, Rs, ps, pe, a_mkt=am)
            report["stress"][sn][name] = pack(e)
        b = report["stress"][sn]["base_v40"]
        c = report["stress"][sn]["candidate"]
        print(f"  {sn:<18} v40 {b['cagr']*100:6.2f}/{b['max_dd']*100:6.2f}/{b['calmar']:5.2f}  cand {c['cagr']*100:6.2f}/{c['max_dd']*100:6.2f}/{c['calmar']:5.2f}", flush=True)

    print("== WFO ==", flush=True)
    for name, ps, pe in (("train_2014_2021", PROXY_START, "2021-12-31"), ("test_2022_2026", OOS_START, "2026-11-10")):
        report["wfo"][name] = {}
        for label, cfg in (("base_v40", base), ("candidate", cand)):
            _, e = run(cfg, R, ps, pe)
            report["wfo"][name][label] = pack(e)
            print(f"  {name:<18} {label:10} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    json.dump(report, open(os.path.join(OUT, f"{TAG}_validation.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[ok] out/{TAG}_validation.json", flush=True)


if __name__ == "__main__":
    main()
