# -*- coding: utf-8 -*-
"""v41 组合搜索（2026-11-11，数据至 2026-11-10）

目的：在 v40 上剔除自由现金流槽（159232/159222），并检验哪些风控层/执行层
对收益与 Calmar 有真实贡献。先做单机制筛选，再对通过项做组合。

口径与生产一致：signal_lag=1、premium_shift=2、strict=True、周五 1 笔、
费用单边万 5、现金 100% 511260、instrument_swaps 生效。

用法: python3 scripts/exp_v41_combo.py
输出: out/exp_v41_combo.json
"""
import copy
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from swaps import pnl_panel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
CFG_FILE = os.path.join(ROOT, "references", "final_cfg_v40.json")

PROXY_START = "2014-06-23"
REAL_START = "2025-04-23"
OOS_START = "2022-01-04"

BASE = json.load(open(CFG_FILE))
R_PROXY, _ = build_panel("proxy")
R_REAL, _ = build_panel("real")


def C(**kw):
    cfg = copy.deepcopy(BASE)
    cfg.update(kw)
    return cfg


def run_one(cfg, R_sig, start):
    R = pnl_panel(R_sig, "real" if R_sig is R_REAL else "proxy", cfg)
    bond = rets_from(read_table(f"{cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav") \
        if cfg.get("cash_bond_pct") else None
    ds = DynamicStrategy(R_sig, cfg=cfg)
    res = run_backtest(
        R,
        target_weights_fn=ds.target_fn(),
        daily_override_fn=ds.daily_fn(),
        start=start,
        name="DYN",
        min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond,
        cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True,
    )
    e = evaluate(res)
    return {
        "cagr": e["cagr"],
        "mdd": e["max_dd"],
        "calmar": e["calmar"],
        "sharpe": e["sharpe"],
        "turnover": e["turnover"],
        "cash": e["avg_cash"],
    }


def metrics(cfg):
    return {
        "proxy": run_one(cfg, R_PROXY, PROXY_START),
        "real": run_one(cfg, R_REAL, REAL_START),
        "oos": run_one(cfg, R_PROXY, OOS_START),
    }


DROP = {"exclude": ["159232"]}
NO_FAST = {"market_dd": {"CN": [9.99, 0.05, 0.18, 0.14],
                         "US": [9.99, 0.12, 0.24, 0.15]}}
NO_DEEP = {"market_dd": {"CN": [0.07, 0.05, 9.99, 0.14],
                         "US": [0.13, 0.12, 9.99, 0.15]}}

CANDS = [
    ("base_v40", C()),
    ("drop_cf", C(**DROP)),
    ("drop_cf_cn0", C(exclude=["159232"], floor_pct={"cn": 0.0, "us": 10.0})),
    ("drop_cf_cn1", C(exclude=["159232"], floor_pct={"cn": 1.0, "us": 10.0})),
    ("drop_cf_us9", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 9.0})),
    ("drop_cf_us11", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 11.0})),
    ("drop_cf_us12", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0})),
    ("drop_cf_us13", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 13.0})),
    ("drop_cf_maxeq100", C(exclude=["159232"], max_eq=1.0)),
    ("drop_cf_maxeq100_cash0", C(exclude=["159232"], max_eq=1.0, min_cash=0.0)),
    ("drop_cf_bull35", C(exclude=["159232"], growth_split_bull={"159952": 0.35, "159941": 0.58, "513500": 0.07})),
    ("drop_cf_bull34", C(exclude=["159232"], growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07})),
    ("drop_cf_bull32_63_5", C(exclude=["159232"], growth_split_bull={"159952": 0.32, "159941": 0.63, "513500": 0.05})),
    ("drop_cf_bull32_58_10", C(exclude=["159232"], growth_split_bull={"159952": 0.32, "159941": 0.58, "513500": 0.10})),
    ("drop_cf_state75", C(exclude=["159232"], state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                                          "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                                          "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]})),
    ("drop_cf_no_corr", C(exclude=["159232"], corr_risk=False)),
    ("drop_cf_no_prem_gate", C(exclude=["159232"], premium_gate=False)),
    ("drop_cf_no_prem_tilt", C(exclude=["159232"], premium_tilt=False)),
    ("drop_cf_no_cool", C(exclude=["159232"], cool_off_weeks=0)),
    ("drop_cf_cool2", C(exclude=["159232"], cool_off_weeks=2)),
    ("drop_cf_cool4", C(exclude=["159232"], cool_off_weeks=4)),
    ("drop_cf_no_fast", C(**{**DROP, **NO_FAST})),
    ("drop_cf_no_deep", C(**{**DROP, **NO_DEEP})),
    ("drop_cf_gate80", C(exclude=["159232"], gate_win=80)),
    ("drop_cf_gate120", C(exclude=["159232"], gate_win=120)),
    ("drop_cf_mindelta02", C(exclude=["159232"], min_delta=0.02)),
    ("drop_cf_mindelta05", C(exclude=["159232"], min_delta=0.05)),
    ("drop_cf_prem_rotate", C(exclude=["159232"], premium_rotate=True)),
    ("drop_cf_us_rotation", C(exclude=["159232"], us_rotation=True)),
    ("drop_cf_recovery", C(exclude=["159232"], recovery_ramp=True, recovery_ramp_min=0.3)),
    ("drop_cf_reentry1", C(exclude=["159232"], reentry_step=1.0)),
    ("drop_cf_us12_state75", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                               state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                          "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                          "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]})),
    ("drop_cf_us12_bull34", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                              growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07})),
    ("drop_cf_us12_bull35", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                              growth_split_bull={"159952": 0.35, "159941": 0.58, "513500": 0.07})),
    ("drop_cf_us12_state75_bull34", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                                      growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07},
                                      state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                                 "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                                 "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]})),
    ("drop_cf_us12_state80_bull34", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                                      growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07},
                                      state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                                 "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                                 "6": [66, 14], "7": [80, 10], "8": [97, 6], "9": [93, 3]})),
    ("drop_cf_us12_state75_bull34_cash0", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                                            growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07},
                                            state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                                       "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                                       "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]},
                                            min_cash=0.0, max_eq=1.0)),
    ("drop_cf_us12_state75_bull34_no_corr", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                                              growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07},
                                              state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                                         "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                                         "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]},
                                              corr_risk=False)),
    ("drop_cf_us12_state75_bull34_md02", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                                           growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07},
                                           state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                                      "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                                      "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]},
                                           min_delta=0.02)),
    ("drop_cf_us13_state75_bull34", C(exclude=["159232"], floor_pct={"cn": 2.0, "us": 13.0},
                                      growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07},
                                      state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                                                 "3": [36, 26], "4": [39, 21], "5": [44, 16],
                                                 "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]})),
]

BEST_RET = dict(exclude=["159232"], floor_pct={"cn": 2.0, "us": 12.0},
                growth_split_bull={"159952": 0.34, "159941": 0.59, "513500": 0.07},
                state_map={"0": [5, 30], "1": [15, 29], "2": [25, 28],
                           "3": [36, 26], "4": [39, 21], "5": [44, 16],
                           "6": [66, 14], "7": [75, 10], "8": [95, 6], "9": [91, 3]},
                min_cash=0.0, max_eq=1.0)
CANDS += [
    ("best_ret_volgate", C(**BEST_RET, vol_gate=True)),
    ("best_ret_regime", C(**BEST_RET, regime_gate=True)),
    ("best_ret_asset_sb", C(**BEST_RET, asset_sb=True)),
    ("best_ret_am", C(**BEST_RET, am_gate=True)),
    ("best_ret_rsrs", C(**BEST_RET, rsrs_gate=True)),
    ("best_ret_valuation", C(**BEST_RET, valuation_gate=True)),
    ("best_ret_breadth", C(**BEST_RET, breadth_gate=True)),
    ("best_ret_adx", C(**BEST_RET, adx_gate=True)),
    ("best_ret_mkt_sb", C(**BEST_RET, speed_brake_mkt=True)),
    ("best_ret_cool4", C(**BEST_RET, cool_off_weeks=4)),
]


def fmt(x, pct=False):
    return f"{x*100:.2f}%" if pct else f"{x:.3f}"


def main():
    rows = {}
    print(f"{'候选':<26} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>7} {'real CAGR':>11} {'MDD':>8} {'Cal':>7} {'OOS Cal':>8}")
    for name, cfg in CANDS:
        m = metrics(cfg)
        rows[name] = {"cfg": cfg, "metrics": m}
        p, r, o = m["proxy"], m["real"], m["oos"]
        print(f"{name:<26} {p['cagr']*100:10.2f}% {p['mdd']*100:7.2f}% {p['calmar']:7.2f} "
              f"{r['cagr']*100:10.2f}% {r['mdd']*100:7.2f}% {r['calmar']:7.2f} {o['calmar']:8.2f}")

    base = rows["base_v40"]["metrics"]
    bp, br, bo = base["proxy"], base["real"], base["oos"]
    print("\n相对基线：")
    survivors = []
    for name, row in rows.items():
        if name == "base_v40":
            continue
        p, r, o = row["metrics"]["proxy"], row["metrics"]["real"], row["metrics"]["oos"]
        d_cagr = (p["cagr"] - bp["cagr"]) * 100
        d_cal = p["calmar"] - bp["calmar"]
        d_real = (r["cagr"] - br["cagr"]) * 100
        d_oos = o["calmar"] - bo["calmar"]
        ok = (d_cagr >= 0.3 and d_cal >= -0.02 and d_real >= 0.0 and d_oos >= -0.05
              and p["mdd"] >= -0.095 and r["mdd"] >= -0.045)
        if ok:
            survivors.append(name)
        print(f"{name:<26} ΔCAGR {d_cagr:+6.2f}pp ΔCal {d_cal:+6.3f} "
              f"Δreal {d_real:+6.2f}pp ΔOOS {d_oos:+6.3f} {'★' if ok else ''}")

    print(f"\n一关通过 {len(survivors)} 个: {survivors}")
    json.dump(rows, open(os.path.join(OUT, "exp_v41_combo.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print("[ok] out/exp_v41_combo.json")


if __name__ == "__main__":
    main()
