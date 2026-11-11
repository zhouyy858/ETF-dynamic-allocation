# -*- coding: utf-8 -*-
"""v42 进攻版的现金层/新增资产研究（纸面，不执行下单）。

比较：
1. 现金层 511260 vs 511010 vs 511380(可转债) vs 511360(短融)；
2. 现金层混合与固定比例可转债/短融 overlay；
3. 用 511380 替换创业板槽；
4. 去掉标普 513500（仅保留纳指+创业板）。

口径：data 截至 2026-11-10；signal_lag=1、strict=True、周五 1 笔、单边万 5。
511380/511360 上市前用 511260 拼接，避免未来函数；overlay 按日再平衡、从现金层出资。
"""
import copy
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from metrics import summary
from strategy import DynamicStrategy
from swaps import pnl_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(ROOT, "references", "final_cfg_v42.json")))
R_PROXY, _ = build_panel("proxy")
R_REAL, _ = build_panel("real")
PROXY_START = "2014-06-23"
REAL_START = "2025-04-23"
OOS_START = "2022-01-04"
REPO_D = (1 + 0.022) ** (1 / 252) - 1


def cash_series(code):
    s = rets_from(read_table(f"{code}_nav.csv"), "cum_nav")
    base = rets_from(read_table("511260_nav.csv"), "cum_nav")
    return s.reindex(base.index.union(s.index)).fillna(base).sort_index()


CASH = {c: cash_series(c) for c in ("511260", "511010", "511380", "511360")}


def blend(name, weight):
    a = CASH["511260"]
    b = CASH[name]
    idx = a.index.union(b.index).sort_values()
    aa = a.reindex(idx).fillna(REPO_D)
    bb = b.reindex(idx).fillna(aa)
    return (1.0 - weight) * aa + weight * bb


def run_cfg(cfg, Rs, start, cash=None, overlay=None, overlay_w=0.0):
    layer = "real" if Rs is R_REAL else "proxy"
    cash = CASH["511260"] if cash is None else cash
    pnl = pnl_panel(Rs, layer, cfg)
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(
        pnl,
        target_weights_fn=ds.target_fn(),
        daily_override_fn=ds.daily_fn(),
        start=start,
        name="DYN",
        min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=cash,
        cash_bond_pct=1.0,
        rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True,
    )
    e = evaluate(res)
    if overlay is None or overlay_w <= 0:
        return e
    r = res["rets"]
    c = cash.reindex(r.index).ffill().fillna(REPO_D)
    a = overlay.reindex(r.index)
    a = a.fillna(c)
    r2 = r + overlay_w * (a - c)
    s = summary(r2, name="overlay")
    return {
        "cagr": s["cagr"],
        "max_dd": s["max_dd"],
        "calmar": s["calmar"],
        "sharpe": s["sharpe"],
        "avg_cash": max(0.0, e["avg_cash"] - overlay_w),
        "turnover": e["turnover"],
    }


def metrics(cfg, cash=None, overlay=None, overlay_w=0.0):
    return {
        "proxy": run_cfg(cfg, R_PROXY, PROXY_START, cash, overlay, overlay_w),
        "real": run_cfg(cfg, R_REAL, REAL_START, cash, overlay, overlay_w),
        "oos": run_cfg(cfg, R_PROXY, OOS_START, cash, overlay, overlay_w),
    }


def no_sp500(premium=True):
    c = copy.deepcopy(BASE)
    c["exclude"] = ["159232", "515100", "513500"]
    c["floor_pct"] = {"cn": 0.0, "us": 28.0}
    c["growth_split_bull"] = {"159952": 0.40, "159941": 0.60, "513500": 0.0}
    c["growth_split_bear"] = {"159952": 0.45, "159941": 0.55, "513500": 0.0}
    c["premium_tilt"] = False
    c["premium_gate"] = bool(premium)
    c["version"] = "v42-no-sp500" + ("" if premium else "-no-premium")
    return c


def main():
    rows = {}
    variants = [
        ("base_v42", BASE, None, None, 0.0),
        ("cash_511010", BASE, CASH["511010"], None, 0.0),
        ("cash_511380", BASE, CASH["511380"], None, 0.0),
        ("cash_511360", BASE, CASH["511360"], None, 0.0),
        ("cash_90_10_380", BASE, blend("511380", 0.10), None, 0.0),
        ("cash_80_20_380", BASE, blend("511380", 0.20), None, 0.0),
        ("cash_90_10_360", BASE, blend("511360", 0.10), None, 0.0),
        ("cash_80_20_360", BASE, blend("511360", 0.20), None, 0.0),
        ("overlay_380_5", BASE, None, CASH["511380"], 0.05),
        ("overlay_380_10", BASE, None, CASH["511380"], 0.10),
        ("overlay_380_15", BASE, None, CASH["511380"], 0.15),
        ("overlay_380_20", BASE, None, CASH["511380"], 0.20),
        ("overlay_360_5", BASE, None, CASH["511360"], 0.05),
        ("overlay_360_10", BASE, None, CASH["511360"], 0.10),
        ("swap_cyb_to_380", dict(BASE, instrument_swaps={"159952": "511380"}, version="v42-swap-cyb-380"), None, None, 0.0),
        ("no_sp500", no_sp500(True), None, None, 0.0),
        ("no_sp500_no_premium", no_sp500(False), None, None, 0.0),
    ]
    print(f"{'variant':<24} {'proxy CAGR':>10} {'MDD':>8} {'Cal':>6} | {'real CAGR':>10} {'MDD':>8} {'Cal':>6} | {'OOS Cal':>7}")
    for name, cfg, cash, overlay, ow in variants:
        m = metrics(cfg, cash, overlay, ow)
        rows[name] = {"cfg": cfg, "metrics": m}
        p, r, o = m["proxy"], m["real"], m["oos"]
        print(f"{name:<24} {p['cagr']*100:9.2f}% {p['max_dd']*100:8.2f}% {p['calmar']:6.3f} | "
              f"{r['cagr']*100:9.2f}% {r['max_dd']*100:8.2f}% {r['calmar']:6.3f} | {o['calmar']:7.3f}", flush=True)

    stats = {}
    for code in ("511260", "511010", "511380", "511360"):
        s = CASH[code]
        s = s[s.index >= "2020-09-25"]
        x = summary(s, name=code)
        stats[code] = {k: x[k] for k in ("cagr", "max_dd", "vol", "sharpe", "calmar")}
        print(f"[asset] {code}: CAGR={x['cagr']*100:.2f}% MDD={x['max_dd']*100:.2f}% Vol={x['vol']*100:.2f}% Cal={x['calmar']:.2f}")

    json.dump({"rows": rows, "asset_stats": stats}, open(os.path.join(OUT, "exp_v42_alt_assets.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    print("[ok] out/exp_v42_alt_assets.json")


if __name__ == "__main__":
    main()
