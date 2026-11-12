# -*- coding: utf-8 -*-
"""v43 底仓参数重训（场内价，纸面研究）。

只扫描 floor_pct：国内按 159232 剔除后的有效底仓计，海外按 12% 名义底仓计。
训练段选参，OOS 与真实 ETF 窗口只做验证；输出不含净值口径。
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from engine import run_backtest, evaluate
from run_backtest import exec_panel
from strategy import DynamicStrategy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(REF, "final_cfg_v43.json")))

TRAIN_START = V.PROXY_START
TRAIN_END = "2021-12-31"
OOS_START = V.OOS_START

# 国内有效底仓 = floor_pct.cn / 2（159232 被排除后只剩 515100/563020）
CN_EFF = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
US = (8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)


def cfg(cn_eff, us):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": float(cn_eff) * 2.0, "us": float(us)}
    c["version"] = f"v43_cn{cn_eff:g}_us{us:g}"
    return c


def run(c, sig, pnl, start, end=None, fee=0.0005):
    ds = DynamicStrategy(sig, cfg=c)
    res = run_backtest(
        pnl,
        target_weights_fn=ds.target_fn(),
        daily_override_fn=ds.daily_fn(),
        start=start,
        end=end,
        name="DYN",
        fee=fee,
        min_delta=c.get("min_delta", 0.02),
        repo=c.get("repo_rate", 0.022),
        tranche_weights=c.get("tranche_weights"),
        cash_bond_rets=V.BOND,
        cash_bond_pct=c.get("cash_bond_pct", 0.0),
        rebal_weekday=c.get("rebal_weekday", 4),
        rebal_freq=c.get("rebal_freq", "weekly"),
        strict=True,
    )
    return res, evaluate(res)


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"grid": {"cn_eff": list(CN_EFF), "us": list(US)}, "metrics": {}}

    print("== 底仓网格（训练段选参，场内价） ==", flush=True)
    print(f"{'cn_eff':>6} {'us':>5} {'train CAGR':>11} {'MDD':>8} {'Cal':>6} "
          f"{'OOS CAGR':>10} {'MDD':>8} {'Cal':>6} {'real Cal':>9}", flush=True)
    for cn in CN_EFF:
        for us in US:
            c = cfg(cn, us)
            key = f"cn{cn:g}_us{us:g}"
            row = {}
            for layer, sig, pnl, start, end in (
                ("train", V.R, P_PROXY, TRAIN_START, TRAIN_END),
                ("oos", V.R, P_PROXY, OOS_START, None),
                ("proxy", V.R, P_PROXY, TRAIN_START, None),
                ("real", V.Rr, P_REAL, V.REAL_START, None),
            ):
                _, e = run(c, sig, pnl, start, end)
                row[layer] = V.pack(e)
            report["metrics"][key] = row
            print(f"{cn:6.1f} {us:5.1f} {row['train']['cagr']*100:10.2f}% "
                  f"{row['train']['max_dd']*100:7.2f}% {row['train']['calmar']:6.2f} "
                  f"{row['oos']['cagr']*100:9.2f}% {row['oos']['max_dd']*100:7.2f}% "
                  f"{row['oos']['calmar']:6.2f} {row['real']['calmar']:9.2f}", flush=True)

    json.dump(report, open(os.path.join(OUT, "exp_v43_floors.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_floors.json", flush=True)


if __name__ == "__main__":
    main()
