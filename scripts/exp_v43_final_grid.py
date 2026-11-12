# -*- coding: utf-8 -*-
"""v43 最终底仓折中网格（含 2015/2018 压力，场内价）。"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import BASE, run
from run_backtest import exec_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


def C(cn, us):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c["market_dd"] = {"CN": [0.07, 0.05, 0.15, 0.14],
                      "US": [0.13, 0.12, 0.21, 0.15]}
    c["am_cut"] = 0.90
    c["asset_sb_cut"] = 0.20
    c["premium_cut"] = [0.30, 0.15, 0.06]
    return c


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}}
    print(f"{'floor':<12} {'full CAGR':>10} {'MDD':>8} {'Cal':>6} {'train':>6} "
          f"{'OOS':>6} {'real':>6} {'2015':>7} {'2018':>7}", flush=True)
    for cn in (3.5, 4.0, 4.5, 5.0):
        for us in (11.0, 11.5):
            c = C(cn, us)
            row = {}
            for layer, sig, pnl, start, end in (
                ("proxy", V.R, P_PROXY, V.PROXY_START, None),
                ("train", V.R, P_PROXY, V.PROXY_START, "2021-12-31"),
                ("oos", V.R, P_PROXY, V.OOS_START, None),
                ("real", V.Rr, P_REAL, V.REAL_START, None),
                ("2015", V.R, P_PROXY, "2015-06-15", "2016-02-29"),
                ("2018", V.R, P_PROXY, "2018-01-02", "2019-01-03"),
            ):
                _, e = run(c, sig, pnl, start, end)
                row[layer] = V.pack(e)
            report["metrics"][f"cn{cn:g}_us{us:g}"] = row
            p = row["proxy"]
            print(f"cn{cn:<4g}/us{us:<4g} {p['cagr']*100:9.2f}% {p['max_dd']*100:7.2f}% {p['calmar']:6.3f} "
                  f"{row['train']['calmar']:6.3f} {row['oos']['calmar']:6.3f} {row['real']['calmar']:6.3f} "
                  f"{row['2015']['max_dd']*100:6.2f}% {row['2018']['max_dd']*100:6.2f}%", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_final_grid.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_final_grid.json", flush=True)


if __name__ == "__main__":
    main()
