# -*- coding: utf-8 -*-
"""v43 成长拆分 × 底仓的二维扫描（场内价）。"""
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


def C(cn, us, chi):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c["growth_split_bull"] = {"159952": float(chi), "159941": 1.0 - float(chi) - 0.07,
                               "513500": 0.07}
    return c


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}}
    print(f"{'floor':<10} {'chi':>5} {'full CAGR':>10} {'MDD':>8} {'Cal':>6} "
          f"{'train Cal':>9} {'OOS Cal':>8} {'real Cal':>9}", flush=True)
    for cn, us in ((1.0, 12.0), (4.5, 11.0), (5.0, 11.5)):
        for chi in (0.30, 0.32, 0.34, 0.36, 0.38, 0.40):
            c = C(cn, us, chi)
            row = {}
            for layer, sig, pnl, start, end in (
                ("proxy", V.R, P_PROXY, V.PROXY_START, None),
                ("train", V.R, P_PROXY, V.PROXY_START, "2021-12-31"),
                ("oos", V.R, P_PROXY, V.OOS_START, None),
                ("real", V.Rr, P_REAL, V.REAL_START, None),
            ):
                _, e = run(c, sig, pnl, start, end)
                row[layer] = V.pack(e)
            report["metrics"][f"cn{cn:g}_us{us:g}_chi{chi:.2f}"] = row
            p, t, o, r = row["proxy"], row["train"], row["oos"], row["real"]
            print(f"{cn:g}/{us:g}   {chi:5.2f} {p['cagr']*100:9.2f}% {p['max_dd']*100:7.2f}% "
                  f"{p['calmar']:6.3f} {t['calmar']:9.3f} {o['calmar']:8.3f} {r['calmar']:9.3f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_split.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_split.json", flush=True)


if __name__ == "__main__":
    main()
