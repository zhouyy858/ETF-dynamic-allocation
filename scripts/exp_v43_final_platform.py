# -*- coding: utf-8 -*-
"""v43 最终候选的 AM/急跌/深熊边界检查（场内价）。"""
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


def C(cn=5.0, us=11.5, deep=15.0, am=0.95, sb=0.15):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c["market_dd"] = {"CN": [0.07, 0.05, float(deep) / 100.0, 0.14],
                      "US": [0.13, 0.12, float(deep + 6) / 100.0, 0.15]}
    c["am_cut"] = float(am)
    c["asset_sb_cut"] = float(sb)
    return c


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"am": {}, "sb": {}, "deep": {}}

    def measure(tag, c, bucket):
        row = {}
        for layer, sig, pnl, start, end in (
            ("proxy", V.R, P_PROXY, V.PROXY_START, None),
            ("train", V.R, P_PROXY, V.PROXY_START, "2021-12-31"),
            ("oos", V.R, P_PROXY, V.OOS_START, None),
            ("real", V.Rr, P_REAL, V.REAL_START, None),
        ):
            _, e = run(c, sig, pnl, start, end)
            row[layer] = V.pack(e)
        report[bucket][tag] = row
        p, t, o, r = row["proxy"], row["train"], row["oos"], row["real"]
        print(f"{tag:<18} {p['cagr']*100:9.2f}% {p['max_dd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{t['calmar']:9.3f} {o['calmar']:8.3f} {r['calmar']:9.3f}", flush=True)

    print("== AM 边界（cn5/us11.5, deep15, sb15） ==", flush=True)
    for am in (0.90, 0.95, 0.975, 0.99):
        measure(f"am{am:.3f}", C(am=am), "am")
    print("== 急跌边界 ==", flush=True)
    for sb in (0.10, 0.15, 0.20):
        measure(f"sb{sb:.2f}", C(sb=sb), "sb")
    print("== 深熊边界 ==", flush=True)
    for d in (14.0, 15.0, 16.0):
        measure(f"deep{d:g}/{d+6:g}", C(deep=d), "deep")

    json.dump(report, open(os.path.join(OUT, "exp_v43_final_platform.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_final_platform.json", flush=True)


if __name__ == "__main__":
    main()
