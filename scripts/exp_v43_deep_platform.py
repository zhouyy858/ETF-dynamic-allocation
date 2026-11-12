# -*- coding: utf-8 -*-
"""v43 深熊阈值/AM/急跌/底仓邻域平台检查（场内价）。"""
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


def C(cn=4.5, us=11.0, deep_cn=15.0, deep_us=21.0, am=0.90, sb=0.20, prem=None):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c["market_dd"] = {"CN": [0.07, 0.05, float(deep_cn) / 100.0, 0.14],
                      "US": [0.13, 0.12, float(deep_us) / 100.0, 0.15]}
    c["am_cut"] = float(am)
    c["asset_sb_cut"] = float(sb)
    if prem is not None:
        c["premium_cut"] = list(prem)
    return c


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"deep": {}, "am": {}, "sb": {}, "floor": {}, "prem": {}}

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
        print(f"{tag:<24} {p['cagr']*100:9.2f}% {p['max_dd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{t['calmar']:9.3f} {o['calmar']:8.3f} {r['calmar']:9.3f}", flush=True)

    print("== deep 阈值（CN / US=CN+6） ==", flush=True)
    for d in (14.0, 15.0, 16.0, 17.0, 18.0):
        measure(f"deep{d:g}/{d+6:g}", C(deep_cn=d, deep_us=d + 6), "deep")
    print("== AM 邻域 ==", flush=True)
    for a in (0.85, 0.90, 0.95):
        measure(f"am{a:.2f}", C(am=a), "am")
    print("== 急跌 cut 邻域 ==", flush=True)
    for s in (0.15, 0.20, 0.25, 0.30):
        measure(f"sb{s:.2f}", C(sb=s), "sb")
    print("== 底仓邻域 ==", flush=True)
    for cn in (4.0, 4.5, 5.0):
        for us in (10.5, 11.0, 11.5):
            measure(f"cn{cn:g}/us{us:g}", C(cn=cn, us=us), "floor")
    print("== 溢价 0.30 变体 ==", flush=True)
    for tag, prem in (("prem30", [0.30, 0.15, 0.06]), ("prem35", [0.35, 0.175, 0.07])):
        measure(tag, C(prem=prem), "prem")

    json.dump(report, open(os.path.join(OUT, "exp_v43_deep_platform.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_deep_platform.json", flush=True)


if __name__ == "__main__":
    main()
