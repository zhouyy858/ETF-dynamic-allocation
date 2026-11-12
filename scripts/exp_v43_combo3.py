# -*- coding: utf-8 -*-
"""v43 第三轮：AM/深熊锁/急跌/溢价的组合（场内价）。"""
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

DEEP_TIGHT = {"CN": [0.07, 0.05, 0.16, 0.14], "US": [0.13, 0.12, 0.22, 0.15]}
DEEP_TIGHTER = {"CN": [0.07, 0.05, 0.15, 0.14], "US": [0.13, 0.12, 0.21, 0.15]}
DEEP_MILD = {"CN": [0.07, 0.05, 0.17, 0.14], "US": [0.13, 0.12, 0.23, 0.15]}


def C(cn=4.5, us=11.0, **kw):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c.update(kw)
    return c


def candidates():
    rows = [
        ("f45_am90", C(am_cut=0.90)),
        ("f45_am100", C(am_gate=False)),
        ("f45_deep", C(market_dd=DEEP_TIGHT)),
        ("f45_deep_am90", C(am_cut=0.90, market_dd=DEEP_TIGHT)),
        ("f45_deep_am100", C(am_gate=False, market_dd=DEEP_TIGHT)),
        ("f45_deep_am90_sb20", C(am_cut=0.90, asset_sb_cut=0.20, market_dd=DEEP_TIGHT)),
        ("f45_deep_am90_prem30", C(am_cut=0.90, premium_cut=[0.30, 0.15, 0.06], market_dd=DEEP_TIGHT)),
        ("f45_deep_am90_sb20_prem30", C(am_cut=0.90, asset_sb_cut=0.20,
                                         premium_cut=[0.30, 0.15, 0.06], market_dd=DEEP_TIGHT)),
        ("f45_am90_sb20_prem30", C(am_cut=0.90, asset_sb_cut=0.20,
                                    premium_cut=[0.30, 0.15, 0.06])),
        ("f5_deep_am90_sb20", C(5.0, 11.5, am_cut=0.90, asset_sb_cut=0.20, market_dd=DEEP_TIGHT)),
        ("f525_deep_am90_sb20", C(5.25, 11.5, am_cut=0.90, asset_sb_cut=0.20, market_dd=DEEP_TIGHT)),
        ("f525_deep_am90_sb20_prem30", C(5.25, 11.5, am_cut=0.90, asset_sb_cut=0.20,
                                          premium_cut=[0.30, 0.15, 0.06], market_dd=DEEP_TIGHT)),
        ("f5_deep_am100_sb20", C(5.0, 11.5, am_gate=False, asset_sb_cut=0.20, market_dd=DEEP_TIGHT)),
        ("f525_deep_am100_sb20", C(5.25, 11.5, am_gate=False, asset_sb_cut=0.20, market_dd=DEEP_TIGHT)),
        ("f45_deepmild_am90_sb20", C(am_cut=0.90, asset_sb_cut=0.20, market_dd=DEEP_MILD)),
        ("f45_deeptighter_am90_sb20", C(am_cut=0.90, asset_sb_cut=0.20, market_dd=DEEP_TIGHTER)),
        ("base_deep_am90_sb20", C(1.0, 12.0, am_cut=0.90, asset_sb_cut=0.20, market_dd=DEEP_TIGHT)),
    ]
    for name, c in rows:
        c["version"] = "v43_combo3_" + name
    return rows


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}}
    print(f"{'variant':<28} {'full CAGR':>10} {'MDD':>8} {'Cal':>6} "
          f"{'train Cal':>9} {'OOS Cal':>8} {'real Cal':>9}", flush=True)
    for name, c in candidates():
        row = {}
        for layer, sig, pnl, start, end in (
            ("proxy", V.R, P_PROXY, V.PROXY_START, None),
            ("train", V.R, P_PROXY, V.PROXY_START, "2021-12-31"),
            ("oos", V.R, P_PROXY, V.OOS_START, None),
            ("real", V.Rr, P_REAL, V.REAL_START, None),
        ):
            _, e = run(c, sig, pnl, start, end)
            row[layer] = V.pack(e)
        report["metrics"][name] = row
        p, t, o, r = row["proxy"], row["train"], row["oos"], row["real"]
        print(f"{name:<28} {p['cagr']*100:9.2f}% {p['max_dd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{t['calmar']:9.3f} {o['calmar']:8.3f} {r['calmar']:9.3f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_combo3.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_combo3.json", flush=True)


if __name__ == "__main__":
    main()
