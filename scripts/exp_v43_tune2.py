# -*- coding: utf-8 -*-
"""v43 在 floor 4.5/11 上的单轴参数扫描（场内价）。"""
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


def C(**kw):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": 9.0, "us": 11.0}
    c.update(kw)
    return c


def state7(g):
    sm = copy.deepcopy(BASE["state_map"])
    sm["7"] = [g, 10]
    sm["8"] = [min(99, g + 20), 6]
    sm["9"] = [min(99, g + 16), 3]
    return sm


def candidates():
    rows = [
        ("base_f45", C()),
        ("state71", C(state_map=state7(71))),
        ("state73", C(state_map=state7(73))),
        ("state75", C(state_map=state7(75))),
        ("state77", C(state_map=state7(77))),
        ("prem30", C(premium_cut=[0.30, 0.15, 0.06])),
        ("prem35", C(premium_cut=[0.35, 0.175, 0.07])),
        ("prem40", C(premium_cut=[0.40, 0.20, 0.08])),
        ("am70", C(am_cut=0.70)),
        ("am80", C(am_cut=0.80)),
        ("am90", C(am_cut=0.90)),
        ("sb20", C(asset_sb_cut=0.20)),
        ("sb30", C(asset_sb_cut=0.30)),
        ("sb40", C(asset_sb_cut=0.40)),
        ("md02", C(min_delta=0.02)),
        ("md035", C(min_delta=0.035)),
        ("md05", C(min_delta=0.05)),
        ("hyst60", C(hyst_up=0.60, hyst_down=0.20)),
        ("hyst66", C(hyst_up=0.66, hyst_down=0.24)),
        ("hyst72", C(hyst_up=0.72, hyst_down=0.28)),
        ("fast_tight", C(market_dd={"CN": [0.06, 0.05, 0.18, 0.14],
                                    "US": [0.12, 0.12, 0.24, 0.15]})),
        ("deep_tight", C(market_dd={"CN": [0.07, 0.05, 0.16, 0.14],
                                    "US": [0.13, 0.12, 0.22, 0.15]})),
        ("both_tight", C(market_dd={"CN": [0.06, 0.05, 0.16, 0.14],
                                    "US": [0.12, 0.12, 0.22, 0.15]})),
        ("deep_loose", C(market_dd={"CN": [0.07, 0.05, 0.20, 0.14],
                                    "US": [0.13, 0.12, 0.26, 0.15]})),
        ("confirm1", C(score_confirm_weeks=1)),
        ("confirm2", C(score_confirm_weeks=2)),
        ("confirm3", C(score_confirm_weeks=3)),
    ]
    for name, c in rows:
        c["version"] = "v43_tune2_" + name
    return rows


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}}
    print(f"{'variant':<14} {'full CAGR':>10} {'MDD':>8} {'Cal':>6} "
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
        print(f"{name:<14} {p['cagr']*100:9.2f}% {p['max_dd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{t['calmar']:9.3f} {o['calmar']:8.3f} {r['calmar']:9.3f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_tune2.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_tune2.json", flush=True)


if __name__ == "__main__":
    main()
