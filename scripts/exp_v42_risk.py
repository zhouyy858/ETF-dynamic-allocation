# -*- coding: utf-8 -*-
"""v42 research: risk-layer ablation/add-back on the no-dividend universe.

Two allocation baselines are tested:
- balanced: US floor 14%, half of the former defense bucket converted to growth
- high-return: US floor 14%, all defense bucket converted, ChiNext 40%

The production default remains v41. Pure paper research.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_v41_combo import metrics
from exp_v42_nodiv import nodiv, state_realloc, split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

BALANCED = nodiv(floor_pct={"cn": 0.0, "us": 14.0},
                 state_map=state_realloc(0.50))
HIGH = nodiv(floor_pct={"cn": 0.0, "us": 14.0},
             state_map=state_realloc(1.0), growth_split_bull=split(0.40))

NO_FAST = {"CN": [9.99, 0.05, 0.18, 0.14], "US": [9.99, 0.12, 0.24, 0.15]}
NO_DEEP = {"CN": [0.07, 0.05, 9.99, 0.14], "US": [0.13, 0.12, 9.99, 0.15]}


def cfg(base, **kw):
    c = copy.deepcopy(base)
    c.update(kw)
    c["version"] = "v42-risk-search"
    return c


def loo(base):
    return [
        ("all_on", cfg(base)),
        ("no_fast", cfg(base, market_dd=NO_FAST)),
        ("no_deep", cfg(base, market_dd=NO_DEEP)),
        ("no_gate", cfg(base, struct_gate=False)),
        ("no_premium", cfg(base, premium_gate=False, premium_tilt=False)),
        ("no_cool", cfg(base, cool_off_weeks=0)),
        ("no_asset_sb", cfg(base, asset_sb=False)),
        ("no_am", cfg(base, am_gate=False)),
        ("no_vol", cfg(base, vol_gate=False)),
    ]


def addback(base):
    return [
        ("add_corr", cfg(base, corr_risk=True)),
        ("add_prem_rotate", cfg(base, premium_rotate=True)),
        ("add_us_rotation", cfg(base, us_rotation=True)),
        ("add_regime", cfg(base, regime_gate=True)),
        ("add_recovery", cfg(base, recovery_ramp=True)),
        ("add_reentry10", cfg(base, reentry_step=0.10)),
        ("add_speed_brake", cfg(base, speed_brake=True)),
        ("gate_band1", cfg(base, gate_band=0.01)),
        ("gate_band2", cfg(base, gate_band=0.02)),
    ]


def main():
    out = {}
    for bname, base in (("balanced", BALANCED), ("high", HIGH)):
        rows = loo(base) + addback(base)
        print(f"\n== {bname} ==", flush=True)
        print(f"{'variant':<20} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>6} {'real CAGR':>11} {'MDD':>8} {'Cal':>6} {'OOS Cal':>8}")
        for name, c in rows:
            m = metrics(c)
            out[f"{bname}:{name}"] = {"cfg": c, "metrics": m}
            p, r, o = m["proxy"], m["real"], m["oos"]
            print(f"{name:<20} {p['cagr']*100:10.2f}% {p['mdd']*100:7.2f}% {p['calmar']:6.3f} "
                  f"{r['cagr']*100:10.2f}% {r['mdd']*100:7.2f}% {r['calmar']:6.3f} {o['calmar']:8.3f}", flush=True)
    json.dump(out, open(os.path.join(OUT, "exp_v42_risk.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v42_risk.json")


if __name__ == "__main__":
    main()
