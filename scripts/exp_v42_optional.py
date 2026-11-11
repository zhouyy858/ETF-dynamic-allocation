# -*- coding: utf-8 -*-
"""v42 research: optional-layer combinations on the high-return no-dividend base.

Mandatory: fast brake, deep-bear lock, structural gate, QDII premium gate/tilt.
Optional: cooldown, asset crash brake, 120d absolute momentum, vol gate, speed brake.
"""
import copy
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_v41_combo import metrics
from exp_v42_risk import HIGH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
KEYS = ["cool", "asset", "am", "vol", "speed"]


def build(bits):
    c = copy.deepcopy(HIGH)
    c["cool_off_weeks"] = 3 if bits[0] else 0
    c["asset_sb"] = bool(bits[1])
    c["am_gate"] = bool(bits[2])
    c["vol_gate"] = bool(bits[3])
    c["speed_brake"] = bool(bits[4])
    c["version"] = "v42-optional"
    return c


def label(bits):
    return "+".join(k for k, b in zip(KEYS, bits) if b) or "core_only"


def main():
    rows = {}
    print(f"{'variant':<34} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>6} {'real CAGR':>11} {'MDD':>8} {'Cal':>6} {'OOS Cal':>8}")
    for bits in itertools.product([False, True], repeat=len(KEYS)):
        c = build(bits)
        m = metrics(c)
        name = label(bits)
        rows[name] = {"bits": bits, "cfg": c, "metrics": m}
        p, r, o = m["proxy"], m["real"], m["oos"]
        print(f"{name:<34} {p['cagr']*100:10.2f}% {p['mdd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{r['cagr']*100:10.2f}% {r['mdd']*100:7.2f}% {r['calmar']:6.3f} {o['calmar']:8.3f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "exp_v42_optional.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v42_optional.json")


if __name__ == "__main__":
    main()
