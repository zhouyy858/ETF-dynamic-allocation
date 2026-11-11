# -*- coding: utf-8 -*-
"""v42 research: exhaustive 2^8 risk-switch enumeration on no-dividend high base."""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_v41_combo import R_PROXY, R_REAL, run_one, PROXY_START, REAL_START
from exp_v42_risk import HIGH, NO_DEEP, NO_FAST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
OOS_START = "2022-01-04"

KEYS = ["fast", "deep", "gate", "premium", "cool", "asset", "am", "vol"]


def build(bits):
    c = copy.deepcopy(HIGH)
    md = copy.deepcopy(c["market_dd"])
    if not bits[0]:
        md["CN"] = list(NO_FAST["CN"])
        md["US"] = list(NO_FAST["US"])
    if not bits[1]:
        md["CN"][2] = NO_DEEP["CN"][2]
        md["US"][2] = NO_DEEP["US"][2]
    c["market_dd"] = md
    c["struct_gate"] = bool(bits[2])
    c["premium_gate"] = bool(bits[3])
    c["premium_tilt"] = bool(bits[3])
    c["cool_off_weeks"] = 3 if bits[4] else 0
    c["asset_sb"] = bool(bits[5])
    c["am_gate"] = bool(bits[6])
    c["vol_gate"] = bool(bits[7])
    c["version"] = "v42-enum"
    return c


def label(bits):
    return "+".join(k for k, b in zip(KEYS, bits) if b) or "none"


def main():
    rows = []
    for mask in range(1 << len(KEYS)):
        bits = tuple(bool(mask >> i & 1) for i in range(len(KEYS)))
        c = build(bits)
        p = run_one(c, R_PROXY, PROXY_START)
        rows.append({"mask": mask, "bits": bits, "label": label(bits),
                     "cfg": c, "proxy": p})
        if (mask + 1) % 32 == 0:
            print(f"  {mask + 1}/256", flush=True)

    by_cal = sorted(rows, key=lambda x: x["proxy"]["calmar"], reverse=True)
    by_ret = sorted((x for x in rows if x["proxy"]["mdd"] >= -0.09),
                    key=lambda x: x["proxy"]["cagr"], reverse=True)
    short = []
    for x in by_cal[:8] + by_ret[:8]:
        if x["mask"] not in {y["mask"] for y in short}:
            short.append(x)
    all_on = next(x for x in rows if x["label"] == "+".join(KEYS))
    if all_on["mask"] not in {y["mask"] for y in short}:
        short.append(all_on)

    print("\nTop proxy Calmar:")
    for x in by_cal[:10]:
        p = x["proxy"]
        print(f"  {x['label']:<42} {p['cagr']*100:6.2f}% {p['mdd']*100:7.2f}% {p['calmar']:6.3f}")
    print("\nTop proxy CAGR with MDD <= -9%:")
    for x in by_ret[:10]:
        p = x["proxy"]
        print(f"  {x['label']:<42} {p['cagr']*100:6.2f}% {p['mdd']*100:7.2f}% {p['calmar']:6.3f}")

    print("\nShortlist: proxy + OOS + real")
    print(f"{'label':<42} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>6} {'OOS Cal':>8} {'real Cal':>8}")
    for x in short:
        c = x["cfg"]
        o = run_one(c, R_PROXY, OOS_START)
        r = run_one(c, R_REAL, REAL_START)
        # Real-panel validation is added only for the final shortlist.
        x["oos"] = o
        x["real"] = r
        p = x["proxy"]
        print(f"{x['label']:<42} {p['cagr']*100:10.2f}% {p['mdd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{o['calmar']:8.3f} {r['calmar']:8.3f}", flush=True)

    json.dump({"keys": KEYS, "rows": rows, "shortlist": short},
              open(os.path.join(OUT, "exp_v42_enum.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v42_enum.json")


if __name__ == "__main__":
    main()
