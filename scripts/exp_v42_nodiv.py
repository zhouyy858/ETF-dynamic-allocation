# -*- coding: utf-8 -*-
"""v42 research: completely remove dividend low-vol (515100/563020).

Tests whether the defense bucket that becomes cash after excluding the CN
defense slot can be reallocated to growth / US floor without breaking the
drawdown envelope. Pure paper research; production default stays v41.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_v41_combo import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(ROOT, "references", "final_cfg_v41.json")))


def nodiv(**kw):
    c = copy.deepcopy(BASE)
    c["exclude"] = ["159232", "515100"]
    c["floor_pct"] = {"cn": 0.0, "us": 12.0}
    c["instrument_swaps"] = {}
    c["version"] = "v42-nodiv-search"
    c.update(kw)
    return c


def state_realloc(low, high=None):
    high = low if high is None else high
    sm = copy.deepcopy(BASE["state_map"])
    for k, (g, d) in list(sm.items()):
        lam = low if int(k) <= 6 else high
        total = float(g) + float(d)
        ng = min(100.0, float(g) + lam * float(d))
        sm[k] = [ng, max(0.0, total - ng)]
    return sm


def split(chi):
    return {"159952": chi, "159941": 1.0 - chi - 0.07, "513500": 0.07}


def variants():
    rows = [("nodiv_base", nodiv())]
    for us in (10.0, 12.0, 14.0, 16.0):
        for lam in (0.25, 0.50, 0.75, 1.00):
            rows.append((f"us{us:g}_lam{lam:.2f}", nodiv(
                floor_pct={"cn": 0.0, "us": us}, state_map=state_realloc(lam))))
    for us in (12.0, 14.0, 16.0):
        for chi in (0.32, 0.36, 0.40):
            rows.append((f"us{us}_lam1_chi{chi:.2f}", nodiv(
                floor_pct={"cn": 0.0, "us": us}, state_map=state_realloc(1.0),
                growth_split_bull=split(chi))))
    for low, high in ((1.0, 0.0), (0.75, 0.25), (0.50, 1.00), (0.75, 0.75)):
        rows.append((f"us12_low{low:.2f}_high{high:.2f}", nodiv(
            state_map=state_realloc(low, high))))
    for md in (0.02, 0.035, 0.05):
        rows.append((f"us14_lam1_md{md:.3f}", nodiv(
            floor_pct={"cn": 0.0, "us": 14.0}, state_map=state_realloc(1.0),
            min_delta=md)))
    return rows


def main():
    out = {}
    print(f"{'variant':<25} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>6} {'real CAGR':>11} {'MDD':>8} {'Cal':>6} {'OOS Cal':>8}")
    for name, cfg in variants():
        m = metrics(cfg)
        out[name] = {"cfg": cfg, "metrics": m}
        p, r, o = m["proxy"], m["real"], m["oos"]
        print(f"{name:<25} {p['cagr']*100:10.2f}% {p['mdd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{r['cagr']*100:10.2f}% {r['mdd']*100:7.2f}% {r['calmar']:6.3f} {o['calmar']:8.3f}", flush=True)
    json.dump(out, open(os.path.join(OUT, "exp_v42_nodiv.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v42_nodiv.json")


if __name__ == "__main__":
    main()
