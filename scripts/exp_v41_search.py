# -*- coding: utf-8 -*-
"""v41 最终组合的近邻搜索（2026-11-11，数据至 2026-11-10）。

只围绕已通过的 v41 候选做单参数/少量组合扰动，重点检查：
- 去掉自由现金流后，资金在红利、纳指、标普、创业板之间的重新分配；
- 风险覆盖层是否仍有贡献；
- 是否存在收益更高且 OOS/真实窗口不明显恶化的组合。
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


def cfg(**kw):
    c = copy.deepcopy(BASE)
    c.update(kw)
    c["version"] = "v41-search"
    return c


def state7(g, d=10):
    sm = copy.deepcopy(BASE["state_map"])
    sm["7"] = [g, d]
    sm["8"] = [min(99, g + 20), 6]
    sm["9"] = [max(0, g + 16), 3]
    return sm


def split(chi):
    return {"159952": chi, "159941": 1.0 - chi - 0.07, "513500": 0.07}


def variants():
    rows = []
    for us in (10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0):
        rows.append((f"floor_us_{us:g}", cfg(floor_pct={"cn": 2.0, "us": us})))
    for chi in (0.32, 0.34, 0.35, 0.36, 0.37, 0.38):
        rows.append((f"bull_chi_{chi:.2f}", cfg(growth_split_bull=split(chi))))
    for g in (73, 75, 77, 79, 80):
        rows.append((f"state7_{g}", cfg(state_map=state7(g))))
    for cuts in ([0.90, 0.75, 0.55], [0.88, 0.72, 0.52], [0.86, 0.70, 0.48], [0.84, 0.65, 0.40], [0.92, 0.80, 0.60]):
        rows.append(("vol_" + "_".join(str(x) for x in cuts), cfg(vol_gate_cuts=cuts)))
    for v in (0.20, 0.30, 0.40):
        rows.append((f"asset_cut_{v:.2f}", cfg(asset_sb_cut=v)))
    for v in (0.70, 0.80, 0.90):
        rows.append((f"am_cut_{v:.2f}", cfg(am_cut=v)))
    for v in (0.02, 0.035, 0.05):
        rows.append((f"min_delta_{v:.3f}", cfg(min_delta=v)))
    for v in (2, 3, 4):
        rows.append((f"cool_{v}", cfg(cool_off_weeks=v)))

    # 少量组合：优先看收益，同时保留当前风险覆盖
    rows += [
        ("combo_us12_chi36", cfg(floor_pct={"cn": 2.0, "us": 12.0}, growth_split_bull=split(0.36))),
        ("combo_us12_chi38", cfg(floor_pct={"cn": 2.0, "us": 12.0}, growth_split_bull=split(0.38))),
        ("combo_us12_state77", cfg(floor_pct={"cn": 2.0, "us": 12.0}, state_map=state7(77))),
        ("combo_us12_chi36_state77", cfg(floor_pct={"cn": 2.0, "us": 12.0}, growth_split_bull=split(0.36), state_map=state7(77))),
        ("combo_us12_vol_mild", cfg(floor_pct={"cn": 2.0, "us": 12.0}, vol_gate_cuts=[0.90, 0.75, 0.55])),
        ("combo_us12_asset40", cfg(floor_pct={"cn": 2.0, "us": 12.0}, asset_sb_cut=0.40)),
        ("combo_us12_am70", cfg(floor_pct={"cn": 2.0, "us": 12.0}, am_cut=0.70)),
        ("combo_us12_cool2", cfg(floor_pct={"cn": 2.0, "us": 12.0}, cool_off_weeks=2)),
    ]
    return rows


def main():
    out = {}
    print(f"{'variant':<28} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>6} {'real CAGR':>11} {'MDD':>8} {'Cal':>6} {'OOS Cal':>8}")
    for name, c in variants():
        m = metrics(c)
        out[name] = {"cfg": c, "metrics": m}
        p, r, o = m["proxy"], m["real"], m["oos"]
        print(f"{name:<28} {p['cagr']*100:10.2f}% {p['mdd']*100:7.2f}% {p['calmar']:6.2f} "
              f"{r['cagr']*100:10.2f}% {r['mdd']*100:7.2f}% {r['calmar']:6.2f} {o['calmar']:8.2f}")
    json.dump(out, open(os.path.join(OUT, "exp_v41_search.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v41_search.json")


if __name__ == "__main__":
    main()
