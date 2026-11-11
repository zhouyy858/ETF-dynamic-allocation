# -*- coding: utf-8 -*-
"""v41 高收益骨架上的风险覆盖层微调（2026-11-11）

骨架 = 剔除自由现金流 + US 底仓 12% + state7-9 75/95/91 + bull 34/59/7 + 现金下限 0。
只调整风险覆盖强度，寻找收益不明显下降、Calmar/MDD 改善的组合。

用法: python3 scripts/exp_v41_overlay.py
输出: out/exp_v41_overlay.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_v41_combo import C, metrics, BEST_RET

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")


def cfg(**kw):
    d = dict(BEST_RET)
    d.update(kw)
    return C(**d)


CANDS = [
    ("best_ret", cfg()),
    ("vol_mild", cfg(vol_gate=True, vol_gate_cuts=[0.85, 0.65, 0.45])),
    ("vol_mid", cfg(vol_gate=True, vol_gate_cuts=[0.80, 0.55, 0.30])),
    ("vol_hi_band", cfg(vol_gate=True, vol_gate_bands=[0.35, 0.45, 0.55], vol_gate_cuts=[0.80, 0.55, 0.30])),
    ("rsrs70", cfg(rsrs_gate=True, rsrs_cut=0.70)),
    ("rsrs_thr0", cfg(rsrs_gate=True, rsrs_thr=0.0, rsrs_cut=0.50)),
    ("asset_sb06", cfg(asset_sb=True, asset_sb_thr=0.06, asset_sb_cut=0.60)),
    ("asset_sb10", cfg(asset_sb=True, asset_sb_thr=0.10, asset_sb_cut=0.50)),
    ("asset_sb_cut3", cfg(asset_sb=True, asset_sb_thr=0.08, asset_sb_cut=0.30)),
    ("am_rsrs70", cfg(am_gate=True, rsrs_gate=True, rsrs_cut=0.70)),
    ("am_asset", cfg(am_gate=True, asset_sb=True)),
    ("vol_mild_asset", cfg(vol_gate=True, vol_gate_cuts=[0.85, 0.65, 0.45], asset_sb=True)),
    ("vol_mild_am", cfg(vol_gate=True, vol_gate_cuts=[0.85, 0.65, 0.45], am_gate=True)),
    ("vol_mild_rsrs70", cfg(vol_gate=True, vol_gate_cuts=[0.85, 0.65, 0.45], rsrs_gate=True, rsrs_cut=0.70)),
    ("all3", cfg(vol_gate=True, vol_gate_cuts=[0.85, 0.65, 0.45], am_gate=True, asset_sb=True)),
    ("vol_mid_asset", cfg(vol_gate=True, vol_gate_cuts=[0.80, 0.55, 0.30], asset_sb=True)),
    ("vol_mid_am", cfg(vol_gate=True, vol_gate_cuts=[0.80, 0.55, 0.30], am_gate=True)),
    ("vol_mid_all3", cfg(vol_gate=True, vol_gate_cuts=[0.80, 0.55, 0.30], am_gate=True, asset_sb=True)),
    ("sb_cut2", cfg(asset_sb=True, asset_sb_cut=0.20)),
    ("sb_cut4", cfg(asset_sb=True, asset_sb_cut=0.40)),
    ("sb_cut3_am", cfg(asset_sb=True, asset_sb_cut=0.30, am_gate=True)),
    ("sb_cut3_vol_mild", cfg(asset_sb=True, asset_sb_cut=0.30,
                             vol_gate=True, vol_gate_cuts=[0.85, 0.65, 0.45])),
    ("sb_cut3_rsrs70", cfg(asset_sb=True, asset_sb_cut=0.30, rsrs_gate=True, rsrs_cut=0.70)),
    ("sb_cut3_am_vol_mild", cfg(asset_sb=True, asset_sb_cut=0.30, am_gate=True,
                                vol_gate=True, vol_gate_cuts=[0.85, 0.65, 0.45])),
    ("sb_cut3_no_corr", cfg(asset_sb=True, asset_sb_cut=0.30, corr_risk=False)),
    ("sb_cut3_us13", C(**{**BEST_RET, "floor_pct": {"cn": 2.0, "us": 13.0}}, asset_sb=True, asset_sb_cut=0.30)),
    ("sb_cut3_am_no_corr", cfg(asset_sb=True, asset_sb_cut=0.30, am_gate=True, corr_risk=False)),
]


def main():
    rows = {}
    print(f"{'候选':<20} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>7} {'real CAGR':>11} {'MDD':>8} {'Cal':>7} {'OOS Cal':>8}")
    for name, c in CANDS:
        m = metrics(c)
        rows[name] = {"cfg": c, "metrics": m}
        p, r, o = m["proxy"], m["real"], m["oos"]
        print(f"{name:<20} {p['cagr']*100:10.2f}% {p['mdd']*100:7.2f}% {p['calmar']:7.2f} "
              f"{r['cagr']*100:10.2f}% {r['mdd']*100:7.2f}% {r['calmar']:7.2f} {o['calmar']:8.2f}")
    json.dump(rows, open(os.path.join(OUT, "exp_v41_overlay.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print("[ok] out/exp_v41_overlay.json")


if __name__ == "__main__":
    main()
