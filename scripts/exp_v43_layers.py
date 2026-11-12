# -*- coding: utf-8 -*-
"""v43 风险层/轮动层消融与加回（场内价，纸面研究）。"""
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

NO_FAST = {"CN": [9.99, 0.05, 0.18, 0.14], "US": [9.99, 0.12, 0.24, 0.15]}
NO_DEEP = {"CN": [0.07, 0.05, 9.99, 0.14], "US": [0.13, 0.12, 9.99, 0.15]}


def C(**kw):
    c = copy.deepcopy(BASE)
    c.update(kw)
    return c


def candidates():
    rows = [
        ("base_v43", C()),
        ("no_fast", C(market_dd=NO_FAST)),
        ("no_deep", C(market_dd=NO_DEEP)),
        ("no_gate", C(struct_gate=False)),
        ("no_premium", C(premium_gate=False, premium_tilt=False)),
        ("no_premium_tilt", C(premium_tilt=False)),
        ("no_cool", C(cool_off_weeks=0)),
        ("no_asset_sb", C(asset_sb=False)),
        ("no_am", C(am_gate=False)),
        ("no_vol", C(vol_gate=False)),
        ("no_fast_deep", C(market_dd={"CN": [9.99, 0.05, 9.99, 0.14],
                                      "US": [9.99, 0.12, 9.99, 0.15]})),
        ("vol_mild", C(vol_gate=True, vol_gate_cuts=[0.90, 0.78, 0.60])),
        ("vol_mid", C(vol_gate=True, vol_gate_cuts=[0.88, 0.72, 0.50])),
        ("vol_severe", C(vol_gate=True, vol_gate_cuts=[0.80, 0.55, 0.30])),
        ("vol_win15", C(vol_gate=True, vol_gate_win=15)),
        ("vol_win30", C(vol_gate=True, vol_gate_win=30)),
        ("rsrs50", C(rsrs_gate=True, rsrs_cut=0.50)),
        ("rsrs60", C(rsrs_gate=True, rsrs_cut=0.60)),
        ("rsrs70", C(rsrs_gate=True, rsrs_cut=0.70)),
        ("rsrs80", C(rsrs_gate=True, rsrs_cut=0.80)),
        ("rsrs70_thr0", C(rsrs_gate=True, rsrs_cut=0.70, rsrs_thr=0.0)),
        ("rsrs70_thr10", C(rsrs_gate=True, rsrs_cut=0.70, rsrs_thr=0.10)),
        ("rsrs70_all", C(rsrs_gate=True, rsrs_cut=0.70, rsrs_gate_all=True)),
        ("rsrs70_no_vol", C(rsrs_gate=True, rsrs_cut=0.70, vol_gate=False)),
        ("rsrs70_vol_mild", C(rsrs_gate=True, rsrs_cut=0.70,
                              vol_gate=True, vol_gate_cuts=[0.90, 0.78, 0.60])),
        ("rsrs70_no_sb", C(rsrs_gate=True, rsrs_cut=0.70, asset_sb=False)),
        ("rsrs70_no_am", C(rsrs_gate=True, rsrs_cut=0.70, am_gate=False)),
        ("rsrs70_no_tilt", C(rsrs_gate=True, rsrs_cut=0.70, premium_tilt=False)),
        ("corr", C(corr_risk=True)),
        ("corr_cond", C(corr_risk=True, corr_conditional=True)),
        ("prem_rotate", C(premium_rotate=True)),
        ("us_rotation", C(us_rotation=True)),
        ("regime", C(regime_gate=True)),
        ("recovery", C(recovery_ramp=True)),
        ("reentry10", C(reentry_step=0.10)),
        ("reentry20", C(reentry_step=0.20)),
        ("speed_brake", C(speed_brake=True)),
        ("speed_brake_mkt", C(speed_brake_mkt=True)),
        ("gate_band1", C(gate_band=0.01)),
        ("gate_band2", C(gate_band=0.02)),
        ("growth_rotation", C(growth_rotation=True)),
        ("growth_iv", C(growth_iv=True)),
        ("reversal", C(reversal_filter=True)),
        ("valuation", C(valuation_gate=True)),
        ("adx", C(adx_gate=True)),
        ("breadth", C(breadth_gate=True)),
        ("hh_stop", C(hh_stop=True)),
        ("defense_momentum", C(defense_momentum=True)),
    ]
    for name, c in rows:
        c["version"] = "v43_" + name
    return rows


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}}
    print(f"{'variant':<22} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>6} "
          f"{'OOS CAGR':>10} {'MDD':>8} {'Cal':>6} {'real Cal':>9}", flush=True)
    for name, c in candidates():
        row = {}
        try:
            for layer, sig, pnl, start in (
                ("proxy", V.R, P_PROXY, V.PROXY_START),
                ("oos", V.R, P_PROXY, V.OOS_START),
                ("real", V.Rr, P_REAL, V.REAL_START),
            ):
                _, e = run(c, sig, pnl, start)
                row[layer] = V.pack(e)
        except Exception as ex:
            print(f"  [err] {name}: {ex}", flush=True)
            continue
        report["metrics"][name] = row
        p, o, r = row["proxy"], row["oos"], row["real"]
        print(f"{name:<22} {p['cagr']*100:10.2f}% {p['max_dd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{o['cagr']*100:9.2f}% {o['max_dd']*100:7.2f}% {o['calmar']:6.3f} {r['calmar']:9.3f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_layers.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_layers.json", flush=True)


if __name__ == "__main__":
    main()
