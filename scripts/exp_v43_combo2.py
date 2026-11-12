# -*- coding: utf-8 -*-
"""v43 第二轮：底仓 × vol_win15 × 轮动/恢复/RSRS/估值 组合（场内价）。"""
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


def C(cn=None, us=None, **kw):
    c = copy.deepcopy(BASE)
    if cn is not None and us is not None:
        c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c.update(kw)
    return c


def candidates():
    rows = [
        ("base", C()),
        ("vol15", C(vol_gate_win=15)),
        ("f45_11", C(4.5, 11.0)),
        ("f45_11_vol15", C(4.5, 11.0, vol_gate_win=15)),
        ("f5_115_vol15", C(5.0, 11.5, vol_gate_win=15)),
        ("f525_115_vol15", C(5.25, 11.5, vol_gate_win=15)),
        ("f45_11_usrot", C(4.5, 11.0, us_rotation=True)),
        ("f45_11_reentry20", C(4.5, 11.0, reentry_step=0.20)),
        ("f45_11_vol15_usrot", C(4.5, 11.0, vol_gate_win=15, us_rotation=True)),
        ("f45_11_vol15_reentry20", C(4.5, 11.0, vol_gate_win=15, reentry_step=0.20)),
        ("f5_115_vol15_usrot", C(5.0, 11.5, vol_gate_win=15, us_rotation=True)),
        ("f5_115_vol15_reentry20", C(5.0, 11.5, vol_gate_win=15, reentry_step=0.20)),
        ("vol15_usrot", C(vol_gate_win=15, us_rotation=True)),
        ("vol15_reentry20", C(vol_gate_win=15, reentry_step=0.20)),
        ("vol15_rsrs70_novol", C(vol_gate_win=15, rsrs_gate=True, rsrs_cut=0.70, vol_gate=False)),
        ("f45_11_vol15_rsrs70_novol", C(4.5, 11.0, vol_gate_win=15, rsrs_gate=True,
                                         rsrs_cut=0.70, vol_gate=False)),
        ("valuation", C(valuation_gate=True)),
        ("valuation_f45_11", C(4.5, 11.0, valuation_gate=True)),
        ("valuation_f525_115_vol15", C(5.25, 11.5, valuation_gate=True, vol_gate_win=15)),
        ("valuation_usrot", C(valuation_gate=True, us_rotation=True)),
    ]
    for name, c in rows:
        c["version"] = "v43_combo2_" + name
    return rows


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}}
    print(f"{'variant':<28} {'proxy CAGR':>11} {'MDD':>8} {'Cal':>6} "
          f"{'OOS CAGR':>10} {'MDD':>8} {'Cal':>6} {'real Cal':>9}", flush=True)
    for name, c in candidates():
        row = {}
        for layer, sig, pnl, start in (
            ("proxy", V.R, P_PROXY, V.PROXY_START),
            ("oos", V.R, P_PROXY, V.OOS_START),
            ("real", V.Rr, P_REAL, V.REAL_START),
        ):
            _, e = run(c, sig, pnl, start)
            row[layer] = V.pack(e)
        report["metrics"][name] = row
        p, o, r = row["proxy"], row["oos"], row["real"]
        print(f"{name:<28} {p['cagr']*100:10.2f}% {p['max_dd']*100:7.2f}% {p['calmar']:6.3f} "
              f"{o['cagr']*100:9.2f}% {o['max_dd']*100:7.2f}% {o['calmar']:6.3f} {r['calmar']:9.3f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_combo2.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_combo2.json", flush=True)


if __name__ == "__main__":
    main()
