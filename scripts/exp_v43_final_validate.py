# -*- coding: utf-8 -*-
"""v43 改进候选完整验证（场内价）：分段、费用、压力、自助法。"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import BASE, run
from engine import evaluate
from run_backtest import exec_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


def C(cn=None, us=None, **kw):
    c = copy.deepcopy(BASE)
    if cn is not None and us is not None:
        c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c.update(kw)
    return c


CANDIDATES = {
    "base": C(),
    "A_return": C(5.25, 11.5, vol_gate_win=15),
    "A2_return": C(5.0, 11.5, vol_gate_win=15),
    "B_balanced": C(4.5, 11.0, vol_gate_win=15, us_rotation=True),
    "B_no_usrot": C(4.5, 11.0, vol_gate_win=15),
    "B_no_vol15": C(4.5, 11.0, us_rotation=True),
    "C_defensive": C(5.25, 11.5, vol_gate_win=15, valuation_gate=True),
}

STRESS = (
    ("2015股灾", "proxy", "2015-06-15", "2016-02-29"),
    ("2018熊市", "proxy", "2018-01-02", "2019-01-03"),
    ("2021-2022熊市", "proxy", "2021-02-19", "2022-10-31"),
    ("2019-2021牛市", "proxy", "2019-01-04", "2021-02-18"),
    ("2023-2024震荡", "proxy", "2023-01-03", "2024-08-30"),
    ("2025-2026真实", "real", "2025-04-23", "2026-11-10"),
)


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}, "periods": {}, "stress": {}, "costs": {}, "bootstrap": {}}

    print("== 双窗口 / OOS / 训练段 ==", flush=True)
    for name, c in CANDIDATES.items():
        report["metrics"][name] = {}
        for layer, sig, pnl, start, end in (
            ("proxy", V.R, P_PROXY, V.PROXY_START, None),
            ("real", V.Rr, P_REAL, V.REAL_START, None),
            ("oos", V.R, P_PROXY, V.OOS_START, None),
            ("train", V.R, P_PROXY, V.PROXY_START, "2021-12-31"),
        ):
            res, e = run(c, sig, pnl, start, end)
            report["metrics"][name][layer] = V.pack(e)
            if layer == "proxy":
                report["periods"][name] = evaluate(res, periods=V.PERIODS)["periods"]
                report["bootstrap"][name] = V.bootstrap(res["rets"])
            print(f"  {name:<12} {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    print("== 费用敏感性 ==", flush=True)
    for name, c in CANDIDATES.items():
        report["costs"][name] = {}
        for bp in (5, 10, 20, 30):
            _, e = run(c, V.R, P_PROXY, V.PROXY_START, None, fee=bp / 10000.0)
            report["costs"][name][f"{bp}bp"] = V.pack(e)
            print(f"  {name:<12} {bp:02d}bp {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    print("== 压力窗口 ==", flush=True)
    for sn, layer, start, end in STRESS:
        sig, pnl = (V.R, P_PROXY) if layer == "proxy" else (V.Rr, P_REAL)
        report["stress"][sn] = {}
        for name, c in CANDIDATES.items():
            _, e = run(c, sig, pnl, start, end)
            report["stress"][sn][name] = V.pack(e)
            print(f"  {sn:<14} {name:<12} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    json.dump(report, open(os.path.join(OUT, "exp_v43_final_validate.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_final_validate.json", flush=True)


if __name__ == "__main__":
    main()
