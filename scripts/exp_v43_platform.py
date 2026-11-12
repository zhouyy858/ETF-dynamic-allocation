# -*- coding: utf-8 -*-
"""v43 两条改进候选的邻域平台检查（场内价）。"""
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


def C(cn, us, vol_win=15, usrot=False):
    c = copy.deepcopy(BASE)
    c["floor_pct"] = {"cn": float(cn) * 2.0, "us": float(us)}
    c["vol_gate_win"] = int(vol_win)
    c["us_rotation"] = bool(usrot)
    return c


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"A": {}, "B": {}}
    print("== A 收益档：floor x vol_win ==", flush=True)
    for cn in (4.75, 5.0, 5.25):
        for us in (11.0, 11.5, 12.0):
            c = C(cn, us, 15, False)
            row = {}
            for layer, sig, pnl, start in (("proxy", V.R, P_PROXY, V.PROXY_START),
                                           ("oos", V.R, P_PROXY, V.OOS_START),
                                           ("real", V.Rr, P_REAL, V.REAL_START)):
                _, e = run(c, sig, pnl, start)
                row[layer] = V.pack(e)
            report["A"][f"cn{cn:g}_us{us:g}_vol15"] = row
            p, o, r = row["proxy"], row["oos"], row["real"]
            print(f"cn{cn:<5g} us{us:<5g} full {p['cagr']*100:5.2f}/{p['max_dd']*100:6.2f}/{p['calmar']:5.3f} "
                  f"oos {o['cagr']*100:5.2f}/{o['max_dd']*100:6.2f}/{o['calmar']:5.3f} real {r['calmar']:5.3f}", flush=True)
    for vw in (13, 14, 15, 16, 17):
        c = C(5.0, 11.5, vw, False)
        row = {}
        for layer, sig, pnl, start in (("proxy", V.R, P_PROXY, V.PROXY_START),
                                       ("oos", V.R, P_PROXY, V.OOS_START),
                                       ("real", V.Rr, P_REAL, V.REAL_START)):
            _, e = run(c, sig, pnl, start)
            row[layer] = V.pack(e)
        report["A"][f"cn5_us11.5_vol{vw}"] = row
        p, o, r = row["proxy"], row["oos"], row["real"]
        print(f"cn5     us11.5 vol{vw:<2d} full {p['cagr']*100:5.2f}/{p['max_dd']*100:6.2f}/{p['calmar']:5.3f} "
              f"oos {o['cagr']*100:5.2f}/{o['max_dd']*100:6.2f}/{o['calmar']:5.3f} real {r['calmar']:5.3f}", flush=True)

    print("== B 回撤档：floor x vol_win + US轮动 ==", flush=True)
    for cn in (4.0, 4.5, 5.0):
        for us in (10.5, 11.0, 11.5):
            c = C(cn, us, 15, True)
            row = {}
            for layer, sig, pnl, start in (("proxy", V.R, P_PROXY, V.PROXY_START),
                                           ("oos", V.R, P_PROXY, V.OOS_START),
                                           ("real", V.Rr, P_REAL, V.REAL_START)):
                _, e = run(c, sig, pnl, start)
                row[layer] = V.pack(e)
            report["B"][f"cn{cn:g}_us{us:g}_vol15_usrot"] = row
            p, o, r = row["proxy"], row["oos"], row["real"]
            print(f"cn{cn:<5g} us{us:<5g} full {p['cagr']*100:5.2f}/{p['max_dd']*100:6.2f}/{p['calmar']:5.3f} "
                  f"oos {o['cagr']*100:5.2f}/{o['max_dd']*100:6.2f}/{o['calmar']:5.3f} real {r['calmar']:5.3f}", flush=True)
    for vw in (14, 15, 16):
        c = C(4.5, 11.0, vw, True)
        row = {}
        for layer, sig, pnl, start in (("proxy", V.R, P_PROXY, V.PROXY_START),
                                       ("oos", V.R, P_PROXY, V.OOS_START),
                                       ("real", V.Rr, P_REAL, V.REAL_START)):
            _, e = run(c, sig, pnl, start)
            row[layer] = V.pack(e)
        report["B"][f"cn4.5_us11_vol{vw}_usrot"] = row
        p, o, r = row["proxy"], row["oos"], row["real"]
        print(f"cn4.5   us11   vol{vw:<2d} full {p['cagr']*100:5.2f}/{p['max_dd']*100:6.2f}/{p['calmar']:5.3f} "
              f"oos {o['cagr']*100:5.2f}/{o['max_dd']*100:6.2f}/{o['calmar']:5.3f} real {r['calmar']:5.3f}", flush=True)

    json.dump(report, open(os.path.join(OUT, "exp_v43_platform.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_platform.json", flush=True)


if __name__ == "__main__":
    main()
