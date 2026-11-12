# -*- coding: utf-8 -*-
"""v43 底仓重训的边界扩展（场内价，纸面研究）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import BASE, cfg, run
from run_backtest import exec_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

CN_EFF = (3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
US = (10.0, 11.0, 11.5, 12.0, 12.5, 13.0, 14.0)


def main():
    P_PROXY = exec_panel(V.R, "proxy", BASE, "price")
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {"grid": {"cn_eff": list(CN_EFF), "us": list(US)}, "metrics": {}}
    print("== 底仓边界扩展（训练段选参，场内价） ==", flush=True)
    for cn in CN_EFF:
        for us in US:
            c = cfg(cn, us)
            row = {}
            for layer, sig, pnl, start, end in (
                ("train", V.R, P_PROXY, V.PROXY_START, "2021-12-31"),
                ("oos", V.R, P_PROXY, V.OOS_START, None),
                ("proxy", V.R, P_PROXY, V.PROXY_START, None),
                ("real", V.Rr, P_REAL, V.REAL_START, None),
            ):
                _, e = run(c, sig, pnl, start, end)
                row[layer] = V.pack(e)
            report["metrics"][f"cn{cn:g}_us{us:g}"] = row
            f, t, o, r = row["proxy"], row["train"], row["oos"], row["real"]
            print(f"cn{cn:<4g} us{us:<4g} full {f['cagr']*100:5.2f}/{f['max_dd']*100:6.2f}/{f['calmar']:5.3f} "
                  f"train {t['calmar']:5.3f} oos {o['calmar']:5.3f} real {r['calmar']:5.3f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_floors_ext.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_floors_ext.json", flush=True)


if __name__ == "__main__":
    main()
