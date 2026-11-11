# -*- coding: utf-8 -*-
"""Detailed comparison: v41 vs no-dividend variants (paper research only)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v42_risk import HIGH
from exp_v42_nodiv import nodiv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
CFGS = {
    "v41": json.load(open(os.path.join(REF, "final_cfg_v41.json"))),
    "nodiv_base": nodiv(),
    "nodiv_current8": HIGH,
    "nodiv_balanced": json.load(open(os.path.join(REF, "candidate_cfg_v42_nodiv_balanced.json"))),
    "nodiv_aggressive": json.load(open(os.path.join(REF, "candidate_cfg_v42_nodiv_aggressive.json"))),
}


def main():
    report = {"metrics": {}, "costs": {}, "stress": {}, "bootstrap": {}, "wfo": {}}
    print("== proxy / real / OOS ==", flush=True)
    for name, cfg in CFGS.items():
        report["metrics"][name] = {}
        for layer, Rs, start, periods in (
            ("proxy", V.R, V.PROXY_START, V.PERIODS),
            ("real", V.Rr, V.REAL_START, None),
            ("oos", V.R, V.OOS_START, None),
        ):
            res, e = V.run(cfg, Rs, start, periods=periods)
            report["metrics"][name][layer] = {"base": V.pack(e), "periods": e.get("periods", {})}
            print(f"  {name:<17} {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)
            if layer == "proxy":
                report["bootstrap"][name] = V.bootstrap(res["rets"])

    print("== fee sensitivity ==", flush=True)
    for name, cfg in CFGS.items():
        report["costs"][name] = {}
        for bp in (5, 10, 15, 20):
            _, e = V.run(cfg, V.R, V.PROXY_START, fee=bp / 10000.0)
            report["costs"][name][f"{bp}bp"] = V.pack(e)

    print("== stress ==", flush=True)
    scenarios = [
        ("2019-2021牛市", V.R, None, "2019-01-04", "2021-02-18"),
        ("2025-2026牛市", V.Rr, None, "2025-04-23", "2026-07-31"),
        ("2023-2024震荡", V.R, None, "2023-01-03", "2024-08-30"),
        ("2015股灾", V.R, None, "2015-06-15", "2016-02-29"),
        ("2018熊市", V.R, None, "2018-01-02", "2019-01-03"),
        ("2021-2022熊市", V.R, None, "2021-02-19", "2022-10-31"),
        ("共振熊市_合成", V.SYNTH, V.S_IDX, "2024-09-02", "2026-07-31"),
    ]
    for sn, Rs, am, ps, pe in scenarios:
        report["stress"][sn] = {}
        for name, cfg in CFGS.items():
            _, e = V.run(cfg, Rs, ps, pe, a_mkt=am)
            report["stress"][sn][name] = V.pack(e)
            print(f"  {sn:<15} {name:<17} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    print("== WFO ==", flush=True)
    for wn, ps, pe in (("train_2014_2021", V.PROXY_START, "2021-12-31"),
                       ("test_2022_2026", V.OOS_START, "2026-11-10")):
        report["wfo"][wn] = {}
        for name, cfg in CFGS.items():
            _, e = V.run(cfg, V.R, ps, pe)
            report["wfo"][wn][name] = V.pack(e)
            print(f"  {wn:<17} {name:<17} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    json.dump(report, open(os.path.join(OUT, "v42_compare.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/v42_compare.json")


if __name__ == "__main__":
    main()
