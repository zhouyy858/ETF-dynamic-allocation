# -*- coding: utf-8 -*-
"""v41 去标普 513500 的纸面对比（不改默认配置）。"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(REF, "final_cfg_v41.json")))


def variant(name):
    c = copy.deepcopy(BASE)
    if name == "no_sp500_cash":
        c["exclude"] = ["159232", "513500"]
    elif name == "no_sp500_to_nasdaq":
        c["instrument_swaps"] = dict(c.get("instrument_swaps", {}), **{"513500": "159941"})
    elif name == "no_sp500_no_floor":
        c["exclude"] = ["159232", "513500"]
        c["floor_pct"]["us"] = 0.0
    elif name == "no_sp500_split_36_64":
        c["exclude"] = ["159232", "513500"]
        c["growth_split_bull"] = {"159952": 0.36, "159941": 0.64, "513500": 0.0}
        c["growth_split_bear"] = {"159952": 0.45, "159941": 0.55, "513500": 0.0}
    elif name == "no_sp500_nasdaq_floor":
        c["exclude"] = ["159232", "513500"]
        c["floor_pct"]["us"] = 24.0
        c["growth_split_bull"] = {"159952": 0.36, "159941": 0.64, "513500": 0.0}
        c["growth_split_bear"] = {"159952": 0.45, "159941": 0.55, "513500": 0.0}
    c["version"] = "v41_" + name
    return c


CFGS = {
    "v41_base": BASE,
    "no_sp500_cash": variant("no_sp500_cash"),
    "no_sp500_no_floor": variant("no_sp500_no_floor"),
    "no_sp500_split_36_64": variant("no_sp500_split_36_64"),
    "no_sp500_nasdaq_floor": variant("no_sp500_nasdaq_floor"),
}


def main():
    report = {"metrics": {}, "stress": {}, "bootstrap": {}}
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
            print(f"  {name:<22} {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)
            if layer == "proxy":
                report["bootstrap"][name] = V.bootstrap(res["rets"])

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
            print(f"  {sn:<15} {name:<22} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)

    json.dump(report, open(os.path.join(OUT, "exp_v41_no_sp500.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v41_no_sp500.json")


if __name__ == "__main__":
    main()
