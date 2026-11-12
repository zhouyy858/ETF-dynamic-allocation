# -*- coding: utf-8 -*-
"""v43 去掉创业板 159952 的组合搜索（场内价，纸面研究）。"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from engine import run_backtest, evaluate
from run_backtest import exec_panel
from strategy import DynamicStrategy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(REF, "final_cfg_v43.json")))


def make(name, split=(0.57, 0.07), floor_us=12.0, state_scale=1.0,
         vol_gate=None, speed_brake=None):
    c = copy.deepcopy(BASE)
    c["exclude"] = ["159232", "159952"]
    c["growth_split_bull"] = {"159952": 0.0, "159941": split[0], "513500": split[1]}
    c["growth_split_bear"] = {"159952": 0.0, "159941": split[0], "513500": split[1]}
    c["floor_pct"]["us"] = float(floor_us)
    if state_scale != 1.0:
        c["state_map"] = {k: [min(100.0, round(g * state_scale, 2)), d]
                          for k, (g, d) in c["state_map"].items()}
    if vol_gate is not None:
        c["vol_gate"] = bool(vol_gate)
    if speed_brake is not None:
        c["speed_brake"] = bool(speed_brake)
    c["version"] = "v43_" + name
    return c


CFGS = {
    "v43_base": BASE,
    "no_cyb": make("no_cyb"),
    "no_cyb_n100": make("no_cyb_n100", split=(1.0, 0.0)),
    "no_cyb_s100": make("no_cyb_s100", split=(0.0, 1.0)),
    "no_cyb_70_30": make("no_cyb_70_30", split=(0.7, 0.3)),
    "no_cyb_50_50": make("no_cyb_50_50", split=(0.5, 0.5)),
    "no_cyb_floor14": make("no_cyb_floor14", floor_us=14.0),
    "no_cyb_floor16": make("no_cyb_floor16", floor_us=16.0),
    "no_cyb_n100_floor14": make("no_cyb_n100_floor14", split=(1.0, 0.0), floor_us=14.0),
    "no_cyb_70_30_floor14": make("no_cyb_70_30_floor14", split=(0.7, 0.3), floor_us=14.0),
    "no_cyb_n100_vol_off": make("no_cyb_n100_vol_off", split=(1.0, 0.0), vol_gate=False),
    "no_cyb_n100_speed_on": make("no_cyb_n100_speed_on", split=(1.0, 0.0), speed_brake=True),
    "no_cyb_n100_state90": make("no_cyb_n100_state90", split=(1.0, 0.0), state_scale=0.9),
}


def run(cfg, sig_R, pnl_R, start, periods=None):
    ds = DynamicStrategy(sig_R, cfg=cfg)
    res = run_backtest(
        pnl_R,
        target_weights_fn=ds.target_fn(),
        daily_override_fn=ds.daily_fn(),
        start=start,
        name="DYN",
        fee=0.0005,
        min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=V.BOND,
        cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True,
    )
    return res, evaluate(res, periods=periods)


def main():
    report = {"metrics": {}}
    print("== proxy / real / OOS (price) ==", flush=True)
    for name, cfg in CFGS.items():
        proxy_pnl = exec_panel(V.R, "proxy", cfg, "price")
        real_pnl = exec_panel(V.Rr, "real", cfg, "price")
        report["metrics"][name] = {}
        for layer, sig, pnl, start in (
            ("proxy", V.R, proxy_pnl, V.PROXY_START),
            ("real", V.Rr, real_pnl, V.REAL_START),
            ("oos", V.R, proxy_pnl, V.OOS_START),
        ):
            _, e = run(cfg, sig, pnl, start)
            report["metrics"][name][layer] = V.pack(e)
            print(f"  {name:<24} {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.2f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v43_no_chinext.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v43_no_chinext.json")


if __name__ == "__main__":
    main()
