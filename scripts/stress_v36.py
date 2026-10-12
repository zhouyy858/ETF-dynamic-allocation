# -*- coding: utf-8 -*-
"""v30 vs v36候选(v27+熊纳指40%) 七情景压力对照 (2026-09-15, 数据至09-14)
严格口径; 情景窗口与 stress_test.py 完全一致。输出 out/stress_v36.json
用法: python3 stress_v36.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stress_test as st
from strategy import DynamicStrategy

OUT = os.path.join(os.getcwd(), "out"); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
V30 = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v30.json")))
V36 = json.load(open(os.path.join(OUT, "cfg_v36_maxcal.json")))


def run_one(cfg, ps, pe, R, a_mkt=None):
    ds = DynamicStrategy(R, cfg=cfg, a_mkt_override=a_mkt)
    bond = st.rets_from(st.read_table("511010_nav.csv"), "cum_nav") if cfg.get("cash_bond_pct") else None
    res = st.run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                          start=ps, end=pe, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                          repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                          cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                          rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                          strict=True)
    return st.evaluate(res)


def main():
    R, _ = st.build_panel("proxy")
    Rr, _ = st.build_panel("real")
    synth, s_idx = st.synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
    SCEN = [
        ("牛市_2019-2021", "2019-01-04", "2021-02-18", R, None),
        ("牛市_真实窗口2025-2026", "2025-04-23", "2026-07-31", Rr, None),
        ("震荡市_2023-2024", "2023-01-03", "2024-08-30", R, None),
        ("熊市_2015股灾", "2015-06-15", "2016-02-29", R, None),
        ("熊市_2018", "2018-01-02", "2019-01-03", R, None),
        ("熊市_2021-2022", "2021-02-19", "2022-10-31", R, None),
        ("共振熊市_合成US-30+A股-20", "2024-09-02", "2026-07-31", synth, s_idx),
    ]
    out = {}
    print(f"{'情景':<28} {'v30 CAGR/MDD':>22} {'v36 CAGR/MDD':>22}  Cal变化")
    for name, ps, pe, Rs, am in SCEN:
        e30 = run_one(V30, ps, pe, Rs, am)
        e36 = run_one(V36, ps, pe, Rs, am)
        out[name] = {"v30": {k: e30[k] for k in ("cagr", "max_dd", "sharpe", "calmar", "total_ret")},
                     "v36": {k: e36[k] for k in ("cagr", "max_dd", "sharpe", "calmar", "total_ret")}}
        print(f"{name:<28} {e30['cagr']*100:6.2f}%/{e30['max_dd']*100:6.2f}%      "
              f"{e36['cagr']*100:6.2f}%/{e36['max_dd']*100:6.2f}%      "
              f"{e36['calmar']-e30['calmar']:+.2f}")
    json.dump(out, open(os.path.join(OUT, "stress_v36.json"), "w"), ensure_ascii=False, indent=1)
    print("\n[ok] out/stress_v36.json")


if __name__ == "__main__":
    main()
