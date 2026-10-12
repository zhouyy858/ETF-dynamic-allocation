# -*- coding: utf-8 -*-
"""v34 候选预注册评估(2026-09-15, 数据至2026-09-14): max-Calmar 方向整体优化
用户决策: 目标=最大化 Calmar(收益/回撤比), 前提仅保留③(纳指AI超配), ①②④放开给数据。
起点 = v27(纯历史最优定稿, 曾过全套三关); 前提③约束操作化:
  bull 桶纳指>=50%(第一大), bear 桶纳指>=40%(v29 水平, AI 敞口熊市不消失)。
预注册变体 12 个(单机制改动), 严格口径(signal_lag=1/溢价T-2/先计提后成交)。
选择规则(预注册): proxy Calmar 最高且 OOS Cal >= v27 基线 OOS 的 95% 且 real MDD <= -4.5% 线内。
用法: python3 exp_maxcal.py  (输出 out/exp_maxcal.json)
"""
import sys, os, json, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

OUT = os.path.join(os.getcwd(), "out"); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
V27 = os.path.join(HERE, "..", "references", "final_cfg_v27.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"


def split(bull, bear, sp):
    """bull/bear = (cyb, nasdaq, sp500); 校验前提③: bull纳指>=0.5 且 bear纳指>=0.4"""
    for lst, thr, tag in ((bull, 0.5, "bull"), (bear, 0.4, "bear")):
        assert lst[1] >= thr, f"前提③违反: {tag} 纳指 {lst[1]} < {thr}"
    return {"growth_split_bull": {"159952": bull[0], "159941": bull[1], "513500": bull[2]},
            "growth_split_bear": {"159952": bear[0], "159941": bear[1], "513500": bear[2]}}


CANDS = {
    "V0_v27原样": {},  # 参照系(其bear纳指0.30不满足前提③, 仅对照)
    "V1_v27修正熊纳指": {"growth_split_bear": {"159952": 0.4, "159941": 0.4, "513500": 0.2}},
    "V2_V1+关速度刹车": {"growth_split_bear": {"159952": 0.4, "159941": 0.4, "513500": 0.2}, "speed_brake": False},
    "V3_V1+us底仓10": {"growth_split_bear": {"159952": 0.4, "159941": 0.4, "513500": 0.2}, "floor_pct": {"cn": 0.0, "us": 10.0}},
    "V4_V1+us底仓5":  {"growth_split_bear": {"159952": 0.4, "159941": 0.4, "513500": 0.2}, "floor_pct": {"cn": 0.0, "us": 5.0}},
    "V5_V1+牛纳指60": split((0.30, 0.60, 0.10), (0.4, 0.4, 0.2), "V5"),
    "V6_V1+牛创板40": split((0.40, 0.50, 0.10), (0.4, 0.4, 0.2), "V6"),
    "V7_V1+牛创板30": split((0.30, 0.55, 0.15), (0.4, 0.4, 0.2), "V7"),
    "V8_V1+熊纳指45": split((0.35, 0.55, 0.10), (0.35, 0.45, 0.2), "V8"),
    "V9_V1+tilt全额": {"growth_split_bear": {"159952": 0.4, "159941": 0.4, "513500": 0.2}, "premium_tilt_max": 1.0},
    "V10_V1+mindelta02": {"growth_split_bear": {"159952": 0.4, "159941": 0.4, "513500": 0.2}, "min_delta": 0.02},
    "V11_V1+熊纳指40US10牛60": {"growth_split_bear": {"159952": 0.4, "159941": 0.4, "513500": 0.2},
                                 "growth_split_bull": {"159952": 0.3, "159941": 0.6, "513500": 0.1},
                                 "floor_pct": {"cn": 0.0, "us": 10.0}},
}


def run_cfg(cfg, R, start):
    ds = DynamicStrategy(R, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    return evaluate(res)


def metrics(cfg, R, start):
    e = run_cfg(cfg, R, start)
    return {"cagr": e["cagr"], "mdd": e["max_dd"], "cal": e["calmar"],
            "sharpe": e["sharpe"], "to": e["turnover"]}


def main():
    base = json.load(open(V27))
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    rows = {}
    for name, ov in CANDS.items():
        cfg = copy.deepcopy(base); cfg.update(ov)
        p = metrics(cfg, R, PROXY_START); r = metrics(cfg, Rr, REAL_START)
        o = metrics(cfg, R, OOS_START)
        rows[name] = {"override": ov, "proxy": p, "real": r, "oos": o}
        print(f"{name:<24} proxy Cal {p['cal']:5.2f} MDD {p['mdd']*100:6.2f}% | "
              f"real Cal {r['cal']:5.2f} MDD {r['mdd']*100:6.2f}% | OOS Cal {o['cal']:5.2f} MDD {o['mdd']*100:6.2f}%")
    b = rows["V0_v27原样"]
    print(f"\n参照 v27原样: proxy Cal {b['proxy']['cal']:.2f} / OOS Cal {b['oos']['cal']:.2f}")
    ok = {n: r for n, r in rows.items()
          if n != "V0_v27原样"
          and r["oos"]["cal"] >= b["oos"]["cal"] * 0.95
          and r["real"]["mdd"] > -0.045
          and r["proxy"]["mdd"] > -0.10}
    ranked = sorted(ok.items(), key=lambda kv: -kv[1]["proxy"]["cal"])
    print(f"\n[预注册选择规则存活] {len(ranked)} 个, 按 proxy Cal 排序:")
    for n, r in ranked:
        print(f"  {n:<24} proxy Cal {r['proxy']['cal']:.2f}  OOS {r['oos']['cal']:.2f}  real Cal {r['real']['cal']:.2f}")
    json.dump(rows, open(os.path.join(OUT, "exp_maxcal.json"), "w"), ensure_ascii=False, indent=1)
    print("\n[ok] out/exp_maxcal.json")


if __name__ == "__main__":
    main()
