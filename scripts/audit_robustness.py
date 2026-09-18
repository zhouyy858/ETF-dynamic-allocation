# -*- coding: utf-8 -*-
"""v15 参数扰动鲁棒性审计 (anti-overfit)
对每个关键参数做 ±10% / ±20% 扰动(每次只动一个), 在全历史代理窗口重跑,
统计 CAGR/MDD/Sharpe/Calmar/换手 的分布, 检验参数是否位于稳定平台而非悬崖。
用法: python3 audit_robustness.py [cfg.json] [tag]
"""
import sys, os, json, copy, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
if not os.path.isdir(OUT):
    OUT = os.path.join(HERE, "out")
_p1 = os.path.join(HERE, "..", "references", "final_cfg_v30.json")
_p2 = os.path.join(HERE, "references", "final_cfg_v30.json")
CFG = json.load(open(sys.argv[1] if len(sys.argv) > 1 else (_p1 if os.path.exists(_p1) else _p2)))
TAG = sys.argv[2] if len(sys.argv) > 2 else "v21"
TW = json.loads(sys.argv[3]) if len(sys.argv) > 3 else CFG.get("tranche_weights")
R, W = build_panel("proxy")
START = "2014-06-23"

def run(cfg):
    ds = DynamicStrategy(R, cfg=cfg)
    bond = rets_from(read_table(f"{cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav") if cfg.get("cash_bond_pct") else None
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=START, name="DYN", min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
                       tranche_weights=TW, cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 2), rebal_freq=cfg.get("rebal_freq", "weekly"), strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"],
                calmar=e["calmar"], turnover=e["turnover"], cash=e["avg_cash"]*100)

base = run(CFG)
print(f"基线 {TAG}: CAGR {base['cagr']:.2f}% MDD {base['mdd']:.2f}% Sharpe {base['sharpe']:.2f} "
      f"Calmar {base['calmar']:.2f} TO {base['turnover']:.1f}")

def mut_scalar(cfg, key, f):
    c = copy.deepcopy(cfg); c[key] = round(c[key] * f, 6); return c

def mut_list(cfg, key, f):
    c = copy.deepcopy(cfg); c[key] = [round(v * f, 6) for v in c[key]]; return c

def mut_state_map(cfg, f):
    c = copy.deepcopy(cfg)
    sm = {}
    for k, (g, d) in c["state_map"].items():
        sm[k] = [max(4, min(98, round(g * f))), d]
    c["state_map"] = sm
    return c

def mut_dd_eq_cap(cfg, f):
    c = copy.deepcopy(cfg)
    c["dd_eq_cap"] = [[round(thr * f, 4), int(cap)] for thr, cap in c["dd_eq_cap"]]
    return c

def mut_floor(cfg, key, f):
    c = copy.deepcopy(cfg)
    fp = dict(c.get("floor_pct", {"cn": 15.0, "us": 15.0}))
    fp[key] = max(0.0, round(fp[key] * f, 4))
    c["floor_pct"] = fp
    return c

def mut_market_dd(cfg, mkt, f):
    c = copy.deepcopy(cfg)
    md = dict(c["market_dd"]); md[mkt] = [round(v * f, 4) if i in (0, 2) else v for i, v in enumerate(md[mkt])]
    c["market_dd"] = md
    return c


def mut_growth_split(cfg, key, f):
    """扰动创业板(159952)权重, 其余两个成长资产按原比例归一。
    陷阱: 早期版本对三资产同时乘 f 再归一 = 恒等变换(空转轴), 已修正为只动 159952 份额。"""
    c = copy.deepcopy(cfg)
    gs = dict(c.get(key))
    w = min(0.95, max(0.05, float(gs.get("159952", 0.3)) * f))
    rest = {s: float(v) for s, v in gs.items() if s != "159952"}
    tot = sum(rest.values()) or 1.0
    gs = {s: round((1.0 - w) * v / tot, 6) for s, v in rest.items()}
    gs["159952"] = round(w, 6)
    c[key] = gs
    return c
PERTURBS = [
    ("state_map_growth", lambda c, f: mut_state_map(c, f)),
    ("vol_target", lambda c, f: mut_scalar(c, "vol_target", f)),
    ("min_cash", lambda c, f: mut_scalar(c, "min_cash", f)),
    ("max_eq", lambda c, f: mut_scalar(c, "max_eq", f)),
    ("dd_eq_cap_thr", lambda c, f: mut_dd_eq_cap(c, f)),
    ("hyst_up", lambda c, f: mut_scalar(c, "hyst_up", f)),
    ("hyst_down", lambda c, f: mut_scalar(c, "hyst_down", f)),
    ("gate_win", lambda c, f: mut_scalar(c, "gate_win", f)),
    ("market_dd_CN", lambda c, f: mut_market_dd(c, "CN", f)),
    ("market_dd_US", lambda c, f: mut_market_dd(c, "US", f)),
    ("defense_momentum_win", lambda c, f: mut_scalar(c, "defense_momentum_win", f)),
    ("defense_momentum_t", lambda c, f: mut_scalar(c, "defense_momentum_t", f)),
    ("defense_clamp", lambda c, f: mut_list(c, "defense_clamp", f)),
    ("premium_thr", lambda c, f: mut_list(c, "premium_thr", f)),
    ("premium_cut", lambda c, f: mut_list(c, "premium_cut", f)),
    ("corr_risk_thr", lambda c, f: mut_list(c, "corr_risk_thr", f)),
    ("corr_risk_cut", lambda c, f: mut_list(c, "corr_risk_cut", f)),
    ("speed_brake_thr", lambda c, f: mut_scalar(c, "speed_brake_thr", f)),
    ("speed_brake_cut", lambda c, f: mut_scalar(c, "speed_brake_cut", f)),
    ("speed_brake_recover", lambda c, f: mut_scalar(c, "speed_brake_recover", f)),
    ("min_delta", lambda c, f: mut_scalar(c, "min_delta", f)),
    ("growth_split_bull", lambda c, f: mut_growth_split(c, "growth_split_bull", f)),
    ("growth_split_bear", lambda c, f: mut_growth_split(c, "growth_split_bear", f)),
    ("floor_cn", lambda c, f: mut_floor(c, "cn", f)),
    ("floor_us", lambda c, f: mut_floor(c, "us", f)),
    ("premium_tilt_thr", lambda c, f: mut_scalar(c, "premium_tilt_thr", f)),
    ("premium_tilt_cap", lambda c, f: mut_scalar(c, "premium_tilt_cap", f)),
    ("premium_tilt_max", lambda c, f: mut_scalar(c, "premium_tilt_max", f)),
]
FACTORS = [0.8, 0.9, 1.1, 1.2]

rows = []
for name, fn in PERTURBS:
    for f in FACTORS:
        try:
            cfg2 = fn(CFG, f)
        except Exception as ex:
            print(f"[skip] {name} x{f}: {ex}"); continue
        try:
            r = run(cfg2)
            rows.append({"param": name, "factor": f, **r})
            print(f"{name:24s} x{f:<5.2f} CAGR {r['cagr']:6.2f}% MDD {r['mdd']:7.2f}% Sharpe {r['sharpe']:.2f} Calmar {r['calmar']:.2f} TO {r['turnover']:5.1f}")
        except Exception as ex:
            print(f"[err] {name} x{f}: {ex}")

df = pd.DataFrame(rows)
summary = df.groupby("param").agg(
    cagr_min=("cagr", "min"), cagr_med=("cagr", "median"), cagr_max=("cagr", "max"),
    mdd_worst=("mdd", "min"), calmar_min=("calmar", "min"), calmar_med=("calmar", "median"),
    to_min=("turnover", "min"), to_max=("turnover", "max"),
).round(2)
print("\n===== 扰动汇总 (每个参数 ±10/±20%) =====")
print(summary.to_string())
n_bad = int((df["calmar"] < base["calmar"] * 0.9).sum())
print(f"\nCalmar 恶化超10%的扰动次数: {n_bad}/{len(df)}")
out = {"tag": TAG, "base": base, "rows": df.to_dict("records"), "summary": summary.to_dict("index")}
json.dump(out, open(os.path.join(OUT, f"audit_{TAG}.json"), "w"), ensure_ascii=False, indent=1)
print(f"[ok] out/audit_{TAG}.json")
