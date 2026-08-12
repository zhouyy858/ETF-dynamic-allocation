# -*- coding: utf-8 -*-
"""未来函数审计(可复现, v22): 时序口径对照 + 旧口径复现校验
生产策略(scripts/strategy.py)已硬编码"信号最小滞后1个交易日"(SignalSet._idx), 配置无法绕过;
旧版"当日信号"泄漏行为仅在本审计脚本内以私有子类(LeakySignalSet)复现, 生产代码无泄漏路径。

口径:
  OLD 旧发布口径(含未来函数) : Leaky信号(当日) + 引擎post计提(成交计当日收益) + 1笔成交
  A0  仅修计提·同日信号      : Leaky信号 + pre计提 + 1笔成交
  B   前日信号+当日成交(1笔) : 严格信号 + pre计提 + 1笔成交
  C   当日信号+次日成交      : Leaky信号 + pre计提 + 次日成交
  D   前日信号+次日成交      : 严格信号 + pre计提 + 次日成交
  v21/v22 严格(定稿)         : 严格信号 + pre计提 + 每周三3周三笔 + 溢价T-2

用法: python3 scripts/audit_lookahead.py
输出: 终端对照表 + out/audit_lookahead.json; 断言旧口径=15.39%/27.17%、严格口径=v21-v30发布值
      (2026-08-12 数据管道确定性去重修复后统一重定基线)
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy, SignalSet
from stress_test import synthetic_resonance

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
if not os.path.isdir(SKILL_REF):
    SKILL_REF = os.path.join(HERE, "references")
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)
CFG20 = json.load(open(f"{SKILL_REF}/final_cfg_v20.json"))
CFG21 = json.load(open(f"{SKILL_REF}/final_cfg_v21.json"))
CFG22 = json.load(open(f"{SKILL_REF}/final_cfg_v22.json"))
CFG22B = json.load(open(f"{SKILL_REF}/final_cfg_v22b.json"))
CFG24 = json.load(open(f"{SKILL_REF}/final_cfg_v24.json"))
CFG25 = json.load(open(f"{SKILL_REF}/final_cfg_v25.json"))
CFG26 = json.load(open(f"{SKILL_REF}/final_cfg_v26.json"))
CFG27 = json.load(open(f"{SKILL_REF}/final_cfg_v27.json"))
CFG28 = json.load(open(f"{SKILL_REF}/final_cfg_v28.json"))
CFG29 = json.load(open(f"{SKILL_REF}/final_cfg_v29.json"))
CFG30 = json.load(open(f"{SKILL_REF}/final_cfg_v30.json"))
bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")

class LeakySignalSet(SignalSet):
    """仅审计用: 复现旧版"决策日使用当日收盘数据"的泄漏口径(生产策略无此路径)"""
    def _idx(self, dt):
        i = self.R.index.get_indexer([dt], method="ffill")[0]
        return max(0, i)

class LeakyStrategy(DynamicStrategy):
    SIG_CLASS = LeakySignalSet

def run(R, start, tag, cfg, strat_cls, accrual="pre", exec_lag=0, tw=None, strict=False):
    c = dict(cfg)
    ds = strat_cls(R, cfg=c)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name=tag, min_delta=0.02, repo=0.022,
                       tranche_weights=tw, cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       exec_lag=exec_lag, accrual_mode=accrual, strict=strict,
                       rebal_weekday=cfg.get("rebal_weekday", 2), rebal_freq=cfg.get("rebal_freq", "weekly"))
    return evaluate(res)

print("===== 未来函数口径对照 · proxy 全历史 2014-06-23 起 =====")
MODES = [
    ("OLD 旧发布口径(含未来函数)",  CFG20, LeakyStrategy, "post", 0, [1.0],   False),
    ("A0  仅修计提·同日信号",       CFG20, LeakyStrategy, "pre",  0, [1.0],   False),
    ("B   前日信号+当日成交(1笔)",  CFG20, DynamicStrategy, "pre", 0, [1.0],   True),
    ("C   当日信号+次日成交",       CFG20, LeakyStrategy, "pre",  1, [1.0],   False),
    ("D   前日信号+次日成交",       CFG20, DynamicStrategy, "pre", 1, [1.0],   False),
    ("v21 严格(3周三笔)",            CFG21, DynamicStrategy, "pre", 0, [1/3]*3, True),
    ("v22 严格(周五weekly1笔)",      CFG22, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v22b 严格(候选定稿)",          CFG22B, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v24 严格(v23+现金债0.75)",      CFG24, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v25 严格(v24+溢价倾斜)",        CFG25, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v26 严格(v25+溢价门控削减增强)", CFG26, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v27 严格(v26+恢复期12)",        CFG27, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v28 严格(前提驱动)",          CFG28, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v29 严格(AI超配)",           CFG29, DynamicStrategy, "pre", 0, [1.0],   True),
    ("v30 严格(定稿)",              CFG30, DynamicStrategy, "pre", 0, [1.0],   True),
]
res = {}
for name, cfg, cls, am, el, tw, st in MODES:
    e = run(R, "2014-06-23", "p", cfg, cls, am, el, tw, st)
    res[name] = {"proxy": {m: e[m] for m in ["cagr", "max_dd", "sharpe", "calmar", "turnover"]}}
    print(f"  {name:<30} CAGR {e['cagr']*100:6.2f}%  MDD {e['max_dd']*100:6.2f}%  Sharpe {e['sharpe']:.2f}  "
          f"Calmar {e['calmar']:4.2f}  TO {e['turnover']:6.1f}")

print("\n===== 真实窗口 2025-04-23 起 =====")
for name, cfg, cls, am, el, tw, st in MODES:
    e = run(Rr, "2025-04-23", "r", cfg, cls, am, el, tw, st)
    res[name]["real"] = {m: e[m] for m in ["cagr", "max_dd", "sharpe", "calmar", "turnover"]}
    print(f"  {name:<30} CAGR {e['cagr']*100:6.2f}%  MDD {e['max_dd']*100:6.2f}%  Calmar {e['calmar']:4.2f}")

old_p = res["OLD 旧发布口径(含未来函数)"]["proxy"]["cagr"]
old_r = res["OLD 旧发布口径(含未来函数)"]["real"]["cagr"]
v_p = res["v21 严格(3周三笔)"]["proxy"]["cagr"]
v_r = res["v21 严格(3周三笔)"]["real"]["cagr"]
w_p = res["v22 严格(周五weekly1笔)"]["proxy"]["cagr"]
w_r = res["v22 严格(周五weekly1笔)"]["real"]["cagr"]
b_p = res["v22b 严格(候选定稿)"]["proxy"]["cagr"]
b_r = res["v22b 严格(候选定稿)"]["real"]["cagr"]
v24_p = res["v24 严格(v23+现金债0.75)"]["proxy"]["cagr"]
v24_r = res["v24 严格(v23+现金债0.75)"]["real"]["cagr"]
v25_p = res["v25 严格(v24+溢价倾斜)"]["proxy"]["cagr"]
v25_r = res["v25 严格(v24+溢价倾斜)"]["real"]["cagr"]
# 2026-08-12 数据管道修复(代理序列去重改为稳定排序)后统一重定基线
assert abs(old_p - 0.1539) < 0.005, f"旧口径proxy复现失败: {old_p:.4f}"
assert abs(old_r - 0.2717) < 0.01, f"旧口径real复现失败: {old_r:.4f}"
assert abs(v_p - 0.0974) < 0.005, f"严格proxy回归失败(v21): {v_p:.4f}"
assert abs(v_r - 0.2317) < 0.01, f"严格real回归失败(v21): {v_r:.4f}"
assert abs(w_p - 0.1009) < 0.01, f"严格proxy回归失败(v22): {w_p:.4f}"
assert abs(w_r - 0.2737) < 0.01, f"严格real回归失败(v22): {w_r:.4f}"
assert abs(b_p - 0.1026) < 0.01, f"严格proxy回归失败(v22b): {b_p:.4f}"  # 2026-08-03数据修正后
assert abs(b_r - 0.2784) < 0.01, f"严格real回归失败(v22b): {b_r:.4f}"
assert abs(v24_p - 0.1052) < 0.01, f"严格proxy回归失败(v24): {v24_p:.4f}"
assert abs(v24_r - 0.2783) < 0.01, f"严格real回归失败(v24): {v24_r:.4f}"
assert abs(v25_p - 0.1063) < 0.01, f"严格proxy回归失败(v25): {v25_p:.4f}"
assert abs(v25_r - 0.2797) < 0.01, f"严格real回归失败(v25): {v25_r:.4f}"
v26_p = res["v26 严格(v25+溢价门控削减增强)"]["proxy"]["cagr"]
v26_r = res["v26 严格(v25+溢价门控削减增强)"]["real"]["cagr"]
assert abs(v26_p - 0.1078) < 0.01, f"严格proxy回归失败(v26): {v26_p:.4f}"
assert abs(v26_r - 0.2933) < 0.01, f"严格real回归失败(v26): {v26_r:.4f}"
v27_p = res["v27 严格(v26+恢复期12)"]["proxy"]["cagr"]
v27_r = res["v27 严格(v26+恢复期12)"]["real"]["cagr"]
assert abs(v27_p - 0.1070) < 0.01, f"严格proxy回归失败(v27): {v27_p:.4f}"
assert abs(v27_r - 0.2933) < 0.01, f"严格real回归失败(v27): {v27_r:.4f}"
v28_p = res["v28 严格(前提驱动)"]["proxy"]["cagr"]
v28_r = res["v28 严格(前提驱动)"]["real"]["cagr"]
assert abs(v28_p - 0.1047) < 0.01, f"严格proxy回归失败(v28): {v28_p:.4f}"
assert abs(v28_r - 0.2816) < 0.01, f"严格real回归失败(v28): {v28_r:.4f}"
v29_p = res["v29 严格(AI超配)"]["proxy"]["cagr"]
v29_r = res["v29 严格(AI超配)"]["real"]["cagr"]
assert abs(v29_p - 0.1062) < 0.01, f"严格proxy回归失败(v29): {v29_p:.4f}"
assert abs(v29_r - 0.2859) < 0.01, f"严格real回归失败(v29): {v29_r:.4f}"
v30_p = res["v30 严格(定稿)"]["proxy"]["cagr"]
v30_r = res["v30 严格(定稿)"]["real"]["cagr"]
assert abs(v30_p - 0.1049) < 0.01, f"严格proxy回归失败(v30): {v30_p:.4f}"
assert abs(v30_r - 0.2914) < 0.01, f"严格real回归失败(v30): {v30_r:.4f}"
print("\n[ok] 旧口径复现(15.39%/27.17%) 通过; 严格口径复现 v21/v22/v22b/v24/v25/v26/v27/v28/v29/v30 通过")

print("\n===== 压力测试 (严格口径 vs 旧口径) =====")
synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
scen = [("2015股灾", R, "2015-06-15", "2016-02-29", None), ("2018熊市", R, "2018-01-02", "2019-01-03", None),
        ("2021-22熊", R, "2021-02-19", "2022-10-31", None), ("2019-21牛", R, "2019-01-04", "2021-02-18", None),
        ("共振熊", synth, "2024-09-02", "2026-07-31", s_idx)]
print(f"{'情景':<10}{'OLD CAGR/MDD':>20}{'严格 CAGR/MDD':>22}")
for name, Rs, ps, pe, am in scen:
    a = run(Rs, ps, "old", CFG20, LeakyStrategy, "post", 0, [1.0], False)
    b = run(Rs, ps, "b", CFG20, DynamicStrategy, "pre", 0, [1.0], True)
    print(f"{name:<10}{a['cagr']*100:10.2f}%/{a['max_dd']*100:6.2f}%{b['cagr']*100:12.2f}%/{b['max_dd']*100:6.2f}%")
    res.setdefault("stress", {})[name] = {"OLD": [round(a["cagr"], 4), round(a["max_dd"], 4)],
                                          "严格": [round(b["cagr"], 4), round(b["max_dd"], 4)]}

json.dump(res, open(f"{OUT}/audit_lookahead.json", "w"), ensure_ascii=False, indent=1, default=str)
print("\n[ok] out/audit_lookahead.json")
