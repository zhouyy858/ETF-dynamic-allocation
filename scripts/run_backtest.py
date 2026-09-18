# -*- coding: utf-8 -*-
"""最终运行器(ETF动态配置skill v26): 每周五用前一日信号决策、目标缺口1笔当日收盘成交(v22 S阶段45组网格实证)
输出: 全历史/真实窗口指标、基准对比、分阶段、仓位现状、终端ASCII图表(不生成图片文件)"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate, fmt_eval, SLOTS
from strategy import DynamicStrategy
from metrics import drawdown_series

OUT = os.path.join(os.getcwd(), "out")
os.makedirs(OUT, exist_ok=True)
R_PROXY, W_PROXY = build_panel("proxy")
R_REAL, W_REAL = build_panel("real")
BACKTEST_START = "2014-06-23"
REAL_START = "2025-04-23"
PERIODS = {
    "2014H2-2015牛市": ("2014-06-23", "2015-06-12"),
    "2015股灾": ("2015-06-15", "2016-02-29"),
    "2016-2017修复": ("2016-03-01", "2017-12-29"),
    "2018熊市": ("2018-01-02", "2019-01-03"),
    "2019-2021牛市": ("2019-01-04", "2021-02-18"),
    "2021-2022熊市": ("2021-02-19", "2022-10-31"),
    "2023-2024震荡": ("2023-01-03", "2024-08-30"),
    "2024Q4-2026双牛": ("2024-09-02", "2026-07-31"),
}

def make_periods(end_date):
    """阶段窗口: 末段结束日随面板数据动态延伸(不再写死 2026-07-31)"""
    d = str(end_date)
    return dict(PERIODS, **{"2024Q4-2026双牛": ("2024-09-02", d)})
BENCHMARKS = {
    "B1等权20": {s: 20 for s in SLOTS},
    "B2保守防御": {"159232": 25, "515100": 25, "159941": 20, "513500": 20, "159952": 10},
    "B3均衡": {"159232": 15, "515100": 15, "159941": 25, "513500": 20, "159952": 25},
    "B4成长进攻": {"159232": 10, "515100": 10, "159941": 30, "513500": 20, "159952": 30},
    "B5价值60/成长40": {"159232": 30, "515100": 30, "159941": 15, "513500": 15, "159952": 10},
}
ETF_NAMES = {"159232": "自由现金流", "515100": "红利低波100", "159941": "纳指100", "513500": "标普500", "159952": "创业板"}

def risk_parity_weights(R, start, end):
    mask = R.index >= start if start else pd.Series(True, index=R.index)
    if end:
        mask = mask & (R.index <= end)
    sub = R[mask]
    vols = sub.std() * np.sqrt(252)
    w = 1 / vols
    return {s: float(w[s] / w.sum() * 100) for s in SLOTS}

def run_all(R, label, start, end, dyn_cfg):
    results = {}
    wealths = {}
    periods = make_periods(R.index[-1].date() if end is None else pd.Timestamp(end).date())
    for bname, bw in BENCHMARKS.items():
        res = run_backtest(R, fixed_weights=bw, start=start, end=end, name=bname, min_delta=0.0002)
        results[bname] = evaluate(res, periods=periods)
        wealths[bname] = res["wealth"]
    rpw = risk_parity_weights(R, start, end)
    res = run_backtest(R, fixed_weights=rpw, start=start, end=end, name="B6风险平价", min_delta=0.0002)
    results["B6风险平价"] = evaluate(res, periods=periods)
    wealths["B6风险平价"] = res["wealth"]
    ew = {s: 20 for s in SLOTS}
    res = run_backtest(R, fixed_weights=ew, start=start, end=end, name="B7等权买持", min_delta=1.0)
    results["B7等权买持"] = evaluate(res, periods=periods)
    wealths["B7等权买持"] = res["wealth"]
    ds = DynamicStrategy(R, cfg=dyn_cfg)
    bond = None
    if dyn_cfg.get("cash_bond_pct"):
        bond = rets_from(read_table(f"{dyn_cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav")
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=end, name=f"DYN {label}",
                       min_delta=dyn_cfg.get("min_delta", 0.02), repo=dyn_cfg.get("repo_rate", 0.022),
                       tranche_weights=dyn_cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=dyn_cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=dyn_cfg.get("rebal_weekday", 2), rebal_freq=dyn_cfg.get("rebal_freq", "weekly"),
                       strict=True)
    results[f"DYN {label}"] = evaluate(res, periods=periods)
    wealths[f"DYN {label}"] = res["wealth"]
    results["_weights"] = res["weights"]
    results["_rets"] = res["rets"]
    results["_ds"] = ds
    results["_wealths"] = wealths
    return results

# ---------- 终端ASCII图表 (不生成图片) ----------
def term_line_chart(series, title, height=16, width=86, log=True, syms=None):
    syms = syms or {}
    idxs = sorted(set(np.concatenate([s.dropna().index.values for s in series.values()])))
    if not idxs:
        return
    d0, d1 = pd.Timestamp(idxs[0]), pd.Timestamp(idxs[-1])
    span = (d1.value - d0.value) / (width - 1)
    tgrid = [pd.Timestamp(int(d0.value + span * c)) for c in range(width)]
    allv = np.concatenate([s.dropna().values for s in series.values()])
    if log:
        allv = np.log(np.maximum(allv, 1e-12))
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    grid = [[" "] * width for _ in range(height)]
    order = list(series.keys())
    order.sort(key=lambda l: (l != "DYN", l))  # DYN 后画, 保证覆盖优先
    for lab in order:
        s = series[lab]
        ch = syms.get(lab, "D" if lab == "DYN" else lab[0])
        for c in range(width):
            v = s.asof(tgrid[c])
            if pd.isna(v):
                continue
            if log:
                v = np.log(max(float(v), 1e-12))
            r = int(round((v - lo) / (hi - lo) * (height - 1)))
            r = max(0, min(height - 1, r))
            row = height - 1 - r
            if grid[row][c] == " ":
                grid[row][c] = ch
    print(f"── {title} ──")
    def fmt(v):
        v = np.exp(v) if log else v
        return f"{v:.2f}" if not log else f"{v:.3f}"
    for r in range(height):
        v = hi - (hi - lo) * r / (height - 1)
        tick = fmt(v) if r in (0, height // 2, height - 1) else ""
        print(f"{tick:>7} |{''.join(grid[r])}|")
    xlab = f"{d0.date()} " + " " * (width - len(str(d0.date())) - len(str(d1.date()))) + f"{d1.date()}"
    print(f"{'':>7} |{xlab}|")
    print(f"  图例: " + "  ".join(f"{syms.get(l, 'D' if l=='DYN' else l[0])}={l}" for l in series))
    print()

def term_dd_chart(series, title, height=12, width=86):
    idxs = sorted(set(np.concatenate([s.dropna().index.values for s in series.values()])))
    if not idxs:
        return
    d0, d1 = pd.Timestamp(idxs[0]), pd.Timestamp(idxs[-1])
    span = (d1.value - d0.value) / (width - 1)
    tgrid = [pd.Timestamp(int(d0.value + span * c)) for c in range(width)]
    allv = np.concatenate([s.dropna().values for s in series.values()])
    lo, hi = float(np.nanmin(allv)), 0.0
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    grid = [[" "] * width for _ in range(height)]
    for lab in series:
        s = series[lab]
        ch = "D" if lab == "DYN" else lab[0]
        for c in range(width):
            v = s.asof(tgrid[c])
            if pd.isna(v):
                continue
            r = int(round((v - lo) / (hi - lo) * (height - 1)))
            r = max(0, min(height - 1, r))
            row = height - 1 - r
            if grid[row][c] == " ":
                grid[row][c] = ch
    print(f"── {title} ──")
    for r in range(height):
        v = hi - (hi - lo) * r / (height - 1)
        tick = f"{v*100:.1f}%" if r in (0, height // 2, height - 1) else ""
        print(f"{tick:>7} |{''.join(grid[r])}|")
    xlab = f"{d0.date()} " + " " * (width - len(str(d0.date())) - len(str(d1.date()))) + f"{d1.date()}"
    print(f"{'':>7} |{xlab}|")
    print(f"  图例: " + "  ".join(f"{('D' if l=='DYN' else l[0])}={l}" for l in series))
    print()

def term_bars(rows, title, width=42):
    """rows: [(label, value, display)]"""
    print(f"── {title} ──")
    maxv = max(abs(v) for _, v, _ in rows) or 1
    for lab, v, disp in rows:
        k = int(round(abs(v) / maxv * width))
        print(f"  {lab:<7} {disp:>11} |{'█' * k}")
    print()

# ---------- 主流程 ----------
def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    _p1 = os.path.join(_here, "..", "references", "final_cfg_v30.json")
    _p2 = os.path.join(_here, "references", "final_cfg_v30.json")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else (_p1 if os.path.exists(_p1) else _p2)
    tag = sys.argv[2] if len(sys.argv) > 2 else "v30"
    cfg = json.load(open(cfg_file))
    wd = cfg.get("rebal_weekday", 2); wd_cn = "一二三四五"[wd]
    freq = cfg.get("rebal_freq", "weekly"); n_tw = len(cfg.get("tranche_weights", [1.0]))
    print(f"===== 迭代 {tag} | 周{wd_cn}-{freq}-{n_tw}笔(前一日信号, 无未来函数, strict) =====")
    rp = run_all(R_PROXY, tag, BACKTEST_START, None, cfg)
    rr = run_all(R_REAL, tag, REAL_START, None, cfg)
    json.dump({k: v for k, v in rp.items() if not k.startswith("_")},
              open(f"{OUT}/iter_{tag}_proxy.json", "w"), ensure_ascii=False, indent=2, default=str)
    json.dump({k: v for k, v in rr.items() if not k.startswith("_")},
              open(f"{OUT}/iter_{tag}_real.json", "w"), ensure_ascii=False, indent=2, default=str)

    print(f"\n===== 扩展代理回测 ({BACKTEST_START} ~ {R_PROXY.index[-1].date()}, 含2015/2018/2021-22三轮大熊) =====")
    for k, e in rp.items():
        if not k.startswith("_"):
            print(fmt_eval(e))
    print(f"\n===== 真实ETF回测 ({REAL_START} ~ {R_REAL.index[-1].date()}) =====")
    for k, e in rr.items():
        if not k.startswith("_"):
            print(fmt_eval(e))

    dyn = rp["DYN " + tag]
    print("\n===== DYN 分阶段(全历史) =====")
    for pn, pe in dyn["periods"].items():
        print(f"  {pn:<18} CAGR={pe['cagr']*100:7.2f}%  MDD={pe['max_dd']*100:6.2f}%  total={pe['total']*100:8.2f}%")

    itfile = f"{OUT}/iterations.json"
    its = json.load(open(itfile)) if os.path.exists(itfile) else {}
    er = rr["DYN " + tag]
    its[tag] = {"proxy_cagr": dyn["cagr"], "proxy_mdd": dyn["max_dd"], "proxy_sharpe": dyn["sharpe"],
                "proxy_calmar": dyn["calmar"], "real_cagr": er["cagr"], "real_mdd": er["max_dd"],
                "real_sharpe": er["sharpe"], "real_calmar": er["calmar"],
                "avg_cash": dyn["avg_cash"], "turnover": dyn["turnover"]}
    json.dump(its, open(itfile, "w"), ensure_ascii=False, indent=2)

    wdf = rp["_weights"]; rets = rp["_rets"]; ds = rp["_ds"]
    w_last = wdf.iloc[-1]
    print("\n===== 当前仓位管理现状 (数据截至 %s) =====" % str(wdf.index[-1].date()))
    print("  实际持仓(含权重漂移, 以最新交易日收盘计):")
    for s in SLOTS + ["cash"]:
        nm = ETF_NAMES.get(s, "现金(逆回购25%+511010 75%)")
        print(f"    {s:<8} {nm:<12} {w_last[s]*100:6.2f}%")
    ds_live = DynamicStrategy(R_PROXY, cfg=dict(cfg, signal_lag=0))
    # 实盘口径: 最新数据日=T-1, 以lag=0实例在T-1数据上计算"下一个周三目标"(等价于周三早盘决策)
    tgt_last = ds_live.regular_target(wdf.index[-1], {"pf_rets": rets})
    print(f"  目标权重(下一个周{wd_cn}决策目标, 用最新可得数据):")
    for s in SLOTS + ["cash"]:
        nm = ETF_NAMES.get(s, "现金(逆回购25%+511010 75%)")
        print(f"    {s:<8} {nm:<12} {tgt_last[s]*100:6.2f}%")
    sc_last = ds.state_log[-1] if ds.state_log else None
    print(f"  最近信号: 有效打分={sc_last[1] if sc_last else '-'} (0-9), CN深熊锁={ds._lock['CN']}, US深熊锁={ds._lock['US']}")
    for mkt in ["CN", "US"]:
        dd, ok20, ok20_60, rec, ok120 = ds.sig.mkt_info(wdf.index[-1], mkt, ds.gate_win)
        print(f"    {mkt}: 市场3年窗回撤={dd*100:.1f}%  站上SMA20={ok20}  双均线多头={ok20_60}  站上SMA{ds.gate_win}={ok120}  恢复度={rec*100:.0f}%")

    # ---- 终端ASCII图表 ----
    navs = {"DYN": rp["_wealths"]["DYN " + tag]}
    for b in ["B1等权20", "B3均衡", "B4成长进攻", "B6风险平价", "B7等权买持"]:
        navs[b] = rp["_wealths"][b]
    term_line_chart(navs, "收益图: DYN vs 静态基准 (全历史, 对数坐标, 起点=1)", log=True)
    dds = {"DYN": drawdown_series(navs["DYN"]), "B3均衡": drawdown_series(navs["B3均衡"]),
           "B4成长进攻": drawdown_series(navs["B4成长进攻"])}
    term_dd_chart(dds, "回撤图: DYN vs B3/B4 (全历史)")
    navr = {"DYN": rr["_wealths"]["DYN " + tag], "B3均衡": rr["_wealths"]["B3均衡"]}
    term_line_chart(navr, "收益图: 真实ETF窗口 (2025-04-23起, 对数坐标)", log=True)
    its2 = json.load(open(itfile))
    canon = [("v10", "v10"), ("v15", "v15check2"), ("v17", "v17final"), ("v21", "v21"), ("v22", "v22"), ("v23", "v23"), ("v24", "v24"), ("v25", "v25"), ("v26", "v26")]
    tags_ok = [lab for lab, k in canon if k in its2]
    def val(lab):
        k = dict(canon)[lab]
        return its2[k]
    rows_c = [(t, val(t)["proxy_cagr"] * 100, f"{val(t)['proxy_cagr']*100:.2f}%") for t in tags_ok]
    term_bars(rows_c, "迭代优化进展: 全历史年化收益 (v10→%s)" % tag)
    rows_cr = [(t, val(t)["real_cagr"] * 100, f"{val(t)['real_cagr']*100:.2f}%") for t in tags_ok if "real_cagr" in val(t)]
    term_bars(rows_cr, "迭代优化进展: 真实窗口年化收益 (v10→%s)" % tag)
    rows_m = [(t, val(t)["proxy_mdd"] * 100, f"{val(t)['proxy_mdd']*100:.2f}%") for t in tags_ok]
    term_bars(rows_m, "迭代优化进展: 全历史最大回撤 (越短越好)")
    print("\n[ok] 已保存: %s/iter_%s_proxy.json, %s/iter_%s_real.json, %s/iterations.json" % (OUT, tag, OUT, tag, OUT))

if __name__ == "__main__":
    main()
