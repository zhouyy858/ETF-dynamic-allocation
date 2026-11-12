# -*- coding: utf-8 -*-
"""最终运行器: 每周五用前一日信号决策、目标缺口1笔当日收盘成交(v22 S阶段45组网格实证)
输出: 全历史/真实窗口指标、基准对比、分阶段、仓位现状、终端ASCII图表(不生成图片文件)
损益口径: 默认 price(QDII 场内价), 第三个参数传 nav 切换净值口径"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate, fmt_eval, SLOTS
from strategy import DynamicStrategy, live_rebalance_target
from metrics import drawdown_series
from swaps import pnl_panel, traded_code, traded_name

OUT = os.path.join(os.getcwd(), "out")
os.makedirs(OUT, exist_ok=True)
R_PROXY, W_PROXY = build_panel("proxy")
R_REAL, W_REAL = build_panel("real")
BACKTEST_START = "2014-06-23"
REAL_START = "2025-04-23"
PRICE_FILES = {"159941": "qdii_price_159941.csv", "513500": "qdii_price_513500.csv"}
PRICE_DROP = {"159941": ["2022-07-05"], "513500": ["2022-03-30"]}


def price_rets(code):
    s = read_table(PRICE_FILES[code])["close"].sort_index()
    r = s.pct_change().dropna()
    return r.drop([pd.Timestamp(d) for d in PRICE_DROP[code]], errors="ignore")


def exec_panel(R, layer, cfg, caliber):
    P = pnl_panel(R, layer, cfg).copy()
    if caliber == "price":
        for code in PRICE_FILES:
            P[code] = price_rets(code).reindex(P.index).combine_first(P[code])
    return P


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

def run_all(R, label, start, end, dyn_cfg, R_pnl=None):
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
    res = run_backtest(R if R_pnl is None else R_pnl, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
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
            t = pd.Timestamp(tgrid[c]).as_unit(s.index.unit)
            v = s.asof(t)
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
            t = pd.Timestamp(tgrid[c]).as_unit(s.index.unit)
            v = s.asof(t)
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
    _p1 = os.path.join(_here, "..", "references", "final_cfg_v45.json")
    _p2 = os.path.join(_here, "references", "final_cfg_v45.json")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else (_p1 if os.path.exists(_p1) else _p2)
    tag = sys.argv[2] if len(sys.argv) > 2 else "v45"
    caliber = sys.argv[3].lower() if len(sys.argv) > 3 else "price"
    if caliber not in ("price", "nav"):
        raise SystemExit("第三个参数必须是 price 或 nav")
    cfg = json.load(open(cfg_file))
    R_proxy_pnl = exec_panel(R_PROXY, "proxy", cfg, caliber)
    R_real_pnl = exec_panel(R_REAL, "real", cfg, caliber)
    wd = cfg.get("rebal_weekday", 2); wd_cn = "一二三四五"[wd]
    freq = cfg.get("rebal_freq", "weekly"); n_tw = len(cfg.get("tranche_weights", [1.0]))
    cal_cn = "场内价" if caliber == "price" else "净值"
    cal_suffix = "" if caliber == "nav" else "_price"
    print(f"===== 迭代 {tag} | 周{wd_cn}-{freq}-{n_tw}笔(前一日信号, 无未来函数, strict) | 损益口径={cal_cn} =====")
    rp = run_all(R_PROXY, tag, BACKTEST_START, None, cfg, R_proxy_pnl)
    rr = run_all(R_REAL, tag, REAL_START, None, cfg, R_real_pnl)
    ds_oos = DynamicStrategy(R_PROXY, cfg=cfg)
    bond_oos = rets_from(read_table(f"{cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav") if cfg.get("cash_bond_pct") else None
    res_oos = run_backtest(R_proxy_pnl, target_weights_fn=ds_oos.target_fn(), daily_override_fn=ds_oos.daily_fn(),
                           start="2022-01-04", name="DYN OOS", min_delta=cfg.get("min_delta", 0.02),
                           repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                           cash_bond_rets=bond_oos, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                           rebal_weekday=cfg.get("rebal_weekday", 2), rebal_freq=cfg.get("rebal_freq", "weekly"),
                           strict=True)
    oos_dyn = evaluate(res_oos)
    print(f"\n===== OOS（proxy 独立起点 2022-01-04）DYN CAGR={oos_dyn['cagr']*100:.2f}%  "
          f"MDD={oos_dyn['max_dd']*100:.2f}%  Calmar={oos_dyn['calmar']:.4f} =====")
    json.dump({k: v for k, v in rp.items() if not k.startswith("_")},
              open(f"{OUT}/iter_{tag}{cal_suffix}_proxy.json", "w"), ensure_ascii=False, indent=2, default=str)
    json.dump({k: v for k, v in rr.items() if not k.startswith("_")},
              open(f"{OUT}/iter_{tag}{cal_suffix}_real.json", "w"), ensure_ascii=False, indent=2, default=str)

    print(f"\n===== 扩展代理回测 ({BACKTEST_START} ~ {R_PROXY.index[-1].date()}, 含2015/2018/2021-22三轮大熊; DYN {cal_cn}口径, 基准净值口径) =====")
    for k, e in rp.items():
        if not k.startswith("_"):
            print(fmt_eval(e))
    print(f"\n===== 真实ETF回测 ({REAL_START} ~ {R_REAL.index[-1].date()}; DYN {cal_cn}口径, 基准净值口径) =====")
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
    its[tag] = {"caliber": caliber,
                "proxy_cagr": dyn["cagr"], "proxy_mdd": dyn["max_dd"], "proxy_sharpe": dyn["sharpe"],
                "proxy_calmar": dyn["calmar"], "real_cagr": er["cagr"], "real_mdd": er["max_dd"],
                "real_sharpe": er["sharpe"], "real_calmar": er["calmar"],
                "avg_cash": dyn["avg_cash"], "turnover": dyn["turnover"]}
    if caliber == "price":
        its[tag].update({"price_proxy_cagr": dyn["cagr"], "price_proxy_mdd": dyn["max_dd"],
                         "price_proxy_calmar": dyn["calmar"], "price_real_cagr": er["cagr"],
                         "price_real_mdd": er["max_dd"], "price_real_calmar": er["calmar"]})
    json.dump(its, open(itfile, "w"), ensure_ascii=False, indent=2)

    wdf = rp["_weights"]; rets = rp["_rets"]; ds = rp["_ds"]
    w_last = wdf.iloc[-1]
    bond_code = cfg.get("cash_bond_code", "511010")
    bond_pct = float(cfg.get("cash_bond_pct", 0.0))
    if bond_pct >= 1.0 - 1e-9:
        cash_nm = f"现金(100% {bond_code})"
    else:
        cash_nm = f"现金(逆回购{1-bond_pct:.0%}+{bond_code} {bond_pct:.0%})"
    active_slots = [s for s in SLOTS if s not in set(cfg.get("exclude", []))]
    print("\n===== 当前仓位管理现状 (数据截至 %s, 配置 %s) =====" % (str(wdf.index[-1].date()), cfg_file))
    print("  实际持仓(含权重漂移, 以最新交易日收盘计; 代码是成交标的):")
    for s in active_slots + ["cash"]:
        code = "cash" if s == "cash" else traded_code(s, cfg)
        nm = cash_nm if s == "cash" else traded_name(s, cfg)
        print(f"    {code:<8} {nm:<22} {w_last[s]*100:6.2f}%")
    # 与引擎同一条路径: 调仓日目标只用前一交易日收盘. 数据若已含该周五, 取回测记下的目标;
    # 数据若停在周四(周五 09:00), 用走完的状态现算, 信号落在周四. 不用冷启动的 signal_lag=0.
    decision, signal_dt, tgt_last = live_rebalance_target(ds, rets, wd)
    print(f"  目标权重(决策日 {decision.date()}，信号截至 {signal_dt.date()}，周五 09:00 清单；不含决策日收盘):")
    for s in active_slots + ["cash"]:
        code = "cash" if s == "cash" else traded_code(s, cfg)
        nm = cash_nm if s == "cash" else traded_name(s, cfg)
        print(f"    {code:<8} {nm:<22} {tgt_last[s]*100:6.2f}%")
    _ver = cfg.get("version", "")
    if _ver.startswith("v45"):
        _prev_name = "final_cfg_v44.json"
    elif _ver.startswith("v44"):
        _prev_name = "final_cfg_v43.json"
    else:
        _prev_name = "final_cfg_v41.json"
    prev_file = os.path.join(os.path.dirname(cfg_file), _prev_name)
    if _ver.startswith(("v43", "v44", "v45")) and os.path.exists(prev_file):
        cfg_prev = json.load(open(prev_file))
        ds_prev = DynamicStrategy(R_PROXY, cfg=cfg_prev)
        bond_prev = rets_from(read_table(f"{cfg_prev.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav") if cfg_prev.get("cash_bond_pct") else None
        res_prev = run_backtest(R_PROXY, target_weights_fn=ds_prev.target_fn(), daily_override_fn=ds_prev.daily_fn(),
                             start=BACKTEST_START, name="v41", min_delta=cfg_prev.get("min_delta", 0.02),
                             repo=cfg_prev.get("repo_rate", 0.022), tranche_weights=cfg_prev.get("tranche_weights"),
                             cash_bond_rets=bond_prev, cash_bond_pct=cfg_prev.get("cash_bond_pct", 0.0),
                             rebal_weekday=cfg_prev.get("rebal_weekday", 2), rebal_freq=cfg_prev.get("rebal_freq", "weekly"),
                             strict=True)
        w_prev = res_prev["weights"].iloc[-1]
        print(f"  相对上一默认 {cfg_prev.get('version', 'prev')} 持仓的切换（纸面，不下单）:")
        for s in SLOTS:
            if s in cfg.get("exclude", []):
                continue
            code = traded_code(s, cfg)
            prev_code = traded_code(s, cfg_prev)
            held = float(w_prev[s]) * 100
            tgt = float(tgt_last[s]) * 100
            if prev_code != code:
                print(f"    卖出 {prev_code} {held:.2f}%  →  买入 {code} {tgt:.2f}%")
            else:
                print(f"    {code} 维持  现仓 {held:.2f}%  目标 {tgt:.2f}%  差额 {tgt-held:+.2f}pp")
        print(f"    现金 {float(w_prev['cash'])*100:.2f}% → 目标 {float(tgt_last['cash'])*100:.2f}%")
    sc_last = ds.state_log[-1] if ds.state_log else None
    print(f"  最近信号: 决策日={decision.date()} 信号截至={signal_dt.date()} 有效打分={sc_last[1] if sc_last else '-'} (0-9), CN深熊锁={ds._lock['CN']}, US深熊锁={ds._lock['US']}")
    for mkt in ["CN", "US"]:
        dd, ok20, ok20_60, rec, ok120 = ds.sig.mkt_info(decision, mkt, ds.gate_win)
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
    if tags_ok:
        rows_c = [(t, val(t)["proxy_cagr"] * 100, f"{val(t)['proxy_cagr']*100:.2f}%") for t in tags_ok]
        term_bars(rows_c, "迭代优化进展: 全历史年化收益 (v10→%s)" % tag)
        rows_cr = [(t, val(t)["real_cagr"] * 100, f"{val(t)['real_cagr']*100:.2f}%") for t in tags_ok if "real_cagr" in val(t)]
        if rows_cr:
            term_bars(rows_cr, "迭代优化进展: 真实窗口年化收益 (v10→%s)" % tag)
        rows_m = [(t, val(t)["proxy_mdd"] * 100, f"{val(t)['proxy_mdd']*100:.2f}%") for t in tags_ok]
        term_bars(rows_m, "迭代优化进展: 全历史最大回撤 (越短越好)")
    else:
        print("  (out/iterations.json 没有 v10–v26 历史键, 跳过迭代图)")
    print("\n[ok] 已保存: %s/iter_%s%s_proxy.json, %s/iter_%s%s_real.json, %s/iterations.json" %
          (OUT, tag, cal_suffix, OUT, tag, cal_suffix, OUT))

if __name__ == "__main__":
    main()
