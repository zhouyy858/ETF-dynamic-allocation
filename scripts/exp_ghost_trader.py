# -*- coding: utf-8 -*-
"""幽灵交易系统（Ghost Trader）纸面回测。研究实验，不改默认策略。

只读 references/final_cfg_v30.json，不写回。不接入 run_backtest 默认路径。
A 股 ETF 只做多；空头侧不实现。

规则（每只 ETF 独立）：
  趋势：EMA9 > EMA19 才允许开多
  动量：入场时 RSI(9) < 阈值（默认 70，另测 65/75）
  入场：收盘创 N 日新高（默认 20 日唐奇安上轨，另测 10/55；通道不含当日）
  出场：收盘跌破 20 日唐奇安下轨（真实与影子相同，出场窗口不随 N 变）
  幽灵过滤：影子账户始终按上述规则成交；只有「最近一笔已平影子交易净盈亏 < 0」
            时，下一笔信号才做真实交易。没有已平影子交易时不做真实交易。
成交时点与 v30 严格口径对齐：信号用当日收盘，下一交易日收盘成交，
新仓位从成交日的下一交易日起计提收益（signal_lag=1 + accrual pre）。
组合：当日有真实持仓的 ETF 等权，其余为现金。
现金层与 v30 相同：25% 逆回购 2.2% + 75% 511010。费用单边万 5。
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from, ETF_META
from engine import run_backtest, evaluate, FEE, SLOTS
from strategy import DynamicStrategy
from metrics import TRADING_DAYS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CFG_FILE = os.path.join(ROOT, "references", "final_cfg_v30.json")
OUT_DIR = os.path.join(ROOT, "out")
REF_JSON = os.path.join(ROOT, "references", "exp_ghost_trader.json")
REF_MD = os.path.join(ROOT, "references", "exp_ghost_trader.md")

PROXY_START = "2014-06-23"
REAL_START = "2025-04-23"
OOS_START = "2022-01-04"
REPO_RATE = 0.022          # v30 repo_rate，现金层逆回购
CASH_BOND_PCT = 0.75       # v30 cash_bond_pct
EXIT_N = 20
EMA_FAST, EMA_SLOW, RSI_N = 9, 19, 9
ENTRY_NS = (10, 20, 55)
RSI_MAXS = (65, 70, 75)
DEFAULT_N, DEFAULT_RSI = 20, 70
# 与 scripts/exp_opt31.py 的 v30 升级三关相同
ACCEPT = {"dcal": 0.15, "proxy_mdd": -0.10, "real_mdd": -0.041}
PLAT = {"win": 0.10, "cliff": 0.5, "min_ok": 3}
OOS_BAD = 0.05
BROAD_CODE = "510300"
BROAD_NAME = "沪深300ETF"
NAMES = {
    "159232": "自由现金流", "515100": "红利低波100", "159941": "纳指100",
    "513500": "标普500", "159952": "创业板", "510300": "沪深300ETF",
}


def wilder_rsi(price, n):
    d = price.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    ad = dn.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + au / ad)
    rsi = rsi.where(ad > 0, np.where(au > 0, 100.0, np.nan))
    return rsi


def signal_frame(price, entry_n, rsi_max, exit_n=EXIT_N):
    """因果信号。唐奇安用前 N 根收盘，不含当日，否则『新高/跌破』恒成立或恒不成立。"""
    ema_f = price.ewm(span=EMA_FAST, min_periods=EMA_FAST, adjust=False).mean()
    ema_s = price.ewm(span=EMA_SLOW, min_periods=EMA_SLOW, adjust=False).mean()
    rsi = wilder_rsi(price, RSI_N)
    prior_high = price.shift(1).rolling(entry_n, min_periods=entry_n).max()
    prior_low = price.shift(1).rolling(exit_n, min_periods=exit_n).min()
    entry = (price > prior_high) & (ema_f > ema_s) & (rsi < rsi_max)
    exit_ = price < prior_low
    return entry.fillna(False).astype(bool), exit_.fillna(False).astype(bool)


def run_books(price, entry, exit_, ghost, fee=FEE):
    """影子/真实持仓状态机。price/entry/exit_ 与评估日历对齐。

    held[i] 为 True 表示该日收益由真实持仓计提（仓位在上一收盘已建立）。
    信号在 bar s 收盘确认，bar s+1 收盘成交，最早计提 s+2 的收益。
    同一根 K 线先平后不再开，避免平仓价上立即来回。
    """
    price = np.asarray(price, float)
    entry = np.asarray(entry, bool)
    exit_ = np.asarray(exit_, bool)
    n = len(price)
    shadow_in = real_in = False
    last_pnl = None
    sh_px = sh_i = re_px = re_i = None
    shadow_trades, real_trades = [], []
    held = np.zeros(n, dtype=bool)
    flips = 0

    def net_pnl(entry_px, exit_px):
        gross = exit_px / entry_px - 1.0
        net = (exit_px * (1.0 - fee)) / (entry_px * (1.0 + fee)) - 1.0
        return gross, net

    for i in range(n):
        held[i] = real_in
        if i == 0:
            continue
        px = float(price[i])
        if not np.isfinite(px) or px <= 0:
            continue
        sig = i - 1
        exited = False
        if shadow_in and exit_[sig]:
            gross, pnl = net_pnl(sh_px, px)
            if gross > 0 >= pnl:
                flips += 1
            shadow_trades.append({
                "entry_i": int(sh_i), "exit_i": int(i), "pnl": float(pnl), "gross": float(gross),
            })
            last_pnl = pnl
            shadow_in = False
            exited = True
        if real_in and exit_[sig]:
            gross, pnl = net_pnl(re_px, px)
            real_trades.append({
                "entry_i": int(re_i), "exit_i": int(i), "pnl": float(pnl), "gross": float(gross),
            })
            real_in = False
        if (not exited) and (not shadow_in) and bool(entry[sig]):
            prev = last_pnl
            shadow_in = True
            sh_px, sh_i = px, i
            take = (not ghost) or (prev is not None and prev < 0.0)
            if take and not real_in:
                real_in = True
                re_px, re_i = px, i
    if shadow_in:
        shadow_trades.append({"entry_i": int(sh_i), "exit_i": None, "pnl": None, "gross": None})
    if real_in:
        real_trades.append({"entry_i": int(re_i), "exit_i": None, "pnl": None, "gross": None})
    return {"held": held, "shadow": shadow_trades, "real": real_trades, "fee_sign_flips": flips}


def simulate_portfolio(asset_rets, held, cash_ret, fee, names):
    """等权于『下一交易日仍持有』的真实仓位。会员集合变化才再平衡，其间权重随收益漂移。
    费用在成交日从当日收益扣除，新权重从下一交易日起计提，与 engine accrual_mode=pre 一致。
    """
    asset_rets = np.asarray(asset_rets, float)
    held = np.asarray(held, bool)
    cash_ret = np.asarray(cash_ret, float)
    n, k = asset_rets.shape
    if held.shape != (n, k):
        raise ValueError("held shape mismatch")
    if held[0].any():
        raise ValueError("首日不能有持仓：成交不可能早于第一根 K 线")
    w = np.zeros(k)
    active = frozenset()
    rets = np.zeros(n)
    w_hist = np.zeros((n, k + 1))
    turnover = 0.0
    exposed = np.zeros(n, dtype=bool)
    for i in range(n):
        exposed[i] = float(w.sum()) > 1e-8
        grown = w * (1.0 + asset_rets[i])
        cash_g = (1.0 - float(w.sum())) * (1.0 + cash_ret[i])
        factor = float(grown.sum() + cash_g)
        if factor <= 0:
            raise RuntimeError("组合净值为非正，价格路径异常")
        day = factor - 1.0
        w = grown / factor
        nxt = active
        if i + 1 < n:
            nxt = frozenset(j for j in range(k) if held[i + 1, j])
        fee_notional = 0.0
        if nxt != active:
            target = np.zeros(k)
            if nxt:
                target[list(nxt)] = 1.0 / len(nxt)
            fee_notional = float(np.abs(target - w).sum())
            w = target
            active = nxt
        if np.any(w < -1e-10) or float(w.sum()) > 1.0 + 1e-8:
            raise RuntimeError("权重越界（只做多、不满仓杠杆）")
        day -= fee_notional * fee
        turnover += fee_notional
        rets[i] = day
        w_hist[i, :k] = w
        w_hist[i, k] = 1.0 - float(w.sum())
    return rets, w_hist, turnover, exposed


def align_asset(native_ret, panel_index):
    """原生收益 → 评估日历上的价格与收益。缺失日价格保持不变、收益记 0。
    窗口首日收益相对窗口前最后一根原生收盘，指标因此能用到窗口前的历史。"""
    r = native_ret.dropna().sort_index()
    r = r[~r.index.duplicated(keep="last")]
    px = (1.0 + r).cumprod()
    px_win = px.reindex(panel_index).ffill()
    ret = px_win.pct_change()
    prior = px[px.index < panel_index[0]]
    if len(prior) and pd.notna(px_win.iloc[0]):
        ret.iloc[0] = float(px_win.iloc[0] / float(prior.iloc[-1]) - 1.0)
    else:
        ret.iloc[0] = 0.0
    listed = px_win.notna().to_numpy()
    ret = ret.where(px_win.notna(), 0.0).fillna(0.0)
    return px, px_win.to_numpy(float), ret.to_numpy(float), listed


def cash_on(index, bond):
    repo_d = (1.0 + REPO_RATE) ** (1.0 / TRADING_DAYS) - 1.0
    b = bond.reindex(index).ffill().fillna(repo_d)
    return ((1.0 - CASH_BOND_PCT) * repo_d + CASH_BOND_PCT * b).to_numpy(float)


def yearly_returns(rets):
    s = pd.Series(rets)
    out = {}
    for y, sub in s.groupby(s.index.year):
        out[str(int(y))] = float((1.0 + sub).prod() - 1.0)
    return out


def trade_stats(trades):
    closed = [t for t in trades if t["exit_i"] is not None]
    if not closed:
        return {"n": 0, "n_open": sum(t["exit_i"] is None for t in trades),
                "win_rate": None, "avg_hold": None, "avg_pnl": None}
    wins = sum(t["pnl"] > 0 for t in closed)
    holds = [t["exit_i"] - t["entry_i"] for t in closed]
    return {
        "n": len(closed),
        "n_open": sum(t["exit_i"] is None for t in trades),
        "win_rate": wins / len(closed),
        "avg_hold": float(np.mean(holds)),
        "avg_pnl": float(np.mean([t["pnl"] for t in closed])),
    }


def run_ghost(native_map, panel_index, bond, entry_n, rsi_max, ghost, names):
    k = len(names)
    rets = np.zeros((len(panel_index), k))
    prices = np.zeros((len(panel_index), k))
    held = np.zeros((len(panel_index), k), dtype=bool)
    per, shadow_all, real_all = {}, [], []
    flips = 0
    for j, name in enumerate(names):
        native_px, px_w, ret_w, listed = align_asset(native_map[name], panel_index)
        entry, exit_ = signal_frame(native_px, entry_n, rsi_max)
        entry = entry.reindex(panel_index, fill_value=False).to_numpy(bool) & listed
        exit_ = exit_.reindex(panel_index, fill_value=False).to_numpy(bool) & listed
        # 未上市前价格为 nan，状态机会跳过成交
        book = run_books(np.where(listed, px_w, np.nan), entry, exit_, ghost)
        real_entries = {t["entry_i"] for t in book["real"]}
        shadow_entries = {t["entry_i"] for t in book["shadow"]}
        if not real_entries <= shadow_entries:
            raise RuntimeError(f"{name} 真实交易不是影子交易的子集")
        held[:, j] = book["held"]
        rets[:, j] = ret_w
        prices[:, j] = px_w
        st_r, st_s = trade_stats(book["real"]), trade_stats(book["shadow"])
        per[name] = {"real": st_r, "shadow": st_s}
        real_all.extend(book["real"])
        shadow_all.extend(book["shadow"])
        flips += book["fee_sign_flips"]
    cash = cash_on(panel_index, bond)
    port_rets, w_hist, turnover, exposed = simulate_portfolio(rets, held, cash, FEE, names)
    rs = pd.Series(port_rets, index=panel_index)
    wdf = pd.DataFrame(w_hist, index=panel_index, columns=list(names) + ["cash"])
    ev = evaluate({"rets": rs, "wealth": (1.0 + rs).cumprod(), "weights": wdf,
                   "turnover": float(turnover), "name": "ghost"})
    years = len(rs) / TRADING_DAYS
    real_st, sh_st = trade_stats(real_all), trade_stats(shadow_all)
    return {
        "cagr": ev["cagr"], "vol": ev["vol"], "max_dd": ev["max_dd"],
        "sharpe": ev["sharpe"], "calmar": ev["calmar"],
        "turnover": ev["turnover"],
        "turnover_ann": (ev["turnover"] / years) if years else None,
        "avg_equity": ev["avg_equity"],
        "time_in_mkt": float(exposed.mean()) if len(exposed) else None,
        "n_days": int(len(rs)),
        "start": str(panel_index.min().date()), "end": str(panel_index.max().date()),
        "n_trades": real_st["n"], "n_open": real_st["n_open"],
        "win_rate": real_st["win_rate"], "avg_hold": real_st["avg_hold"],
        "avg_pnl": real_st["avg_pnl"],
        "shadow_trades": sh_st["n"], "shadow_win_rate": sh_st["win_rate"],
        "shadow_avg_hold": sh_st["avg_hold"],
        "fee_sign_flips": int(flips),
        "yearly": yearly_returns(rs),
        "per_asset": per,
        "final_wealth": ev["final_wealth"],
    }


def slice_index(index, start):
    idx = index[index >= pd.Timestamp(start)]
    if len(idx) < 30:
        raise RuntimeError(f"窗口 {start} 样本过短: {len(idx)}")
    return idx


def run_v30(R, bond, start, cfg):
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(
        R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=None, name="v30", min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", REPO_RATE), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    ev = evaluate(res)
    rs = res["rets"]
    years = len(rs) / TRADING_DAYS
    return {
        "cagr": ev["cagr"], "vol": ev["vol"], "max_dd": ev["max_dd"],
        "sharpe": ev["sharpe"], "calmar": ev["calmar"],
        "turnover": ev["turnover"], "turnover_ann": ev["turnover"] / years,
        "avg_equity": ev["avg_equity"],
        "time_in_mkt": float((res["weights"][SLOTS].sum(axis=1) > 1e-8).mean()),
        "n_days": int(len(rs)),
        "start": str(rs.index.min().date()), "end": str(rs.index.max().date()),
        "yearly": yearly_returns(rs), "final_wealth": ev["final_wealth"],
    }


def run_bh(R, start, weights):
    """引擎 B7：首日建仓后 min_delta=1 不再平衡，单边万 5。满仓，无现金层。"""
    res = run_backtest(R, fixed_weights=weights, start=start, end=None,
                       name="BH", min_delta=1.0)
    ev = evaluate(res)
    rs = res["rets"]
    years = len(rs) / TRADING_DAYS
    return {
        "cagr": ev["cagr"], "vol": ev["vol"], "max_dd": ev["max_dd"],
        "sharpe": ev["sharpe"], "calmar": ev["calmar"],
        "turnover": ev["turnover"], "turnover_ann": ev["turnover"] / years,
        "avg_equity": ev["avg_equity"],
        "time_in_mkt": float((res["weights"][SLOTS].sum(axis=1) > 1e-8).mean()),
        "n_days": int(len(rs)),
        "start": str(rs.index.min().date()), "end": str(rs.index.max().date()),
        "yearly": yearly_returns(rs), "final_wealth": ev["final_wealth"],
    }


def fnum(x, nd=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def fpct(x, nd=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{nd}f}%"


def cell_passes_gate1(g, v30):
    dp = g["proxy"]["calmar"] - v30["proxy"]["calmar"]
    dr = g["real"]["calmar"] - v30["real"]["calmar"]
    ok = (np.isfinite(dp) and np.isfinite(dr) and dp >= ACCEPT["dcal"] and dr >= ACCEPT["dcal"]
          and g["proxy"]["max_dd"] > ACCEPT["proxy_mdd"]
          and g["real"]["max_dd"] > ACCEPT["real_mdd"])
    return bool(ok), float(dp), float(dr)


def evaluate_gates(default_runs, v30, grid):
    """三关。第二关只看预注册默认点的单轴邻居，不把网格赢家当成通过。"""
    g1, dp, dr = cell_passes_gate1(
        {"proxy": default_runs["proxy"], "real": default_runs["real"]}, v30)
    reasons = []
    if not (np.isfinite(dp) and dp >= ACCEPT["dcal"]):
        reasons.append(f"proxy ΔCalmar {dp:+.3f} < +{ACCEPT['dcal']:.2f}")
    if not (np.isfinite(dr) and dr >= ACCEPT["dcal"]):
        reasons.append(f"real ΔCalmar {dr:+.3f} < +{ACCEPT['dcal']:.2f}")
    if not (default_runs["proxy"]["max_dd"] > ACCEPT["proxy_mdd"]):
        reasons.append(f"proxy MDD {default_runs['proxy']['max_dd']*100:.2f}% 未优于 -10%")
    if not (default_runs["real"]["max_dd"] > ACCEPT["real_mdd"]):
        reasons.append(f"real MDD {default_runs['real']['max_dd']*100:.2f}% 未优于 -4.1%")

    neighbors = []
    for n, rsi in ((10, DEFAULT_RSI), (55, DEFAULT_RSI), (DEFAULT_N, 65), (DEFAULT_N, 75)):
        neighbors.append(grid[(n, rsi, True)])
    # 沿 N 与沿 RSI 的相邻落差（含默认点）
    cliff = False
    cliff_notes = []
    for axis, keys in (
        ("N", [(10, DEFAULT_RSI, True), (20, DEFAULT_RSI, True), (55, DEFAULT_RSI, True)]),
        ("RSI", [(DEFAULT_N, 65, True), (DEFAULT_N, 70, True), (DEFAULT_N, 75, True)]),
    ):
        seq = [grid[k]["proxy"]["calmar"] for k in keys]
        for a, b, ka, kb in zip(seq, seq[1:], keys, keys[1:]):
            if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > PLAT["cliff"]:
                cliff = True
                cliff_notes.append(f"{axis} {ka[0] if axis=='N' else ka[1]}→{kb[0] if axis=='N' else kb[1]} "
                                   f"proxy Calmar {a:.3f}→{b:.3f}")
    ok_n = 0
    neighbor_rows = []
    for n, rsi in ((10, DEFAULT_RSI), (55, DEFAULT_RSI), (DEFAULT_N, 65), (DEFAULT_N, 75)):
        cell = grid[(n, rsi, True)]
        d1 = cell["proxy"]["calmar"] - v30["proxy"]["calmar"]
        d2 = cell["real"]["calmar"] - v30["real"]["calmar"]
        hit = bool(np.isfinite(d1) and np.isfinite(d2) and d1 > PLAT["win"] and d2 > PLAT["win"])
        ok_n += int(hit)
        neighbor_rows.append({"entry_n": n, "rsi_max": rsi, "dcal_proxy": d1, "dcal_real": d2, "hit": hit})
    g2 = (not cliff) and ok_n >= PLAT["min_ok"]
    g2_reason = []
    if cliff:
        g2_reason.append("相邻参数 Calmar 落差 > 0.5：" + "；".join(cliff_notes))
    if ok_n < PLAT["min_ok"]:
        g2_reason.append(f"邻居中双窗口 ΔCalmar>{PLAT['win']:.2f} 的只有 {ok_n} 个，需要 ≥{PLAT['min_ok']}")

    oo = default_runs["oos"]["calmar"] - v30["oos"]["calmar"]
    g3 = bool(np.isfinite(oo) and oo >= -OOS_BAD)
    g3_reason = None if g3 else (
        f"OOS Calmar {default_runs['oos']['calmar']:.3f}，v30 {v30['oos']['calmar']:.3f}，"
        f"差 {oo:+.3f}（低于 v30 超过 {OOS_BAD:.2f} 即未过）")

    any_pass = []
    for key, cell in grid.items():
        ok, dpx, drl = cell_passes_gate1(cell, v30)
        oo_c = cell["oos"]["calmar"] - v30["oos"]["calmar"]
        oos_ok = bool(np.isfinite(oo_c) and oo_c >= -OOS_BAD)
        if ok and oos_ok:
            any_pass.append({"entry_n": key[0], "rsi_max": key[1], "ghost": key[2],
                             "dcal_proxy": dpx, "dcal_real": drl, "dcal_oos": oo_c})
    return {
        "gate1": g1, "gate1_reasons": reasons, "dcal_proxy": dp, "dcal_real": dr,
        "gate2": g2, "gate2_reasons": g2_reason, "neighbor_hits": ok_n,
        "neighbors": neighbor_rows, "cliff": cliff,
        "gate3": g3, "gate3_reason": g3_reason, "dcal_oos": oo,
        "all_pass": bool(g1 and g2 and g3),
        "grid_cells_passing_g1_and_g3": any_pass,
    }


def _row(label, window, s):
    return (f"| {label} | {window} | {fpct(s['cagr'])} | {fpct(s['max_dd'])} | {fnum(s['sharpe'])} | "
            f"{fnum(s['calmar'])} | {s.get('n_trades', '—')} | {fpct(s.get('win_rate'))} | "
            f"{fnum(s.get('avg_hold'), 1)} | {fnum(s.get('turnover_ann'), 2)} | {fpct(s.get('time_in_mkt'), 1)} |")


def write_report(payload):
    g, ng, v30, bh = payload["ghost"], payload["noghost"], payload["v30"], payload["bh"]
    gates = payload["gates"]
    meta = payload["meta"]
    verdict = "三关全过" if gates["all_pass"] else "未通过三关"
    lines = []
    a = lines.append
    a("# 幽灵交易系统纸面回测")
    a("")
    a(f"**结论：预注册默认参数（20 日新高、RSI(9)<70、幽灵过滤开启）{verdict}，不能替换 v30。** "
      f"proxy 年化 {fpct(g['proxy']['cagr'])}，与 v30 的 {fpct(v30['proxy']['cagr'])} 接近，"
      f"但最大回撤 {fpct(g['proxy']['max_dd'])}（v30 {fpct(v30['proxy']['max_dd'])}），"
      f"Calmar {fnum(g['proxy']['calmar'])} 对 {fnum(v30['proxy']['calmar'])}。"
      f"real 窗口年化 {fpct(g['real']['cagr'])}（v30 {fpct(v30['real']['cagr'])}），"
      f"OOS 年化 {fpct(g['oos']['cagr'])}（v30 {fpct(v30['oos']['cagr'])}）。"
      f"本实验只读 `final_cfg_v30.json`，没有改默认策略，也没有实盘下单。")
    a("")
    a("数据 " + meta["proxy_start"] + " → " + meta["data_end"]
      + f"（proxy {g['proxy']['n_days']} 个交易日；real {g['real']['n_days']} 日；"
      + f"OOS {g['oos']['n_days']} 日）。费用单边万 5。空仓现金 = 25% 逆回购 2.2% + 75% 国债 ETF 511010。"
      + "信号当日收盘确认，下一交易日收盘成交，成交当日收益仍归旧仓。")
    a("")
    a("## 规则")
    a("")
    a("- 只做多。空头规则没有实现。")
    a("- 入场同时满足：EMA9 > EMA19，RSI(9) 未超买，收盘创前 N 日新高。出场：收盘跌破前 20 日最低收盘。N 与 RSI 阈值见敏感性表，出场窗口固定 20。")
    a("- 影子账户每笔信号都做。真实账户只在最近一笔**已平**影子交易净盈亏 < 0 时接下一次信号。还没有任何已平影子交易时，真实账户不做。净盈亏含双边万 5。")
    a("- 组合权重 = 当前有真实持仓的 ETF 等权，其余现金。持仓集合不变时不每日再平衡。")
    a("- 对照：同一规则关掉幽灵过滤、v30（同一窗口、引擎原样）、五只等权买入持有（引擎 B7，建仓后不再平衡）。")
    a("")
    a("## 主结果")
    a("")
    a("换手/年 = 全样本 Σ|Δ权重| ÷ 年数（与引擎的换手同口径，再按年化）。胜率、平均持仓只统计已平真实交易，盈亏是该 ETF 自身的双边费用后收益，不是组合贡献。在场 = 当日收益由持仓计提的天数占比。")
    a("")
    a("| 方案 | 窗口 | CAGR | MaxDD | Sharpe | Calmar | 笔数 | 胜率 | 平均持仓(日) | 换手/年 | 在场 |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab, block in (("幽灵·默认", g), ("无幽灵过滤", ng), ("v30", v30), ("等权买入持有", bh)):
        for wlab, key in (("proxy 全历史", "proxy"), ("real 2025-04 起", "real"), ("OOS 2022 起", "oos")):
            a(_row(lab, wlab, block[key]))
    a("")
    a("Sharpe 无风险利率 1.8%，与 `engine.evaluate` 一致。v30 与买入持有没有『笔数/胜率/持仓天数』——它们不是这笔突破系统的回合交易。"
      f"v30 在场 100% 是因为底仓使权益权重始终大于 0，平均权益只有 {fpct(v30['proxy']['avg_equity'], 0)}（proxy）；"
      f"幽灵默认的平均权益是 {fpct(g['proxy']['avg_equity'], 0)}，和在场比例相同，有仓的日子几乎满仓。")
    a("")
    a(f"等权买入持有 proxy 年化 {fpct(bh['proxy']['cagr'])}、回撤 {fpct(bh['proxy']['max_dd'])}、Calmar {fnum(bh['proxy']['calmar'])}："
      f"收益高于幽灵，回撤更深，Calmar 仍低于幽灵的 proxy {fnum(g['proxy']['calmar'])}。"
      f"real 与 OOS 上买持的 Calmar（{fnum(bh['real']['calmar'])}、{fnum(bh['oos']['calmar'])}）都高于幽灵。")
    a("")
    # filter isolation
    a("## 幽灵过滤本身")
    a("")
    a("下表是默认参数下，打开过滤相对关掉过滤的差。过滤的设计假设是：上一笔影子亏损之后，下一笔更值得做，所以真实胜率应高于影子胜率。")
    a("")
    a("| 窗口 | ΔCAGR | ΔMaxDD（正=回撤变浅） | ΔCalmar | 真实笔数 / 影子笔数 | 真实胜率 | 影子胜率 |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for wlab, key in (("proxy", "proxy"), ("real", "real"), ("OOS", "oos")):
        a("| {w} | {c} | {d} | {k} | {nt}/{ns} | {wr} | {sw} |".format(
            w=wlab,
            c=fpct(g[key]["cagr"] - ng[key]["cagr"]),
            d=fpct(g[key]["max_dd"] - ng[key]["max_dd"]),
            k=fnum(g[key]["calmar"] - ng[key]["calmar"]),
            nt=g[key]["n_trades"], ns=g[key]["shadow_trades"],
            wr=fpct(g[key]["win_rate"]), sw=fpct(g[key]["shadow_win_rate"]),
        ))
    a("")
    flips = g["proxy"]["fee_sign_flips"]
    a(f"proxy 上，费用把毛利由正打成净亏的影子交易有 {flips} 笔（过滤用的是净盈亏）。"
      f"真实交易占影子交易的比例：proxy {g['proxy']['n_trades']}/{g['proxy']['shadow_trades']}，"
      f"real {g['real']['n_trades']}/{g['real']['shadow_trades']}，"
      f"OOS {g['oos']['n_trades']}/{g['oos']['shadow_trades']}。")
    a("")
    a(f"胜率没有按过滤的假设抬起来。proxy 真实胜率 {fpct(g['proxy']['win_rate'])}，影子 {fpct(g['proxy']['shadow_win_rate'])}，"
      f"差约 2 个百分点；109 笔、胜率五成时标准误大约 4.8 个百分点，这个差看不出来。"
      f"real 窗口真实胜率 {fpct(g['real']['win_rate'])}，低于影子的 {fpct(g['real']['shadow_win_rate'])}（只有 {g['real']['n_trades']} 笔）。"
      f"OOS 真实 {fpct(g['oos']['win_rate'])}、影子 {fpct(g['oos']['shadow_win_rate'])}，样本 {g['oos']['n_trades']} 笔。"
      f"proxy 上 Calmar 从 {fnum(ng['proxy']['calmar'])} 到 {fnum(g['proxy']['calmar'])}，对应的是在场从 {fpct(ng['proxy']['time_in_mkt'], 0)} 降到 {fpct(g['proxy']['time_in_mkt'], 0)}，少挨了几段回撤，不是胜率变高。"
      f"real 窗口过滤把 Calmar 从 {fnum(ng['real']['calmar'])} 打到 {fnum(g['real']['calmar'])}。")
    a("")
    a("## 单 ETF 与宽基")
    a("")
    a("单 ETF 是该标的 100% / 现金，规则与组合相同，默认参数。『样本内最好』按 proxy Calmar 从五只里挑出，**只是描述，不是预先指定的交易标的**。宽基是 510300 沪深300ETF 净值，同一套规则和三个日期窗口（它没有 proxy/real 两层数据）。")
    a("")
    a("| 标的 | 过滤 | CAGR | MaxDD | Sharpe | Calmar | 笔数 | 胜率 | 持仓日 | 在场 |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    best = payload["best_etf"]
    for code in SLOTS:
        s = payload["singles"][code]["ghost"]["proxy"]
        mark = " ←样本内最好" if code == best else ""
        a(f"| {code} {NAMES[code]}{mark} | 开 | {fpct(s['cagr'])} | {fpct(s['max_dd'])} | {fnum(s['sharpe'])} | {fnum(s['calmar'])} | {s['n_trades']} | {fpct(s['win_rate'])} | {fnum(s['avg_hold'], 1)} | {fpct(s['time_in_mkt'], 1)} |")
    a("")
    bs = payload["singles"][best]
    bb = payload["single_bh"][best]
    a(f"样本内最好的 {best} {NAMES[best]}，三个窗口（幽灵 / 无过滤 / 买入持有）：")
    a("")
    a("| 窗口 | 幽灵 CAGR | 幽灵 MDD | 幽灵 Calmar | 无过滤 Calmar | 买入持有 CAGR | 买入持有 MDD |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for wlab, key in (("proxy", "proxy"), ("real", "real"), ("OOS", "oos")):
        a(f"| {wlab} | {fpct(bs['ghost'][key]['cagr'])} | {fpct(bs['ghost'][key]['max_dd'])} | {fnum(bs['ghost'][key]['calmar'])} | {fnum(bs['noghost'][key]['calmar'])} | {fpct(bb[key]['cagr'])} | {fpct(bb[key]['max_dd'])} |")
    a("")
    a("510300 沪深300ETF：")
    a("")
    a("| 窗口 | 幽灵 CAGR | MDD | Calmar | 笔数 | 胜率 | 无过滤 Calmar | 买入持有 CAGR | 买入持有 MDD |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    br = payload["broad"]
    for wlab, key in (("proxy 同期", "proxy"), ("real 同期", "real"), ("OOS", "oos")):
        cal = fnum(br["ghost"][key]["calmar"])
        if br["ghost"][key]["n_trades"] == 0:
            cal += "†"
        a(f"| {wlab} | {fpct(br['ghost'][key]['cagr'])} | {fpct(br['ghost'][key]['max_dd'])} | {cal} | {br['ghost'][key]['n_trades']} | {fpct(br['ghost'][key]['win_rate'])} | {fnum(br['noghost'][key]['calmar'])} | {fpct(br['bh'][key]['cagr'])} | {fpct(br['bh'][key]['max_dd'])} |")
    a("")
    a(f"510300 净值文件止于 {br['ghost']['proxy']['end']}，五只 ETF 的面板止于 {meta['data_end']}，宽基少了尾部大约三个月。"
      f"real 同期真实交易 {br['ghost']['real']['n_trades']} 笔、在场 {fpct(br['ghost']['real']['time_in_mkt'], 0)}，"
      f"年化 {fpct(br['ghost']['real']['cagr'])}、回撤 {fpct(br['ghost']['real']['max_dd'])} 来自现金层（逆回购 + 511010）。"
      f"标 † 的 Calmar 是空仓对上国债 ETF 的微小回撤，不能当成系统有效。")
    a("")
    a("## 分年收益（proxy）")
    a("")
    years = sorted(set(g["proxy"]["yearly"]) | set(v30["proxy"]["yearly"]))
    a("| 年 | 幽灵 | 无过滤 | v30 | 等权买持 |")
    a("|---|---:|---:|---:|---:|")
    for y in years:
        label = y
        if y == years[0]:
            label = f"{y}（自 {meta['proxy_start'][5:]}）"
        elif y == years[-1]:
            label = f"{y}（至 {meta['data_end'][5:]}）"
        a(f"| {label} | {fpct(g['proxy']['yearly'].get(y))} | {fpct(ng['proxy']['yearly'].get(y))} | {fpct(v30['proxy']['yearly'].get(y))} | {fpct(bh['proxy']['yearly'].get(y))} |")
    a("")
    a("2014 与 2026 不是完整日历年。")
    wealth = 1.0
    for v in g["proxy"]["yearly"].values():
        wealth *= 1.0 + v
    y2015 = g["proxy"]["yearly"].get("2015")
    if y2015 is not None and wealth > 1:
        without = wealth / (1.0 + y2015)
        share = float(np.log(1.0 + y2015) / np.log(wealth))
        a(f"2015 年幽灵 {fpct(y2015)}，占全历史对数收益的 {share*100:.0f}%。去掉这一年，期末财富从 {wealth:.2f} 降到 {without:.2f}。")
    a("")
    a("## 参数敏感性")
    a("")
    a("3×3×2 = 18 格，全部是任务事先写明的邻域（N=10/20/55，RSI 上限 65/70/75，过滤开/关），不是搜索出来的。表内是 Calmar。默认格加粗。")
    a("")
    a("| N | RSI | 过滤 | proxy | real | OOS |")
    a("|---:|---:|---|---:|---:|---:|")
    for n in ENTRY_NS:
        for rsi in RSI_MAXS:
            for ghost_on in (True, False):
                cell = payload["grid_rows"][f"{n}|{rsi}|{int(ghost_on)}"]
                mark = "**开（默认）**" if (n, rsi, ghost_on) == (DEFAULT_N, DEFAULT_RSI, True) else ("开" if ghost_on else "关")
                def _c(window, bold=False):
                    t = fnum(cell[window]["calmar"])
                    if cell[window]["n_trades"] == 0:
                        t += "†"
                    return f"**{t}**" if bold else t
                bold = (n, rsi, ghost_on) == (DEFAULT_N, DEFAULT_RSI, True)
                a(f"| {n} | {rsi} | {mark} | {_c('proxy', bold)} | {_c('real', bold)} | {_c('oos', bold)} |")
    a("")
    a("† 该窗口真实交易 0 笔。Calmar 是现金层收益除以国债 ETF 的微小回撤，不是择时结果。")
    a("")
    cals = [payload["grid_rows"][f"{n}|{rsi}|1"]["proxy"]["calmar"] for n in ENTRY_NS for rsi in RSI_MAXS]
    cals = [c for c in cals if np.isfinite(c)]
    spread = (max(cals) - min(cals)) if cals else float("nan")
    best_key, best_cal = None, -1e9
    real_filter_better = []
    real_traded_worse = 0
    real_traded = 0
    for n in ENTRY_NS:
        for rsi in RSI_MAXS:
            gc = payload["grid_rows"][f"{n}|{rsi}|1"]
            nc = payload["grid_rows"][f"{n}|{rsi}|0"]
            if gc["proxy"]["calmar"] is not None and gc["proxy"]["calmar"] > best_cal:
                best_cal = gc["proxy"]["calmar"]
                best_key = (n, rsi)
            g_real, n_real = gc["real"], nc["real"]
            if g_real["n_trades"] and n_real["calmar"] is not None and g_real["calmar"] is not None:
                real_traded += 1
                if g_real["calmar"] > n_real["calmar"]:
                    real_filter_better.append((n, rsi))
                else:
                    real_traded_worse += 1
            elif g_real["n_trades"] == 0 and n_real["n_trades"]:
                real_filter_better.append((n, rsi, "empty"))
    rsi65 = payload["grid_rows"]["20|65|1"]["proxy"]["calmar"]
    a(f"过滤开启的 9 格里，proxy Calmar 从 {fnum(min(cals))} 到 {fnum(max(cals))}，极差 {spread:.2f}。"
      f"默认格{'就是' if payload['default_is_grid_max'] else '不是'}这 9 格里最高的"
      + ("" if payload["default_is_grid_max"] else f"（最高在 N={best_key[0]}、RSI<{best_key[1]}，Calmar {fnum(best_cal)}）")
      + f"。把 RSI 上限从 70 收到 65，proxy Calmar {fnum(g['proxy']['calmar'])} → {fnum(rsi65)}，"
      f"落差 {abs(g['proxy']['calmar'] - rsi65):.2f}，贴着平台规则的 0.5 悬崖线。")
    empty_wins = [x for x in real_filter_better if len(x) == 3]
    traded_wins = [x for x in real_filter_better if len(x) == 2]
    a(f"过滤相对无过滤：proxy Calmar 更高的 {payload['filter_proxy_wins']}/9，"
      f"OOS 更高的 {payload['filter_oos_wins']}/9，real 数字更高的 {payload['filter_real_wins']}/9。"
      f"real 上真正有交易、且过滤 Calmar 仍更高的格子是 {len(traded_wins)} 个；"
      f"有交易但过滤更差的是 {real_traded_worse} 个。"
      + (f"另外 {len(empty_wins)} 格过滤侧是 0 笔交易，Calmar 来自现金层，不能算过滤有效。" if empty_wins else ""))
    a("")
    a("## 三关")
    a("")
    a("判定与 `scripts/exp_opt31.py` 里 v30 的升级线相同，基准是本次用同一数据重跑的 v30，不是 README 里的旧打印数。")
    a("")
    a(f"1. **双窗口**（proxy 与 real 的 Calmar 都要 ≥ v30 + 0.15，且 proxy MDD 浅于 -10%、real MDD 浅于 -4.1%）："
      f"{'通过' if gates['gate1'] else '未过'}。"
      + ("；".join(gates["gate1_reasons"]) if gates["gate1_reasons"] else "两项 ΔCalmar 与两条回撤线都满足。"))
    a(f"2. **平台**（默认点的四个单轴邻居里，至少 3 个双窗口 ΔCalmar > 0.10，且 N 轴、RSI 轴相邻 Calmar 落差不超过 0.5）："
      f"{'通过' if gates['gate2'] else '未过'}。"
      + ("；".join(gates["gate2_reasons"]) if gates["gate2_reasons"] else "邻居与落差都在线内。"))
    a(f"3. **OOS**（proxy 自 2022-01-04 起，指标用得到之前的价格，但影子盈亏状态从零开始，与 v30 重开状态变量的方式一致。Calmar 不得低于 v30 OOS − 0.05）："
      f"{'通过' if gates['gate3'] else '未过'}。"
      + (gates["gate3_reason"] or f"ΔCalmar {gates['dcal_oos']:+.3f}。"))
    a("")
    n_luck = len(gates["grid_cells_passing_g1_and_g3"])
    if n_luck:
        a(f"18 格里有 {n_luck} 格同时满足第一关和第三关的数字。这不构成通过：格子是事后看见的，第二关也不允许把尖峰当成平台。默认策略保持 v30。")
    else:
        a("18 格里没有任何一格同时满足第一关和第三关。不存在一个『只是没被选为默认』的隐藏赢家。")
    a("")
    a("## 过拟合与做不到的事")
    a("")
    a("- 默认参数是任务给定的，不是这 18 格里挑出来的。即便如此，N 与 RSI 各只有三个点，平台很窄；极差大就说明收益对参数敏感，不能外推。")
    a("- 真实交易次数少。proxy 109 笔的胜率标准误大约 5 个百分点，过滤相对影子只高约 2 个百分点；real 窗口过滤后的胜率低于影子。过滤没有表现出它声称的『亏完下一笔更值得做』。")
    a("- 『上一笔亏了下一笔就做』假定交易盈亏负相关。趋势突破没有这个结构。本样本里它主要是少做交易、降低在场时间；这个效果在 proxy 上看起来减了回撤，在 real 窗口是减了收益。")
    a("- real 窗口大约一年半，并且落在 2025–2026。这个窗口的 Calmar 不能当成系统的长期水平。")
    a("- 五只里挑 Calmar 最高的一只是样本内选择。宽基 510300 是同一规则的外部对照，不是第二个优化目标。")
    a("- 等权于『当前有仓的标的』意味着只持有一只时仓位是 100%。回撤里有很大一部分是集中度，不是过滤能消掉的。")
    a("- 回测没有涨跌停、QDII 溢价、停牌顺延。万 5 是佣金加滑点的合并假设，和 v30 相同，不是券商账单。")
    a("")
    a("## 复现")
    a("")
    a("```bash")
    a("python3 scripts/exp_ghost_trader.py")
    a("```")
    a("")
    a("脚本把同一份数字写入 `references/exp_ghost_trader.json` 和本文件。`out/` 在 `.gitignore` 里，不作为记录。")
    a("")
    text = "\n".join(lines)
    with open(REF_MD, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        x = float(o)
        return None if not np.isfinite(x) else x
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    return o


def self_test():
    # 唐奇安：新高 / 跌破用前 N 根，不含当日
    px = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 6], dtype=float)
    prior_high = px.shift(1).rolling(3, min_periods=3).max()
    prior_low = px.shift(1).rolling(3, min_periods=3).min()
    assert bool(px.iloc[3] > prior_high.iloc[3])
    assert not bool(px.iloc[4] > prior_high.iloc[4] and px.iloc[4] <= prior_high.iloc[4])
    assert float(prior_high.iloc[4]) == 4
    assert bool(px.iloc[7] < prior_low.iloc[7])
    assert float(prior_low.iloc[7]) == 3

    # 未来数据不改历史信号
    base = pd.Series(np.linspace(100, 130, 80) + np.sin(np.arange(80)), dtype=float)
    e1, x1 = signal_frame(base, 20, 70)
    spiked = pd.concat([base, pd.Series([1000.0])], ignore_index=True)
    e2, x2 = signal_frame(spiked, 20, 70)
    assert np.array_equal(e1.to_numpy(), e2.to_numpy()[:-1])
    assert np.array_equal(x1.to_numpy(), x2.to_numpy()[:-1])

    # 幽灵过滤：亏→做，赢→跳过；第一笔没有前序，不做
    n = 12
    price = np.array([10, 10, 10, 9, 9, 9, 12, 12, 12, 11, 11, 10], float)
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    entry[[0, 4, 8]] = True
    exit_[[2, 6, 10]] = True
    on = run_books(price, entry, exit_, ghost=True)
    off = run_books(price, entry, exit_, ghost=False)
    real_on = [t for t in on["real"] if t["exit_i"] is not None]
    real_off = [t for t in off["real"] if t["exit_i"] is not None]
    sh = [t for t in on["shadow"] if t["exit_i"] is not None]
    assert len(sh) == 3, sh
    assert sh[0]["pnl"] < 0 and sh[1]["pnl"] > 0 and sh[2]["pnl"] < 0
    assert len(real_on) == 1 and real_on[0]["entry_i"] == 5 and real_on[0]["pnl"] > 0
    assert len(real_off) == 3
    # 成交在信号下一根，收益从再下一根开始：入场成交 i=5，持有 i=6 和 i=7（出场成交）
    assert on["held"][5] == False and on["held"][6] and on["held"][7] and not on["held"][8]

    # 同一根同时有进出场：平掉，不在同一价格上重开
    entry2 = np.zeros(8, dtype=bool)
    exit2 = np.zeros(8, dtype=bool)
    entry2[1] = True
    exit2[3] = True
    entry2[3] = True
    px2 = np.array([10, 10, 11, 12, 9, 9, 9, 9], float)
    both = run_books(px2, entry2, exit2, ghost=False)
    closed = [t for t in both["real"] if t["exit_i"] is not None]
    assert len(closed) == 1 and closed[0]["exit_i"] == 4
    assert not both["held"][5]

    # 组合：成交日扣费，收益日与 held 对齐
    rets = np.zeros((10, 1))
    rets[6, 0] = 0.10
    rets[7, 0] = -0.05
    held = np.zeros((10, 1), dtype=bool)
    held[6:8, 0] = True
    cash = np.zeros(10)
    pr, _, turnover, exposed = simulate_portfolio(rets, held, cash, 0.0005, ["A"])
    assert abs(pr[5] + 0.0005) < 1e-12
    assert abs(pr[6] - 0.10) < 1e-12
    assert abs(pr[7] - (-0.05 - 0.0005)) < 1e-12
    assert exposed[6] and exposed[7] and not exposed[5]
    assert abs(turnover - 2.0) < 1e-12

    # 两只同时持有：从现金到 50/50，名义换手为 1
    rets2 = np.zeros((6, 2))
    rets2[3, 0] = 0.02
    rets2[3, 1] = 0.04
    held2 = np.zeros((6, 2), dtype=bool)
    held2[3, :] = True
    pr2, wh, to2, _ = simulate_portfolio(rets2, held2, np.zeros(6), 0.0005, ["A", "B"])
    assert abs(pr2[2] + 0.0005) < 1e-12
    # 当日既计提收益，也在收盘平回现金，费用打在同一天
    assert abs(pr2[3] - (0.5 * 0.02 + 0.5 * 0.04 - 0.0005)) < 1e-12
    assert abs(to2 - 2.0) < 1e-12
    assert wh[3, 2] > 0.99
    print("[self-test] ok")


def main():
    self_test()
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = json.load(open(CFG_FILE, encoding="utf-8"))
    assert cfg.get("version") == "v30"
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    broad_ret = rets_from(read_table("510300_nav.csv"), "cum_nav")

    windows = {
        "proxy": (R, slice_index(R.index, PROXY_START)),
        "real": (Rr, slice_index(Rr.index, REAL_START)),
        "oos": (R, slice_index(R.index, OOS_START)),
    }

    def native_of(frame):
        return {s: frame[s].dropna() for s in SLOTS}

    grid = {}
    print("[grid] 18 cells × 3 windows")
    for entry_n in ENTRY_NS:
        for rsi_max in RSI_MAXS:
            for ghost in (True, False):
                cell = {}
                for wname, (frame, idx) in windows.items():
                    cell[wname] = run_ghost(native_of(frame), idx, bond, entry_n, rsi_max, ghost, SLOTS)
                grid[(entry_n, rsi_max, ghost)] = cell
                p = cell["proxy"]
                print(f"  N={entry_n:<2} RSI<{rsi_max:<2} ghost={int(ghost)}  "
                      f"proxy Cal {p['calmar']:.2f} CAGR {p['cagr']*100:.1f}% MDD {p['max_dd']*100:.1f}%  "
                      f"trades {p['n_trades']}/{p['shadow_trades']}")

    print("[v30 + buyhold]")
    v30, bh = {}, {}
    for wname, start in (("proxy", PROXY_START), ("real", REAL_START), ("oos", OOS_START)):
        frame = Rr if wname == "real" else R
        v30[wname] = run_v30(frame, bond, start, cfg)
        bh[wname] = run_bh(frame, start, {s: 20 for s in SLOTS})
        print(f"  {wname} v30 Cal {v30[wname]['calmar']:.2f} CAGR {v30[wname]['cagr']*100:.1f}% "
              f"MDD {v30[wname]['max_dd']*100:.1f}% | BH Cal {bh[wname]['calmar']:.2f}")

    print("[singles + 510300]")
    singles = {}
    for code in SLOTS:
        singles[code] = {}
        for ghost, tag in ((True, "ghost"), (False, "noghost")):
            singles[code][tag] = {}
            for wname, (frame, idx) in windows.items():
                singles[code][tag][wname] = run_ghost(
                    {code: frame[code].dropna()}, idx, bond, DEFAULT_N, DEFAULT_RSI, ghost, [code])
        print(f"  {code} ghost proxy Cal {singles[code]['ghost']['proxy']['calmar']:.2f} "
              f"trades {singles[code]['ghost']['proxy']['n_trades']}")
    best = max(SLOTS, key=lambda c: (
        singles[c]["ghost"]["proxy"]["calmar"]
        if np.isfinite(singles[c]["ghost"]["proxy"]["calmar"]) else -1e9,
        singles[c]["ghost"]["proxy"]["cagr"],
    ))
    single_bh = {}
    for code in SLOTS:
        weights = {s: (100 if s == code else 0) for s in SLOTS}
        single_bh[code] = {}
        for wname, start in (("proxy", PROXY_START), ("real", REAL_START), ("oos", OOS_START)):
            frame = Rr if wname == "real" else R
            single_bh[code][wname] = run_bh(frame, start, weights)

    broad_idx = {
        "proxy": slice_index(broad_ret.index, PROXY_START),
        "real": slice_index(broad_ret.index, REAL_START),
        "oos": slice_index(broad_ret.index, OOS_START),
    }
    broad = {"ghost": {}, "noghost": {}, "bh": {}}
    for ghost, tag in ((True, "ghost"), (False, "noghost")):
        for wname, idx in broad_idx.items():
            broad[tag][wname] = run_ghost(
                {BROAD_CODE: broad_ret}, idx, bond, DEFAULT_N, DEFAULT_RSI, ghost, [BROAD_CODE])
    for wname, idx in broad_idx.items():
        # 买入持有：首日现金，当日收盘买入，之后一直持有。费用万 5，空仓才用现金层。
        n = len(idx)
        held = np.zeros((n, 1), dtype=bool)
        held[1:, 0] = True
        native_px, px_w, ret_w, listed = align_asset(broad_ret, idx)
        cash = cash_on(idx, bond)
        pr, wh, to, exposed = simulate_portfolio(ret_w.reshape(-1, 1), held, cash, FEE, [BROAD_CODE])
        rs = pd.Series(pr, index=idx)
        wdf = pd.DataFrame(wh, index=idx, columns=[BROAD_CODE, "cash"])
        ev = evaluate({"rets": rs, "wealth": (1 + rs).cumprod(), "weights": wdf,
                       "turnover": float(to), "name": "bh"})
        years = len(rs) / TRADING_DAYS
        broad["bh"][wname] = {
            "cagr": ev["cagr"], "max_dd": ev["max_dd"], "sharpe": ev["sharpe"], "calmar": ev["calmar"],
            "turnover": ev["turnover"], "turnover_ann": ev["turnover"] / years,
            "time_in_mkt": float(exposed.mean()), "avg_equity": ev["avg_equity"],
            "n_days": int(len(rs)), "n_trades": None, "win_rate": None, "avg_hold": None,
            "yearly": yearly_returns(rs),
        }

    ghost = grid[(DEFAULT_N, DEFAULT_RSI, True)]
    noghost = grid[(DEFAULT_N, DEFAULT_RSI, False)]
    gates = evaluate_gates(ghost, v30, grid)

    def _strip(block):
        """json 里保留指标，去掉不可序列化的残留；per_asset 已是纯 dict。"""
        return block

    grid_rows = {}
    filter_wins = {"proxy": 0, "real": 0, "oos": 0}
    ghost_cals = []
    for entry_n in ENTRY_NS:
        for rsi_max in RSI_MAXS:
            for ghost_on in (True, False):
                cell = grid[(entry_n, rsi_max, ghost_on)]
                grid_rows[f"{entry_n}|{rsi_max}|{int(ghost_on)}"] = {
                    w: {k: cell[w][k] for k in
                        ("cagr", "max_dd", "sharpe", "calmar", "n_trades", "win_rate",
                         "avg_hold", "turnover_ann", "time_in_mkt", "shadow_trades", "shadow_win_rate")}
                    for w in ("proxy", "real", "oos")}
            gcell = grid[(entry_n, rsi_max, True)]
            ncell = grid[(entry_n, rsi_max, False)]
            ghost_cals.append(gcell["proxy"]["calmar"])
            for w in ("proxy", "real", "oos"):
                if np.isfinite(gcell[w]["calmar"]) and np.isfinite(ncell[w]["calmar"]):
                    if gcell[w]["calmar"] > ncell[w]["calmar"]:
                        filter_wins[w] += 1
    default_cal = ghost["proxy"]["calmar"]
    default_is_max = bool(np.isfinite(default_cal) and default_cal >= max(ghost_cals) - 1e-12)

    # 缺失收益诊断：窗口内各标的原生收益没有覆盖到的评估日
    miss = {}
    for wname, (frame, idx) in windows.items():
        miss[wname] = {s: int(frame[s].reindex(idx).isna().sum()) for s in SLOTS}

    payload = {
        "meta": {
            "proxy_start": PROXY_START, "real_start": REAL_START, "oos_start": OOS_START,
            "data_end": str(R.index.max().date()),
            "fee": FEE, "repo_rate": REPO_RATE, "cash_bond_pct": CASH_BOND_PCT,
            "cash_bond": "511010", "exit_n": EXIT_N,
            "ema": [EMA_FAST, EMA_SLOW], "rsi_n": RSI_N,
            "default": {"entry_n": DEFAULT_N, "rsi_max": DEFAULT_RSI, "ghost": True},
            "short_side": "disabled",
            "cfg_version": cfg.get("version"),
            "cfg_path": "references/final_cfg_v30.json",
            "missing_return_days": miss,
            "gates_source": "scripts/exp_opt31.py ACCEPT/PLAT/OOS_BAD",
            "note": "研究回测。未修改 final_cfg_v30.json，未修改 strategy.py / engine.py 的默认路径。",
        },
        "ghost": _strip(ghost),
        "noghost": _strip(noghost),
        "v30": v30,
        "bh": bh,
        "singles": singles,
        "single_bh": single_bh,
        "best_etf": best,
        "broad": broad,
        "grid_rows": grid_rows,
        "gates": gates,
        "filter_proxy_wins": filter_wins["proxy"],
        "filter_real_wins": filter_wins["real"],
        "filter_oos_wins": filter_wins["oos"],
        "default_is_grid_max": default_is_max,
    }
    blob = _clean(payload)
    for path in (REF_JSON, os.path.join(OUT_DIR, "exp_ghost_trader.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=1)
    write_report(blob)
    print(f"[best single by proxy Calmar] {best}")
    print(f"[gates] g1={gates['gate1']} g2={gates['gate2']} g3={gates['gate3']} all={gates['all_pass']}")
    print(f"[wrote] {REF_MD}")


if __name__ == "__main__":
    main()
