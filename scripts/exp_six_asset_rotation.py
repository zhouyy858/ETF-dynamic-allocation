# -*- coding: utf-8 -*-
"""六大类资产轮动纸面回测（研究专用，2026-11-09）。

不修改 references/final_cfg_v30.json，不改 DynamicStrategy / 引擎默认路径。
只读 v30 配置，用与 scripts/exp_opt31.py 相同的三关窗口和通过线做对照。

规则（只做多）:
  每个决策日对池内标的按 N 日总收益排名。
  空仓时只在「第 1 名且收盘 > SMA」时买入。
  持仓退出: 名次差于门槛，或收盘 < SMA；然后若新的第 1 名站上均线就换入，否则进 v30 现金袖套。
  主规则 exit_rank=2: 跌到第 2 名及更差就退出（只持有第 1 名），对应条文
  「drops to rank #2 or worse」。
  敏感性 exit_rank=3: 第 2 名仍可持有，跌出前两名才退出。

执行（与 v30 严格口径一致，timing="strict"）:
  决策日用截至前一交易日的收盘做排名和均线；当日收盘成交；
  先按旧仓计提当日收益，再扣费；新仓从下一交易日起计收益。
  费用单边万 5。空仓 = 25% 逆回购 2.2% + 75% 511010。
  timing="same_close" 只作对照: 用当日收盘发信号并假定能以该收盘价成交，仍从次日起计收益。

用法: python3 scripts/exp_six_asset_rotation.py
输出: references/exp_six_asset_rotation.md
      references/exp_six_asset_rotation.json
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from data_prep import read_table, rets_from, build_panel, build_returns
from engine import run_backtest, evaluate, FEE
from strategy import DynamicStrategy
from metrics import TRADING_DAYS

OUT_MD = os.path.join(ROOT, "references", "exp_six_asset_rotation.md")
OUT_JSON = os.path.join(ROOT, "references", "exp_six_asset_rotation.json")
CFG_FILE = os.path.join(ROOT, "references", "final_cfg_v30.json")

PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
CLAIM_START, CLAIM_END = "2021-03-01", "2026-07-31"
LIST_511260 = "2017-08-24"  # fetch_cash_bond.py: 此前为 511010 拼接，不作为可交易 10Y
FEE_ONE_WAY = FEE  # 0.0005
REPO = 0.022
CASH_BOND_PCT = 0.75
# 三关数字与 exp_opt31.py 相同。夏普仍走 evaluate() 默认 rf=1.8%，与该脚本一致。
ACCEPT = {"dcal": 0.15, "proxy_mdd": -0.10, "real_mdd": -0.041}
PLAT = {"win": 0.10, "cliff": 0.5, "min_ok": 3}
OOS_BAD = 0.05
# 卫星门槛在出数前写死，不按结果回改。10% 指每日再平衡到固定比例。
SAT_CAGR = 0.004
SAT_CORR = 0.80
SAT_REAL_SLACK = 0.01

BASE = {"lookback": 20, "ma": 28, "exit_rank": 2}
GRID_LB = (10, 20, 60)
GRID_MA = (20, 28, 60)
GRID_EXIT = (2, 3)

# 列顺序即排名并列时的决胜顺序（靠前优先）
CODES = ["hs300", "chinext", "cnet", "nasdaq", "gold", "cgb10"]
NAMES = {
    "hs300": "沪深300", "chinext": "创业板", "cnet": "中概/恒生科技",
    "nasdaq": "纳指100", "gold": "黄金", "cgb10": "10Y国债",
    "CASH": "现金袖套",
}


def _read_col(fn, col):
    df = pd.read_csv(os.path.join(ROOT, "assets", "data", fn), parse_dates=["date"])
    s = df.set_index("date")[col]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s.index.day_of_week < 5].astype(float)
    return s


def _on_calendar(s, calendar):
    s = s[~s.index.duplicated(keep="last")].sort_index()
    union = calendar.union(s.index).sort_values()
    return s.reindex(union).ffill().reindex(calendar)


def _level_from_rets(r, calendar):
    """把收益铺到交易日历上再累成价格，停牌日收益为 0。首个有效日前为 NaN。"""
    if len(r) == 0:
        return pd.Series(np.nan, index=calendar)
    rr = r.reindex(calendar)
    first = rr.first_valid_index()
    rr = rr.fillna(0.0)
    lv = (1.0 + rr).cumprod()
    if first is not None:
        lv.loc[lv.index < first] = np.nan
    return lv


def load_market():
    hs300 = _read_col("510300_nav.csv", "cum_nav")
    chi_idx = _read_col("index_sz399006.csv", "close")
    chi_etf = _read_col("159952_nav.csv", "cum_nav")
    cnet = _read_col("513180_nav.csv", "cum_nav")
    nasdaq_etf = _read_col("159941_nav.csv", "cum_nav")
    nasdaq_feed = _read_col("270042_nav.csv", "cum_nav")
    gold = _read_col("518880_nav.csv", "cum_nav")
    cgb = _read_col("511260_nav.csv", "cum_nav")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    px = _read_col("qdii_price_159941.csv", "close")
    nav_u = _read_col("159941_nav.csv", "unit_nav")

    # 共同截止: 黄金 / 沪深300 / 恒生科技净值都停在同一天，晚于这天的 v30 样本轮动看不到
    ends = {
        "510300": hs300.index.max(), "518880": gold.index.max(), "513180": cnet.index.max(),
        "511260": cgb.index.max(), "159941": nasdaq_etf.index.max(),
        "270042": nasdaq_feed.index.max(), "399006": chi_idx.index.max(),
        "159952": chi_etf.index.max(), "511010": bond.index.max(),
    }
    match_end = min(ends["510300"], ends["518880"], ends["513180"])

    calendar = hs300[(hs300.index >= "2013-01-01") & (hs300.index <= match_end)].index
    calendar = calendar.sort_values().unique()
    calendar = pd.DatetimeIndex(calendar)

    def put(s):
        return _on_calendar(s, calendar)

    cgb_lv = put(cgb)
    cgb_lv.loc[cgb_lv.index < LIST_511260] = np.nan  # 上市前是 511010 拼接，不入池

    proxy_lv = pd.DataFrame({
        "hs300": put(hs300),
        "chinext": put(chi_idx),
        "cnet": put(cnet),
        "nasdaq": put(nasdaq_feed),
        "gold": put(gold),
        "cgb10": cgb_lv,
    }, index=calendar)
    real_lv = proxy_lv.copy()
    real_lv["chinext"] = put(chi_etf)
    real_lv["nasdaq"] = put(nasdaq_etf)

    # 159941 场内价: 份额折算日用净值收益替换，避免把拆分当成暴跌
    both = pd.concat([px.rename("px"), nav_u.rename("nav")], axis=1, sort=True).dropna()
    pr, nr = both["px"].pct_change(), both["nav"].pct_change()
    bad = ((pr < -0.30) & (nr > -0.15)) | ((pr > 0.50) & (nr < 0.15))
    adj = pr.copy()
    adj.loc[bad] = nr.loc[bad]
    split_days = [str(d.date()) for d in bad[bad].index]
    px_r = adj.dropna()
    nav_r = put(nasdaq_etf).pct_change()
    spliced = nav_r.copy()
    px_on = px_r.reindex(calendar)
    use = px_on.notna()
    spliced.loc[use] = px_on.loc[use]
    px_lv = _level_from_rets(spliced.dropna(), calendar)
    # 信号用场内价、盈亏也用场内价
    price_lv = real_lv.copy()
    price_lv["nasdaq"] = px_lv

    prem = (both["px"] / both["nav"] - 1.0).drop(pd.Timestamp("2022-07-04"), errors="ignore")
    prem = _on_calendar(prem, calendar)
    # 决策日可用溢价 = T-2（与 v30 premium_shift=2 一致）。缺失不拦截。
    prem_known = prem.shift(2)

    # 与 v30 proxy 纳指是否同一条（270042 在重叠期压过 159941）
    v30_nq, _ = build_returns("159941", "proxy")
    mine = put(nasdaq_feed).pct_change()
    common = pd.concat([v30_nq.rename("v"), mine.rename("m")], axis=1, sort=True).dropna()
    nq_corr = float(common["v"].corr(common["m"])) if len(common) > 50 else float("nan")

    meta = {
        "match_end": str(pd.Timestamp(match_end).date()),
        "series_end": {k: str(pd.Timestamp(v).date()) for k, v in ends.items()},
        "calendar_start": str(calendar.min().date()),
        "calendar_end": str(calendar.max().date()),
        "n_days": int(len(calendar)),
        "split_days_replaced": split_days,
        "nasdaq_proxy_corr_vs_v30": nq_corr,
        "substitutions": [
            {"sleeve": "沪深300", "requested": "510300", "used": "510300_nav.cum_nav",
             "note": "代码在库内，净值覆盖 2012-05 起，本窗口够用。"},
            {"sleeve": "创业板", "requested": "159915", "used_proxy": "index_sz399006.close",
             "used_real": "159952_nav.cum_nav",
             "note": "159915 不在库。159952 是库内创业板 ETF，与 159915 都跟踪创业板指。proxy 用指数，与 v30 的 159952 代理相同；指数不含分红（创业板股息率<0.5%，与 data_prep 口径一致）。"},
            {"sleeve": "中国互联网", "requested": "513050", "used": "513180_nav.cum_nav",
             "note": "513050（中证海外中国互联网）不在库，也没有对应长指数。513180 恒生科技是库内最近的境外中国科技 ETF，2021-05-25 才上市，成分与中概互联不同，不能当成同一标的。"},
            {"sleeve": "纳指QDII", "requested": "159941",
             "used_proxy": "270042_nav.cum_nav", "used_real": "159941_nav.cum_nav",
             "note": "proxy 与 v30 相同：联接基金 270042 在重叠期覆盖 159941。159941 于 2015-06 上市，2014 窗口前半段交易的不是这只 ETF。场内价见 qdii_price_159941，拆分日已替换。"},
            {"sleeve": "黄金", "requested": "518880", "used": "518880_nav.cum_nav",
             "note": "代码在库内。净值停在与 510300、513180 相同的末日。"},
            {"sleeve": "10Y国债", "requested": "511260", "used": "511260_nav.cum_nav，2017-08-24 前剔除",
             "note": "文件在上市前用 511010 拼接。拿去排名会和现金袖套里的 511010 变成同一个资产，所以上市前不入池。空仓现金袖套始终是 25% 逆回购 + 75% 511010，不是 511260。"},
        ],
    }
    bundle = {
        "proxy": proxy_lv, "real": real_lv, "price": price_lv, "px_ret": px_r,
        "bond": bond, "prem": prem, "prem_known": prem_known, "meta": meta,
    }
    return bundle


def _arrays(levels, codes, lookback, ma):
    lv = levels[codes]
    mom = lv.pct_change(lookback)
    sma = lv.rolling(ma, min_periods=ma).mean()
    valid = (mom.notna() & sma.notna() & lv.notna()).to_numpy(dtype=bool)
    above = ((lv > sma) & mom.notna() & sma.notna()).to_numpy(dtype=bool)
    below = ((lv < sma) & mom.notna() & sma.notna()).to_numpy(dtype=bool)
    return mom.to_numpy(dtype=float), above, below, valid


def _decide(mom, above, below, valid, holding, exit_rank, require_full, blocked, codes):
    idx = np.flatnonzero(valid)
    n_elig = int(idx.size)
    if n_elig == 0 or (require_full and n_elig < len(codes)):
        return "CASH", n_elig
    order = np.lexsort((idx, -mom[idx]))
    ranked = idx[order]
    rank = {int(a): r + 1 for r, a in enumerate(ranked)}
    top = int(ranked[0])
    hold_i = None if holding == "CASH" else codes.index(holding)
    if hold_i is not None and hold_i in rank and rank[hold_i] < exit_rank and not below[hold_i]:
        new_i = hold_i
    else:
        new_i = top if above[top] else None
    nq = codes.index("nasdaq") if "nasdaq" in codes else None
    if new_i is not None and nq is not None and new_i == nq and hold_i != nq and blocked:
        new_i = None
    return ("CASH" if new_i is None else codes[new_i]), n_elig


def simulate(levels, bond_rets, lookback, ma, exit_rank, timing="strict",
             require_full=False, block_nasdaq=None, pnl_levels=None, codes=None,
             fee=FEE_ONE_WAY, repo=REPO, cash_bond_pct=CASH_BOND_PCT):
    """timing=strict: 决策日信号用昨日收盘（sig=i-1），今日收盘成交，新仓明日计收益。
    因此「赚到第 i 日收益的仓位」在第 i-1 日收盘确定，用的是截至 i-2 的价格。
    这与 engine 的 signal_lag=1 + accrual=pre + 收盘成交相同。
    """
    codes = list(codes or CODES)
    if timing not in ("strict", "same_close"):
        raise ValueError(timing)
    sig_lag = 1 if timing == "strict" else 0  # 决策日 i 使用的价格下标 = i - sig_lag
    dates = levels.index
    n = len(dates)
    mom, above, below, valid = _arrays(levels, codes, lookback, ma)
    px_lv = levels if pnl_levels is None else pnl_levels
    rt = px_lv[codes].pct_change().to_numpy(dtype=float)
    repo_d = (1.0 + repo) ** (1.0 / TRADING_DAYS) - 1.0
    bond = bond_rets.reindex(dates).ffill().fillna(repo_d).to_numpy(dtype=float)
    cash_r = (1.0 - cash_bond_pct) * repo_d + cash_bond_pct * bond
    if block_nasdaq is None:
        blocked = np.zeros(n, dtype=bool)
    else:
        b = block_nasdaq.reindex(dates)
        blocked = (b > 0).fillna(False).to_numpy(dtype=bool) if b.dtype != bool else b.fillna(False).to_numpy(dtype=bool)
        if not blocked.dtype == bool:
            blocked = np.nan_to_num(block_nasdaq.reindex(dates).to_numpy(dtype=float), nan=-1.0) > 0

    pos = "CASH"
    rets = np.zeros(n)
    earned = np.empty(n, dtype=object)
    turnover = np.zeros(n)
    n_elig = np.zeros(n, dtype=int)
    nan_hold = 0
    code_i = {c: k for k, c in enumerate(codes)}
    for i in range(n):
        if pos == "CASH":
            r = float(cash_r[i])
        else:
            r = rt[i, code_i[pos]]
            if not np.isfinite(r):
                r = 0.0
                nan_hold += 1
        sig = i - sig_lag
        if sig < 0 or sig >= n:
            new = "CASH"
            n_el = 0
        else:
            new, n_el = _decide(mom[sig], above[sig], below[sig], valid[sig], pos,
                                exit_rank, require_full, bool(blocked[i]), codes)
        traded = 0.0
        if new != pos:
            if pos != "CASH":
                traded += 1.0
            if new != "CASH":
                traded += 1.0
        rets[i] = r - traded * fee
        turnover[i] = traded
        earned[i] = pos
        n_elig[i] = n_el
        pos = new
    out = pd.DataFrame({"ret": rets, "pos": earned, "to": turnover, "n_elig_sig": n_elig}, index=dates)
    out.attrs["nan_hold"] = nan_hold
    return out


def slice_frame(df, start, end):
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def run_lengths(pos):
    lens = []
    prev, n = None, 0
    for p in pos:
        if p == "CASH":
            if prev is not None:
                lens.append(n)
            prev, n = None, 0
        elif p == prev:
            n += 1
        else:
            if prev is not None:
                lens.append(n)
            prev, n = p, 1
    if prev is not None:
        lens.append(n)
    return lens


def pack_metrics(df):
    r = df["ret"].astype(float)
    n_nan = int(r.isna().sum())
    r = r.fillna(0.0)
    pos = df["pos"]
    cash = (pos == "CASH").astype(float)
    wealth = (1.0 + r).cumprod()
    ev = evaluate({"rets": r, "wealth": wealth,
                   "weights": pd.DataFrame({"cash": cash.to_numpy()}, index=r.index),
                   "turnover": float(df["to"].sum()), "name": "rot"})
    lens = run_lengths(pos.tolist())
    holds = pos.value_counts(normalize=True)
    ev.update({
        "start": str(r.index.min().date()) if len(r) else None,
        "end": str(r.index.max().date()) if len(r) else None,
        "n": int(len(r)),
        "n_nan_ret": n_nan,
        "turnover_ann": float(ev["turnover"] / (len(r) / TRADING_DAYS)) if len(r) else np.nan,
        "time_in_market": float(1.0 - ev["avg_cash"]),
        "avg_hold_days": float(np.mean(lens)) if lens else 0.0,
        "n_round_trips": int(len(lens)),
        "holdings": {str(k): float(v) for k, v in holds.items()},
    })
    return ev


def yearly_returns(rets):
    rows = []
    r = rets.dropna()
    for y, sub in r.groupby(r.index.year):
        if len(sub) < 20:
            continue
        rows.append({"year": int(y), "ret": float((1.0 + sub).prod() - 1.0), "n": int(len(sub))})
    return rows


def valid_frame(levels, lookback, ma, codes=None):
    codes = list(codes or CODES)
    lv = levels[codes]
    mom = lv.pct_change(lookback)
    sma = lv.rolling(ma, min_periods=ma).mean()
    return mom.notna() & sma.notna() & lv.notna()


def overlay_nasdaq_price(levels, price_ret):
    """信号仍用 levels。盈亏里的纳指在有场内价的日子改用拆分调整后的价格收益，其余日子沿用原净值收益。"""
    lv = levels.copy()
    base_r = levels["nasdaq"].pct_change()
    use = price_ret.reindex(levels.index)
    spliced = base_r.copy()
    ok = use.notna()
    spliced.loc[ok] = use.loc[ok]
    lv["nasdaq"] = _level_from_rets(spliced.dropna(), levels.index)
    return lv


def first_full_date(levels, lookback, ma):
    v = valid_frame(levels, lookback, ma)
    ok = v.all(axis=1)
    if not ok.any():
        return None
    return ok[ok].index[0]


def coverage(levels, lookback, ma, start, end):
    v = valid_frame(levels, lookback, ma).loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(v) == 0:
        return {"n": 0, "frac_full": None, "mean_elig": None}
    return {"n": int(len(v)), "frac_full": float(v.all(axis=1).mean()),
            "mean_elig": float(v.sum(axis=1).mean())}


def run_v30(cfg, panel, start, end):
    ds = DynamicStrategy(panel, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    res = run_backtest(
        panel, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="v30", min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", REPO), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    ev = evaluate(res)
    ev["start"] = str(res["rets"].index.min().date())
    ev["end"] = str(res["rets"].index.max().date())
    ev["n"] = int(len(res["rets"]))
    ev["turnover_ann"] = float(ev["turnover"] / (ev["n"] / TRADING_DAYS))
    ev["time_in_market"] = float(ev["avg_equity"])
    return ev, res


def bh_stats(levels, start, end):
    out = {}
    rets = levels.pct_change()
    for c in CODES:
        sub = rets[c].loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
        if len(sub) < 40:
            continue
        w = (1.0 + sub).cumprod()
        ev = evaluate({"rets": sub, "wealth": w,
                       "weights": pd.DataFrame({"cash": np.zeros(len(sub))}, index=sub.index),
                       "turnover": 0.0, "name": c})
        out[c] = {"cagr": ev["cagr"], "mdd": ev["max_dd"], "sharpe": ev["sharpe"],
                  "calmar": ev["calmar"], "n": int(len(sub))}
    return out


def mix_stats(a, b, w_b):
    """a、b 为日收益。每日再平衡到 (1-w_b)*a + w_b*b。"""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 40:
        return None
    r = (1.0 - w_b) * df["a"] + w_b * df["b"]
    w = (1.0 + r).cumprod()
    ev = evaluate({"rets": r, "wealth": w,
                   "weights": pd.DataFrame({"cash": np.zeros(len(r))}, index=r.index),
                   "turnover": 0.0, "name": "mix"})
    ev["n"] = int(len(r))
    ev["corr"] = float(df["a"].corr(df["b"]))
    ev["start"] = str(r.index.min().date())
    ev["end"] = str(r.index.max().date())
    return ev


def _self_test():
    idx = pd.bdate_range("2020-01-01", periods=90)
    n = len(idx)
    a = 100.0 * (1.003) ** np.arange(n)
    b = np.full(n, 100.0)
    b[55:] = 100.0 * (1.02) ** np.arange(n - 55)
    c = np.full(n, 50.0)
    levels = pd.DataFrame({"A": a, "B": b, "C": c, "nasdaq": c, "gold": c, "cgb10": c}, index=idx)
    # 只用 A/B/C 三列做规则测试
    codes = ["A", "B", "C"]
    levels = levels[codes]
    bond = pd.Series(0.0, index=idx)
    kw = dict(lookback=10, ma=10, codes=codes)
    s1 = simulate(levels, bond, exit_rank=2, timing="strict", **kw)
    s3 = simulate(levels, bond, exit_rank=3, timing="strict", **kw)
    mom = levels.pct_change(10)
    sma = levels.rolling(10, min_periods=10).mean()
    above = levels > sma
    found = False
    for s in range(30, n - 3):
        if not (bool(above.iloc[s]["A"]) and bool(above.iloc[s]["B"])):
            continue
        if not (mom.iloc[s]["B"] > mom.iloc[s]["A"] > 0):
            continue
        earn = s + 2  # strict: 看到 s 的成交在 s+1，收益从 s+2 开始
        assert s1.iloc[earn]["pos"] == "B", (s, s1.iloc[earn]["pos"])
        assert s3.iloc[earn]["pos"] == "A", (s, s3.iloc[earn]["pos"])
        found = True
        break
    assert found, "rank-split day not found"

    # 跳空日 T 的收益不得被吃到。strict 最早吃到 T+2；same_close 最早吃到 T+1。
    idx2 = pd.bdate_range("2021-01-01", periods=40)
    aa = 100.0 * (1.0005) ** np.arange(40)
    bb = np.full(40, 100.0)
    bb[25:] = 140.0 * (1.002) ** np.arange(15)
    lv2 = pd.DataFrame({"A": aa, "B": bb, "C": np.full(40, 80.0)}, index=idx2)
    bond2 = pd.Series(0.0, index=idx2)
    st = simulate(lv2, bond2, lookback=5, ma=5, exit_rank=2, timing="strict", codes=["A", "B", "C"])
    sc = simulate(lv2, bond2, lookback=5, ma=5, exit_rank=2, timing="same_close", codes=["A", "B", "C"])
    T = 25
    assert st.iloc[T]["pos"] != "B" and st.iloc[T + 1]["pos"] != "B", st.iloc[T:T + 3]["pos"].tolist()
    assert st.iloc[T + 2]["pos"] == "B", st.iloc[T:T + 4]["pos"].tolist()
    assert sc.iloc[T]["pos"] != "B"
    assert sc.iloc[T + 1]["pos"] == "B", sc.iloc[T:T + 3]["pos"].tolist()

    # 第 1 名跌破均线 → 现金，不顺延第 2 名（第 2 名也在均线下）
    px = np.array([100 * (1.01) ** i for i in range(25)] + [160 * (0.97) ** i for i in range(20)], dtype=float)
    lv3 = pd.DataFrame({"A": px, "B": np.full(len(px), 100.0)}, index=pd.bdate_range("2022-01-03", periods=len(px)))
    # B 全程贴着均线（相等不算站上），A 下跌后仍可能暂时排名第一
    sm = simulate(lv3, pd.Series(0.0, index=lv3.index), lookback=5, ma=8, exit_rank=2,
                  timing="strict", codes=["A", "B"])
    sma3 = lv3["A"].rolling(8, min_periods=8).mean()
    crossed = None
    for s in range(10, len(lv3) - 3):
        if lv3["A"].iloc[s] < sma3.iloc[s] and lv3["A"].iloc[s - 1] >= sma3.iloc[s - 1]:
            crossed = s
            break
    assert crossed is not None
    assert sm.iloc[crossed + 2]["pos"] == "CASH", (crossed, sm.iloc[crossed:crossed + 4]["pos"].tolist())

    # 买入费记在成交日，成交日仍吃旧仓（现金）收益
    lv4 = pd.DataFrame({"A": 100 * (1.01) ** np.arange(30)}, index=pd.bdate_range("2023-01-02", periods=30))
    bond4 = pd.Series(0.0, index=lv4.index)
    sm4 = simulate(lv4, bond4, lookback=5, ma=5, exit_rank=2, timing="strict", codes=["A"],
                   repo=0.0, cash_bond_pct=0.0)
    buy = sm4.index[(sm4["pos"] == "CASH") & (sm4["to"] > 0)]
    assert len(buy) >= 1
    # repo=0 且债券收益 0 → 现金日毛收益 0，买入扣 5bps
    assert abs(sm4.loc[buy[0], "ret"] + FEE_ONE_WAY) < 1e-12, sm4.loc[buy[0], "ret"]
    print("[self-test] ok")


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(round(float(x), nd))


def slim(ev):
    keys = ["start", "end", "n", "cagr", "max_dd", "mdd_peak", "mdd_trough", "sharpe", "calmar",
            "vol", "turnover", "turnover_ann", "avg_cash", "avg_equity", "time_in_market",
            "avg_hold_days", "n_round_trips", "holdings", "n_nan_ret"]
    out = {}
    for k in keys:
        if k not in ev:
            continue
        v = ev[k]
        if k in ("mdd_peak", "mdd_trough"):
            out[k] = None if v is None else str(pd.Timestamp(v).date())
        elif k == "holdings":
            out[k] = {kk: _f(vv, 4) for kk, vv in v.items()}
        elif k in ("start", "end", "n", "n_round_trips", "n_nan_ret"):
            out[k] = v
        else:
            out[k] = _f(v, 6)
    return out


def pct(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:.2f}%"


def num(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def gate1(rot_p, rot_r, base_p, base_r):
    dcp = rot_p["calmar"] - base_p["calmar"]
    dcr = rot_r["calmar"] - base_r["calmar"]
    ok = (dcp >= ACCEPT["dcal"] and dcr >= ACCEPT["dcal"]
          and rot_p["max_dd"] > ACCEPT["proxy_mdd"] and rot_r["max_dd"] > ACCEPT["real_mdd"])
    return {"dcal_proxy": dcp, "dcal_real": dcr, "pass": bool(ok),
            "proxy_mdd_ok": bool(rot_p["max_dd"] > ACCEPT["proxy_mdd"]),
            "real_mdd_ok": bool(rot_r["max_dd"] > ACCEPT["real_mdd"]),
            "dcal_proxy_ok": bool(dcp >= ACCEPT["dcal"]),
            "dcal_real_ok": bool(dcr >= ACCEPT["dcal"])}


def gate2(rows):
    """邻域 = 主规则一次只动一个旋钮。悬崖看均线轴和回看轴的相邻格。"""
    by = {(r["lookback"], r["ma"], r["exit_rank"]): r for r in rows}
    neighbors = [(10, 28, 2), (60, 28, 2), (20, 20, 2), (20, 60, 2), (20, 28, 3)]
    detail = []
    n_ok = 0
    for key in neighbors:
        r = by[key]
        ok = r["dcal_proxy"] > PLAT["win"] and r["dcal_real"] > PLAT["win"]
        n_ok += int(ok)
        detail.append({"lookback": key[0], "ma": key[1], "exit_rank": key[2],
                       "dcal_proxy": r["dcal_proxy"], "dcal_real": r["dcal_real"], "wide_ok": ok,
                       "proxy_calmar": r["proxy"]["calmar"]})
    cliff = False
    cliff_notes = []
    for axis, keys in (
        ("ma", [(20, 20, 2), (20, 28, 2), (20, 60, 2)]),
        ("lookback", [(10, 28, 2), (20, 28, 2), (60, 28, 2)]),
    ):
        cals = [by[k]["proxy"]["calmar"] for k in keys]
        for i in range(len(cals) - 1):
            gap = abs(cals[i + 1] - cals[i])
            if gap > PLAT["cliff"]:
                cliff = True
                cliff_notes.append(f"{axis} {keys[i]}→{keys[i+1]} proxy Calmar 落差 {gap:.2f}")
    passed = (n_ok >= PLAT["min_ok"]) and (not cliff)
    return {"n_ok": n_ok, "need": PLAT["min_ok"], "cliff": cliff, "cliff_notes": cliff_notes,
            "pass": bool(passed), "neighbors": detail}


def judge(g1, g2, g3_pass, coverage_frac, mixes):
    """replace 要三关全过，且 proxy 窗口里六资产同时可交易的天数占比 >= 80%。
    覆盖率不够时，长样本不是这条规则的确认样本，不能升级默认策略。
    satellite 要 10% 混合在整段 proxy、纯六资产子样本、real 三段都过写死的门槛。
    """
    cov_ok = coverage_frac is not None and coverage_frac >= 0.80
    if g1["pass"] and g2["pass"] and g3_pass and cov_ok:
        return "replace", "三关全过，且 proxy 窗口六资产覆盖率≥80%。按仓库升级线可以替换 v30。本脚本仍不改配置。"
    reasons = []
    if not g1["pass"]:
        reasons.append("第一关未过（双窗口 Calmar 各 +0.15 且 proxy MDD 浅于 -10%、real MDD 浅于 -4.1%）")
    if not g2["pass"]:
        reasons.append("第二关平台未过（邻域至少 3 格双窗口 ΔCal>0.10，且均线/回看轴无 >0.5 的 Calmar 悬崖）")
    if not g3_pass:
        reasons.append("第三关未过（OOS Calmar 不得比 v30 差过 0.05）")
    if not cov_ok:
        reasons.append(f"proxy 窗口六资产同时可用的天数只占 {pct(coverage_frac)}，长样本不是完整六资产测试，不能据此替换 v30")
    sat_ok = True
    sat_why = []
    for name, need in (
        ("proxy", "整段"),
        ("proxy_full6", "纯六资产子样本"),
        ("real", "real"),
    ):
        m = mixes.get(name)
        if m is None or m.get("mix10") is None:
            sat_ok = False
            sat_why.append(f"{need}混合无法计算")
            continue
        mix, base = m["mix10"], m["v30"]
        if name == "real":
            if mix["max_dd"] < base["max_dd"] - SAT_REAL_SLACK:
                sat_ok = False
                sat_why.append(f"real 混合 MDD {pct(mix['max_dd'])} 比 v30 {pct(base['max_dd'])} 深过 1pp")
            if mix["cagr"] < base["cagr"] - SAT_REAL_SLACK:
                sat_ok = False
                sat_why.append(f"real 混合 CAGR {pct(mix['cagr'])} 比 v30 差过 1pp")
        else:
            if not (mix["max_dd"] > ACCEPT["proxy_mdd"]):
                sat_ok = False
                sat_why.append(f"{need} 10% 混合 MDD {pct(mix['max_dd'])} 破 -10% 线")
            if mix["cagr"] < base["cagr"] + SAT_CAGR:
                sat_ok = False
                sat_why.append(f"{need} 10% 混合 CAGR {pct(mix['cagr'])} 未高于 v30 {pct(base['cagr'])} 达 0.40pp")
            if mix["calmar"] < base["calmar"]:
                sat_ok = False
                sat_why.append(f"{need} 10% 混合 Calmar {num(mix['calmar'])} 低于 v30 {num(base['calmar'])}")
            corr = m["mix10"]["corr"]
            if corr is None or corr >= SAT_CORR:
                sat_ok = False
                sat_why.append(f"{need} 日收益相关 {num(corr)} 不低于 0.80")
    note = ""
    if g3_pass:
        note = "第三关过了（OOS 的 Calmar 没有比 v30 差过 0.05），前两关没过，不能升级。"
    if sat_ok and not (g1["pass"] and g2["pass"] and g3_pass and cov_ok):
        return "satellite", "不替换 v30。" + "；".join(reasons) + "。" + note + "10% 卫星混合过了写死门槛，只作为研究观察，不改默认配置。"
    return "reject", "不替换 v30，也不作为卫星。" + "；".join(reasons) + "。" + note + (
        ("卫星未过：" + "；".join(sat_why) + "。") if sat_why else "")


def row_line(label, ev):
    return (f"| {label} | {pct(ev['cagr'])} | {pct(ev['max_dd'])} | {num(ev['sharpe'])} | "
            f"{num(ev['calmar'])} | {num(ev['turnover_ann'])} | {pct(ev['time_in_market'])} |")


def render(doc):
    m = doc["meta"]
    v30, base = doc["v30"], doc["base"]
    g = doc["gates"]
    lines = []
    A = lines.append
    A("# 六大类资产轮动：纸面回测（不改 v30）")
    A("")
    A(f"**判决：{doc['verdict_zh']}。** {doc['verdict_reason']}")
    A("")
    A("这是研究脚本的一次性对照，不是新的默认策略。`references/final_cfg_v30.json` 与 `DynamicStrategy` 都没有改。")
    A("")
    A("## 规则怎么落地")
    A("")
    A("主规则（预先指定，不从网格里挑最好的一格）：20 个交易日总收益排名，只买第 1 名，且收盘高于 28 日简单均线；名次落到第 2 名及更差，或收盘跌破均线，就退出。新的第 1 名也站上均线才换入，否则 100% 现金袖套。没有合格标的也是 100% 现金袖套。不把第 2 名递补上来。")
    A("")
    A("条文写的是「跌到第 2 名或更差就退出」，也就是只持有第 1 名。另有一档敏感性：第 2 名继续持有，跌出前两名才退出。提示里「below #1 vs below #2 (as written)」和条文「rank #2 or worse」对不齐，本报告以条文原句为主规则，把「前两名都可持有」放到敏感性里。")
    A("")
    A("执行与 v30 严格口径相同，不是收盘后用当天收盘价立刻成交：")
    A("")
    A("- 决策日只用前一交易日收盘（排名、均线都截止到昨天）。")
    A("- 当天收盘价成交，单边费用万 5。从 A 换到 B 计双边，一进一出各万 5。")
    A("- 当天收益先按旧仓计提，新仓从下一交易日才开始计收益。信号收盘到新仓收益之间隔着成交日，跳空当天的收益吃不到。")
    A("- 现金袖套与 v30 相同：25% 国债逆回购（年化 2.2%，按 252 日复利）+ 75% 511010。空仓不是买入 511260。511260 只在它自己排到第 1 名且站上均线时作为风险资产满仓。")
    A("- 不做 v30 的 `min_delta` 过滤。这条规则是 0/100 切换，每次换仓都是满仓进出。")
    A("- 夏普的无风险利率沿用 `evaluate()` 默认的 1.8%，和 `exp_opt31.py` 同一函数。Calmar = CAGR / |MaxDD|。")
    A("")
    A("「收盘信号、假定能以该收盘价成交、次日才计收益」只作为对照行，不参与三关。它比仓库口径少一天滞后。")
    A("")
    A("## 标的替代与样本边界")
    A("")
    A("| 袖套 | 规则代码 | 实际用的序列 |")
    A("|---|---|---|")
    A("| 沪深300 | 510300 | `510300_nav` 累计净值（库内就有） |")
    A("| 创业板 | 159915 | **库内没有 159915。** proxy 用创业板指 `index_sz399006`（与 v30 的创业板代理相同）；real / 营销窗口用创业板 ETF `159952` 净值。两只 ETF 都跟踪创业板指。 |")
    A("| 中国互联网 | 513050 | **库内没有 513050，也没有中证海外中国互联网指数。** 用恒生科技 ETF `513180`（2021-05-25 上市）。这是境外中国科技，不是中概互联，成分不同。 |")
    A("| 纳指 QDII | 159941 | proxy 用联接基金 `270042` 净值，和 v30 proxy 同一条（相关 "
      f"{num(m['nasdaq_proxy_corr_vs_v30'], 4)}）。real / 营销窗口用 `159941` 净值。ETF 2015-06 才上市。 |")
    A("| 黄金 | 518880 | `518880_nav` |")
    A("| 10Y 国债 | 511260 | `511260_nav`，**2017-08-24 上市前剔除**。此前文件是 511010 拼接，拿来排名会和现金层撞车。 |")
    A("")
    A(f"轮动能用的共同末日是 **{m['match_end']}**。510300、518880、513180 的净值都停在这一天；v30 核心面板走到 {m['v30_proxy_end']}。对照全部截到 {m['match_end']}，否则会把 2026 年 8 月之后的行情（含 10 月创业板继续下跌、v30 触发深熊锁）算进只有一边看得到的样本。三关的起点与 `exp_opt31.py` 相同：proxy {PROXY_START}，real {REAL_START}，OOS {OOS_START}（OOS 用 proxy 面板，从该日重新起步的是 v30；轮动是连续持仓切出来的收益）。")
    A("")
    A(f"主规则在 proxy 窗口里，六只同时具备均线和动量的天数占 **{pct(doc['coverage']['proxy_base']['frac_full'])}**，平均在池个数 {num(doc['coverage']['proxy_base']['mean_elig'], 2)}。恒生科技要到 {doc['coverage']['first_full_proxy']} 之后六只才齐。2014–2021 这段不是六资产规则，只是「当时已经上市的那几只」。")
    A("")
    A("## 与 v30 对照（主规则：20 日 / MA28 / 只持有第 1 名）")
    A("")
    A("权益暴露：轮动是「当天满仓风险资产」的天数占比；v30 是平均权益权重（它很少空仓到 0，也不满仓到 100%）。换手是单边成交额 / 年（从 A 换到 B 记 2）。")
    A("")
    A("| 样本 | CAGR | MaxDD | Sharpe | Calmar | 单边换手/年 | 权益暴露 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    A(row_line("轮动 proxy", base["proxy"]))
    A(row_line("v30 proxy", v30["proxy"]))
    A(row_line("轮动 real（ETF 净值）", base["real"]))
    A(row_line("v30 real", v30["real"]))
    A(row_line("轮动 OOS（proxy，2022 起）", base["oos"]))
    A(row_line("v30 OOS", v30["oos"]))
    A(row_line("轮动 proxy 六资产齐了之后", base["proxy_full6"]))
    A(row_line("v30 同期", v30["proxy_full6"]))
    A("")
    A(f"v30 若放到各自面板最后一天（不截断）：proxy CAGR {pct(v30['proxy_open']['cagr'])} / MDD {pct(v30['proxy_open']['max_dd'])} / Calmar {num(v30['proxy_open']['calmar'])}；"
      f"real CAGR {pct(v30['real_open']['cagr'])} / MDD {pct(v30['real_open']['max_dd'])} / Calmar {num(v30['real_open']['calmar'])}。发布口径的差异主要来自 2026-08-03 之后的行情，不用于本次过关。")
    A("")
    A(f"real 窗口只有 {v30['real']['start']} 至 {v30['real']['end']}，不到一年半。v30 在这段的 Calmar {num(v30['real']['calmar'])} 来自 {pct(v30['real']['max_dd'])} 的回撤，对短样本极度敏感，不能外推成常态。")
    A("")
    A(f"主规则 proxy 回撤谷底 {base['proxy'].get('mdd_trough')}（峰 {base['proxy'].get('mdd_peak')}）。"
      f"平均一段持仓 {num(base['proxy'].get('avg_hold_days'), 1)} 个交易日，完整进出 {base['proxy'].get('n_round_trips')} 次。")
    A("")
    A("proxy 窗口持仓天数占比：")
    A("")
    holds = base["proxy"].get("holdings") or {}
    A("、".join(f"{NAMES.get(k, k)} {pct(v)}" for k, v in sorted(holds.items(), key=lambda kv: -kv[1])))
    A("")
    A("## 年度收益")
    A("")
    A("当年收益率，不是年化。样本年内不足 20 个交易日的年份略去。proxy 对照截到共同末日。2014 从 6 月 23 日算起，2026 停在共同末日，这两格都不是完整自然年。")
    A("")
    years = sorted(set([y["year"] for y in doc["yearly"]["rot_proxy"]] + [y["year"] for y in doc["yearly"]["v30_proxy"]]))
    rp = {y["year"]: y["ret"] for y in doc["yearly"]["rot_proxy"]}
    vp = {y["year"]: y["ret"] for y in doc["yearly"]["v30_proxy"]}
    rc = {y["year"]: y["ret"] for y in doc["yearly"]["rot_claim"]}
    A("| 年 | 轮动 proxy | v30 proxy | 轮动 ETF 净值（营销窗口内） |")
    A("|---:|---:|---:|---:|")
    for y in years:
        A(f"| {y} | {pct(rp.get(y))} | {pct(vp.get(y))} | {pct(rc.get(y))} |")
    A("")
    A("## 敏感性")
    A("")
    shallow = max(r["proxy"]["max_dd"] for r in doc["grid"])
    n_both_pos = sum(1 for r in doc["grid"] if r["dcal_proxy"] > 0 and r["dcal_real"] > 0)
    A("网格是事先写好的三档，不是搜索出来再当主规则。ΔCal 相对截断后的 v30。`proxy6` 是该组参数自己的六资产齐备日之后到共同末日。")
    A("")
    A(f"18 格里双窗口 ΔCal 同时为正的有 {n_both_pos} 格。proxy 回撤最浅的一格是 {pct(shallow)}，仍然深于 -10%。没有一格能靠换参数过第一关。回看从 10 到 20 再到 60，Calmar 落差超过 0.5，但这是问询里给定的粗网格，不是 `exp_opt31.py` 那种 1 个百分点的邻域。")
    A("")
    A("| 回看 | 均线 | 退出 | proxy CAGR | proxy MDD | proxy Cal | ΔCal p | real Cal | ΔCal r | OOS Cal | proxy6 Cal | 换手/年 | 在场 |")
    A("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in doc["grid"]:
        mark = " **主规则**" if r["is_base"] else ""
        exit_zh = "跌到第2名就换" if r["exit_rank"] == 2 else "跌出前2才换"
        A(f"| {r['lookback']} | {r['ma']} | {exit_zh}{mark} | {pct(r['proxy']['cagr'])} | {pct(r['proxy']['max_dd'])} | "
          f"{num(r['proxy']['calmar'])} | {r['dcal_proxy']:+.2f} | {num(r['real']['calmar'])} | {r['dcal_real']:+.2f} | "
          f"{num(r['oos']['calmar'])} | {num(r['proxy6']['calmar'])} | {num(r['proxy']['turnover_ann'])} | {pct(r['proxy']['time_in_market'])} |")
    A("")
    A("## 三关（与 exp_opt31.py 同一条线）")
    A("")
    A(f"第一关：proxy 与 real 的 Calmar 都至少比 v30 高 {ACCEPT['dcal']:.2f}，且 proxy MDD 浅于 {pct(ACCEPT['proxy_mdd'])}、real MDD 浅于 {pct(ACCEPT['real_mdd'])}。")
    A("")
    g1 = g["gate1"]
    A(f"- proxy ΔCal {g1['dcal_proxy']:+.3f}（{'过' if g1['dcal_proxy_ok'] else '不过'}），real ΔCal {g1['dcal_real']:+.3f}（{'过' if g1['dcal_real_ok'] else '不过'}）。")
    A(f"- proxy MDD {pct(base['proxy']['max_dd'])}（{'浅于 -10%' if g1['proxy_mdd_ok'] else '深于或等于 -10%，不过'}），real MDD {pct(base['real']['max_dd'])}（{'浅于 -4.1%' if g1['real_mdd_ok'] else '深于或等于 -4.1%，不过'}）。")
    A(f"- 第一关：**{'通过' if g1['pass'] else '不通过'}**。")
    A("")
    g2 = g["gate2"]
    A(f"第二关：主规则周围 5 个一次一旋钮的邻居里，双窗口 ΔCal > {PLAT['win']:.2f} 的有 {g2['n_ok']} 个，需要至少 {g2['need']} 个。均线轴与回看轴相邻 Calmar 落差不得超过 {PLAT['cliff']}。")
    if g2["cliff_notes"]:
        A("悬崖：" + "；".join(g2["cliff_notes"]) + "。")
    else:
        A("这两条轴上没有超过 0.5 的 Calmar 落差。")
    A(f"- 第二关：**{'通过' if g2['pass'] else '不通过'}**。一关没过时，`exp_opt31.py` 不会继续跑二关；这里照样算了，避免只报一个总开关。")
    A("")
    g3 = g["gate3"]
    A(f"第三关：OOS Calmar {num(g3['rot'])} 对比 v30 {num(g3['v30'])}，变差超过 {OOS_BAD:.2f} 即拒绝。")
    A(f"- 第三关：**{'通过' if g3['pass'] else '不通过'}**。")
    A("")
    A(f"三关合计：**{'全过' if g['all_pass'] else '未过'}**。")
    A("")
    A("## 营销数字：2021-03 至 2026-07")
    A("")
    A("声称大约是 CAGR 27.5%、MaxDD 21%、Calmar 0.88。按本仓库的定义 Calmar = CAGR / |MaxDD|，27.5% / 21% ≈ 1.31，对不上 0.88。若 Calmar 真是 0.88 且 CAGR 是 27.5%，隐含回撤约 31%；若回撤真是 21% 且 Calmar 是 0.88，隐含 CAGR 约 18.5%。下面不替这组数字圆场，只放同一窗口上测到的结果。")
    A("")
    A("测量用 ETF 净值面板（创业板用 159952，纳指用 159941 净值，互联网用 513180）。513180 的 28 日均线要到 "
      f"{doc['coverage']['first_full_real']} 才齐，所以 2021 年 3 月到大约 7 月并不是六只一起转。")
    A("")
    A("| 测法 | 区间 | CAGR | MaxDD | Sharpe | Calmar | 换手/年 | 在场 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, label in (
        ("claim_partial", "上市即可入池（齐之前只有 5 只）"),
        ("claim_strict", "未齐之前持有现金袖套"),
        ("claim_full6", "六只都齐之后到 2026-07-31"),
        ("claim_same_close", "对照：当日收盘发信号并成交（少 1 日滞后）"),
        ("claim_px", "159941 用场内价做信号和盈亏"),
    ):
        ev = doc["claim"][key]
        A(f"| {label} | {ev['start']} ~ {ev['end']} | {pct(ev['cagr'])} | {pct(ev['max_dd'])} | {num(ev['sharpe'])} | {num(ev['calmar'])} | {num(ev['turnover_ann'])} | {pct(ev['time_in_market'])} |")
    A("")
    A("同期单资产买入持有（不扣费，只作参照，六资产齐备日到 2026-07-31）：")
    A("")
    A("| 资产 | CAGR | MaxDD | Calmar |")
    A("|---|---:|---:|---:|")
    for c, ev in doc["bh_claim"].items():
        A(f"| {NAMES.get(c, c)} | {pct(ev['cagr'])} | {pct(ev['mdd'])} | {num(ev['calmar'])} |")
    A("")
    A("和声称对不齐的原因，按影响从大到小：")
    A("")
    A("1. 513050 不在库，513180 既晚上市又不是同一指数。2021-03 到均线预热完成之前，要么缺这只，要么持有现金。")
    A("2. 回测没有用信号当天的收盘价去吃当天涨跌。少一天滞后的对照在这个窗口上更差，不是更好，所以和营销数字的差距不能解释成「我们把成交做严了所以吃亏」。对照仍然不是「先看到收盘、再按这个收盘成交、还把当天收益记到新仓上」。")
    A("3. 费用按单边万 5 扣在换仓上。这条规则换手远高于 v30。")
    A("4. 159941 的净值收益不是场内成交价。溢价、折价和限购见下一节。")
    A("5. 创业板用的是 159952 / 创业板指，不是 159915 本身。同一指数下的差异远小于互联网这只替代。")
    A("6. 空仓有现金袖套收益，不是零收益，但也不会凭空造出权益牛市的年化。")
    A("")
    A("## 159941 溢价和限购")
    A("")
    A("净值序列里没有溢价，也没有「溢价高了买不进」。仓库手册的执行约束是溢价 >2% 停止买入、>5% 场内减量或暂缓。QDII 净值发布偏晚，拦截用的是决策日可见的 T-2 溢价，和 v30 `premium_shift=2` 相同。已持有的可以继续拿到卖出信号，限购限制的是申购方向。拆分过渡日 2022-07-04 从溢价序列里剔除；场内价在份额折算日改用净值收益，替换的日期："
      + ("、".join(m["split_days_replaced"]) if m["split_days_replaced"] else "无") + "。")
    A("")
    pr = doc["premium"]
    A(f"全样本未截断溢价（{pr['full']['start']} ~ {pr['full']['end']}，{pr['full']['n']} 天）：均值 {pct(pr['full']['mean'])}，中位数 {pct(pr['full']['median'])}，最高 {pct(pr['full']['max'])}，最低 {pct(pr['full']['min'])}。高于 2% 的天数 {pct(pr['full']['gt_2'])}，高于 5% 的 {pct(pr['full']['gt_5'])}，高于 10% 的 {pct(pr['full']['gt_10'])}。")
    A("")
    A(f"主规则 proxy 持有纳指的日子（{pr['held']['n']} 天）同期溢价均值 {pct(pr['held']['mean'])}，高于 2% 占 {pct(pr['held']['gt_2'])}，高于 5% 占 {pct(pr['held']['gt_5'])}。新开仓 {pr['entries']['n']} 次，其中 T-2 溢价已知且 >2% 的有 {pr['entries']['known_gt_2']} 次，>5% 的有 {pr['entries']['known_gt_5']} 次。")
    A("")
    A("| 处理 | 样本 | CAGR | MaxDD | Calmar |")
    A("|---|---|---:|---:|---:|")
    for key, label, sample in (
        ("block2_proxy", "T-2 溢价>2% 禁止新开纳指", "proxy"),
        ("block5_proxy", "T-2 溢价>5% 禁止新开纳指", "proxy"),
        ("px_pnl_proxy", "信号仍用净值，持有纳指时按场内价计收益", "proxy"),
        ("block2_claim", "T-2 溢价>2% 禁止新开纳指", "营销窗口 ETF"),
        ("px_signal_claim", "信号和收益都改用场内价", "营销窗口 ETF"),
    ):
        ev = pr["runs"][key]
        A(f"| {label} | {sample} | {pct(ev['cagr'])} | {pct(ev['max_dd'])} | {num(ev['calmar'])} |")
    A("")
    A("513180 也是 QDII，库里没有它的场内价，恒生科技的溢价和限购同样没有进净值曲线。上面的修正只覆盖 159941。")
    A("")
    A("## 若只做 10% 卫星")
    A("")
    A("混合是每日再平衡回固定比例，纸面结果会比买完放着更平滑。门槛写在脚本顶部：整段 proxy 和纯六资产子样本都要求混合 MDD 浅于 -10%、CAGR 至少比 v30 高 0.40 个百分点、Calmar 不低于 v30、日收益相关低于 0.80；real 混合的 CAGR 和 MDD 不得比 v30 差过 1 个百分点。")
    A("")
    A("| 样本 | 相关 | 10% CAGR | 10% MDD | 10% Cal | v30 CAGR | v30 MDD | v30 Cal | 20% MDD |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, label in (("proxy", "proxy 全段"), ("proxy_full6", "proxy 六资产齐之后"), ("real", "real")):
        mxx = doc["mix"][key]
        a, b, c = mxx["mix10"], mxx["v30"], mxx["mix20"]
        A(f"| {label} | {num(a['corr'])} | {pct(a['cagr'])} | {pct(a['max_dd'])} | {num(a['calmar'])} | "
          f"{pct(b['cagr'])} | {pct(b['max_dd'])} | {num(b['calmar'])} | {pct(c['max_dd'])} |")
    A("")
    A("## 判决")
    A("")
    A(doc["verdict_reason"])
    A("")
    if doc["verdict"] == "replace":
        A("按事先写死的线，这条规则可以替换 v30。配置文件仍然没有改，替换要另开变更。")
    elif doc["verdict"] == "satellite":
        A("可以放在观察名单上当小比例卫星看，不能进默认配置。纸面混合每天再平衡，而且互联网这只是恒生科技不是中概互联。")
    else:
        A("拒绝替换 v30，也拒绝放进默认组合当卫星。主规则在能对齐的长样本上 CAGR 更高，但最大回撤约 -27%，Calmar 低于 v30，换手大约是 v30 的八到九倍，在场时间超过 90%。18 格敏感性没有一格把回撤收到 -10% 以内。")
        A("")
        A("营销窗口（2021-03 至 2026-07）测到的是大约 18%–22% CAGR、最大回撤 -26%，不是声称的 27.5% 和 -21%。六资产齐了之后的 Calmar 约 0.74，数值上靠近声称的 0.88，但那是更低的收益配上更深的回撤，不是同一组结果。2025 年轮动收益高于任何一只的全年买入持有，是因为黄金和创业板的上涨月份错开，仍远低于事后按月挑选赢家。")
        A("")
        A("10% 资金每日再平衡进这条规则，长样本最大回撤会破 v30 的 -10% 线（测到约 -10.2%）。2021 年 7 月以后的子样本上，同样的 10% 混合回撤更浅、Calmar 更高，但事先写死的卫星条件要求整段和子样本都过，整段没过。")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("python3 scripts/exp_six_asset_rotation.py")
    A("```")
    A("")
    A("脚本先跑合成价格自测（排名退出、不吃信号日跳空、跌破均线回现金、费用记在旧仓），再跑网格。数字写在 `references/exp_six_asset_rotation.json`。")
    A("")
    return "\n".join(lines)


def main():
    _self_test()
    print("[data] loading")
    bundle = load_market()
    meta = bundle["meta"]
    match_end = meta["match_end"]
    cfg = json.load(open(CFG_FILE))
    print(f"[data] calendar {meta['calendar_start']}..{meta['calendar_end']} n={meta['n_days']} match_end={match_end}")
    print(f"[data] nasdaq proxy corr vs v30 build_returns = {meta['nasdaq_proxy_corr_vs_v30']:.6f}")
    assert meta["nasdaq_proxy_corr_vs_v30"] > 0.999

    print("[v30] proxy/real/oos + open end")
    R_proxy, _ = build_panel("proxy")
    R_real, _ = build_panel("real")
    v30 = {}
    v30_res = {}
    for key, panel, start, end in (
        ("proxy", R_proxy, PROXY_START, match_end),
        ("real", R_real, REAL_START, match_end),
        ("oos", R_proxy, OOS_START, match_end),
        ("proxy_open", R_proxy, PROXY_START, None),
        ("real_open", R_real, REAL_START, None),
    ):
        ev, res = run_v30(cfg, panel, start, end)
        v30[key] = ev
        v30_res[key] = res
        print(f"  v30 {key:<10} {ev['cagr']*100:6.2f}% {ev['max_dd']*100:7.2f}% Cal {ev['calmar']:.2f} "
              f"Sh {ev['sharpe']:.2f} TO/yr {ev['turnover_ann']:.1f} eq {ev['time_in_market']*100:.1f}%")

    bond = bundle["bond"]
    proxy_lv, real_lv, price_lv = bundle["proxy"], bundle["real"], bundle["price"]
    prem_known = bundle["prem_known"]

    def run_one(levels, lookback, ma, exit_rank, **kw):
        return simulate(levels, bond, lookback, ma, exit_rank, **kw)

    print("[grid] 18 x proxy/real")
    rows = []
    cache = {}
    for lb in GRID_LB:
        for ma in GRID_MA:
            for ex in GRID_EXIT:
                sp = run_one(proxy_lv, lb, ma, ex, timing="strict")
                sr = run_one(real_lv, lb, ma, ex, timing="strict")
                cache[(lb, ma, ex)] = (sp, sr)
                own_full = first_full_date(proxy_lv, lb, ma)
                evp = pack_metrics(slice_frame(sp, PROXY_START, match_end))
                evr = pack_metrics(slice_frame(sr, REAL_START, match_end))
                evo = pack_metrics(slice_frame(sp, OOS_START, match_end))
                ev6 = pack_metrics(slice_frame(sp, own_full, match_end)) if own_full is not None else None
                rec = {
                    "lookback": lb, "ma": ma, "exit_rank": ex,
                    "is_base": lb == BASE["lookback"] and ma == BASE["ma"] and ex == BASE["exit_rank"],
                    "first_full": None if own_full is None else str(pd.Timestamp(own_full).date()),
                    "proxy": evp, "real": evr, "oos": evo, "proxy6": ev6,
                    "dcal_proxy": evp["calmar"] - v30["proxy"]["calmar"],
                    "dcal_real": evr["calmar"] - v30["real"]["calmar"],
                }
                rows.append(rec)
                print(f"  lb{lb:>2} ma{ma:>2} exit{ex}  pCal {evp['calmar']:.2f} ({rec['dcal_proxy']:+.2f}) "
                      f"MDD {evp['max_dd']*100:.1f}%  rCal {evr['calmar']:.2f} ({rec['dcal_real']:+.2f}) "
                      f"OOS {evo['calmar']:.2f}")

    base_sp, base_sr = cache[(BASE["lookback"], BASE["ma"], BASE["exit_rank"])]
    ff_p = first_full_date(proxy_lv, BASE["lookback"], BASE["ma"])
    ff_r = first_full_date(real_lv, BASE["lookback"], BASE["ma"])
    cov_p = coverage(proxy_lv, BASE["lookback"], BASE["ma"], PROXY_START, match_end)

    # v30 切到纯六资产子样本（连续区间，不删中间日）
    v30_full6_rets = v30_res["proxy"]["rets"]
    v30_full6_rets = v30_full6_rets.loc[pd.Timestamp(ff_p):pd.Timestamp(match_end)]
    v30["proxy_full6"] = pack_metrics(pd.DataFrame({
        "ret": v30_full6_rets, "pos": "CASH", "to": 0.0,
    }))
    # pack_metrics 把 pos 全记成现金会把权益暴露打成 0。v30 这段改用引擎权重。
    w_full = v30_res["proxy"]["weights"].loc[v30_full6_rets.index]
    # 重新用 evaluate，权益用真实权重
    wf = (1.0 + v30_full6_rets).cumprod()
    ev6 = evaluate({"rets": v30_full6_rets, "wealth": wf, "weights": w_full,
                    "turnover": float("nan"), "name": "v30"})
    # 换手只在子样本上没有逐日分解（引擎给的是全程合计）。用权重日变化近似。
    dw = w_full.drop(columns=["cash"]).diff().abs().sum(axis=1).fillna(0.0)
    ev6["turnover"] = float(dw.sum())
    ev6["turnover_ann"] = float(ev6["turnover"] / (len(v30_full6_rets) / TRADING_DAYS))
    ev6["time_in_market"] = float(ev6["avg_equity"])
    ev6["start"] = str(v30_full6_rets.index.min().date())
    ev6["end"] = str(v30_full6_rets.index.max().date())
    ev6["n"] = int(len(v30_full6_rets))
    v30["proxy_full6"] = ev6

    base_metrics = {
        "proxy": pack_metrics(slice_frame(base_sp, PROXY_START, match_end)),
        "real": pack_metrics(slice_frame(base_sr, REAL_START, match_end)),
        "oos": pack_metrics(slice_frame(base_sp, OOS_START, match_end)),
        "proxy_full6": pack_metrics(slice_frame(base_sp, ff_p, match_end)),
    }

    # 营销窗口与执行对照
    print("[claim] etf panel variants")
    claim_partial = pack_metrics(slice_frame(base_sr, CLAIM_START, CLAIM_END))
    strict_r = run_one(real_lv, BASE["lookback"], BASE["ma"], BASE["exit_rank"],
                       timing="strict", require_full=True)
    claim_strict = pack_metrics(slice_frame(strict_r, CLAIM_START, CLAIM_END))
    claim_full6 = pack_metrics(slice_frame(strict_r, ff_r, CLAIM_END))
    same_r = run_one(real_lv, BASE["lookback"], BASE["ma"], BASE["exit_rank"], timing="same_close")
    claim_same = pack_metrics(slice_frame(same_r, CLAIM_START, CLAIM_END))
    price_sig = run_one(price_lv, BASE["lookback"], BASE["ma"], BASE["exit_rank"], timing="strict")
    claim_px = pack_metrics(slice_frame(price_sig, CLAIM_START, CLAIM_END))

    # 溢价
    print("[premium]")
    prem = bundle["prem"]
    held_idx = slice_frame(base_sp, PROXY_START, match_end)
    held_mask = held_idx["pos"] == "nasdaq"
    held_prem = prem.reindex(held_idx.index)[held_mask].dropna()
    # 开仓：赚到收益的第一天的前一日是成交日
    pos = held_idx["pos"]
    entries = (pos == "nasdaq") & (pos.shift(1).fillna("CASH") != "nasdaq")
    entry_trade_dates = entries[entries].index
    known_on_entry = []
    for dt in entry_trade_dates:
        loc = held_idx.index.get_loc(dt)
        if loc == 0:
            continue
        trade_dt = held_idx.index[loc - 1]
        if trade_dt in prem_known.index and np.isfinite(prem_known.loc[trade_dt]):
            known_on_entry.append(float(prem_known.loc[trade_dt]))

    def prem_desc(s):
        if len(s) == 0:
            return {"n": 0, "mean": None, "median": None, "max": None, "min": None,
                    "gt_2": None, "gt_5": None, "gt_10": None, "start": None, "end": None}
        return {"n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
                "max": float(s.max()), "min": float(s.min()),
                "gt_2": float((s > 0.02).mean()), "gt_5": float((s > 0.05).mean()),
                "gt_10": float((s > 0.10).mean()),
                "start": str(s.index.min().date()), "end": str(s.index.max().date())}

    def blocked_series(thr):
        return prem_known.where(prem_known.notna(), -1.0) > thr

    b2p = run_one(proxy_lv, **BASE, timing="strict", block_nasdaq=blocked_series(0.02))
    b5p = run_one(proxy_lv, **BASE, timing="strict", block_nasdaq=blocked_series(0.05))
    b2c = run_one(real_lv, **BASE, timing="strict", block_nasdaq=blocked_series(0.02))
    # 信号仍走 proxy 净值/指数；只有纳指的盈亏在有场内价的日子改用拆分调整后的价格收益
    pnl_proxy_lv = overlay_nasdaq_price(proxy_lv, bundle["px_ret"])
    px_pnl = run_one(proxy_lv, **BASE, timing="strict", pnl_levels=pnl_proxy_lv)
    premium_runs = {
        "block2_proxy": pack_metrics(slice_frame(b2p, PROXY_START, match_end)),
        "block5_proxy": pack_metrics(slice_frame(b5p, PROXY_START, match_end)),
        "px_pnl_proxy": pack_metrics(slice_frame(px_pnl, PROXY_START, match_end)),
        "block2_claim": pack_metrics(slice_frame(b2c, CLAIM_START, CLAIM_END)),
        "px_signal_claim": claim_px,
    }

    # 混合。轮动用自己的日历，v30 用面板日历，内连接。
    print("[mix]")
    mixes = {}
    pairs = {
        "proxy": (v30_res["proxy"]["rets"], slice_frame(base_sp, PROXY_START, match_end)["ret"], v30["proxy"]),
        "proxy_full6": (v30_full6_rets, slice_frame(base_sp, ff_p, match_end)["ret"], v30["proxy_full6"]),
        "real": (v30_res["real"]["rets"], slice_frame(base_sr, REAL_START, match_end)["ret"], v30["real"]),
    }
    for key, (va, rb, v30_ev) in pairs.items():
        m10 = mix_stats(va, rb, 0.10)
        m20 = mix_stats(va, rb, 0.20)
        mixes[key] = {"v30": v30_ev, "mix10": m10, "mix20": m20}
        print(f"  mix {key} corr {m10['corr']:.2f} 10% {m10['cagr']*100:.2f}%/{m10['max_dd']*100:.2f}% "
              f"Cal {m10['calmar']:.2f} vs v30 {v30_ev['cagr']*100:.2f}%/{v30_ev['max_dd']*100:.2f}%")

    g1 = gate1(base_metrics["proxy"], base_metrics["real"], v30["proxy"], v30["real"])
    g2 = gate2(rows)
    g3_pass = base_metrics["oos"]["calmar"] >= v30["oos"]["calmar"] - OOS_BAD
    g3 = {"rot": base_metrics["oos"]["calmar"], "v30": v30["oos"]["calmar"], "pass": bool(g3_pass)}
    verdict, reason = judge(g1, g2, g3_pass, cov_p["frac_full"], mixes)
    verdict_zh = {"replace": "替换 v30", "satellite": "不替换，只可观察为卫星", "reject": "拒绝"}[verdict]
    print(f"[verdict] {verdict} {reason}")

    bh = bh_stats(real_lv, ff_r, CLAIM_END)

    def clean_ev(ev):
        return slim(ev)

    doc = {
        "meta": meta,
        "verdict": verdict,
        "verdict_zh": verdict_zh,
        "verdict_reason": reason,
        "coverage": {
            "proxy_base": cov_p,
            "first_full_proxy": str(pd.Timestamp(ff_p).date()),
            "first_full_real": str(pd.Timestamp(ff_r).date()),
        },
        "v30": {k: clean_ev(v) for k, v in v30.items()},
        "base": {k: clean_ev(v) for k, v in base_metrics.items()},
        "grid": [{
            "lookback": r["lookback"], "ma": r["ma"], "exit_rank": r["exit_rank"],
            "is_base": r["is_base"], "first_full": r["first_full"],
            "dcal_proxy": _f(r["dcal_proxy"], 4), "dcal_real": _f(r["dcal_real"], 4),
            "proxy": clean_ev(r["proxy"]), "real": clean_ev(r["real"]),
            "oos": clean_ev(r["oos"]), "proxy6": clean_ev(r["proxy6"]),
        } for r in rows],
        "gates": {"gate1": {k: (_f(v, 4) if isinstance(v, float) else v) for k, v in g1.items()},
                  "gate2": g2, "gate3": {**g3, "rot": _f(g3["rot"], 4), "v30": _f(g3["v30"], 4)},
                  "all_pass": bool(g1["pass"] and g2["pass"] and g3_pass)},
        "claim": {
            "claim_partial": clean_ev(claim_partial),
            "claim_strict": clean_ev(claim_strict),
            "claim_full6": clean_ev(claim_full6),
            "claim_same_close": clean_ev(claim_same),
            "claim_px": clean_ev(claim_px),
        },
        "bh_claim": bh,
        "premium": {
            "full": prem_desc(prem.dropna()),
            "held": prem_desc(held_prem),
            "entries": {
                "n": int(entries.sum()),
                "known_n": len(known_on_entry),
                "known_gt_2": int(sum(x > 0.02 for x in known_on_entry)),
                "known_gt_5": int(sum(x > 0.05 for x in known_on_entry)),
            },
            "runs": {k: clean_ev(v) for k, v in premium_runs.items()},
        },
        "mix": {},
        "yearly": {
            "rot_proxy": yearly_returns(slice_frame(base_sp, PROXY_START, match_end)["ret"]),
            "v30_proxy": yearly_returns(v30_res["proxy"]["rets"]),
            "rot_claim": yearly_returns(slice_frame(base_sr, CLAIM_START, CLAIM_END)["ret"]),
        },
    }
    for key, mxx in mixes.items():
        doc["mix"][key] = {
            "v30": clean_ev(mxx["v30"]),
            "mix10": clean_ev(mxx["mix10"]),
            "mix20": clean_ev(mxx["mix20"]),
        }
        doc["mix"][key]["mix10"]["corr"] = _f(mxx["mix10"]["corr"], 4)
        doc["mix"][key]["mix20"]["corr"] = _f(mxx["mix20"]["corr"], 4)

    # v30 proxy_open 的 time_in_market 已在 clean 里
    doc["meta"]["v30_proxy_end"] = v30["proxy_open"]["end"]
    doc["meta"]["v30_real_end"] = v30["real_open"]["end"]

    # gate2 neighbors 里的 float 收成可序列化
    for item in doc["gates"]["gate2"]["neighbors"]:
        for k in ("dcal_proxy", "dcal_real", "proxy_calmar"):
            item[k] = _f(item[k], 4)

    text = render(doc)
    with open(OUT_MD, "w") as f:
        f.write(text)
    with open(OUT_JSON, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, default=str)
    print(f"[ok] {OUT_MD}")
    print(f"[ok] {OUT_JSON}")


if __name__ == "__main__":
    main()
