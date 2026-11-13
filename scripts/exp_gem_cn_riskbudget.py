# -*- coding: utf-8 -*-
"""旁路纸面实验: GEM-CN 双动量 + 四资产风险预算，对照 v30。

不修改 references/final_cfg_v30.json，不改引擎默认路径。
严格口径: 信号只用调仓日之前的收盘；先计提当日收益再收盘成交；单边万5。

用法（仓库根目录）:
    python3 scripts/exp_gem_cn_riskbudget.py

写出:
    out/exp_gem_rb_summary.json
    out/exp_gem_rb_monthly.csv
    references/exp_gem_rb_summary.json
    references/exp_gem_rb_monthly.csv
"""
import hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, build_returns, read_table, rets_from
from engine import run_backtest, evaluate
from metrics import annualized_ret, annualized_vol, calmar, max_drawdown, sharpe
from strategy import DynamicStrategy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
CFG_PATH = os.path.join(ROOT, "references", "exp_gem_rb_cfg.json")
V30_PATH = os.path.join(ROOT, "references", "final_cfg_v30.json")
FEE = 0.0005
LOOKBACK = 252
VOL_WIN = 63
RF = 0.018

PERIODS = [
    ("2014H2-2015牛市", "2014-06-23", "2015-06-12"),
    ("2015股灾", "2015-06-15", "2016-02-29"),
    ("2016-2017修复", "2016-03-01", "2017-12-29"),
    ("2018熊市", "2018-01-02", "2019-01-03"),
    ("2019-2021牛市", "2019-01-04", "2021-02-18"),
    ("2021-2022熊市", "2021-02-19", "2022-10-31"),
    ("2023-2024震荡", "2023-01-03", "2024-08-30"),
    ("2024Q4-重叠窗末", "2024-09-02", "2026-12-31"),
]


def file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def nav_rets(fn):
    df = read_table(fn)
    df = df[df.index.day_of_week < 5]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df["cum_nav"].astype(float).pct_change().dropna()


def weekday(s):
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index.day_of_week < 5]


def cash_sleeve(bond, index, repo_rate, bond_pct):
    repo_d = (1.0 + repo_rate) ** (1.0 / 252.0) - 1.0
    b = bond.reindex(index).ffill()
    filled = int(b.isna().sum())
    b = b.fillna(repo_d)
    return (1.0 - bond_pct) * repo_d + bond_pct * b, filled, float(repo_d)


def gem_weights(hist, eq_cols, lookback):
    cols = list(eq_cols) + ["CASH"]
    w = pd.Series(0.0, index=cols)
    if hist is None or len(hist) < lookback:
        w["CASH"] = 1.0
        return w
    window = hist.iloc[-lookback:]
    moms = {c: float((1.0 + window[c]).prod() - 1.0) for c in cols}
    best = max(eq_cols, key=lambda c: (moms[c], 1.0 if c == "513500" else 0.0))
    if moms[best] <= moms["CASH"]:
        w["CASH"] = 1.0
    else:
        w[best] = 1.0
    return w


def inv_vol_weights(hist, cols, lookback):
    cols = list(cols)
    if hist is None or len(hist) < lookback:
        return pd.Series(1.0 / len(cols), index=cols)
    sd = hist.iloc[-lookback:].std(ddof=1).reindex(cols)
    sd = sd.clip(lower=1e-8)
    if not np.isfinite(sd).all():
        return pd.Series(1.0 / len(cols), index=cols)
    inv = 1.0 / sd
    return inv / inv.sum()


def erc_weights_from_cov(cov):
    """长多头等风险贡献，坐标下降。

    债券与股票相关为负时，w*(Σw) 可能为负。乘性迭代 w←w·mean(rc)/rc 会把负贡献打到角落，
    不能当作 ERC。这里对每个资产解 a w_i^2 + b w_i = 组合方差/n 的正根，再归一。
    """
    cov = np.array(cov, dtype=float)
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T) + np.eye(n) * 1e-12
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    w = 1.0 / vol
    w = w / w.sum()
    for _ in range(32):
        for i in range(n):
            a = float(cov[i, i])
            b = float(cov[i] @ w - a * w[i])
            target = float(w @ cov @ w) / n
            disc = b * b + 4.0 * a * target
            if a <= 0 or disc < 0:
                continue
            w[i] = max(0.0, (-b + np.sqrt(disc)) / (2.0 * a))
        s = float(w.sum())
        w = np.ones(n) / n if s <= 0 else w / s
    return w


def erc_weights(hist, cols, lookback):
    cols = list(cols)
    if hist is None or len(hist) < lookback:
        return pd.Series(1.0 / len(cols), index=cols)
    window = hist.iloc[-lookback:][cols]
    if window.isna().any().any() or len(window) < lookback:
        return inv_vol_weights(hist, cols, lookback)
    w = erc_weights_from_cov(window.cov().values)
    return pd.Series(w, index=cols)


def equal_weights(hist, cols, lookback):
    return pd.Series(1.0 / len(cols), index=list(cols))


def make_fn(frame, fn, cols, lookback):
    def wrapped(dt):
        hist = frame.loc[frame.index < dt, cols]
        return fn(hist, cols, lookback)
    return wrapped


def make_gem_fn(frame, eq_cols, lookback):
    """信号帧可以长于盈亏窗。eq_cols 不含 CASH，帧里必须有 CASH 列。"""
    cols = list(eq_cols) + ["CASH"]

    def wrapped(dt):
        hist = frame.loc[frame.index < dt, cols]
        return gem_weights(hist, eq_cols, lookback)
    return wrapped


def is_month_rebal(i, dt):
    return i == 0 or (dt.day_of_week == 4 and dt.day <= 7)


def simulate(pnl, weight_fn, cash_ret, fee=FEE, min_delta=0.002, traded_cols=None):
    """pnl 列是可交易袖套。权重和<1 的残差赚 cash_ret。CASH 列若存在，其变化不计成交费。"""
    dates = pnl.index
    cols = list(pnl.columns)
    traded = traded_cols if traded_cols is not None else [c for c in cols if c != "CASH"]
    t_idx = [cols.index(c) for c in traded]
    k = len(cols)
    w = np.zeros(k)
    if "CASH" in cols:
        w[cols.index("CASH")] = 1.0
    rets = np.zeros(len(dates))
    earn = np.zeros((len(dates), k))
    post = np.zeros((len(dates), k))
    for i, dt in enumerate(dates):
        r = pnl.iloc[i].values.astype(float)
        cr = float(cash_ret.loc[dt])
        earn[i] = w
        gross = w * (1.0 + r)
        cash_w = 1.0 - float(w.sum())
        factor = float(gross.sum() + cash_w * (1.0 + cr))
        if factor <= 0:
            raise RuntimeError(f"净值因子<=0 @ {dt.date()}")
        pf = factor - 1.0
        w = gross / factor
        if is_month_rebal(i, dt):
            tgt = weight_fn(dt).reindex(cols).fillna(0.0).values.astype(float)
            if tgt.sum() > 1.0 + 1e-8:
                tgt = tgt / tgt.sum()
            delta = tgt - w
            if i == 0 or np.max(np.abs(delta[t_idx])) >= min_delta:
                w = w + delta
                pf -= float(np.abs(delta[t_idx]).sum()) * fee
        if w.sum() > 1.0 + 1e-6 or (w < -1e-8).any():
            raise RuntimeError(f"权重越界 @ {dt.date()} sum={w.sum():.6f}")
        rets[i] = pf
        post[i] = w
    ret_s = pd.Series(rets, index=dates, name="ret")
    earn_df = pd.DataFrame(earn, index=dates, columns=cols)
    post_df = pd.DataFrame(post, index=dates, columns=cols)
    return ret_s, earn_df, post_df


def pack(r, name):
    r = r.dropna()
    if len(r) < 5:
        return {"name": name, "n_days": int(len(r))}
    w = (1.0 + r).cumprod()
    ann = float(annualized_ret(r))
    mdd, peak, trough = max_drawdown(w)
    mdd = float(mdd)
    return {
        "name": name,
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "n_days": int(len(r)),
        "cagr": ann,
        "vol": float(annualized_vol(r)),
        "max_dd": mdd,
        "mdd_peak": None if peak is None else str(pd.Timestamp(peak).date()),
        "mdd_trough": None if trough is None else str(pd.Timestamp(trough).date()),
        "sharpe": float(sharpe(r, RF)),
        "calmar": None if not (mdd < 0) else float(calmar(ann, mdd)),
        "total_ret": float(w.iloc[-1] - 1.0),
        "final_wealth": float(w.iloc[-1]),
    }


def slice_pack(r, name, start=None, end=None):
    x = r
    if start is not None:
        x = x[x.index >= pd.Timestamp(start)]
    if end is not None:
        x = x[x.index <= pd.Timestamp(end)]
    return pack(x, name)


def periods_of(r, end_cap):
    out = {}
    for name, a, b in PERIODS:
        b2 = min(pd.Timestamp(b), pd.Timestamp(end_cap))
        sub = r[(r.index >= pd.Timestamp(a)) & (r.index <= b2)]
        if len(sub) < 5:
            continue
        w = (1.0 + sub).cumprod()
        mdd, _, _ = max_drawdown(w)
        out[name] = {
            "start": str(sub.index.min().date()),
            "end": str(sub.index.max().date()),
            "n_days": int(len(sub)),
            "total": float(w.iloc[-1] - 1.0),
            "cagr": float(annualized_ret(sub)),
            "max_dd": float(mdd),
        }
    return out


def month_returns(r):
    r = r.dropna().sort_index()
    g = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
    g.index = g.index.to_timestamp("M")
    return g


def compound_mask(month_r, mask):
    x = month_r[mask.reindex(month_r.index).fillna(False)]
    if len(x) == 0:
        return {"n_months": 0}
    wealth = float((1.0 + x).prod())
    return {
        "n_months": int(len(x)),
        "total": wealth - 1.0,
        "avg_month": float(x.mean()),
        "hit_rate_positive": float((x > 0).mean()),
        "final_wealth": wealth,
    }


def align_cols(frames):
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    idx = idx.sort_values()
    return [f.reindex(idx) for f in frames]


def run_v30(R, cfg, start, end, lock_log):
    ds = DynamicStrategy(R, cfg=cfg)
    orig = ds._market_mult

    def wrapped(dt, market):
        out = orig(dt, market)
        rec = lock_log.setdefault(pd.Timestamp(dt), {})
        rec[market] = bool(ds._lock[market])
        return out

    ds._market_mult = wrapped
    bond = rets_from(read_table(f"{cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav")
    res = run_backtest(
        R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="v30",
        min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True,
    )
    ev = evaluate(res, rf=RF)
    return res, ev


def self_check():
    idx = pd.bdate_range("2020-01-01", periods=400)
    a = pd.Series(0.001, index=idx)
    b = pd.Series(-0.001, index=idx)
    cash = pd.Series(0.0, index=idx)
    hist = pd.DataFrame({"513500": a, "159941": b, "CASH": cash})
    w = gem_weights(hist, ["513500", "159941"], 252)
    assert w["513500"] == 1.0 and w["CASH"] == 0.0, w.to_dict()
    hist2 = hist.copy()
    hist2["513500"] = -0.0005
    hist2["159941"] = -0.0008
    w2 = gem_weights(hist2, ["513500", "159941"], 252)
    assert w2["CASH"] == 1.0, w2.to_dict()
    # 低波动资产权重大
    rng = np.random.default_rng(0)
    noisy = pd.DataFrame({
        "hi": rng.normal(0, 0.02, size=len(idx)),
        "lo": rng.normal(0, 0.005, size=len(idx)),
    }, index=idx)
    iv = inv_vol_weights(noisy, ["hi", "lo"], 63)
    assert iv["lo"] > iv["hi"]
    cov = np.array([
        [0.04, 0.01, 0.0, -0.002],
        [0.01, 0.03, 0.005, -0.001],
        [0.0, 0.005, 0.02, 0.0],
        [-0.002, -0.001, 0.0, 0.0004],
    ], dtype=float)
    w_erc = erc_weights_from_cov(cov)
    rc = w_erc * (cov @ w_erc)
    share = rc / rc.sum()
    if not np.allclose(share, np.full(4, 0.25), atol=1e-6):
        raise AssertionError(f"ERC 风险贡献未对齐 {share}")
    # 未来收益改写不得改变当日权重
    frame = hist.copy()
    fn = make_gem_fn(frame, ["513500", "159941"], 252)
    dt = idx[300]
    w_ok = fn(dt)
    bad = frame.copy()
    bad.loc[bad.index >= dt] = 0.05
    w_bad = make_gem_fn(bad, ["513500", "159941"], 252)(dt)
    if not np.allclose(w_ok.values, w_bad.values):
        raise AssertionError(f"信号用到了当日或未来收益 {w_ok.to_dict()} vs {w_bad.to_dict()}")
    # 含当日的窗口必须看见冲击，否则这道检验没有区分力
    m1 = float((1.0 + frame.loc[frame.index <= dt, "513500"].iloc[-252:]).prod() - 1.0)
    m2 = float((1.0 + bad.loc[bad.index <= dt, "513500"].iloc[-252:]).prod() - 1.0)
    if abs(m1 - m2) < 1e-6:
        raise AssertionError("未来冲击没有进入含当日的窗口，检验无区分力")


def fmt_row(m):
    if "cagr" not in m:
        return f"{m.get('name',''):<16} n={m.get('n_days')}"
    return (f"{m['name']:<16} {m['start']}~{m['end']} n={m['n_days']:<4} "
            f"CAGR={m['cagr']*100:7.2f}%  MDD={m['max_dd']*100:7.2f}%  "
            f"Sharpe={m['sharpe']:5.2f}  Calmar={m['calmar']:5.2f}  "
            f"财富={m['final_wealth']:6.3f}")


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def judge(p1_wealth, p1_dd, p2, p3):
    return bool(p1_wealth or p1_dd) and bool(p2) and bool(p3)


def main():
    cfg_exp = json.load(open(CFG_PATH, encoding="utf-8"))
    assert cfg_exp["gem"]["lookback_days"] == LOOKBACK
    assert cfg_exp["risk_budget"]["vol_lookback"] == VOL_WIN
    assert cfg_exp["risk_budget"]["primary"] == "inverse_vol"
    assert abs(cfg_exp["blend"]["arm2_inv_vol"] - 0.5) < 1e-12
    assert abs(cfg_exp["blend"]["arm1_gem"] - 0.3) < 1e-12
    assert abs(cfg_exp["blend"]["v30"] - 0.2) < 1e-12
    v30_hash_before = file_sha256(V30_PATH)
    self_check()

    v30_cfg = json.load(open(V30_PATH, encoding="utf-8"))
    repo = float(v30_cfg["repo_rate"])
    bond_pct = float(v30_cfg["cash_bond_pct"])
    assert abs(repo - cfg_exp["repo_rate"]) < 1e-12
    assert abs(bond_pct - cfg_exp["cash_bond_pct"]) < 1e-12

    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    spx = weekday(R["513500"])
    ndx = weekday(R["159941"])
    feeder = weekday(nav_rets("050025_nav.csv"))
    pre = feeder[feeder.index < spx.index.min()]
    spx_ext = pd.concat([pre, spx]).sort_index()
    spx_ext = spx_ext[~spx_ext.index.duplicated(keep="last")]
    # 上市后必须与 v30 标普列逐点一致
    both = spx_ext.index.intersection(spx.index)
    if not np.allclose(spx_ext.reindex(both).values, spx.reindex(both).values, equal_nan=False):
        raise AssertionError("标普拼接在 513500 区间与 v30 proxy 不一致")

    bond = weekday(nav_rets("511010_nav.csv"))
    bond10 = weekday(nav_rets("511260_nav.csv"))
    hs300 = weekday(nav_rets("510300_nav.csv"))
    gold = weekday(nav_rets("518880_nav.csv"))
    hstech = weekday(nav_rets("513180_nav.csv"))
    ndx_real = weekday(nav_rets("159941_nav.csv"))
    spx_real = weekday(nav_rets("513500_nav.csv"))

    # 拼接跳变（511260 在 2017-08-24 前用 511010）
    splice = pd.Timestamp("2017-08-24")
    near = bond10[(bond10.index >= splice - pd.Timedelta(days=10)) & (bond10.index <= splice + pd.Timedelta(days=10))]
    splice_note = {
        "date": "2017-08-24",
        "nearby_max_abs_ret": float(near.abs().max()) if len(near) else None,
        "n": int(len(near)),
    }

    proxy_start = cfg_exp["windows"]["proxy_start"]
    oos_start = cfg_exp["windows"]["oos_start"]
    real_start = cfg_exp["windows"]["real_start"]

    # 信号帧用权益共同交易日，现金袖套把 511010 对齐到该日历（缺口回退逆回购）。
    # 若先与债券日历取交集，2014-06-23 之前只剩 246 天，12 个月动量会空转。
    gem_cal = spx_ext.index.intersection(ndx.index)
    cash_g, cash_fill_g, repo_d = cash_sleeve(bond, gem_cal, repo, bond_pct)
    gem_full = pd.DataFrame({
        "513500": spx_ext.reindex(gem_cal),
        "159941": ndx.reindex(gem_cal),
        "CASH": cash_g.reindex(gem_cal),
    }).dropna()
    gem_fn = make_gem_fn(gem_full, ["513500", "159941"], LOOKBACK)
    gem_pnl = gem_full.loc[gem_full.index >= pd.Timestamp(proxy_start)]
    if len(gem_full.loc[gem_full.index < gem_pnl.index[0]]) < LOOKBACK:
        raise RuntimeError("GEM 在重叠窗起点之前不足 252 个交易日")
    gem_ret, gem_earn, gem_post = simulate(gem_pnl, gem_fn, gem_pnl["CASH"], traded_cols=["513500", "159941"])

    # 四资产：标普用 v30 列；其余用场内净值。波动率回看同样使用起点之前的样本。
    rb_cal = hs300.index.intersection(spx.index).intersection(gold.index).intersection(bond10.index)
    cash_rb_full, cash_fill_rb, _ = cash_sleeve(bond, rb_cal, repo, bond_pct)
    rb_full = pd.DataFrame({
        "510300": hs300.reindex(rb_cal),
        "513500": spx.reindex(rb_cal),
        "518880": gold.reindex(rb_cal),
        "511260": bond10.reindex(rb_cal),
    }).dropna()
    if len(rb_full.loc[rb_full.index < pd.Timestamp(proxy_start)]) < VOL_WIN:
        raise RuntimeError("风险预算在重叠窗起点之前不足 63 个交易日")
    rb_pnl = rb_full.loc[rb_full.index >= pd.Timestamp(proxy_start)]
    cash_rb = cash_rb_full.reindex(rb_pnl.index)
    cols4 = list(rb_pnl.columns)
    iv_ret, iv_earn, iv_post = simulate(rb_pnl, make_fn(rb_full, inv_vol_weights, cols4, VOL_WIN), cash_rb)
    erc_ret, erc_earn, erc_post = simulate(rb_pnl, make_fn(rb_full, erc_weights, cols4, VOL_WIN), cash_rb)
    ew_ret, ew_earn, ew_post = simulate(rb_pnl, make_fn(rb_full, equal_weights, cols4, VOL_WIN), cash_rb)

    # 恒生科技短窗：标普 vs 513180，不进主混合
    intl_cal = spx.index.intersection(hstech.index).intersection(bond.index)
    cash_i, _, _ = cash_sleeve(bond, intl_cal, repo, bond_pct)
    intl_pnl = pd.DataFrame({
        "513500": spx.reindex(intl_cal),
        "513180": hstech.reindex(intl_cal),
        "CASH": cash_i.reindex(intl_cal),
    }).dropna()
    # 信号不满 252 日之前的现金段会稀释窗口，从第一天可交易信号开始
    ready_dt = None
    for dt in intl_pnl.index:
        if len(intl_pnl.loc[intl_pnl.index < dt]) >= LOOKBACK:
            ready_dt = dt
            break
    if ready_dt is None:
        raise RuntimeError("恒生科技窗口不足 252 日")
    intl_fn = make_gem_fn(intl_pnl, ["513500", "513180"], LOOKBACK)
    intl_pnl_run = intl_pnl.loc[ready_dt:]
    intl_ret, intl_earn, intl_post = simulate(
        intl_pnl_run, intl_fn, intl_pnl_run["CASH"], traded_cols=["513500", "513180"])

    print("运行 v30 proxy / real …")
    lock_proxy, lock_real = {}, {}
    res_p, ev_p = run_v30(R, v30_cfg, proxy_start, None, lock_proxy)
    res_r, ev_r = run_v30(Rr, v30_cfg, real_start, None, lock_real)
    v30 = weekday(res_p["rets"])
    v30r = weekday(res_r["rets"])
    v30_cash = weekday(res_p["weights"]["cash"])
    v30_w = res_p["weights"].copy()
    v30_w = v30_w[v30_w.index.day_of_week < 5]

    # 重叠窗：三者同时有收益
    ov_idx = v30.index.intersection(gem_ret.index).intersection(iv_ret.index).sort_values()
    if len(ov_idx) < 252:
        raise RuntimeError(f"重叠窗过短 n={len(ov_idx)}")
    g = gem_ret.reindex(ov_idx)
    a2 = iv_ret.reindex(ov_idx)
    base = v30.reindex(ov_idx)
    erc_a = erc_ret.reindex(ov_idx)
    ew_a = ew_ret.reindex(ov_idx)
    # 混合：月度再平衡 50/30/20，v30 缺失才用现金（本窗不缺失）
    blend_pnl = pd.DataFrame({"RB": a2, "GEM": g, "V30": base})
    target = pd.Series({"RB": 0.50, "GEM": 0.30, "V30": 0.20})

    def blend_fn(dt, _frame=blend_pnl):
        return target.copy()

    cash_b, _, _ = cash_sleeve(bond, blend_pnl.index, repo, bond_pct)
    blend_ret, blend_earn, blend_post = simulate(blend_pnl, blend_fn, cash_b, traded_cols=["RB", "GEM", "V30"])
    # ERC 混合只记录、不判定
    blend_erc_pnl = pd.DataFrame({"RB": erc_a, "GEM": g, "V30": base})

    def blend_erc_fn(dt):
        return target.copy()

    cash_be, _, _ = cash_sleeve(bond, blend_erc_pnl.index, repo, bond_pct)
    blend_erc_ret, _, _ = simulate(blend_erc_pnl, blend_erc_fn, cash_be, traded_cols=["RB", "GEM", "V30"])

    end = ov_idx.max()
    names = {
        "v30": base,
        "GEM-CN": g,
        "RB-invvol": a2,
        "RB-erc": erc_a,
        "RB-ew": ew_a,
        "blend_50_30_20": blend_ret,
        "blend_erc_robustness": blend_erc_ret,
    }
    overlap_metrics = {k: pack(v, k) for k, v in names.items()}
    oos_metrics = {k: slice_pack(v, k, oos_start, end) for k, v in names.items()}
    # 全长对照（不受黄金/沪深300截止约束的配对）
    pair_idx = v30.index.intersection(gem_ret.index)
    pairwise = {
        "v30": pack(v30.reindex(pair_idx), "v30"),
        "GEM-CN": pack(gem_ret.reindex(pair_idx), "GEM-CN"),
    }
    pairwise_oos = {
        "v30": slice_pack(v30.reindex(pair_idx), "v30", oos_start, None),
        "GEM-CN": slice_pack(gem_ret.reindex(pair_idx), "GEM-CN", oos_start, None),
    }
    v30_published = pack(v30, "v30_proxy_full")
    v30_published_engine = {
        "cagr": float(ev_p["cagr"]), "max_dd": float(ev_p["max_dd"]),
        "sharpe": float(ev_p["sharpe"]), "calmar": float(ev_p["calmar"]),
        "avg_cash": float(ev_p["avg_cash"]),
        "start": str(res_p["rets"].index.min().date()),
        "end": str(res_p["rets"].index.max().date()),
    }

    # 真实窗：GEM 用场内 159941/513500，不用联接代理
    real_gem_cal = spx_real.index.intersection(ndx_real.index).intersection(bond.index)
    real_gem_cal = real_gem_cal[real_gem_cal >= ndx_real.index.min()]
    cash_rg, _, _ = cash_sleeve(bond, real_gem_cal, repo, bond_pct)
    real_gem_pnl = pd.DataFrame({
        "513500": spx_real.reindex(real_gem_cal),
        "159941": ndx_real.reindex(real_gem_cal),
        "CASH": cash_rg.reindex(real_gem_cal),
    }).dropna()
    # 从真实窗起点往后模拟会丢掉起点之前的动量，所以在全历史实盘净值上跑，再切片
    rg_fn = make_gem_fn(real_gem_pnl, ["513500", "159941"], LOOKBACK)
    rg_ret, _, _ = simulate(real_gem_pnl, rg_fn, real_gem_pnl["CASH"], traded_cols=["513500", "159941"])
    real_idx = v30r.index.intersection(rg_ret.index).intersection(iv_ret.index)
    real_idx = real_idx[real_idx >= pd.Timestamp(real_start)]
    real_metrics = {}
    if len(real_idx) >= 20:
        rv = v30r.reindex(real_idx)
        rg = rg_ret.reindex(real_idx)
        ra = iv_ret.reindex(real_idx)
        real_blend_pnl = pd.DataFrame({"RB": ra, "GEM": rg, "V30": rv})
        cash_real_b, _, _ = cash_sleeve(bond, real_blend_pnl.index, repo, bond_pct)
        rbld, _, _ = simulate(real_blend_pnl, blend_fn, cash_real_b, traded_cols=["RB", "GEM", "V30"])
        real_metrics = {
            "v30": pack(rv, "v30"),
            "GEM-CN": pack(rg, "GEM-CN"),
            "RB-invvol": pack(ra, "RB-invvol"),
            "blend_50_30_20": pack(rbld, "blend_50_30_20"),
            "note": "真实窗 v30 用 real 面板；GEM 用 159941/513500 场内净值；风险预算仍用 510300/518880/511260/513500。截止受黄金与沪深300文件约束。",
        }
    v30_real_full = pack(v30r, "v30_real_full")
    v30_real_engine = {
        "cagr": float(ev_r["cagr"]), "max_dd": float(ev_r["max_dd"]),
        "sharpe": float(ev_r["sharpe"]), "calmar": float(ev_r["calmar"]),
        "avg_cash": float(ev_r["avg_cash"]),
        "start": str(res_r["rets"].index.min().date()),
        "end": str(res_r["rets"].index.max().date()),
    }

    intl_idx = intl_ret.index.intersection(v30.index).intersection(iv_ret.index)
    intl_metrics = {
        "window": {"start": str(intl_ret.index.min().date()), "end": str(intl_ret.index.max().date()),
                   "n": int(len(intl_ret))},
        "overlap_with_v30_rb": {
            "v30": pack(v30.reindex(intl_idx), "v30") if len(intl_idx) else {},
            "GEM-HSTECH": pack(intl_ret.reindex(intl_idx), "GEM-HSTECH") if len(intl_idx) else {},
            "RB-invvol": pack(iv_ret.reindex(intl_idx), "RB-invvol") if len(intl_idx) else {},
        },
        "label": cfg_exp["gem"]["intl_sensitivity"]["label"],
    }
    # 现金占比
    gem_cash_days = float((gem_earn.reindex(ov_idx)["CASH"] > 0.9).mean())
    gem_months = month_returns(g)
    gem_hold = gem_earn.reindex(ov_idx).groupby(ov_idx.to_period("M")).mean()
    months_in_cash = float((gem_hold["CASH"] > 0.9).mean())
    hold_share = {
        "513500": float((gem_hold["513500"] > 0.9).mean()),
        "159941": float((gem_hold["159941"] > 0.9).mean()),
        "CASH": months_in_cash,
    }

    # 锁月 / 高现金月
    lock_rows = []
    for dt, rec in lock_proxy.items():
        if dt.day_of_week >= 5:
            continue
        lock_rows.append({"date": dt, "CN": bool(rec.get("CN", False)), "US": bool(rec.get("US", False))})
    lock_df = pd.DataFrame(lock_rows).set_index("date").sort_index() if lock_rows else pd.DataFrame(columns=["CN", "US"])
    lock_df = lock_df[~lock_df.index.duplicated(keep="last")]
    lock_m = lock_df.groupby(lock_df.index.to_period("M")).mean() if len(lock_df) else pd.DataFrame()
    cash_m = v30_cash.groupby(v30_cash.index.to_period("M")).mean()
    # 月收益
    m_v30 = month_returns(base)
    m_gem = month_returns(g)
    m_rb = month_returns(a2)
    m_erc = month_returns(erc_a)
    m_ew = month_returns(ew_a)
    m_bl = month_returns(blend_ret)
    months = m_v30.index.intersection(m_gem.index).intersection(m_rb.index).intersection(m_bl.index)
    m_v30, m_gem, m_rb = m_v30.reindex(months), m_gem.reindex(months), m_rb.reindex(months)
    m_erc, m_ew, m_bl = m_erc.reindex(months), m_ew.reindex(months), m_bl.reindex(months)
    per = pd.PeriodIndex(months, freq="M")
    # to_timestamp("M") is month-start in some versions; use period as key via the index we built
    cash_on_m = cash_m.copy()
    cash_on_m.index = cash_on_m.index.to_timestamp("M")
    # 对齐到 month_returns 的索引（to_timestamp("M") 在 pandas 2/3 为月末还是月初？我们两边用同一函数）
    # month_returns 用 to_timestamp("M")。cash 也用 to_timestamp("M")。若不一致，改用 period 交集。
    def as_period_series(s):
        out = s.copy()
        out.index = pd.DatetimeIndex(out.index).to_period("M")
        return out

    mv = as_period_series(m_v30)
    mg = as_period_series(m_gem)
    mr = as_period_series(m_rb)
    me = as_period_series(m_erc)
    mw = as_period_series(m_ew)
    mb = as_period_series(m_bl)
    common_p = mv.index.intersection(mg.index).intersection(mr.index).intersection(mb.index)
    mv, mg, mr, me, mw, mb = (x.reindex(common_p) for x in (mv, mg, mr, me, mw, mb))
    cash_p = cash_m.reindex(common_p)
    cn_p = lock_m["CN"].reindex(common_p) if len(lock_m) else pd.Series(np.nan, index=common_p)
    us_p = lock_m["US"].reindex(common_p) if len(lock_m) else pd.Series(np.nan, index=common_p)
    high = cash_p >= cfg_exp["high_cash_month"]
    high70 = cash_p >= cfg_exp["high_cash_strict"]
    cn_lock = cn_p >= cfg_exp["cn_lock_month_frac"]
    # 锁月可能在重叠窗早期没有日度记录（前 40 个交易日 daily_fn 不更新）。NaN 不算锁。
    cn_lock = cn_lock.fillna(False)
    us_lock = (us_p >= cfg_exp["cn_lock_month_frac"]).fillna(False)

    cond = {}
    for label, mask in (("high_cash_50", high), ("high_cash_70", high70), ("cn_lock", cn_lock), ("us_lock", us_lock)):
        cond[label] = {
            "v30": compound_mask(mv, mask),
            "GEM-CN": compound_mask(mg, mask),
            "RB-invvol": compound_mask(mr, mask),
            "blend_50_30_20": compound_mask(mb, mask),
        }
        # 相对 v30 的月均超额
        if int(mask.sum()) > 0:
            cond[label]["avg_excess_vs_v30"] = {
                "GEM-CN": float((mg[mask] - mv[mask]).mean()),
                "RB-invvol": float((mr[mask] - mv[mask]).mean()),
                "blend_50_30_20": float((mb[mask] - mv[mask]).mean()),
            }
            cond[label]["months_beat_v30"] = {
                "GEM-CN": float((mg[mask] > mv[mask]).mean()),
                "RB-invvol": float((mr[mask] > mv[mask]).mean()),
                "blend_50_30_20": float((mb[mask] > mv[mask]).mean()),
            }

    # 判定（预先写在 cfg 里）
    b = overlap_metrics["blend_50_30_20"]
    v = overlap_metrics["v30"]
    p1_wealth = b["final_wealth"] > v["final_wealth"]
    p1_dd = (abs(b["cagr"] - v["cagr"]) <= 0.01) and ((b["max_dd"] - v["max_dd"]) >= 0.01)
    bo = oos_metrics["blend_50_30_20"]
    vo = oos_metrics["v30"]
    p2 = bo.get("cagr", -99) >= vo.get("cagr", 0) - 0.015
    c_blend = cond["high_cash_50"]["blend_50_30_20"]
    c_v30 = cond["high_cash_50"]["v30"]
    p3 = c_blend.get("n_months", 0) > 0 and c_blend.get("final_wealth", 0) > c_v30.get("final_wealth", 0)
    passed = judge(p1_wealth, p1_dd, p2, p3)
    verdict = "PASS" if passed else "FAIL"

    # 相关性
    corr = pd.DataFrame({
        "v30": base, "GEM-CN": g, "RB-invvol": a2, "RB-erc": erc_a, "blend": blend_ret,
    }).corr()

    # 回撤带
    mdd_band = {
        "arms": {
            "v30": v["max_dd"],
            "GEM-CN": overlap_metrics["GEM-CN"]["max_dd"],
            "RB-invvol": overlap_metrics["RB-invvol"]["max_dd"],
        },
        "blend": b["max_dd"],
        "single_arm_best": float(max(v["max_dd"], overlap_metrics["GEM-CN"]["max_dd"], overlap_metrics["RB-invvol"]["max_dd"])),
        "single_arm_worst": float(min(v["max_dd"], overlap_metrics["GEM-CN"]["max_dd"], overlap_metrics["RB-invvol"]["max_dd"])),
    }

    period_table = {k: periods_of(series, end) for k, series in (
        ("v30", base), ("GEM-CN", g), ("RB-invvol", a2), ("blend_50_30_20", blend_ret))}

    # 月末持仓快照（重叠窗最后一个决策后的权重）
    last_dt = ov_idx.max()
    snap = {
        "date": str(last_dt.date()),
        "gem_post": {k: float(gem_post.loc[last_dt, k]) for k in gem_post.columns} if last_dt in gem_post.index else {},
        "rb_invvol_post": {k: float(iv_post.loc[last_dt, k]) for k in iv_post.columns} if last_dt in iv_post.index else {},
        "v30_eod": {k: float(v30_w.loc[last_dt, k]) for k in v30_w.columns} if last_dt in v30_w.index else {},
    }
    # 若最后重叠日不是月末调仓日，给出各序列自身最后一天，供 2026-11 锁仓对照
    last_pair = pair_idx.max()
    snap_pair = {
        "date": str(pd.Timestamp(last_pair).date()),
        "gem_post": {k: float(gem_post.loc[last_pair, k]) for k in gem_post.columns},
        "v30_eod": {k: float(v30_w.loc[last_pair, k]) for k in v30_w.columns} if last_pair in v30_w.index else {},
        "note": "黄金与沪深300在此日之前已经没有净值，风险预算与混合停在重叠窗末。",
    }

    rb_avg_w = {k: float(iv_earn.reindex(ov_idx)[k].mean()) for k in cols4}
    erc_avg_w = {k: float(erc_earn.reindex(ov_idx)[k].mean()) for k in cols4}

    data_notes = {
        "no_nikkei": "assets/data 没有日经或欧洲 QDII。GEM-CN 主结果是标普 vs 纳指。513180 恒生科技只作短窗附录。",
        "spx_proxy_corr_vs_513500": float(feeder.reindex(spx.index).corr(spx)),
        "ndx_proxy_note": build_returns("159941", "proxy")[1],
        "gold_end": str(gold.index.max().date()),
        "hs300_end": str(hs300.index.max().date()),
        "hstech_range": [str(hstech.index.min().date()), str(hstech.index.max().date())],
        "bond10_end": str(bond10.index.max().date()),
        "v30_proxy_end": str(v30.index.max().date()),
        "overlap": [str(ov_idx.min().date()), str(ov_idx.max().date()), int(len(ov_idx))],
        "pairwise_gem_v30": [str(pair_idx.min().date()), str(pd.Timestamp(pair_idx.max()).date()), int(len(pair_idx))],
        "cash_sleeve_bond_days_filled_gem": cash_fill_g,
        "cash_sleeve_bond_days_filled_rb": cash_fill_rb,
        "splice_511260": splice_note,
        "repo_daily": repo_d,
        "policy_bank_bond": "政金债 ETF 不在库内。GEM 防御仓用仓库现金约定（25% 逆回购 + 75% 511010）。风险预算债券腿用 511260（10Y，上市前拼接 511010）。",
        "weekend_nav_rows": "513500/159941/270042/050025 含少量周末估值行，实验在计收益前剔除周六日。v30 面板本身不含周末索引。",
        "v30_cfg_sha256": v30_hash_before,
    }

    summary = {
        "experiment": "exp_gem_rb",
        "verdict": verdict,
        "pass_fail": {
            "rule": cfg_exp["pass_fail"],
            "P1_wealth": p1_wealth,
            "P1_cagr_within_1pp_and_mdd_better_1pp": p1_dd,
            "P1": bool(p1_wealth or p1_dd),
            "P2_oos_cagr_gap_le_1_5pp": p2,
            "P2_oos_cagr_blend_minus_v30": None if "cagr" not in bo else bo["cagr"] - vo["cagr"],
            "P3_high_cash_blend_wealth_gt_v30": p3,
            "P3_n_months": c_v30.get("n_months", 0),
            "verdict": verdict,
        },
        "overlap_metrics": overlap_metrics,
        "oos_metrics": oos_metrics,
        "pairwise_gem_vs_v30_full": pairwise,
        "pairwise_gem_vs_v30_oos": pairwise_oos,
        "v30_proxy_engine": v30_published_engine,
        "v30_proxy_pack": v30_published,
        "real_overlap": real_metrics,
        "v30_real_engine": v30_real_engine,
        "v30_real_pack": v30_real_full,
        "gem_hstech": intl_metrics,
        "gem_months_in_cash": months_in_cash,
        "gem_cash_day_frac": gem_cash_days,
        "gem_month_hold_share": hold_share,
        "rb_avg_earn_weight": rb_avg_w,
        "erc_avg_earn_weight": erc_avg_w,
        "conditional_months": cond,
        "mdd_band": mdd_band,
        "corr_daily": corr.round(4).to_dict(),
        "periods": period_table,
        "snapshot_overlap_end": snap,
        "snapshot_gem_v30_last": snap_pair,
        "data_notes": data_notes,
        "reproduce": "python3 scripts/exp_gem_cn_riskbudget.py",
        "cfg": "references/exp_gem_rb_cfg.json",
    }
    if file_sha256(V30_PATH) != v30_hash_before:
        raise RuntimeError("final_cfg_v30.json 在运行中被改动")

    text = json.dumps(conv(summary), ensure_ascii=False, indent=1)
    for path in (os.path.join(OUT, "exp_gem_rb_summary.json"),
                 os.path.join(ROOT, "references", "exp_gem_rb_summary.json")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    monthly = pd.DataFrame({
        "v30": mv, "gem": mg, "rb_invvol": mr, "rb_erc": me, "rb_ew": mw, "blend": mb,
        "v30_cash": cash_p, "cn_lock_frac": cn_p, "us_lock_frac": us_p,
        "high_cash_50": high.astype(int), "high_cash_70": high70.astype(int),
        "cn_lock": cn_lock.astype(int),
    })
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    for path in (os.path.join(OUT, "exp_gem_rb_monthly.csv"),
                 os.path.join(ROOT, "references", "exp_gem_rb_monthly.csv")):
        monthly.to_csv(path, float_format="%.8f")

    print("\n===== 重叠窗（判定窗口）=====")
    for k in ("v30", "GEM-CN", "RB-invvol", "RB-erc", "RB-ew", "blend_50_30_20"):
        print(fmt_row(overlap_metrics[k]))
    print("\n===== 样本外", oos_start, "至", str(end.date()), "=====")
    for k in ("v30", "GEM-CN", "RB-invvol", "blend_50_30_20"):
        print(fmt_row(oos_metrics[k]))
    print("\n===== GEM vs v30 配对至", str(pd.Timestamp(pair_idx.max()).date()), "=====")
    for k in ("v30", "GEM-CN"):
        print(fmt_row(pairwise[k]))
    print("\n===== 真实窗重叠 =====")
    for k, m in real_metrics.items():
        if isinstance(m, dict) and "cagr" in m:
            print(fmt_row(m))
    print("\n===== 高现金 / 锁月 =====")
    for label, block in cond.items():
        nv = block["v30"].get("n_months")
        print(f"{label} n={nv} v30财富={block['v30'].get('final_wealth')} "
              f"GEM={block['GEM-CN'].get('final_wealth')} RB={block['RB-invvol'].get('final_wealth')} "
              f"blend={block['blend_50_30_20'].get('final_wealth')}")
    print("\n判定", verdict, "P1", bool(p1_wealth or p1_dd), "P1财富", p1_wealth, "P1回撤", p1_dd, "P2", p2, "P3", p3)
    print("GEM 现金月占比", round(months_in_cash, 3), "日占比", round(gem_cash_days, 3))
    print("相关\n", corr.round(3))
    print("v30 proxy 引擎", v30_published_engine)
    print("已写 references/exp_gem_rb_summary.json")


if __name__ == "__main__":
    main()
