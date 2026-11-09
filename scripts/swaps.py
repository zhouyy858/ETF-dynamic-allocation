# -*- coding: utf-8 -*-
"""v39 换券：规则槽位不变，成交标的与损益换成 instrument_swaps。

信号（打分、防御动量、深熊锁）仍用原槽的 proxy/real 面板。
损益从该券自身前复权日收益开始；此前沿用原槽收益（与 exp_v39 的 gap_fill 相同）。
real 窗口不把上市前的原券拼进比较段。没有 instrument_swaps 时原样返回，v37/v30 不受影响。
"""
import os
import pandas as pd
from data_prep import DATA_DIR

REAL_START = "2025-04-23"
TRADED_NAMES = {
    "159232": "自由现金流",
    "159222": "自由现金流国证",
    "515100": "红利低波100",
    "563020": "红利低波",
    "159941": "纳指100",
    "513500": "标普500",
    "159952": "创业板",
}


def swaps_of(cfg):
    raw = (cfg or {}).get("instrument_swaps") or {}
    return {str(k): str(v) for k, v in raw.items() if k and v and str(k) != str(v)}


def traded_code(slot, cfg):
    return swaps_of(cfg).get(slot, slot)


def traded_name(slot, cfg):
    code = traded_code(slot, cfg)
    return TRADED_NAMES.get(code, code)


def qfq_returns(code):
    path = os.path.join(DATA_DIR, f"{code}_qfq.csv")
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()


def _gap_fill(base, own, index):
    """与 scripts/exp_v39_universe.py 的 gap_fill 同一规则。"""
    own = own.dropna().sort_index()
    first = own.index.min()
    out = base.reindex(index).astype(float).copy()
    after = index[index >= first]
    got = own.reindex(after)
    miss = got.isna()
    longest = run = 0
    for v in miss.tolist():
        run = run + 1 if v else 0
        longest = max(longest, run)
    frac = int(miss.sum()) / max(len(after), 1)
    if not (longest <= 5 and frac <= 0.02):
        raise RuntimeError(f"换券缺口过大 longest={longest} frac={frac:.3f} first={first.date()}")
    out.loc[after] = got.fillna(0.0).to_numpy()
    return out


def _real_fill(base, own, index):
    own = own.dropna().sort_index()
    first = own.index.min()
    out = base.reindex(index).astype(float).copy()
    cut = pd.Timestamp(REAL_START if first <= pd.Timestamp(REAL_START) else first)
    after = index[index >= cut]
    got = own.reindex(after)
    miss = got.isna()
    longest = run = 0
    for v in miss.tolist():
        run = run + 1 if v else 0
        longest = max(longest, run)
    frac = int(miss.sum()) / max(len(after), 1)
    if not (longest <= 5 and frac <= 0.02):
        raise RuntimeError(f"换券 real 缺口过大 longest={longest} frac={frac:.3f}")
    out.loc[after] = got.fillna(0.0).to_numpy()
    return out


def pnl_panel(R, layer, cfg):
    swaps = swaps_of(cfg)
    if not swaps:
        return R
    out = R.copy()
    for slot, code in swaps.items():
        if slot not in out.columns:
            raise KeyError(f"instrument_swaps 槽位 {slot} 不在面板里")
        own = qfq_returns(code)
        if layer == "real":
            out[slot] = _real_fill(out[slot], own, out.index)
        else:
            out[slot] = _gap_fill(out[slot], own, out.index)
    return out
