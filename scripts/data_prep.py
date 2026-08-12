# -*- coding: utf-8 -*-
"""统一数据处理: 真实ETF净值层 + 长历史代理层, 构建对齐收益矩阵与财富指数"""
import pandas as pd, numpy as np, json

import os
def _default_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "assets", "data"),
              os.path.join(here, "data"),
              os.path.join(here, "..", "data")):
        if os.path.isdir(p):
            return p
    return os.path.join(here, "..", "assets", "data")
DATA_DIR = os.environ.get("ETF_DATA_DIR") or _default_data_dir()
DATA = DATA_DIR
ETF_META = {
    "159232": {"name": "自由现金流ETF", "type": "A价值", "market": "CN", "track": "中证全指自由现金流指数"},
    "515100": {"name": "红利低波100ETF", "type": "A防御", "market": "CN", "track": "中证红利低波100指数"},
    "159941": {"name": "纳指100ETF", "type": "海外成长", "market": "US", "track": "纳斯达克100"},
    "513500": {"name": "标普500ETF", "type": "海外均衡", "market": "US", "track": "标普500"},
    "159952": {"name": "创业板ETF", "type": "A成长", "market": "CN", "track": "创业板指"},
}
PROXY = {
    "159232": ("index_932365.csv", None),
    "515100": ("index_930955.csv", None),
    "159941": ("270042_nav.csv", "159941_nav.csv"),  # 270042联接净值仅作2012-2015回测数据代理, 实盘不交易场外
    "513500": ("513500_nav.csv", None),
    "159952": ("index_sz399006.csv", "159952_nav.csv"),
}
REPO_YIELD = 0.018
TRADING_DAYS = 252

def read_table(fn):
    return pd.read_csv(f"{DATA}/{fn}", parse_dates=["date"]).set_index("date").sort_index()

def rets_from(df, col):
    s = df[col].dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()

def load_series(spec):
    if spec.endswith(".csv"):
        df = read_table(spec)
        col = "close" if "close" in df.columns else ("level" if "level" in df.columns else "cum_nav")
        return rets_from(df, col)
    df = read_table(f"{spec}_nav.csv")
    return rets_from(df, "cum_nav")

def build_returns(slot, layer="real"):
    if layer == "real":
        return load_series(slot), f"{ETF_META[slot]['name']}净值"
    primary, backup = PROXY[slot]
    parts, note = [], []
    if backup:
        parts.append(load_series(backup)); note.append(backup)
    parts.append(load_series(primary)); note.append(primary)
    # 陷阱: 备用/主代理序列日期重叠时, 默认不稳定排序会让 duplicated(keep="last") 的
    # 胜出者随面板总长度漂移(追加尾部数据会改写历史收益) -> 必须稳定排序保证主代理恒优先
    r = pd.concat(parts).sort_index(kind="stable")
    r = r[~r.index.duplicated(keep="last")]
    r = r[r.index > "1990-12-31"]
    return r, "+".join(note)

def build_panel(layer="real", start=None, end=None):
    R = pd.DataFrame({s: build_returns(s, layer)[0] for s in ETF_META}).sort_index()
    R = R.replace([np.inf, -np.inf], np.nan)
    if start: R = R[R.index >= start]
    if end: R = R[R.index <= end]
    W = (1 + R).cumprod()
    # 每列用首个有效值归一化(避免某资产上市晚于面板起点时首行NaN导致整列NaN)
    W = W.div(W.bfill().iloc[0].replace(0, 1.0), axis=1)
    return R, W

def panel_info(R):
    info = {}
    for s in R.columns:
        rs = R[s].dropna()
        if len(rs) < 2:
            info[s] = {"name": ETF_META[s]["name"], "start": "-", "end": "-", "n": 0, "cagr": "-"}
            continue
        w = (1 + rs).cumprod()
        info[s] = {"name": ETF_META[s]["name"], "start": str(rs.index.min().date()), "end": str(rs.index.max().date()),
                   "n": int(len(rs)), "cagr": f"{(w.iloc[-1] ** (TRADING_DAYS/len(rs) - 1) * 100):.1f}%"}
    return info

def save_cache():
    panels = {}
    for layer in ["real", "proxy"]:
        R, W = build_panel(layer)
        R.to_csv(f"data/panel_{layer}_rets.csv")
        W.to_csv(f"data/panel_{layer}_wealth.csv")
        panels[layer] = panel_info(R)
    with open("data/panels.json", "w") as f:
        json.dump(panels, f, ensure_ascii=False, indent=2)
    return panels

if __name__ == "__main__":
    panels = save_cache()
    for layer, info in panels.items():
        print(f"=== {layer} ===")
        for s, v in info.items():
            print(f"  {s} {v['name']}: {v['start']}~{v['end']} n={v['n']} 全期CAGR={v['cagr']}")
