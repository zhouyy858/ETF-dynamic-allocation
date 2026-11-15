# -*- coding: utf-8 -*-
"""纸面：AI 应用方向对照 v47。不改 references/final_cfg_v47.json，不合并，不下单。

预注册（跑收益之前）：
- 同指数选券：score = ln(规模元) + ln(20日均成交额元) + 0.5*ln(上市至 2026-11-12 的天数)。
  规模用东财 ETF 列表 DEC_NAV（亿元）。三关只看下面 primary 格子。
- 全指数切换：上市前用官方指数收益，上市后用场内价。指数不够长的日期沿用原槽
  （swaps.gap_fill），并在结果里分开报。
- 新增袖没有底仓。纳指槽的底仓仍留在 159941。新增 QDII 袖用 v47 同一组溢价阈值削减。
- 换掉 159941 槽时，溢价门控改读成交标的自己的溢价（该标的没有溢价的日期仍用 159941），
  仍取两只 QDII 的较高者。引擎把溢价裁剪在 [-10%, +15%]。
- 晋升否决：最近 T-2 未裁剪溢价 > 8% 的 QDII，Calmar 再好也不晋升。
"""
import copy, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import DATA_DIR, build_panel, read_table, rets_from
from engine import SLOTS, evaluate, run_backtest
from run_backtest import book_slots, exec_panel
from strategy import DynamicStrategy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
AI = os.path.join(DATA_DIR, "ai_app")
BASE_FILE = os.path.join(REF, "final_cfg_v47.json")
END = pd.Timestamp("2026-11-12")
WINDOWS = (
    ("proxy", "proxy", "2014-06-23", None),
    ("train", "proxy", "2014-06-23", "2021-12-31"),
    ("oos", "proxy", "2022-01-04", None),
    ("real", "real", "2025-04-23", None),
)
# 本脚本在改引擎之前用同一条 v47 路径钉住的场内价数字。改完必须逐位对上。
V47_PRICE = {
    "proxy": {"cagr": 0.16226312167448897, "max_dd": -0.07983071306532696, "calmar": 2.0325901578970744},
    "train": {"cagr": 0.168087, "max_dd": -0.07983071306532696, "calmar": 2.1056},
    "oos": {"cagr": 0.15071, "max_dd": -0.05891, "calmar": 2.5583},
    "real": {"cagr": 0.3408857320012417, "max_dd": -0.043455893447549854, "calmar": 7.844407396954846},
}

SLEEVES = {
    "159852": {"index": "index_930601.csv", "index_code": "930601", "index_name": "中证软件服务"},
    "515230": {"index": "index_H30202.csv", "index_code": "H30202", "index_name": "中证全指软件"},
    "159509": {"index": "index_NDXTMC.csv", "index_code": "NDXTMC", "index_name": "纳斯达克科技市值加权"},
    "161128": {"index": None, "index_code": "SBI", "index_name": "标普500信息科技"},
    "513050": {"index": None, "index_code": "H30533", "index_name": "中国互联网50人民币"},
}

CELLS = [
    {"id": "CN852", "role": "primary", "swaps": {"159952": "159852"}, "overlay": [],
     "note": "创业板槽整段换成中证软件服务 159852。纳指/标普不动。"},
    {"id": "CN230", "role": "primary", "swaps": {"159952": "515230"}, "overlay": [],
     "note": "创业板槽整段换成中证全指软件 515230。全指数切换，不是半仓对调。"},
    {"id": "USfull", "role": "primary", "swaps": {"159941": "159509"}, "overlay": [], "qdii": "159509",
     "note": "纳指槽整段换成 159509。不是应用纯净，硬件权重高。溢价门控改读 159509。"},
    {"id": "BothFull", "role": "primary", "swaps": {"159952": "159852", "159941": "159509"}, "overlay": [],
     "qdii": "159509", "note": "CN852 与 USfull 同时全切换。"},
    {"id": "BothAlt", "role": "primary", "swaps": {"159952": "515230", "159941": "159509"}, "overlay": [],
     "qdii": "159509", "note": "CN230 与 USfull 同时全切换。"},
    {"id": "AddBoth", "role": "primary", "swaps": {}, "overlay": [
        {"parent": "159952", "child": "159852", "share": 0.5},
        {"parent": "159941", "child": "159509", "share": 0.5},
     ], "qdii": "159509",
     "note": "保留四只。创业板成长份额 50/50，纳指成长份额（底仓以上）50/50。新袖无底仓。159509 用自己的溢价门控。"},
    {"id": "APX_161128", "role": "appendix", "swaps": {"159941": "161128"}, "overlay": [], "qdii": "161128",
     "note": "附录：易方达标普信息科技 QDII-LOF，不是 ETF，硬件权重大，不是应用纯净。"},
    {"id": "APX_513050", "role": "appendix", "swaps": {"159941": "513050"}, "overlay": [], "qdii": "513050",
     "note": "附录：海外互联网应用，不是美股 AI 应用。不参与晋升。"},
]


def _close(path):
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index <= END]
    # 中证全指软件 2013-06-28 官方序列 close=0、OHLC 为空，丢掉这一天，不把 -100% 接进收益。
    return s[s > 0]


def _returns_from_prices(px):
    return px.pct_change().dropna()


def splice_returns(index_close, etf_close):
    """上市前用指数，第一根场内收益日起用场内价。指数更短时，序列就从指数第一根收益开始。"""
    etf_r = _returns_from_prices(etf_close)
    first = etf_r.index.min()
    parts = [etf_r]
    if index_close is not None:
        idx_r = _returns_from_prices(index_close)
        parts.append(idx_r[idx_r.index < first])
    own = pd.concat(parts).sort_index()
    own = own[~own.index.duplicated(keep="last")]
    return own.dropna()


def write_level(returns, path):
    """pct_change 能把收益逐日还原。前面垫 1 元，避免第一天收益被 pct_change 吃掉。"""
    base = returns.index[0] - pd.Timedelta(days=1)
    level = (1.0 + returns).cumprod()
    out = pd.concat([pd.Series({base: 1.0}), level])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    pd.DataFrame({"close": out}).rename_axis("date").to_csv(path)
    back = out.pct_change().dropna()
    gap = (back.reindex(returns.index) - returns).abs().max()
    if not np.isfinite(gap) or gap > 1e-10:
        raise RuntimeError(f"splice roundtrip {path} gap={gap}")


def prepare():
    os.makedirs(AI, exist_ok=True)
    listing = {}
    for code, spec in SLEEVES.items():
        etf = _close(os.path.join(AI, f"{code}_qfq.csv"))
        idx = _close(os.path.join(DATA_DIR, spec["index"])) if spec["index"] else None
        price_r = splice_returns(idx, etf)
        write_level(price_r, os.path.join(DATA_DIR, f"{code}_qfq.csv"))
        nav = pd.read_csv(os.path.join(AI, f"{code}_nav.csv"), parse_dates=["date"]).set_index("date").sort_index()
        nav = nav[nav.index <= END]
        nav_px = nav["cum_nav"].dropna()
        nav_r = splice_returns(idx, nav_px)
        write_level(nav_r, os.path.join(DATA_DIR, f"{code}_navsplice.csv"))
        listing[code] = {
            "etf_first": str(etf.index.min().date()),
            "etf_last": str(etf.index.max().date()),
            "splice_first": str(price_r.index.min().date()),
            "index_first": None if idx is None else str(idx.index.min().date()),
            "index_last": None if idx is None else str(idx.index.max().date()),
        }
        if code in ("159509", "161128", "513050"):
            raw = _close(os.path.join(AI, f"{code}_raw.csv"))
            pd.DataFrame({"close": raw}).rename_axis("date").to_csv(
                os.path.join(DATA_DIR, f"qdii_price_{code}.csv"))
            nav.reset_index().to_csv(os.path.join(DATA_DIR, f"{code}_nav.csv"), index=False)
    base = json.load(open(BASE_FILE, encoding="utf-8"))
    for cell in CELLS:
        cfg = copy.deepcopy(base)
        swaps = {"515100": "512890"}
        swaps.update(cell["swaps"])
        cfg["instrument_swaps"] = swaps
        if cell["overlay"]:
            cfg["growth_overlay"] = cell["overlay"]
        cfg["version"] = "paper-ai-" + cell["id"]
        cfg["_note"] = "纸面，不改默认 v47，不下单。" + cell["note"]
        path = os.path.join(REF, f"paper_cfg_v47_ai_{cell['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
            f.write("\n")
        cell["cfg"] = path
    return listing


def run_cfg(cfg, signals, panel, start, end):
    slots = book_slots(cfg)
    slots_arg = slots if slots != list(SLOTS) else None
    ds = DynamicStrategy(signals, cfg=cfg)
    bond = rets_from(read_table("511260_nav.csv"), "cum_nav")
    res = run_backtest(
        panel, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="DYN", min_delta=cfg.get("min_delta", 0.035),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 1.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True, slots=slots_arg,
    )
    ev = evaluate(res)
    return res, ev


def pack(ev):
    return {k: float(ev[k]) for k in ("cagr", "max_dd", "sharpe", "calmar", "turnover", "avg_cash")}


def fmt(p):
    return f"{p['cagr']*100:6.2f}% / {p['max_dd']*100:6.2f}% / S {p['sharpe']:.2f} / Cal {p['calmar']:.3f}"


def unclipped_premium(code):
    px = _close(os.path.join(AI, f"{code}_raw.csv"))
    nav = pd.read_csv(os.path.join(AI, f"{code}_nav.csv"), parse_dates=["date"]).set_index("date").sort_index()
    nav = nav[nav.index <= END]["unit_nav"]
    df = pd.concat([px.rename("px"), nav.rename("nav")], axis=1, sort=True).dropna()
    p = df["px"] / df["nav"] - 1
    dates = px.index.sort_values()
    last = dates[dates <= END][-1]
    t2 = dates[dates.get_loc(last) - 2]
    tail = p.iloc[-20:]
    return {
        "t2_date": str(t2.date()),
        "t2": float(p.loc[t2]) if t2 in p.index else None,
        "last_date": str(p.index[-1].date()),
        "last": float(p.iloc[-1]),
        "med20": float(tail.median()),
        "max20": float(tail.max()),
        "frac_gt_8": float((p > 0.08).mean()),
        "frac_gt_12": float((p > 0.12).mean()),
        "n": int(len(p)),
    }


def main():
    listing = prepare()
    base = json.load(open(BASE_FILE, encoding="utf-8"))
    R = {"proxy": build_panel("proxy")[0], "real": build_panel("real")[0]}
    panels = {}
    for cal in ("price", "nav"):
        panels[cal] = {
            "proxy": exec_panel(R["proxy"], "proxy", base, cal),
            "real": exec_panel(R["real"], "real", base, cal),
        }
    print("== v47 identity ==", flush=True)
    v47 = {"price": {}, "nav": {}}
    for cal in ("price", "nav"):
        for name, layer, start, end in WINDOWS:
            res, ev = run_cfg(base, R[layer], panels[cal][layer], start, end)
            v47[cal][name] = pack(ev)
            v47[cal][name]["_rets_id"] = id(res)
            print(f"v47 {cal:5} {name:5} {fmt(v47[cal][name])}", flush=True)
            if cal == "price" and name in ("proxy", "real"):
                exp = V47_PRICE[name]
                for k in ("cagr", "max_dd", "calmar"):
                    if abs(v47[cal][name][k] - exp[k]) > 1e-9:
                        raise SystemExit(f"v47 {name} {k} drifted {v47[cal][name][k]} vs {exp[k]}")
        # train/oos checked more loosely against the rounded published anchors
        for name, tol in (("train", 5e-4), ("oos", 5e-4)):
            exp = V47_PRICE[name]
            got = v47["price"][name]
            if abs(got["calmar"] - exp["calmar"]) > tol or abs(got["cagr"] - exp["cagr"]) > tol:
                raise SystemExit(f"v47 {name} off published anchor {got} vs {exp}")
    print("v47 price anchors held", flush=True)

    results = {"v47": {cal: {w: {k: v for k, v in m.items() if not k.startswith("_")}
                             for w, m in windows.items()} for cal, windows in v47.items()},
               "listing": listing, "cells": {}, "premiums": {}}
    for code in ("159509", "161128", "513050"):
        results["premiums"][code] = unclipped_premium(code)
        pr = results["premiums"][code]
        print(f"premium {code} T-2 {pr['t2_date']} {pr['t2']*100:.2f}% med20 {pr['med20']*100:.2f}%", flush=True)

    for cell in CELLS:
        cfg = json.load(open(cell["cfg"], encoding="utf-8"))
        built = {}
        for cal in ("price", "nav"):
            built[cal] = {
                "proxy": exec_panel(R["proxy"], "proxy", cfg, cal),
                "real": exec_panel(R["real"], "real", cfg, cal),
            }
        cell_out = {"role": cell["role"], "note": cell["note"], "cfg": os.path.basename(cell["cfg"]),
                    "metrics": {}, "diagnostics": {}}
        for cal in ("price", "nav"):
            cell_out["metrics"][cal] = {}
            for name, layer, start, end in WINDOWS:
                res, ev = run_cfg(cfg, R[layer], built[cal][layer], start, end)
                cell_out["metrics"][cal][name] = pack(ev)
                if name == "proxy" and cal == "price":
                    cell_out["_proxy_wealth"] = float(res["wealth"].iloc[-1])
            print(cell["id"], cal, " ".join(
                f"{w}:{cell_out['metrics'][cal][w]['calmar']:.3f}" for w in ("proxy", "train", "oos", "real")),
                flush=True)
        # 上市前指数 vs 上市后场内：把上市日前的收益改回 v47 原槽，只在场内价 proxy 上做诊断。
        price_p = built["price"]["proxy"]
        base_p = panels["price"]["proxy"]
        half = price_p.copy()
        live_starts = []
        touched = list(cell["swaps"].items()) + [(ov["parent"], ov["child"]) for ov in cell["overlay"]]
        for slot, code in touched:
            first = pd.Timestamp(listing[code]["etf_first"])
            col = code if any(ov["child"] == code for ov in cell["overlay"]) else slot
            src = base_p[slot]
            half.loc[half.index < first, col] = src.reindex(half.index)[half.index < first]
            live_starts.append(first)
        _, ev_half = run_cfg(cfg, R["proxy"], half, "2014-06-23", None)
        cell_out["diagnostics"]["half_swap_proxy"] = pack(ev_half)
        live_start = max(live_starts) if live_starts else None
        if live_start is not None:
            _, ev_live = run_cfg(cfg, R["proxy"], price_p, str(live_start.date()), None)
            _, ev_base_live = run_cfg(base, R["proxy"], base_p, str(live_start.date()), None)
            cell_out["diagnostics"]["live_start"] = str(live_start.date())
            cell_out["diagnostics"]["live_cold"] = pack(ev_live)
            cell_out["diagnostics"]["v47_live_cold"] = pack(ev_base_live)
        px = cell_out["metrics"]["price"]
        base_px = results["v47"]["price"]
        gates = {}
        for w in ("proxy", "oos", "real"):
            gates[w] = bool(px[w]["calmar"] > base_px[w]["calmar"])
        cell_out["gates"] = gates
        cell_out["gates_all"] = all(gates.values())
        cell_out["cagr15_mdd10"] = bool(px["proxy"]["cagr"] >= 0.15 and px["proxy"]["max_dd"] > -0.10)
        qdii = cell.get("qdii")
        if qdii:
            t2 = results["premiums"][qdii]["t2"]
            cell_out["premium_t2"] = t2
            cell_out["premium_blocks_promotion"] = bool(t2 is not None and t2 > 0.08)
        else:
            cell_out["premium_blocks_promotion"] = False
        cell_out["promotion_eligible"] = bool(
            cell["role"] == "primary" and cell_out["gates_all"] and not cell_out["premium_blocks_promotion"])
        results["cells"][cell["id"]] = cell_out
        print(f"  gates {gates} premium_block {cell_out['premium_blocks_promotion']} "
              f"eligible {cell_out['promotion_eligible']}", flush=True)

    out_path = os.path.join(REF, "v47_ai_app_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
