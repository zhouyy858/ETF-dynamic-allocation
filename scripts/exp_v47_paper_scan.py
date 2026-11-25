# -*- coding: utf-8 -*-
"""纸面扫描：在官方引擎上，用刷新后的同一截止日比较 v47 与预注册候选。

不改 references/final_cfg_v47.json，不下单。
 sleeve 只做整槽替换（instrument_swaps），规则只做下面写死的邻域，不做组合搜索。

预注册（先于结果）：
- 三关：proxy / OOS(2022-01-04) / real(2025-04-23) 的 Calmar 都必须严格高于同一数据上的 v47。
- 候选自身行情必须覆盖整个 OOS 窗口（首日 <= 2022-01-04），否则 OOS 被原槽拼接，三关无效。
- QDII 在引擎当前 T-2 日（决策日 2026-11-27 的可用溢价日）的原始溢价 > 8% 则否决，不改用低溢价孪生。
- 同一暴露至少两只场内 ETF（去掉联接）才算有替代；孤品否决。
- 近 20 日成交额中位数 < 2000 万则流动性否决。
- 2022 以来与被替换槽日收益相关 >= 0.90 视为近重复，不作为结构升级。
- 现金只测 511010（5Y 国债复合的代表）和 30 年国债里流动性最好的一只。不复活 110017。
- 防御动量在只剩一个防御槽时是零贡献（v41 记录），本轮不重开。
"""
import copy
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import evaluate, run_backtest
from strategy import DynamicStrategy
from swaps import pnl_panel, traded_code

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "assets", "data")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
CFG_PATH = os.path.join(ROOT, "references", "final_cfg_v47.json")
BASE = json.load(open(CFG_PATH, encoding="utf-8"))

PROXY_START = "2014-06-23"
OOS_START = "2022-01-04"
REAL_START = "2025-04-23"
PREMIUM_DATE = "2026-11-24"  # 决策日 2026-11-27，信号 2026-11-25，premium_shift=2
LIQ_MIN = 2.0e7
CORR_VETO = 0.90
PREMIUM_VETO = 0.08

H = {"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"}
TENCENT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# 名称命中即排除：重述当前书或已证伪的科技/AI 线
SKIP_NAME = re.compile(
    r"联接|LOF|纳指|纳斯达克|标普500|标普生物|标普消费|美国50|道琼斯|"
    r"芯片|半导体|人工智能|科创|创业板|中概|恒生科技|恒生互联网|互联网|"
    r"机器人|软件|云计算|算力|自由现金流"
)

EXPOSURES = [
    # 防御槽 515100：和红利低波不同的结构
    {"id": "gold", "slot": "515100", "kind": "sleeve", "label": "黄金",
     "include": r"黄金ETF|金ETF|上海金ETF", "exclude": r"黄金股"},
    {"id": "dividend", "slot": "515100", "kind": "sleeve", "label": "普通红利(非低波)",
     "include": r"红利ETF", "exclude": r"低波|港股|恒生|增强|质量|智选|主动|成长|现金流"},
    {"id": "bank", "slot": "515100", "kind": "sleeve", "label": "银行",
     "include": r"银行ETF", "exclude": r"港"},
    {"id": "utility", "slot": "515100", "kind": "sleeve", "label": "公用事业",
     "include": r"公用事业ETF", "exclude": r"绿色"},
    {"id": "convertible", "slot": "515100", "kind": "sleeve", "label": "可转债",
     "include": r"可转债ETF", "exclude": r"科创"},
    {"id": "credit", "slot": "515100", "kind": "sleeve", "label": "信用/公司/城投债",
     "include": r"信用债ETF|公司债ETF|城投债ETF", "exclude": r"科创"},
    # 成长槽 159952：替换创业板，不碰芯片/AI/科创
    {"id": "hs300", "slot": "159952", "kind": "sleeve", "label": "沪深300",
     "include": r"沪深300ETF", "exclude": r"增强|价值|成长"},
    {"id": "sse50", "slot": "159952", "kind": "sleeve", "label": "上证50",
     "include": r"上证50ETF", "exclude": r"增强"},
    {"id": "csi500", "slot": "159952", "kind": "sleeve", "label": "中证500",
     "include": r"中证500ETF", "exclude": r"增强|价值|成长"},
    {"id": "csi1000", "slot": "159952", "kind": "sleeve", "label": "中证1000",
     "include": r"中证1000ETF", "exclude": r"增强"},
    {"id": "broker", "slot": "159952", "kind": "sleeve", "label": "证券",
     "include": r"证券ETF|券商ETF", "exclude": r"港|上证"},
    {"id": "healthcare", "slot": "159952", "kind": "sleeve", "label": "医药",
     "include": r"医药ETF", "exclude": r"港|恒生|生物|创业|创新"},
    {"id": "consumer", "slot": "159952", "kind": "sleeve", "label": "消费",
     "include": r"消费ETF", "exclude": r"电子|港|恒生|在线|线上|龙头|50"},
    {"id": "military", "slot": "159952", "kind": "sleeve", "label": "军工",
     "include": r"军工ETF", "exclude": r"龙头"},
    {"id": "appliance", "slot": "159952", "kind": "sleeve", "label": "家电",
     "include": r"家电ETF", "exclude": r"家居|龙头"},
    {"id": "realestate", "slot": "159952", "kind": "sleeve", "label": "房地产",
     "include": r"房地产ETF", "exclude": r""},
    {"id": "oil_equity", "slot": "159952", "kind": "sleeve", "label": "油气股票",
     "include": r"油气|石油", "exclude": r"标普|港"},
    # 海外槽：区域替换，不测纳指/标普孪生
    {"id": "hsi", "slot": "159941", "kind": "qdii", "label": "恒生指数",
     "include": r"恒生(指数)?ETF", "exclude": r"科技|互联网|医疗|医药|消费|国企|红利|生物|央企|中国企业|港股通|汽车"},
    {"id": "nikkei", "slot": "513500", "kind": "qdii", "label": "日经/东证",
     "include": r"日经|日本东证", "exclude": r""},
    {"id": "germany", "slot": "513500", "kind": "qdii", "label": "德国",
     "include": r"德国ETF", "exclude": r""},
    {"id": "sp_energy", "slot": "513500", "kind": "qdii", "label": "标普油气",
     "include": r"标普油气ETF", "exclude": r""},
    # 现金替代。孤品只记账，不进回测。
    {"id": "cash_5y", "slot": "cash", "kind": "cash", "label": "5Y国债复合(511010)",
     "include": r"^国债ETF", "exclude": r"30年|政金", "force_code": "511010"},
    {"id": "cash_30y", "slot": "cash", "kind": "cash", "label": "30年国债",
     "include": r"30年国债ETF", "exclude": r""},
]

# 有名字、但预注册为孤品或已证伪主题，只计数不回测
CITE_ONLY = [
    {"id": "soy", "label": "豆粕", "include": r"豆粕ETF", "exclude": r"", "why": "场内只有一只，无替代"},
    {"id": "chemcycl", "label": "能源化工", "include": r"能源化工ETF", "exclude": r"", "why": "场内只有一只，无替代"},
    {"id": "coal", "label": "煤炭", "include": r"煤炭ETF", "exclude": r"", "why": "场内只有一只，无替代"},
    {"id": "liquor", "label": "酒", "include": r"酒ETF", "exclude": r"", "why": "场内只有一只，无替代"},
    {"id": "france", "label": "法国", "include": r"法国ETF", "exclude": r"", "why": "场内只有一只，无替代"},
    {"id": "saudi", "label": "沙特", "include": r"沙特ETF", "exclude": r"", "why": "场内只有一只，无替代"},
    {"id": "brazil", "label": "巴西", "include": r"巴西ETF", "exclude": r"", "why": "场内只有一只，无替代"},
    {"id": "short_bond", "label": "短融", "include": r"短融ETF", "exclude": r"", "why": "场内只有一只，无替代；不作现金升级"},
]


def tencent_symbol(code):
    return ("sh" if code.startswith("5") else "sz") + code


def fetch_bars(code, start, end, count=400):
    sym = tencent_symbol(code)
    j = requests.get(
        TENCENT, params={"param": f"{sym},day,{start},{end},{count},qfq"},
        headers=H, timeout=30,
    ).json()
    d = (j.get("data") or {}).get(sym) or {}
    day = d.get("qfqday") or d.get("day") or []
    if not day:
        return pd.DataFrame(columns=["date", "close", "vol"])
    df = pd.DataFrame(day, columns=["date", "open", "close", "high", "low", "vol"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    return df.dropna(subset=["close"]).drop_duplicates("date")


def load_universe():
    r = requests.get("https://fund.eastmoney.com/js/fundcode_search.js", headers=H, timeout=40)
    r.raise_for_status()
    text = r.text
    blob = text[text.find("["): text.rfind("]") + 1]
    rows = json.loads(blob)
    keep_types = {"指数型-股票", "指数型-海外股票", "指数型-固收", "指数型-其他"}
    out = []
    for code, _py, name, typ, _py2 in rows:
        if typ not in keep_types:
            continue
        if not re.fullmatch(r"\d{6}", code):
            continue
        if code[:2] not in ("15", "16", "51", "56", "58"):
            continue
        if "联接" in name or "LOF" in name:
            continue
        out.append({"code": code, "name": name, "type": typ})
    return out


def members_of(universe, include, exclude):
    inc = re.compile(include)
    exc = re.compile(exclude) if exclude else None
    rows = []
    for row in universe:
        if not inc.search(row["name"]):
            continue
        if exc and exc.search(row["name"]):
            continue
        if SKIP_NAME.search(row["name"]):
            continue
        rows.append(row)
    return rows


def probe_code(code):
    """腾讯 fqkline 返回窗口尾部，不能把返回的第一行当成上市日。
    历史门槛只问 2022-01-04 之前有没有成交；真正首日等全量 qfq 下载。"""
    early = fetch_bars(code, "2013-01-01", OOS_START, count=2)
    time.sleep(0.12)
    recent = fetch_bars(code, "2026-09-01", "2026-11-26", count=80)
    time.sleep(0.12)
    end = None
    amount = None
    if not recent.empty:
        end = str(recent["date"].max().date())
        last = recent.sort_values("date").tail(20)
        # 腾讯 vol 为手；512890 单日约 620 万手 * 100 * 价格 ≈ 数亿，与已知成交额同量级
        amount = float((last["vol"] * 100.0 * last["close"]).median())
    return {
        "code": code,
        "history_through_oos": (not early.empty),
        "end": end,
        "amount20": amount,
    }


def download_qfq(code):
    path = os.path.join(DATA, f"{code}_qfq.csv")
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["date"])
        if not old.empty and old["date"].max() >= pd.Timestamp("2026-11-25") and "close" in old.columns:
            return path
    rows = []
    cur = pd.Timestamp("2013-01-01")
    end = pd.Timestamp("2026-11-26")
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=360), end)
        df = fetch_bars(code, f"{cur:%Y-%m-%d}", f"{nxt:%Y-%m-%d}", count=400)
        if not df.empty:
            rows.append(df[["date", "close"]])
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(0.12)
    if not rows:
        raise RuntimeError(f"no qfq bars for {code}")
    out = pd.concat(rows, ignore_index=True).drop_duplicates("date").sort_values("date")
    out.to_csv(path, index=False)
    return path


def official_nav(code):
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    j = requests.get(url, params={"fundCode": code, "pageIndex": 1, "pageSize": 20}, headers=H, timeout=30).json()
    rows = []
    for d in ((j.get("Data") or {}).get("LSJZList") or []):
        rows.append({"date": d["FSRQ"], "unit_nav": float(d["DWJZ"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def raw_premium(code, on_date):
    px = pd.read_csv(os.path.join(DATA, f"{code}_qfq.csv"), parse_dates=["date"]).set_index("date")["close"]
    nav = official_nav(code)
    time.sleep(0.15)
    if nav.empty:
        return {"date": None, "premium": None, "error": "no nav"}
    nav = nav.set_index("date")["unit_nav"]
    both = pd.concat([px.rename("px"), nav.rename("nav")], axis=1).dropna().sort_index()
    if both.empty:
        return {"date": None, "premium": None, "error": "no overlap"}
    day = pd.Timestamp(on_date)
    use = both[both.index <= day]
    if use.empty:
        use = both
    dt = use.index[-1]
    prem = float(use.loc[dt, "px"] / use.loc[dt, "nav"] - 1.0)
    return {"date": str(dt.date()), "premium": prem, "error": None}


PRICE_FILES = {"159941": "qdii_price_159941.csv", "513500": "qdii_price_513500.csv"}
PRICE_DROP = {"159941": ["2022-07-05"], "513500": ["2022-03-30"]}


def price_rets(code):
    s = read_table(PRICE_FILES[code])["close"].sort_index()
    return s.pct_change().dropna().drop([pd.Timestamp(d) for d in PRICE_DROP[code]], errors="ignore")


def exec_pnl(returns, layer, cfg):
    """与 run_backtest.exec_panel 相同，但换券后不再把原 QDII 场内价写回该列。"""
    panel = pnl_panel(returns, layer, cfg).copy()
    for slot in PRICE_FILES:
        if traded_code(slot, cfg) == slot:
            panel[slot] = price_rets(slot).reindex(panel.index).combine_first(panel[slot])
    return panel


def run_windows(cfg, cash=None):
    bond = cash if cash is not None else rets_from(read_table("511260_nav.csv"), "cum_nav")
    out = {}
    rets = {}
    for window, signals, layer, start in (
        ("proxy", R_PROXY, "proxy", PROXY_START),
        ("oos", R_PROXY, "proxy", OOS_START),
        ("real", R_REAL, "real", REAL_START),
        ("train", R_PROXY, "proxy", PROXY_START),
    ):
        end = "2021-12-31" if window == "train" else None
        panel = exec_pnl(signals, layer, cfg)
        strategy = DynamicStrategy(signals, cfg=cfg)
        result = run_backtest(
            panel, target_weights_fn=strategy.target_fn(), daily_override_fn=strategy.daily_fn(),
            start=start, end=end, name="DYN", min_delta=cfg.get("min_delta", 0.035),
            repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
            cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 1.0),
            rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
            strict=True,
        )
        ev = evaluate(result)
        out[window] = {k: float(ev[k]) for k in ("cagr", "max_dd", "sharpe", "calmar", "vol", "turnover")}
        if window in ("proxy", "oos", "real"):
            rets[window] = result["rets"]
    return out, rets


def yearly(rets):
    s = rets.dropna()
    rows = {}
    for year, g in s.groupby(s.index.year):
        wealth = float((1.0 + g).prod())
        rows[str(int(year))] = wealth - 1.0
    return rows


def beats(m, base):
    return all(m[w]["calmar"] > base[w]["calmar"] for w in ("proxy", "oos", "real"))


def pack_metrics(m):
    return {
        w: {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)
            for k, v in row.items()}
        for w, row in m.items()
    }


def rule_grid():
    """只扰动已贡献的规则，每个邻居单独测，不交叉。"""
    specs = []
    specs.append(("gate_60_80_100", {"gate_windows": [60, 80, 100], "gate_mode": "majority"}))
    specs.append(("gate_100_120_140", {"gate_windows": [100, 120, 140], "gate_mode": "majority"}))
    specs.append(("gate_all", {"gate_windows": [80, 100, 120], "gate_mode": "all"}))
    specs.append(("gate_any", {"gate_windows": [80, 100, 120], "gate_mode": "any"}))
    specs.append(("cn_fb_trigger_6", {"market_dd": {"CN": [0.06, 0.05, 0.15, 0.14]}}))
    specs.append(("cn_fb_trigger_8", {"market_dd": {"CN": [0.08, 0.05, 0.15, 0.14]}}))
    specs.append(("cn_fb_cut_4", {"market_dd": {"CN": [0.07, 0.04, 0.15, 0.14]}}))
    specs.append(("cn_fb_cut_6", {"market_dd": {"CN": [0.07, 0.06, 0.15, 0.14]}}))
    specs.append(("cooloff_2", {"cool_off_weeks": 2}))
    specs.append(("cooloff_4", {"cool_off_weeks": 4}))
    specs.append(("growth_cap_m4", {"_growth_shift": -4}))
    specs.append(("growth_cap_p4", {"_growth_shift": 4}))
    out = []
    for name, patch in specs:
        cfg = copy.deepcopy(BASE)
        shift = patch.pop("_growth_shift", None)
        if "market_dd" in patch:
            md = copy.deepcopy(cfg["market_dd"])
            md["CN"] = patch.pop("market_dd")["CN"]
            cfg["market_dd"] = md
        cfg.update(patch)
        if shift is not None:
            for key in ("7", "8", "9"):
                g, d = cfg["state_map"][key]
                cfg["state_map"][key] = [g + shift, d]
        cfg["version"] = "v47-paper-" + name
        cfg["_note"] = "纸面邻域，不改默认"
        out.append((name, cfg))
    return out


def slot_corr(code, slot):
    own = pd.read_csv(os.path.join(DATA, f"{code}_qfq.csv"), parse_dates=["date"]).set_index("date")["close"]
    own = own.pct_change().dropna()
    base = exec_pnl(R_PROXY, "proxy", BASE)[slot].dropna()
    both = pd.concat([own.rename("a"), base.rename("b")], axis=1).dropna()
    both = both[both.index >= OOS_START]
    if len(both) < 60:
        return None
    return float(both["a"].corr(both["b"]))


def cash_returns(code):
    if code == "511010":
        return rets_from(read_table("511010_nav.csv"), "cum_nav")
    s = pd.read_csv(os.path.join(DATA, f"{code}_qfq.csv"), parse_dates=["date"]).set_index("date")["close"]
    return s.pct_change().dropna()


def fmt_row(name, m, base):
    def cell(w):
        d = m[w]["calmar"] - base[w]["calmar"]
        return f"{m[w]['cagr']*100:6.2f}/{m[w]['max_dd']*100:6.2f}/{m[w]['calmar']:6.3f} ({d:+.3f})"
    return f"{name:<22} {cell('proxy')} | {cell('oos')} | {cell('real')}"


def main():
    global R_PROXY, R_REAL
    print("== build panels ==", flush=True)
    R_PROXY, _ = build_panel("proxy")
    R_REAL, _ = build_panel("real")
    print("== baseline ==", flush=True)
    base_m, base_rets = run_windows(BASE)
    official = json.load(open(os.path.join(OUT, "iterations.json"), encoding="utf-8"))["v47"]
    for key, window, field in (
        ("proxy_calmar", "proxy", "calmar"),
        ("proxy_cagr", "proxy", "cagr"),
        ("real_calmar", "real", "calmar"),
        ("real_cagr", "real", "cagr"),
    ):
        got = base_m[window][field]
        ref = official[key]
        if abs(got - ref) > 1e-9:
            raise SystemExit(f"baseline mismatch {key}: scan {got} official {ref}")
    print(fmt_row("v47", base_m, base_m), flush=True)

    print("== universe ==", flush=True)
    universe = load_universe()
    print(f"exchange ETFs kept {len(universe)}", flush=True)

    probe_path = os.path.join(OUT, "v47_scan_probe.json")
    probe = json.load(open(probe_path, encoding="utf-8")) if os.path.exists(probe_path) else {}

    def probed(code):
        if code not in probe:
            print(f"  probe {code}", flush=True)
            probe[code] = probe_code(code)
            json.dump(probe, open(probe_path, "w"), ensure_ascii=False)
        return probe[code]

    families = []
    tested = []

    def consider(spec, cite_only=False):
        rows = members_of(universe, spec["include"], spec.get("exclude", ""))
        family = rows
        if spec.get("force_code"):
            rows = [r for r in family if r["code"] == spec["force_code"]] or [
                {"code": spec["force_code"], "name": spec["label"], "type": "forced"}
            ]
        entry = {
            "id": spec["id"], "label": spec["label"], "slot": spec.get("slot"),
            "kind": spec.get("kind", "cite"), "n_members": len(family),
            "members": [{"code": r["code"], "name": r["name"]} for r in family],
            "why": spec.get("why"),
        }
        if cite_only or len(family) < 2:
            entry["status"] = "uniqueness_veto" if len(family) < 2 else "cite_only"
            if len(rows) < 2 and not entry["why"]:
                entry["why"] = "少于两只可替代产品"
            families.append(entry)
            print(f"[veto-unique] {spec['id']} n={len(family)}", flush=True)
            return
        enriched = []
        for row in rows:
            info = dict(row)
            info.update(probed(row["code"]))
            enriched.append(info)
        entry["probes"] = [
            {"code": e["code"], "name": e["name"], "history_through_oos": e["history_through_oos"],
             "end": e["end"], "amount20": e["amount20"]}
            for e in enriched
        ]
        eligible = [
            e for e in enriched
            if e["history_through_oos"] and (e["amount20"] or 0) >= LIQ_MIN
        ]
        if spec.get("slot") in ("515100", "159952"):
            domestic = [e for e in eligible if e.get("type") != "指数型-海外股票"]
            if len(domestic) >= 1:
                eligible = domestic
        eligible.sort(key=lambda e: e["amount20"] or 0, reverse=True)
        if not eligible:
            n_hist = sum(1 for e in enriched if e["history_through_oos"])
            entry["status"] = "history_or_liquidity_veto"
            entry["why"] = (
                f"没有同时满足 2022-01-04 前已交易且20日成交额中位数>={LIQ_MIN:.0f}的成员；"
                f"覆盖到 OOS 的成员 {n_hist}/{len(enriched)}"
            )
            families.append(entry)
            print(f"[veto-hist] {spec['id']} {entry['why']}", flush=True)
            return
        primary = eligible[0]
        entry["primary"] = primary["code"]
        entry["primary_name"] = primary["name"]
        others = [r["code"] for r in family if r["code"] != primary["code"]]
        entry["substitute"] = eligible[1]["code"] if len(eligible) > 1 else (others[0] if others else None)
        if spec["kind"] == "qdii" or primary.get("type") == "指数型-海外股票":
            download_qfq(primary["code"])
            prem = raw_premium(primary["code"], PREMIUM_DATE)
            entry["premium"] = prem
            if prem["premium"] is None or prem["premium"] > PREMIUM_VETO:
                entry["status"] = "premium_veto"
                entry["why"] = f"T-2风格溢价 {prem}"
                families.append(entry)
                print(f"[veto-premium] {spec['id']} {primary['code']} {prem}", flush=True)
                return
        entry["status"] = "tested"
        families.append(entry)
        tested.append((spec, entry, primary))
        print(f"[test] {spec['id']} {primary['code']} {primary['name']} oos_hist {primary['history_through_oos']} amt {primary['amount20']:.0f}", flush=True)

    for spec in EXPOSURES:
        consider(spec, cite_only=False)
    for spec in CITE_ONLY:
        consider(spec, cite_only=True)

    print("== sleeve / cash backtests ==", flush=True)
    results = []
    for spec, entry, primary in tested:
        code = primary["code"]
        if spec["kind"] != "cash" or code != "511010":
            download_qfq(code)
        cfg = copy.deepcopy(BASE)
        cash = None
        corr = None
        own_start = None
        if spec["kind"] == "cash":
            cash = cash_returns(code)
            own_start = str(cash.index.min().date())
            cfg["version"] = f"v47-paper-cash-{code}"
        else:
            own_start = str(pd.read_csv(os.path.join(DATA, f"{code}_qfq.csv"), parse_dates=["date"])["date"].min().date())
            swaps = dict(cfg.get("instrument_swaps") or {})
            swaps[spec["slot"]] = code
            cfg["instrument_swaps"] = swaps
            cfg["version"] = f"v47-paper-swap-{spec['slot']}-{code}"
            corr = slot_corr(code, spec["slot"])
        try:
            metrics, rets = run_windows(cfg, cash=cash)
        except Exception as exc:
            results.append({"id": spec["id"], "code": code, "error": str(exc)[:300]})
            print(f"[fail] {spec['id']} {exc}", flush=True)
            continue
        oos_own = bool(own_start and own_start <= OOS_START)
        corr_veto = corr is not None and corr >= CORR_VETO
        passed = bool(oos_own and not corr_veto and beats(metrics, base_m))
        row = {
            "id": spec["id"], "label": spec["label"], "code": code, "name": primary["name"],
            "slot": spec["slot"], "kind": spec["kind"], "own_start": own_start,
            "amount20": primary["amount20"], "corr_since_2022": corr,
            "oos_own_history": oos_own, "corr_veto": corr_veto,
            "premium": entry.get("premium"),
            "metrics": pack_metrics(metrics),
            "yearly_proxy": yearly(rets["proxy"]),
            "pass_three_calmar": passed,
        }
        results.append(row)
        print(fmt_row(f"{spec['id']}:{code}", metrics, base_m)
              + f" corr={corr if corr is not None else float('nan'):.2f} pass={passed}", flush=True)

    print("== rule neighbors ==", flush=True)
    rules = []
    for name, cfg in rule_grid():
        metrics, rets = run_windows(cfg)
        passed = beats(metrics, base_m)
        rules.append({
            "id": name,
            "metrics": pack_metrics(metrics),
            "yearly_proxy": yearly(rets["proxy"]) if passed else None,
            "pass_three_calmar": passed,
        })
        print(fmt_row(name, metrics, base_m) + f" pass={passed}", flush=True)

    any_pass = any(r.get("pass_three_calmar") for r in results) or any(r["pass_three_calmar"] for r in rules)
    report = {
        "asof": "2026-11-25",
        "premium_date": PREMIUM_DATE,
        "default_config": "references/final_cfg_v47.json",
        "default_unchanged": True,
        "change_default": False,
        "any_three_calmar_pass": any_pass,
        "bars": {
            "liq_min_cny": LIQ_MIN,
            "corr_veto": CORR_VETO,
            "premium_veto": PREMIUM_VETO,
            "oos_start": OOS_START,
        },
        "baseline": {
            "metrics": pack_metrics(base_m),
            "yearly_proxy": yearly(base_rets["proxy"]),
            "yearly_oos": yearly(base_rets["oos"]),
            "yearly_real": yearly(base_rets["real"]),
        },
        "universe_n": len(universe),
        "families": families,
        "tested": results,
        "rules": rules,
        "not_rerun": [
            "US ETFs + cash only: paper PR #17",
            "AI-application sleeves 159852/515230/159509: paper PR #18; 159509 premium veto",
            "rule ablation vs then-v40: PR #16",
            "six-slot recombination vs then-v39: PR #13",
            "cash-duration timing and conditional gold: PR #14",
            "159509 pool scan: PR #11",
            "defense_momentum is zero-contribution with one defense slot (v41 report); not reopened",
            "110017 not retested",
        ],
    }
    dest = os.path.join(ROOT, "references", "v47_optimization_scan_results.json")
    json.dump(report, open(dest, "w"), ensure_ascii=False, indent=1)
    print(f"[ok] {dest} any_pass={any_pass}", flush=True)


if __name__ == "__main__":
    main()
