# -*- coding: utf-8 -*-
"""v39 宇宙枚举：只拉公开数据，不算策略。

来源：
- 东财 fundcode_search.js（全市场基金名称）
- 东财 FundMNDetailInformation（指数、成立日、规模、管理费、托管费）
- 东财 FundMNNBasicInformation 的 TRKERROR 原字段
- 东财 push2his kline（前复权收盘=收益，不复权收盘+单位净值=溢价）
- 东财 pingzhongdata 的累计净值 / 单位净值

没有行情的代码不进入短名单。不编造费率、规模或跟踪误差。
"""
import os, re, json, time, math
import requests
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "data", "v39_universe")
os.makedirs(OUT, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://fund.eastmoney.com/"}
S = requests.Session()
S.headers.update(H)

# PR#8 已经对照过的现金工具，本轮不重复当候选。511010 是 v30/拼接腿，511260 是现任。
CASH_SEEN = {
    "511010", "511260", "511090", "511130", "511520", "511020", "511030",
    "511360", "511380", "518880", "511880", "511990",
}
INCUMBENT = {
    "fcf": "159232", "div": "515100", "nasdaq": "159941",
    "spx": "513500", "cyb": "159952", "cash": "511260",
}


def get(url, timeout=30):
    last = None
    for i in range(5):
        try:
            r = S.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(0.6 + i * 0.8)
    raise last


def fund_list():
    txt = get("https://fund.eastmoney.com/js/fundcode_search.js", timeout=40).text
    return json.loads(re.search(r"var r = (\[.*\]);", txt, re.S).group(1))


def screen(funds):
    """场内 ETF（名称含 ETF、不是联接），关键词覆盖六个槽的同一指数和近替代。"""
    keys = ("现金流", "红利低波", "纳指", "纳斯达克", "标普500", "创业板", "国债", "地方债", "国开债", "政金债")
    out = []
    for code, _py, name, ftype, _full in funds:
        if "ETF" not in name or "联接" in name:
            continue
        if not any(k in name for k in keys):
            continue
        out.append({"code": code, "name": name, "ftype": ftype})
    return out


def detail(code):
    url = ("https://fundmobapi.eastmoney.com/FundMNewApi/FundMNDetailInformation"
           f"?FCODE={code}&deviceid=1&plat=Iphone&product=EFund&version=6.3.8")
    j = get(url, timeout=25).json()
    d = j.get("Datas") or {}
    burl = ("https://fundmobapi.eastmoney.com/FundMNewApi/FundMNNBasicInformation"
            f"?FCODE={code}&deviceid=1&plat=Iphone&product=EFund&version=6.3.8")
    trk = None
    try:
        b = get(burl, timeout=25).json().get("Datas") or {}
        raw = b.get("TRKERROR")
        if raw not in (None, "", "--"):
            trk = float(raw)
    except Exception:
        trk = None
    return {
        "code": code,
        "short": d.get("SHORTNAME"),
        "full": d.get("FULLNAME"),
        "index_code": d.get("INDEXCODE") or "",
        "index_name": d.get("INDEXNAME") or "",
        "bench": d.get("BENCH") or "",
        "estab": d.get("ESTABDATE") or "",
        "aum": _f(d.get("ENDNAV")),
        "aum_date": d.get("FEGMRQ") or "",
        "mgmt": _pct(d.get("MGREXP")),
        "cust": _pct(d.get("TRUSTEXP")),
        "trk_raw": trk,
    }


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def _pct(x):
    if x in (None, "", "--"):
        return None
    s = str(x).strip().replace("%", "")
    try:
        return float(s) / 100.0
    except Exception:
        return None


def secid(code):
    return ("1." if code.startswith(("5", "6")) else "0.") + code


def kline(code, fqt):
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={secid(code)}&klt=101&fqt={fqt}&beg=20000101&end=20261231&lmt=100000"
           "&fields1=f1&fields2=f51,f53,f57")
    j = get(url, timeout=40).json()
    data = j.get("data") or {}
    rows = []
    for line in data.get("klines") or []:
        p = line.split(",")
        if len(p) < 3:
            continue
        rows.append({"date": p[0], "close": float(p[1]), "amount": float(p[2]) if p[2] not in ("", "-") else 0.0})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date").sort_index()


def nav_history(code):
    """单位净值与累计净值。时间戳按东财惯例 +8 小时取中国日期。"""
    txt = get(f"http://fund.eastmoney.com/pingzhongdata/{code}.js", timeout=40).text
    unit = _json_var(txt, "Data_netWorthTrend")
    ac = _json_var(txt, "Data_ACWorthTrend")
    rows = {}
    if isinstance(unit, list):
        for it in unit:
            if not isinstance(it, dict) or "x" not in it or it.get("y") is None:
                continue
            dt = pd.Timestamp(int(it["x"]), unit="ms", tz="UTC").tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            rows.setdefault(dt, {})["unit_nav"] = float(it["y"])
    if isinstance(ac, list):
        for it in ac:
            if not isinstance(it, list) or len(it) < 2:
                continue
            if it[1] is None:
                continue
            dt = pd.Timestamp(int(it[0]), unit="ms", tz="UTC").tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            rows.setdefault(dt, {})["cum_nav"] = float(it[1])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    return df


def _json_var(txt, name):
    i = txt.find(name)
    if i < 0:
        return None
    j = txt.find("=", i)
    k = j + 1
    while k < len(txt) and txt[k] in " \n":
        k += 1
    if k >= len(txt) or txt[k] not in "[{":
        return None
    open_c, close_c = txt[k], ("]" if txt[k] == "[" else "}")
    depth = 0
    for t in range(k, len(txt)):
        if txt[t] == open_c:
            depth += 1
        elif txt[t] == close_c:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[k:t + 1])
                except Exception:
                    return None
    return None


def classify(rec):
    """返回 (slot, family) 或 (None, reason)。family=same 同指数，close 近替代。"""
    code = rec["code"]
    name = rec["name"] or ""
    idxn = rec.get("index_name") or ""
    idxc = (rec.get("index_code") or "").upper()
    blob = idxn + "|" + (rec.get("bench") or "") + "|" + name

    if any(k in name for k in ("国债", "地方债", "国开", "政金")) or "债" in idxn and "ETF" in name:
        if code in CASH_SEEN:
            return None, "cash_already_compared"
        # 只收 5 年及以上国债、或 5/10 年地方债。30 年、短融、0-4 年、国开/政金已在 PR#8 覆盖过类别。
        if "地方债" in blob:
            if any(k in blob for k in ("5年", "10年", "十年")):
                return "cash", "close"
            return None, "cash_tenor_out"
        if "国债" in blob and "政金" not in blob and "国开" not in blob:
            if any(k in blob for k in ("30年", "三十年")):
                return None, "cash_already_compared_30y"
            return "cash", "close"
        return None, "cash_policy_or_other"

    if "现金流" in blob:
        # 同指数：中证全指自由现金流（159232）。近替代：国证自由现金流（980092 一带）和用户点名的 563990。
        if "全指自由现金流" in idxn or idxc in ("932365", "932365CNY010"):
            return "fcf", "same"
        if "980092" in idxc or "国证自由现金流" in idxn:
            return "fcf", "close"
        if code == "563990":
            return "fcf", "close"
        return None, "fcf_other_universe"

    if "红利低波" in blob:
        if any(k in blob for k in ("港股", "恒生", "沪深300", "中证800", "800红利", "A500", "标普", "红利低波50", "低波50")):
            return None, "div_other_universe"
        if "红利低波100" in idxn or "红利低波动100" in idxn or idxc in ("930955",):
            return "div", "same"
        if "红利低波" in idxn or "红利低波动" in idxn:
            return "div", "close"
        return None, "div_unparsed"

    if "纳斯达克100" in idxn or "纳指100" in idxn or idxc in ("NDX", "NDX100", "XNDX"):
        return "nasdaq", "same"
    if code in ("159632", "513300", "513110", "159501") and ("纳斯达克" in blob or "纳指" in blob):
        # 用户点名。指数名若不是 100，标成 close，不假装同指数。
        if "纳斯达克100" in idxn or "纳指100" in idxn:
            return "nasdaq", "same"
        return "nasdaq", "close"
    if "纳斯达克" in blob or "纳指" in blob:
        return None, "nasdaq_not_100"

    if "标普500" in idxn or "标普500" in name:
        if any(k in blob for k in ("ESG", "等权", "低波", "成长", "价值")):
            return None, "spx_other"
        return "spx", "same"

    if "创业板" in blob:
        if "创业板成长" in idxn or "创业板成长" in name:
            return "cyb", "close"
        # 创业板指，排除 50/200/综/行业
        if any(k in idxn for k in ("创业板50", "创业板200", "创业板综合", "创业板综")):
            return None, "cyb_other_index"
        if idxc in ("399006", "SZ399006", "399006.SZ") or idxn in ("创业板指", "创业板指数"):
            return "cyb", "same"
        if "创业板指" in idxn and not any(k in idxn for k in ("50", "200", "综合", "成长", "价值")):
            return "cyb", "same"
        return None, "cyb_other_index"
    return None, "unclassified"


def amt_stats(amount):
    a = amount[amount > 0]
    if a.empty:
        return None, None
    tail = a.iloc[-120:]
    return float(a.mean()), float(tail.mean())


def premium_stats(raw_close, unit_nav):
    df = pd.concat([raw_close.rename("px"), unit_nav.rename("nav")], axis=1).dropna()
    if df.empty:
        return None
    p = df["px"] / df["nav"] - 1.0
    # 份额折算或字段错位：比值离开 [0.5, 1.5] 的日子剔除，不修正。
    bad = (p < -0.5) | (p > 0.5)
    p = p[~bad]
    if len(p) < 20:
        return None
    return {
        "n": int(len(p)),
        "dropped": int(bad.sum()),
        "median": float(p.median()),
        "mean": float(p.mean()),
        "p90": float(p.quantile(0.9)),
        "last": float(p.iloc[-1]),
        "start": str(p.index.min().date()),
        "end": str(p.index.max().date()),
    }


def main():
    funds = fund_list()
    screened = screen(funds)
    print(f"[screen] etf name hits {len(screened)}")
    rows = []
    for i, rec in enumerate(screened):
        try:
            d = detail(rec["code"])
        except Exception as e:
            print(f"[detail fail] {rec['code']} {e}")
            continue
        d["name"] = rec["name"]
        d["ftype"] = rec["ftype"]
        slot, family = classify(d)
        d["slot"] = slot
        d["family"] = family
        rows.append(d)
        if (i + 1) % 25 == 0:
            print(f"  detail {i+1}/{len(screened)}")
        time.sleep(0.12)
    meta = pd.DataFrame(rows)
    eligible = meta[meta["slot"].notna()].copy()
    print("[eligible]")
    print(eligible.groupby(["slot", "family"]).size())

    px_dir = os.path.join(OUT, "px")
    nav_dir = os.path.join(OUT, "nav")
    os.makedirs(px_dir, exist_ok=True)
    os.makedirs(nav_dir, exist_ok=True)

    stats = []
    for _, rec in eligible.iterrows():
        code = rec["code"]
        try:
            q = kline(code, 1)
        except Exception as e:
            print(f"[kline fail] {code} {e}")
            q = pd.DataFrame()
        time.sleep(0.12)
        raw = pd.DataFrame()
        nav = pd.DataFrame()
        if rec["slot"] in ("nasdaq", "spx") or code in INCUMBENT.values():
            try:
                raw = kline(code, 0)
            except Exception as e:
                print(f"[raw fail] {code} {e}")
            time.sleep(0.12)
            try:
                nav = nav_history(code)
            except Exception as e:
                print(f"[nav fail] {code} {e}")
            time.sleep(0.12)
        if not q.empty:
            q.to_csv(os.path.join(px_dir, f"{code}_qfq.csv"))
        if not raw.empty:
            raw.to_csv(os.path.join(px_dir, f"{code}_raw.csv"))
        if not nav.empty:
            nav.to_csv(os.path.join(nav_dir, f"{code}.csv"))
        full_amt, amt120 = (None, None) if q.empty else amt_stats(q["amount"])
        prem = None
        if not raw.empty and not nav.empty and "unit_nav" in nav.columns:
            prem = premium_stats(raw["close"], nav["unit_nav"])
        fee = None
        if rec["mgmt"] is not None and rec["cust"] is not None:
            fee = float(rec["mgmt"]) + float(rec["cust"])
        elif rec["mgmt"] is not None:
            fee = float(rec["mgmt"])
        stats.append({
            "code": code,
            "name": rec["name"],
            "short": rec["short"],
            "slot": rec["slot"],
            "family": rec["family"],
            "index_code": rec["index_code"],
            "index_name": rec["index_name"],
            "estab": rec["estab"],
            "px_start": None if q.empty else str(q.index.min().date()),
            "px_end": None if q.empty else str(q.index.max().date()),
            "px_n": int(len(q)),
            "nav_start": None if nav.empty else str(nav.index.min().date()),
            "nav_end": None if nav.empty else str(nav.index.max().date()),
            "nav_n": int(len(nav)),
            "aum": rec["aum"],
            "aum_date": rec["aum_date"],
            "mgmt": rec["mgmt"],
            "cust": rec["cust"],
            "fee": fee,
            "trk_raw": rec["trk_raw"],
            "amt_full": full_amt,
            "amt_120": amt120,
            "prem": prem,
        })
        print(f"[px] {code} {rec['slot']}/{rec['family']} n={0 if q.empty else len(q)} "
              f"{'' if q.empty else str(q.index.min().date())+'~'+str(q.index.max().date())}")

    # 指数跟踪差：候选前复权 vs 现任前复权，重叠至少 60 日。现任没有行情则留空。
    by = {s["code"]: s for s in stats}
    for s in stats:
        inc = INCUMBENT[s["slot"]]
        s["te_vs_incumbent"] = None
        if s["code"] == inc or s["px_n"] < 60:
            continue
        fp = os.path.join(px_dir, f"{s['code']}_qfq.csv")
        ip = os.path.join(px_dir, f"{inc}_qfq.csv")
        if not (os.path.exists(fp) and os.path.exists(ip)):
            continue
        a = pd.read_csv(fp, parse_dates=["date"]).set_index("date")["close"]
        b = pd.read_csv(ip, parse_dates=["date"]).set_index("date")["close"]
        ra, rb = a.pct_change(), b.pct_change()
        both = pd.concat([ra.rename("a"), rb.rename("b")], axis=1).dropna()
        if len(both) < 60:
            continue
        te = float((both["a"] - both["b"]).std(ddof=1) * math.sqrt(252))
        s["te_vs_incumbent"] = te
        s["te_n"] = int(len(both))

    blob = {"screened": len(screened), "stats": stats,
            "dropped_reason_counts": meta.loc[meta["slot"].isna(), "family"].value_counts().to_dict()}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    print(f"[wrote] {OUT}/meta.json eligible={len(stats)}")

    # 短名单公式（与实验脚本同一份）。只打印，不跑策略。
    print("\n===== RANK =====")
    for slot in ("fcf", "div", "nasdaq", "spx", "cyb", "cash"):
        cand = [s for s in stats if s["slot"] == slot and s["code"] != INCUMBENT[slot] and s["px_n"] >= 40]
        def key(s):
            liq = math.log10(max(s["amt_120"] or 0.0, 1.0))
            fee = s["fee"] if s["fee"] is not None else 0.01
            prem = 0.0
            if slot in ("nasdaq", "spx"):
                prem = -((s["prem"] or {}).get("median") if s["prem"] else 0.05)
            return liq + (-fee) * 50 + prem * 20
        cand.sort(key=key, reverse=True)
        print(f"\n## {slot}")
        for s in cand:
            med = None if not s["prem"] else s["prem"]["median"]
            print(f"  {s['code']} {s['family']:5} key={key(s):.3f} amt120={s['amt_120']} fee={s['fee']} "
                  f"prem={med} aum={s['aum']} estab={s['estab']} px={s['px_start']}~{s['px_end']} "
                  f"idx={s['index_name']} te={s['te_vs_incumbent']}")


if __name__ == "__main__":
    main()
