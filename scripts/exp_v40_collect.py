# -*- coding: utf-8 -*-
"""v40 宇宙普查：只拉公开数据，不算策略。

规则在采集之前写死。成交额、费率、指数名可以看；策略收益不能看。
短名单由本文件的公式从普查结果算出，实验脚本会重算并核对。
"""
import os, sys, json, time, math, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v39_collect as src

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "data", "v40_universe")
os.makedirs(OUT, exist_ok=True)

# 当前默认 v39 的成交标的。它们是基线，不作为新候选。
TRADED_NOW = {"159222", "563020", "159941", "513500", "159952", "511260"}
# v37/v39 已经对照过的现金工具，本轮不重复。
CASH_SEEN = {
    "511010", "511260", "511090", "511130", "511520", "511020", "511030",
    "511360", "511380", "518880", "511880", "511990",
    "511100", "511270", "159972",
}
# 近端日均成交额下限（元）。2025-06 以来最多 120 个成交额>0 的交易日。
AMT_FLOOR = 2.0e7
WORKERS = 5
FORCED = {"159509"}

# 桶 -> 替换的信号槽。现金不换权益槽。
SLOT_OF = {
    "nasdaq": "159941",
    "nasdaq_tech": "159941",
    "nasdaq_bio": "159941",
    "hs_tech": "159941",
    "china_net": "159941",
    "spx": "513500",
    "spx_bio": "513500",
    "fcf": "159232",
    "div": "515100",
    "cyb": "159952",
    "kcb": "159952",
    "hs300": "159952",
    "zz500": "159952",
    "zz1000": "159952",
    "a_tech": "159952",
    "cash": None,
    "wildcard": None,  # 跑数前按日收益相关最高的槽位赋值
}
QDII_BUCKETS = {
    "nasdaq", "nasdaq_tech", "nasdaq_bio", "hs_tech", "china_net", "spx", "spx_bio",
}


def bucket_of(name):
    """名称规则。先匹配者优先。增强、杠杆、联接在进这个函数之前已经丢掉。"""
    n = name or ""
    if "纳指科技" in n or "纳斯达克科技" in n:
        if "生物" in n:
            return "nasdaq_bio"
        return "nasdaq_tech"
    if ("纳指" in n or "纳斯达克" in n) and "生物" in n:
        return "nasdaq_bio"
    if "纳指" in n or "纳斯达克" in n:
        return "nasdaq"
    if "标普" in n and "生物" in n:
        return "spx_bio"
    if "标普500" in n and not any(k in n for k in ("ESG", "等权", "低波", "成长", "价值", "红利")):
        return "spx"
    if "恒生科技" in n:
        return "hs_tech"
    if "中概" in n:
        return "china_net"
    if "现金流" in n:
        return "fcf"
    if any(k in n for k in ("港", "恒生", "标普")):
        pass
    elif "红利低波" in n and not any(k in n for k in ("A500", "300", "800", "50")):
        return "div"
    elif "红利质量" in n:
        return "div"
    elif "中证红利" in n:
        return "div"
    if "科创50" in n and "增强" not in n:
        return "kcb"
    if "创业板" in n and not any(k in n for k in (
            "50", "200", "综", "软件", "算力", "电池", "新能源", "人工智能",
            "科技", "医药", "价值", "增强", "金融", "300")):
        return "cyb"
    if "沪深300" in n and not any(k in n for k in (
            "红利", "成长", "价值", "质量", "增强", "现金流", "医药", "金融", "非银", "ESG", "低碳")):
        return "hs300"
    if ("中证500" in n or n.startswith("500ETF")) and not any(k in n for k in (
            "红利", "成长", "价值", "质量", "增强", "现金流", "医药", "金融")):
        return "zz500"
    if "中证1000" in n and not any(k in n for k in ("红利", "成长", "价值", "质量", "增强", "现金流")):
        return "zz1000"
    if ("科技ETF" in n) and not any(k in n for k in (
            "金融", "生物", "信息", "央企", "港", "恒生", "海外", "香港", "沪港深", "东南亚", "纳指")):
        return "a_tech"
    if ("国债" in n or "地方债" in n) and "货币" not in n:
        return "cash"
    return None


def cash_ok(index_name, name):
    """与 v39 同一套期限。1-3 年、0-4 年、30 年、政金、国开不进。"""
    blob = (index_name or "") + "|" + (name or "")
    if any(k in blob for k in ("1-3年", "0-4年", "30年", "三十年", "政金", "国开")):
        return False
    if "地方债" in blob:
        return any(k in blob for k in ("5年", "10年", "十年"))
    if "国债" in blob:
        return True
    return False


def rank_key(amt, fee, prem_median, qdii):
    liq = math.log10(max(amt or 0.0, 1.0))
    fee_v = fee if fee is not None else 0.01
    prem = 0.0
    if qdii:
        prem = -(prem_median if prem_median is not None else 0.05)
    return liq + (-fee_v) * 50 + prem * 20


_tls = threading.local()


def session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(src.H)
        _tls.s = s
    return s


def fetch(url, timeout=25):
    last = None
    for i in range(4):
        try:
            r = session().get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(0.4 + i * 0.6)
    raise last


def detail_local(code):
    url = ("https://fundmobapi.eastmoney.com/FundMNewApi/FundMNDetailInformation"
           f"?FCODE={code}&deviceid=1&plat=Iphone&product=EFund&version=6.3.8")
    d = fetch(url, timeout=25).json().get("Datas") or {}
    burl = ("https://fundmobapi.eastmoney.com/FundMNewApi/FundMNNBasicInformation"
            f"?FCODE={code}&deviceid=1&plat=Iphone&product=EFund&version=6.3.8")
    trk = None
    try:
        b = fetch(burl, timeout=25).json().get("Datas") or {}
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
        "aum": src._f(d.get("ENDNAV")),
        "mgmt": src._pct(d.get("MGREXP")),
        "cust": src._pct(d.get("TRUSTEXP")),
        "trk_raw": trk,
    }


def kline_local(code, fqt):
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={src.secid(code)}&klt=101&fqt={fqt}&beg=20000101&end=20261231&lmt=100000"
           "&fields1=f1&fields2=f51,f53,f57")
    j = fetch(url, timeout=40).json()
    rows = []
    for line in ((j.get("data") or {}).get("klines") or []):
        p = line.split(",")
        if len(p) < 3:
            continue
        rows.append({"date": p[0], "close": float(p[1]),
                     "amount": float(p[2]) if p[2] not in ("", "-") else 0.0})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date").sort_index()


def nav_local(code):
    txt = fetch(f"http://fund.eastmoney.com/pingzhongdata/{code}.js", timeout=40).text
    return src._nav_from_text(txt) if hasattr(src, "_nav_from_text") else _nav_from_text(txt)


def _nav_from_text(txt):
    unit = src._json_var(txt, "Data_netWorthTrend")
    ac = src._json_var(txt, "Data_ACWorthTrend")
    rows = {}
    if isinstance(unit, list):
        for it in unit:
            if not isinstance(it, dict) or "x" not in it or it.get("y") is None:
                continue
            dt = pd.Timestamp(int(it["x"]), unit="ms", tz="UTC").tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            rows.setdefault(dt, {})["unit_nav"] = float(it["y"])
    if isinstance(ac, list):
        for it in ac:
            if not isinstance(it, list) or len(it) < 2 or it[1] is None:
                continue
            dt = pd.Timestamp(int(it[0]), unit="ms", tz="UTC").tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            rows.setdefault(dt, {})["cum_nav"] = float(it[1])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    return df


def short_amt(code):
    """2025-06 以来的日 K，取最近最多 120 个成交额>0 的日子。不拉全历史。"""
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={src.secid(code)}&klt=101&fqt=1&beg=20250601&end=20261231&lmt=400"
           "&fields1=f1&fields2=f51,f53,f57")
    j = fetch(url, timeout=25).json()
    rows = []
    for line in ((j.get("data") or {}).get("klines") or []):
        p = line.split(",")
        if len(p) < 3:
            continue
        amt = float(p[2]) if p[2] not in ("", "-") else 0.0
        rows.append(amt)
    pos = [a for a in rows if a > 0]
    tail = pos[-120:]
    return {
        "code": code,
        "n_recent": len(rows),
        "amt_120": (sum(tail) / len(tail)) if tail else None,
    }


def load_census(path):
    got = {}
    if not os.path.exists(path):
        return got
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            got[rec["code"]] = rec
    return got


def main():
    t0 = time.time()
    funds = src.fund_list()
    etf = []
    for code, _py, name, ftype, _full in funds:
        if "ETF" not in name or "联接" in name:
            continue
        if any(k in name for k in ("杠杆", "反向", "做空", "两倍", "货币")):
            continue
        etf.append({"code": code, "name": name, "ftype": ftype, "bucket": bucket_of(name)})
    print(f"[list] etf {len(etf)} themed {sum(1 for e in etf if e['bucket'])} sec {time.time()-t0:.1f}")

    census_path = os.path.join(OUT, "census.jsonl")
    census = load_census(census_path)
    todo = [e for e in etf if e["code"] not in census]
    print(f"[census] have {len(census)} todo {len(todo)}")
    lock = threading.Lock()
    done = {"n": 0}

    def one(e):
        try:
            rec = short_amt(e["code"])
        except Exception as ex:
            rec = {"code": e["code"], "n_recent": 0, "amt_120": None, "err": str(ex)[:120]}
        rec["name"] = e["name"]
        rec["ftype"] = e["ftype"]
        rec["bucket"] = e["bucket"]
        with lock:
            with open(census_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            census[e["code"]] = rec
            done["n"] += 1
            if done["n"] % 100 == 0:
                print(f"  census {done['n']}/{len(todo)}", flush=True)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(as_completed([ex.submit(one, e) for e in todo]))
    print(f"[census] done {len(census)} sec {time.time()-t0:.1f}")

    # 需要指数名与费率的代码：主题里过成交额线的，加上成交额最高的非主题缓冲。
    themed_pass = []
    for e in etf:
        if not e["bucket"]:
            continue
        c = census.get(e["code"]) or {}
        amt = c.get("amt_120")
        if e["code"] in FORCED or (amt is not None and amt >= AMT_FLOOR and (c.get("n_recent") or 0) >= 40):
            themed_pass.append(e)
    wild_pool = []
    for e in etf:
        if e["bucket"]:
            continue
        if any(k in e["name"] for k in ("债", "黄金", "白银", "豆粕", "货币")):
            continue
        c = census.get(e["code"]) or {}
        amt = c.get("amt_120")
        if amt is None or amt < AMT_FLOOR or (c.get("n_recent") or 0) < 40:
            continue
        wild_pool.append((amt, e))
    wild_pool.sort(key=lambda x: x[0], reverse=True)
    wild_buf = [e for _, e in wild_pool[:40]]
    need_detail = {e["code"]: e for e in themed_pass + wild_buf}
    print(f"[detail] themed_pass {len(themed_pass)} wild_buf {len(wild_buf)}")

    details = {}
    detail_path = os.path.join(OUT, "details.json")
    if os.path.exists(detail_path):
        details = json.load(open(detail_path))

    def one_detail(code):
        try:
            d = detail_local(code)
        except Exception as ex:
            return code, {"code": code, "err": str(ex)[:160]}
        return code, d

    miss = [c for c in need_detail if c not in details]
    if miss:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, fu in enumerate(as_completed([ex.submit(one_detail, c) for c in miss])):
                code, d = fu.result()
                details[code] = d
                if (i + 1) % 40 == 0:
                    print(f"  detail {i+1}/{len(miss)}", flush=True)
                    json.dump(details, open(detail_path, "w"), ensure_ascii=False)
        json.dump(details, open(detail_path, "w"), ensure_ascii=False)
    print(f"[detail] {len(details)} sec {time.time()-t0:.1f}")

    # 现金再滤期限；主题按 (bucket, index) 取成交额/费率前 5，准备溢价。
    staged = []
    for code, e in need_detail.items():
        d = details.get(code) or {}
        if d.get("err"):
            continue
        c = census.get(code) or {}
        bucket = e["bucket"] or "wildcard"
        if bucket == "cash":
            if code in CASH_SEEN or not cash_ok(d.get("index_name") or "", e["name"]):
                continue
        if code in TRADED_NOW:
            continue
        staged.append({
            "code": code,
            "name": e["name"],
            "ftype": e["ftype"],
            "bucket": bucket,
            "index_code": d.get("index_code") or "",
            "index_name": d.get("index_name") or "",
            "estab": d.get("estab") or "",
            "aum": d.get("aum"),
            "mgmt": d.get("mgmt"),
            "cust": d.get("cust"),
            "fee": (None if d.get("mgmt") is None and d.get("cust") is None
                    else (d.get("mgmt") or 0.0) + (d.get("cust") or 0.0)),
            "trk_raw": d.get("trk_raw"),
            "amt_120": c.get("amt_120"),
            "n_recent": c.get("n_recent"),
            "slot": SLOT_OF.get(bucket),
        })

    # 每个 (bucket, index) 先按不含溢价的键留 5 只，QDII 再补溢价后重排留 3。
    groups = {}
    for s in staged:
        key = (s["bucket"], s["index_name"] or s["name"])
        groups.setdefault(key, []).append(s)
    prelim = []
    for key, rows in groups.items():
        rows.sort(key=lambda s: rank_key(s["amt_120"], s["fee"], None, False), reverse=True)
        keep = 5 if key[0] in QDII_BUCKETS or key[0] == "wildcard" else 3
        prelim.extend(rows[:keep])
    # 159509 必留
    have = {s["code"] for s in prelim}
    for s in staged:
        if s["code"] in FORCED and s["code"] not in have:
            prelim.append(s)

    px_dir = os.path.join(OUT, "px")
    nav_dir = os.path.join(OUT, "nav")
    os.makedirs(px_dir, exist_ok=True)
    os.makedirs(nav_dir, exist_ok=True)

    def ensure_full(s):
        code = s["code"]
        qpath = os.path.join(px_dir, f"{code}_qfq.csv")
        if not os.path.exists(qpath):
            q = kline_local(code, 1)
            if not q.empty:
                q.to_csv(qpath)
        qdii = s["bucket"] in QDII_BUCKETS or "海外" in (s.get("ftype") or "")
        if qdii:
            rpath = os.path.join(px_dir, f"{code}_raw.csv")
            npath = os.path.join(nav_dir, f"{code}.csv")
            if not os.path.exists(rpath):
                raw = kline_local(code, 0)
                if not raw.empty:
                    raw.to_csv(rpath)
            if not os.path.exists(npath):
                nav = nav_local(code)
                if nav is not None and not nav.empty:
                    nav.to_csv(npath)
        return code

    print(f"[full] series {len(prelim)}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(ensure_full, s) for s in prelim]
        for i, fu in enumerate(as_completed(futs)):
            try:
                fu.result()
            except Exception as exn:
                print(f"  full fail {exn}")
            if (i + 1) % 20 == 0:
                print(f"  full {i+1}/{len(prelim)}", flush=True)
    print(f"[full] sec {time.time()-t0:.1f}")

    def load_px(code, kind):
        path = os.path.join(px_dir, f"{code}_{kind}.csv")
        if not os.path.exists(path):
            return None
        s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"]
        return s[~s.index.duplicated(keep="last")].sort_index()

    def load_nav(code):
        path = os.path.join(nav_dir, f"{code}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]

    enriched = []
    for s in prelim:
        q = load_px(s["code"], "qfq")
        s = dict(s)
        s["px_n"] = 0 if q is None else int(len(q))
        s["px_start"] = None if q is None or q.empty else str(q.index.min().date())
        s["px_end"] = None if q is None or q.empty else str(q.index.max().date())
        prem = None
        raw = load_px(s["code"], "raw")
        nav = load_nav(s["code"])
        if raw is not None and nav is not None and "unit_nav" in nav.columns:
            prem = src.premium_stats(raw, nav["unit_nav"])
        s["prem"] = prem
        s["nav_n"] = 0 if nav is None else int(len(nav))
        s["qdii"] = s["bucket"] in QDII_BUCKETS or "海外" in (s.get("ftype") or "")
        enriched.append(s)

    # 终榜：QDII 用溢价重排，每组 3。通配先不在这里切，下面按成交额取 20。
    final = []
    runners = []
    grouped = {}
    wild = []
    for s in enriched:
        if s["px_n"] < 40 and s["code"] not in FORCED:
            continue
        if s["bucket"] == "wildcard":
            wild.append(s)
            continue
        grouped.setdefault((s["bucket"], s["index_name"] or s["name"]), []).append(s)
    for key, rows in grouped.items():
        qdii = key[0] in QDII_BUCKETS
        rows.sort(key=lambda s: rank_key(
            s["amt_120"], s["fee"],
            None if not s["prem"] else s["prem"].get("median"),
            qdii), reverse=True)
        for s in rows[:3]:
            s["rank_key"] = rank_key(
                s["amt_120"], s["fee"],
                None if not s["prem"] else s["prem"].get("median"), qdii)
            s["kept"] = True
            final.append(s)
        for s in rows[3:]:
            s["kept"] = False
            s["rank_key"] = rank_key(
                s["amt_120"], s["fee"],
                None if not s["prem"] else s["prem"].get("median"), qdii)
            runners.append(s)
    # 强制 159509
    if not any(s["code"] == "159509" for s in final):
        for s in enriched:
            if s["code"] == "159509":
                s["kept"] = True
                s["rank_key"] = rank_key(
                    s["amt_120"], s["fee"],
                    None if not s["prem"] else s["prem"].get("median"), True)
                final.append(s)
    wild.sort(key=lambda s: rank_key(s["amt_120"], s["fee"], None, False), reverse=True)
    for s in wild[:20]:
        s["kept"] = True
        s["rank_key"] = rank_key(s["amt_120"], s["fee"], None, False)
        final.append(s)

    # 通配槽位：与五只代理日收益的相关，取最高。相关低于 0.30 的仍保留，但实验里不能晋升。
    R = None
    try:
        from data_prep import build_panel
        R, _ = build_panel("proxy")
    except Exception as ex:
        print("[corr] panel fail", ex)
    for s in final:
        if s["bucket"] != "wildcard":
            continue
        q = load_px(s["code"], "qfq")
        s["slot_corr"] = {}
        s["slot"] = "159952"
        s["max_corr"] = None
        if q is None or R is None:
            continue
        rq = q.pct_change()
        best, best_c = "159952", -1.0
        for col in R.columns:
            both = pd.concat([rq.rename("a"), R[col].rename("b")], axis=1).dropna()
            if len(both) < 60:
                s["slot_corr"][col] = None
                continue
            c = float(both["a"].corr(both["b"]))
            s["slot_corr"][col] = c
            if c > best_c:
                best, best_c = col, c
        s["slot"] = best
        s["max_corr"] = None if best_c < 0 else best_c

    # 普查摘要
    amts = [c.get("amt_120") for c in census.values() if c.get("amt_120")]
    amts.sort(reverse=True)
    summary = {
        "n_etf_name": len(etf),
        "n_census": len(census),
        "n_census_err": sum(1 for c in census.values() if c.get("err")),
        "amt_floor": AMT_FLOOR,
        "n_above_floor": sum(1 for c in census.values() if (c.get("amt_120") or 0) >= AMT_FLOOR),
        "n_themed_name": sum(1 for e in etf if e["bucket"]),
        "n_themed_pass": len(themed_pass),
        "n_shortlist": len(final),
        "forced": sorted(FORCED),
        "amt_floor_note": "近端窗口 2025-06-01 起，最多 120 个成交额>0 的交易日的均值，单位元",
    }
    blob = {"summary": summary, "shortlist": final, "runners": runners, "staged_n": len(staged)}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    print(f"[wrote] shortlist {len(final)} sec {time.time()-t0:.1f}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n===== SHORTLIST =====")
    for s in sorted(final, key=lambda x: (x["bucket"], -(x.get("rank_key") or 0))):
        print(f"  {s['bucket']:12} {s['code']} slot={s.get('slot')} "
              f"amt={None if not s['amt_120'] else round(s['amt_120']/1e8, 2)}亿 "
              f"fee={s['fee']} px={s['px_start']}~{s['px_end']} n={s['px_n']} "
              f"idx={s['index_name']} estab={s['estab']} corr={s.get('max_corr')}")


if __name__ == "__main__":
    main()
