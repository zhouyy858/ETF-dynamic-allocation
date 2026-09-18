# -*- coding: utf-8 -*-
"""现金层债券 ETF 前复权序列(腾讯)生成: 511010(5Y) 与 511260(10Y) 拼接序列.
口径: 腾讯 fqkline qfq 收盘价(前复权=含分红再投资), 与 skill 内 511010_nav.csv 同源
(日收益相关 1.0000, 最大单日差 0.01pp). 511260 上市前(2017-08-24)用 511010 拼接,
引擎侧无需额外回退逻辑, 且拼接前后均无未来函数.
用法: python3 scripts/fetch_cash_bond.py   # 写 assets/data/511260_nav.csv
"""
import os, time, requests, pandas as pd

H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "data")
SPLICE = "2017-08-24"   # 511260 上市首日


def fq(sym, start="2013-01-01", end="2026-12-31"):
    rows, cur, end_dt = [], pd.Timestamp(start), pd.Timestamp(end)
    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=360), end_dt)
        j = requests.get(URL, params={"param": f"{sym},day,{cur:%Y-%m-%d},{nxt:%Y-%m-%d},400,qfq"},
                         headers=H, timeout=30).json()
        d = (j.get("data") or {}).get(sym) or {}
        for r in (d.get("qfqday") or d.get("day") or []):
            rows.append({"date": r[0], "cum_nav": float(r[2])})
        cur = nxt; time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


n5, n10 = fq("sh511010"), fq("sh511260")
r = n10["cum_nav"].pct_change().dropna()
idx = n5.index.union(n10.index).sort_values()
spliced = pd.Series(index=idx, dtype=float)
base = None
for dt in idx:
    if dt < pd.Timestamp(SPLICE):
        base = float(n5.loc[dt, "cum_nav"]) if dt in n5.index else base
    else:
        if base is None:
            base = float(n5.loc[:dt, "cum_nav"].iloc[-1])
        base = base * (1.0 + float(r.loc[dt])) if dt in r.index else base
    spliced[dt] = base
out = pd.DataFrame({"unit_nav": spliced.values, "cum_nav": spliced.values}, index=spliced.index)
out["ret_pct"] = (spliced.pct_change() * 100).round(4).fillna(0.0)
out.index.name = "date"
out.to_csv(os.path.join(OUT, "511260_nav.csv"))
print(f"[OK] 511260_nav.csv(拼接): {out.index.min().date()}~{out.index.max().date()} n={len(out)} "
      f"累计 {out['cum_nav'].iloc[-1]/out['cum_nav'].iloc[0]-1:.2%}")
