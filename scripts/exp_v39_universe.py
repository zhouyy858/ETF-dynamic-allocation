# -*- coding: utf-8 -*-
"""v39 纸面实验：六个槽位在全市场里换更好的工具（2026-11-09）

规则在 PRE 里写死。先提交指纹，再跑数。结果不得回改 PRE、SHORTLIST、RANK。
不改 references/final_cfg_v37.json、README 当前持仓段、SKILL.md、scripts/engine.py。
纸面，不实盘。过关才写 references/final_cfg_v39.json，并标明未激活。

用法:
  python3 scripts/exp_v39_universe.py --fingerprint-only
  python3 scripts/exp_v39_universe.py
"""
import sys, os, json, hashlib, copy, time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from, DATA_DIR
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
UNI = os.path.join(DATA_DIR, "v39_universe")
CFG_FILE = os.path.join(REF, "final_cfg_v37.json")
BOND_PATH = os.path.join(DATA_DIR, "511260_nav.csv")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"

# 公布的 v37 / A100，旧 511260 文件上必须对上。容差 1e-8。
ANCHOR = {
    "proxy": {"cagr": 0.12041732788095327, "mdd": -0.09563306611941624, "cal": 1.2591599617917626},
    "real": {"cagr": 0.24323726154202463, "mdd": -0.03744283692152839, "cal": 6.496229493822656},
    "oos": {"cal": 1.4719682333787487},
}

# 跑数之前按流动性/费用/溢价锁定。排序函数见 rank_key，必须与 meta.json 的前三名一致。
SHORTLIST = {
    "fcf": ["159201", "159222", "563760"],
    "div": ["512890", "563020", "159547"],
    "nasdaq": ["513100", "513300", "159632"],
    "spx": ["513650", "159655", "159612"],
    "cyb": ["159915", "159967", "159977"],
    "cash": ["511100", "511270", "159972"],
}
INCUMBENT = {
    "fcf": "159232", "div": "515100", "nasdaq": "159941",
    "spx": "513500", "cyb": "159952", "cash": "511260",
}
SLOT_OF = {v: k for k, v in INCUMBENT.items()}
NAV_SLOTS = {"nasdaq", "spx"}  # 单换用累计净值，不把溢价涨跌算进收益

PRE_TEXT = """
## 1. 预注册（本节在跑数之前写入，不随结果改写）

对照是 `references/final_cfg_v37.json`。不改这个文件，不改 README 的当前持仓段，不改 SKILL.md，不改 `scripts/engine.py`。纸面，不实盘。默认仍是 v37。

引擎、费用（单边万 5）、信号（`signal_lag=1`、溢价 T-2、先计提后成交、`strict=True`）、周五一笔，与 v37 相同。打分、深熊锁、底仓、溢价门控、溢价倾斜的输入仍用现任五只的原面板。换券只替换该槽的损益序列，不改规则。

窗口：

- proxy 全历史从 2014-06-23 起，终点沿用面板自然末日（与公布 v37 同一日历）
- real 从 2025-04-23 起
- OOS 是 proxy 上的独立起点 2022-01-04（新策略实例）
- 自有样本：从该代码第一根自身收益起到同一末日，基线用同一起点重跑

公布锚（旧的 `assets/data/511260_nav.csv`，未补日期之前）：proxy CAGR 0.12041732788095327、MDD -0.09563306611941624、Calmar 1.2591599617917626；real 0.24323726154202463 / -0.03744283692152839 / 6.496229493822656；OOS Calmar 1.4719682333787487。对不上就中止解释。容差 1e-8。

### 1.1 宇宙怎么搜

东财 `fundcode_search.js` 全名单。场内 ETF（名称含 ETF、不含联接），关键词：现金流、红利低波、纳指、纳斯达克、标普500、创业板、国债、地方债、国开债、政金债。指数名用东财 `FundMNDetailInformation`。管理费、托管费、规模、成立日同一接口。`TRKERROR` 是东财原字段，单位不改写。行情是东财 `push2his` 前复权日 K（收益）和不复权收盘（溢价的分子）。净值是东财 `pingzhongdata` 的单位净值与累计净值。没有前复权行情的代码丢掉，不编造。

资格：

- 自由现金流：指数名含「全指自由现金流」为同指数；「国证自由现金流」或代码 563990 为近替代。300/500/800 现金流其余的只点名，不进短名单。
- 红利低波：指数名含「红利低波100」或「红利低波动100」为同指数；其余 A 股「红利低波/红利低波动」且不含港股、恒生、300、800、A500、50、标普，为近替代。
- 纳指：指数名含「纳斯达克100」或「纳指100」为同指数。
- 标普：指数名含「标普500」，且不含 ESG、等权、低波、成长、价值。
- 创业板：指数是创业板指（399006）为同指数；创业板成长 / 创业板动量成长为近替代。50、200、综指和行业主题排除。
- 现金：只加 PR#8 没比过的。PR#8 已比 511090、511130、511520、511020、511030、511360、511380、518880、511880、511990。511010 与 511260 不是新候选。国开、政金不再加（511520 已代表这一类）。指数名含「1-3年」的短债不进短名单。0-4 年地方债不进。30 年已比过。

跟踪差：候选前复权日收益减现任前复权日收益，重叠至少 60 日，标准差乘 sqrt(252)。这是两只 ETF 之间的差，不是对指数公司公布的跟踪误差。现金现任没有东财文件，跟踪差留空。

### 1.2 短名单（流动性 / 费用 / 溢价，不看策略收益）

现任排除。至少 40 根前复权 K 线。排序键（越大越好）：

`log10(近120日日均成交额) - 50 * 总费率 + （QDII 才有）(-20 * 溢价中位数)`

总费率 = 管理费 + 托管费。缺费率按 1% 计。QDII 缺溢价中位数按 5% 计，用来惩罚没拿到溢价的代码，不拿去填一个假溢价。近 120 日成交额取成交额字段里大于 0 的日子。每个槽取前 3。现金先丢掉指数名含「1-3年」的。

锁定的三只：

- 自由现金流：159201、159222、563760
- 红利低波：512890、563020、159547
- 纳指：513100、513300、159632
- 标普：513650、159655、159612
- 创业板：159915、159967、159977
- 现金：511100、511270、159972

脚本会用 meta.json 重算这张名单，对不上就中止。

### 1.3 收益怎么接

权益单换，一次只动一个槽。

- A 股槽（现金流、红利、创业板）用前复权收盘的日收益（含分红再投资）。
- QDII 槽的单换用累计净值的日收益，好跟 v37 的净值口径比。溢价不进这一臂。
- proxy：自身收益第一天之前，用现任槽的 proxy 收益拼接。拼接日写明。内部缺口最长不超过 5 个交易日、占比不超过 2% 的，那些日子收益记 0 并计数。超过则该臂不能晋升。
- real：不拼接。自身收益盖不住 2025-04-23 到 real 末日的，real 改从双方都有自身收益的第一天起，基线同一起点。这种 real 不能当全窗口，不能过 (a)。
- 现金：不拼接，不接到 511260 或逆回购上。盖不住 2014-06-23 的，proxy 只在缩短窗口上做诊断，不能过 (a)。
- 信号面板始终是现任五只。防御动量、创业板均线、纳指/标普均线都不改用新券。

### 1.4 三关

只评短名单里的主比较，一次换一只。ΔCal 相对同一次跑出来的 v37（511260 已补到最新之后的那条基线）。

(a) proxy 与 real 的 ΔCalmar 都大于 0，至少一个达到 +0.10；proxy MDD 严格好于 -10%（`mdd > -0.10`）。real 必须是标准窗口。proxy 必须从 2014-06-23 可评（现金因此要自身覆盖这段）。

(b) 该槽 2 到 3 只里，至少两只的 proxy ΔCal 与 real ΔCal 都大于 0。只有一只为正，算幸运代码，整槽不晋升。

(c) OOS Calmar 不低于同一次 v37 的 OOS Calmar。

识别（写在跑数之前，防止拼接段假过关）：自有样本不少于 252 个交易日，且这段 proxy ΔCal 与全样本 proxy ΔCal 同号。不满足则 (a)(c) 的数字也不晋升。现金缩短窗口本身已不能过 (a)，识别只用于陈列。

换手只陈列。一个槽里多只都过，晋升其中 proxy Calmar 最高的一只。

### 1.5 QDII 最低溢价载体

与 PR#5 臂 D 不是一回事。臂 D 是纳指 159941 和标普 513500 这两个不同指数之间，T-2 溢价差超过 5 个百分点时，把底仓里至多 2 个百分点从贵的搬到便宜的，收益仍是原来的净值。本臂不搬权重，不换指数。每个周五收盘（以及窗口第一天）在同一指数的 QDII 里，选 T-2 溢价最低的一只，下一交易日起该槽的损益改记这一只的前复权场内价。溢价 = 不复权收盘 / 单位净值 - 1，绝对值大于 50% 的日子丢掉。并列时留现任，否则取代码最小的。没有合格溢价就沿用上一只。

对照臂：两个 QDII 槽始终用现任自己的前复权场内价（上市前仍用 proxy 净值）。规则要晋升，除了 (a)(c) 和识别，还要 proxy Calmar 高于这条「始终现任场内价」。否则多出来的差来自净值改场内价，不是选券。纳指单袖、标普单袖、两袖一起，proxy ΔCal 不得反号（绝对差小于 0.02 视为持平）。决策日里同一只占比超过 85% 的，记成静态等价，不当作一条独立规则晋升。

### 1.6 2026 年以来与 511260

`assets/data/511260_nav.csv` 停在 2026-09-18。仓库里 `scripts/fetch_cash_bond.py` 用腾讯前复权。本实验只在文件末尾追加：用腾讯 `sh511260` 的日收益接到最后一笔累计净值上，不改写 2017-08-24 之前的拼接，也不改写已有日期。重叠段的日收益差只陈列。腾讯失败则不改文件。

YTD 是策略路径在 2026 年的累计收益（状态从 2014 带入），截到权益五槽与 511260 都有原始打印的最后一天。不是年化。

### 1.7 组合与提议

只有槽位过了 (a)(b)(c) 和识别，才进入组合。组合是这些槽同时换成各自晋升的那一只，规则仍是 v37。组合自己要再过 (a)(c) 和识别（组合的自有样本取各槽自有起点里最晚的那天）。过了才写 `references/final_cfg_v39.json`。文件只加 `instrument_swaps`，引擎不读这个字段，默认路径不指向它。单槽过、组合不过，就不写文件。没有任何槽过，也不写。

事先不做的事：不改 v37 的阈值，不重开日频，不把 PR#8 的现金工具再跑一遍，不改 README / SKILL / 引擎。
"""


def fingerprint():
    h = hashlib.sha256()
    h.update(PRE_TEXT.encode())
    h.update(json.dumps(SHORTLIST, sort_keys=True).encode())
    meta = os.path.join(UNI, "meta.json")
    h.update(open(meta, "rb").read())
    return h.hexdigest()[:16]


def rank_key(s, slot):
    liq = float(np.log10(max(s.get("amt_120") or 0.0, 1.0)))
    fee = s["fee"] if s.get("fee") is not None else 0.01
    prem_term = 0.0
    if slot in ("nasdaq", "spx"):
        med = None
        if s.get("prem"):
            med = s["prem"].get("median")
        prem_term = -(med if med is not None else 0.05)
    return liq + (-fee) * 50 + prem_term * 20


def assert_shortlist(meta):
    reasons = []
    for slot, want in SHORTLIST.items():
        cand = []
        for s in meta["stats"]:
            if s["slot"] != slot or s["code"] == INCUMBENT[slot] or s["px_n"] < 40:
                continue
            if slot == "cash" and "1-3年" in (s.get("index_name") or ""):
                continue
            cand.append(s)
        cand.sort(key=lambda s: rank_key(s, slot), reverse=True)
        got = [s["code"] for s in cand[:3]]
        if got != want:
            reasons.append(f"{slot} 短名单 {got} != {want}")
    return reasons


def pct(x):
    return "nan" if x is None or x != x else f"{x * 100:.2f}%"


def calf(x):
    return "nan" if x is None or x != x else f"{x:.2f}"


def dlt(x):
    return "nan" if x is None or x != x else f"{x:+.3f}"


def cell(m):
    if m is None:
        return "—"
    return f"{pct(m['cagr'])} / {pct(m['mdd'])} / {calf(m['cal'])}"


def load_px(code, kind):
    path = os.path.join(UNI, "px", f"{code}_{kind}.csv")
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def load_nav(code):
    path = os.path.join(UNI, "nav", f"{code}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def rets_of(series):
    s = series.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()


def own_return(slot, code):
    if slot in NAV_SLOTS:
        nav = load_nav(code)
        if nav is None or "cum_nav" not in nav.columns:
            return None
        return rets_of(nav["cum_nav"])
    px = load_px(code, "qfq")
    if px is None:
        return None
    return rets_of(px)


def gap_fill(base, own, index):
    """自身收益起点之前用 base。之后小缺口记 0。"""
    own = own.dropna().sort_index()
    first = own.index.min()
    out = base.reindex(index).astype(float).copy()
    after = index[index >= first]
    got = own.reindex(after)
    miss = got.isna()
    n_miss = int(miss.sum())
    longest = 0
    run = 0
    for v in miss.tolist():
        run = run + 1 if v else 0
        longest = max(longest, run)
    frac = n_miss / max(len(after), 1)
    ok = longest <= 5 and frac <= 0.02
    got = got.fillna(0.0)
    out.loc[after] = got.to_numpy()
    info = {
        "first": str(pd.Timestamp(first).date()),
        "n_own_days": int((index >= first).sum()),
        "n_miss": n_miss,
        "longest_gap": longest,
        "gap_frac": frac,
        "gap_ok": bool(ok),
        "spliced": bool(pd.Timestamp(first) > pd.Timestamp(index.min())),
    }
    return out, info


def metrics(res):
    e = evaluate(res)
    return {"cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
            "to": float(e["turnover"]), "end": str(res["rets"].index.max().date()),
            "n": int(len(res["rets"]))}


def run_dyn(R_signal, R_pnl, cfg, start, bond, end=None):
    ds = DynamicStrategy(R_signal, cfg=cfg)
    res = run_backtest(
        R_pnl, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="DYN",
        min_delta=cfg.get("min_delta", 0.035), repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    return metrics(res), res


def ytd_of(rets, strict_end):
    sub = rets[(rets.index.year == 2026) & (rets.index <= strict_end)]
    if len(sub) < 5:
        return None
    w = (1 + sub).cumprod()
    peak = w.cummax()
    mdd = float((w / peak - 1).min())
    return {"total": float(w.iloc[-1] - 1), "mdd": mdd, "n": int(len(sub)),
            "start": str(sub.index.min().date()), "end": str(sub.index.max().date())}


def strict_common(R, bond):
    ok = R.dropna(how="any")
    b = bond.dropna()
    ok = ok[ok.index.isin(b.index)]
    if ok.empty:
        return None
    return ok.index.max()


def refresh_bond(path):
    """只追加腾讯 511260 的新日期。历史行不改。"""
    old = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    last = old.index.max()
    H = {"User-Agent": "Mozilla/5.0"}
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    rows, cur, end_dt = [], pd.Timestamp("2017-08-01"), pd.Timestamp("2026-12-31")
    try:
        while cur < end_dt:
            nxt = min(cur + pd.Timedelta(days=360), end_dt)
            j = requests.get(url, params={"param": f"sh511260,day,{cur:%Y-%m-%d},{nxt:%Y-%m-%d},400,qfq"},
                             headers=H, timeout=30).json()
            d = (j.get("data") or {}).get("sh511260") or {}
            for r in (d.get("qfqday") or d.get("day") or []):
                rows.append({"date": r[0], "close": float(r[2])})
            cur = nxt
            time.sleep(0.2)
    except Exception as e:
        return {"ok": False, "reason": str(e)[:160], "appended": 0, "old_end": str(last.date())}
    if not rows:
        return {"ok": False, "reason": "腾讯空", "appended": 0, "old_end": str(last.date())}
    px = pd.DataFrame(rows).drop_duplicates("date")
    px["date"] = pd.to_datetime(px["date"])
    px = px.set_index("date").sort_index()["close"]
    tr = px.pct_change().dropna()
    old_r = old["cum_nav"].pct_change().dropna()
    both = pd.concat([old_r.rename("o"), tr.rename("n")], axis=1).dropna()
    both = both[both.index >= "2017-08-24"]
    max_abs = float((both["o"] - both["n"]).abs().max()) if len(both) else None
    extra = tr[tr.index > last].dropna()
    nav = float(old["cum_nav"].iloc[-1])
    lines = []
    for dt, ret in extra.items():
        nav = nav * (1.0 + float(ret))
        lines.append(f"{pd.Timestamp(dt).date()},{nav:.10f},{nav:.10f},{float(ret) * 100:.4f}\n")
    if lines:
        with open(path, "a") as f:
            f.writelines(lines)
    new_end = str(extra.index.max().date()) if len(extra) else str(last.date())
    return {"ok": True, "reason": "", "appended": len(lines), "old_end": str(last.date()),
            "new_end": new_end, "overlap_n": int(len(both)), "overlap_max_abs": max_abs}


def premium_series(code):
    raw = load_px(code, "raw")
    nav = load_nav(code)
    if raw is None or nav is None or "unit_nav" not in nav.columns:
        return None
    df = pd.concat([raw.rename("px"), nav["unit_nav"].rename("nav")], axis=1).dropna()
    p = df["px"] / df["nav"] - 1.0
    p = p[(p >= -0.5) & (p <= 0.5)]
    return p


def carrier_book(index, base, peers, incumbent, shift=2, force=None):
    """周五收盘选定的载体，从下一交易日起计提。force 给定则始终用那一只。"""
    prem = pd.DataFrame({c: premium_series(c) for c, _q in peers})
    prem = prem.reindex(index).shift(shift)
    qmap = {c: q.reindex(index) for c, q in peers}
    current = None
    out = []
    decisions = []
    for i, dt in enumerate(index):
        use = current
        if use is None or pd.isna(qmap[use].iloc[i]):
            val = base.iloc[i]
            out.append(0.0 if pd.isna(val) else float(val))
            src = "base"
        else:
            out.append(float(qmap[use].iloc[i]))
            src = use
        is_decision = i == 0 or dt.weekday() == 4
        if is_decision:
            if force:
                if not pd.isna(qmap[force].iloc[i]) or (prem is not None and force in prem.columns and pd.notna(prem[force].iloc[i])):
                    current = force
                pick = force
            else:
                row = prem.iloc[i].dropna()
                # 只选当天之后真有行情可记的
                row = row[[c for c in row.index if c in qmap]]
                if len(row):
                    m = float(row.min())
                    cands = [c for c in row.index if abs(float(row[c]) - m) <= 1e-12]
                    pick = incumbent if incumbent in cands else sorted(cands)[0]
                    current = pick
                else:
                    pick = current
            decisions.append(pick)
        _ = src
    counts = {}
    for d in decisions:
        if d is None:
            continue
        counts[d] = counts.get(d, 0) + 1
    n = sum(counts.values()) or 1
    modal = max(counts, key=counts.get) if counts else None
    share = (counts.get(modal, 0) / n) if modal else 0.0
    return pd.Series(out, index=index), {"n_decisions": len(decisions), "counts": counts,
                                        "modal": modal, "modal_share": share}


def near(m, pin, keys, tol=1e-8):
    bad = []
    for k in keys:
        if abs(m[k] - pin[k]) > tol:
            bad.append(f"{k}:{m[k]:.12g}!={pin[k]:.12g}")
    return bad


def gate_nums(base, row, full_real, full_proxy):
    if row["proxy"] is None or row["real"] is None:
        return {"ok": False, "same": False, "one": False, "mdd": False,
                "dp": None, "dr": None, "full": False}
    dp = row["proxy"]["cal"] - base["proxy"]["cal"]
    dr = row["real"]["cal"] - base["real"]["cal"] if full_real else None
    same = bool(full_real and full_proxy and dp > 0 and dr > 0)
    one = bool(same and max(dp, dr) >= 0.10)
    mdd = bool(row["proxy"]["mdd"] > -0.10)
    return {"ok": bool(same and one and mdd), "same": same, "one": one, "mdd": mdd,
            "dp": dp, "dr": dr, "full": bool(full_real and full_proxy)}


def main():
    digest = fingerprint()
    print(f"[预注册指纹] {digest}")
    meta = json.load(open(os.path.join(UNI, "meta.json")))
    bad_list = assert_shortlist(meta)
    if bad_list:
        print("[短名单不一致]", bad_list)
        if "--fingerprint-only" not in sys.argv:
            sys.exit(2)
    if "--fingerprint-only" in sys.argv:
        return

    cfg = json.load(open(CFG_FILE))
    bond_old = rets_from(read_table("511260_nav.csv"), "cum_nav")
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    print(f"[数据] proxy {R.index.min().date()}~{R.index.max().date()} real {Rr.index.min().date()}~{Rr.index.max().date()}")

    anchor = {}
    anchor_bad = []
    for win, panel, start, keys in (
        ("proxy", R, PROXY_START, ("cagr", "mdd", "cal")),
        ("real", Rr, REAL_START, ("cagr", "mdd", "cal")),
        ("oos", R, OOS_START, ("cal",)),
    ):
        m, _res = run_dyn(panel, panel, cfg, start, bond_old)
        anchor[win] = m
        anchor_bad += [f"{win} " + b for b in near(m, ANCHOR[win], keys)]
    print("[锚]", "通过" if not anchor_bad else anchor_bad)

    info_bond = refresh_bond(BOND_PATH)
    print("[511260]", info_bond)
    bond = rets_from(read_table("511260_nav.csv"), "cum_nav")
    common_p = strict_common(R.loc[R.index >= PROXY_START], bond)
    common_r = strict_common(Rr.loc[Rr.index >= REAL_START], bond)
    common = min(common_p, common_r)
    print("[共同末日]", common.date(), "proxy", common_p.date(), "real", common_r.date())

    base = {}
    base_rets = {}
    for win, panel, start in (("proxy", R, PROXY_START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
        m, res = run_dyn(panel, panel, cfg, start, bond)
        base[win] = m
        base_rets[win] = res["rets"]
        print(f"[基线刷新后] {win} {cell(m)}")

    rows = {}
    # 权益单换
    for slot, codes in SHORTLIST.items():
        if slot == "cash":
            continue
        col = INCUMBENT[slot]
        for code in codes:
            own_p = own_return(slot, code)
            name = f"{slot}:{code}"
            rec = {"slot": slot, "code": code, "kind": "swap"}
            if own_p is None or own_p.empty:
                rec["error"] = "无收益"
                rows[name] = rec
                print("[缺]", name)
                continue
            pnl, ginfo = gap_fill(R[col], own_p, R.index)
            Rp = R.copy()
            Rp[col] = pnl
            # real：不拼接
            own_on_real = own_p.reindex(Rr.index)
            real_start_dt = own_p.index.min()
            full_real = bool(real_start_dt <= pd.Timestamp(REAL_START))
            if full_real:
                # 标准窗口内的缺口
                seg = own_p.reindex(Rr.index[Rr.index >= REAL_START])
                miss = int(seg.isna().sum())
                longest = 0
                run = 0
                for v in seg.isna().tolist():
                    run = run + 1 if v else 0
                    longest = max(longest, run)
                if longest > 5 or miss / max(len(seg), 1) > 0.02:
                    full_real = False
                    rec["real_gap"] = {"miss": miss, "longest": longest}
                Rr2 = Rr.copy()
                Rr2[col] = seg.fillna(0.0).reindex(Rr.index).combine_first(Rr[col])
                # 标准窗口之前的 real 行不会被回测用到。窗口内用自身收益。
                rr_start = REAL_START
            else:
                Rr2 = Rr.copy()
                # 缩短：起点之后全部用自身，不借现任
                cut = Rr.index[Rr.index >= real_start_dt]
                Rr2.loc[cut, col] = own_p.reindex(cut).fillna(0.0)
                rr_start = str(pd.Timestamp(real_start_dt).date())
            rec["splice"] = ginfo
            rec["full_real"] = full_real
            rec["full_proxy"] = True
            rec["real_start"] = rr_start
            mp, resp = run_dyn(R, Rp, cfg, PROXY_START, bond)
            mr, _ = run_dyn(Rr, Rr2, cfg, rr_start, bond)
            mo, _ = run_dyn(R, Rp, cfg, OOS_START, bond)
            # 自有样本
            own_start = str(max(pd.Timestamp(ginfo["first"]), pd.Timestamp(PROXY_START)).date())
            if pd.Timestamp(own_start) <= pd.Timestamp(PROXY_START):
                m_own = mp
                m_own_base = base["proxy"]
            else:
                m_own, _ = run_dyn(R, Rp, cfg, own_start, bond)
                m_own_base, _ = run_dyn(R, R, cfg, own_start, bond)
            rec["proxy"] = mp
            rec["real"] = mr
            rec["oos"] = mo
            rec["own"] = m_own
            rec["own_base"] = m_own_base
            rec["own_start"] = own_start
            rec["own_n"] = m_own["n"]
            rec["proxy_rets"] = resp["rets"]
            rows[name] = rec
            print(f"[换] {name} proxy {cell(mp)} real {cell(mr)} oos {calf(mo['cal'])} splice {ginfo['first']}")

    # 现金
    for code in SHORTLIST["cash"]:
        px = load_px(code, "qfq")
        name = f"cash:{code}"
        rec = {"slot": "cash", "code": code, "kind": "cash"}
        if px is None:
            rec["error"] = "无行情"
            rows[name] = rec
            continue
        br = rets_of(px)
        first = br.index.min()
        full_proxy = bool(first <= pd.Timestamp(PROXY_START))
        start_p = PROXY_START if full_proxy else str(first.date())
        # 标准 real 是否被覆盖
        full_real = bool(first <= pd.Timestamp(REAL_START))
        mp, resp = run_dyn(R, R, cfg, start_p, br)
        mb, _ = run_dyn(R, R, cfg, start_p, bond)
        if full_real:
            mr, _ = run_dyn(Rr, Rr, cfg, REAL_START, br)
            mrb = base["real"]
            rr_start = REAL_START
        else:
            mr, mrb, rr_start = None, None, None
        if full_proxy:
            mo, _ = run_dyn(R, R, cfg, OOS_START, br)
        else:
            # OOS 若晚于自身起点，仍可算；否则从自身起点
            oos_start = OOS_START if first <= pd.Timestamp(OOS_START) else str(first.date())
            mo, _ = run_dyn(R, R, cfg, oos_start, br)
            rec["oos_start"] = oos_start
        rec.update({"proxy": mp, "proxy_base": mb, "real": mr, "real_base": mrb,
                    "oos": mo, "full_proxy": full_proxy, "full_real": full_real,
                    "real_start": rr_start, "own_start": str(first.date()), "own_n": mp["n"],
                    "own": mp, "own_base": mb, "splice": {"first": str(first.date()), "spliced": False,
                                                         "gap_ok": True, "n_miss": 0},
                    "proxy_rets": resp["rets"]})
        rows[name] = rec
        print(f"[现金] {code} 缩短起点 {start_p} {cell(mp)} vs基线 {cell(mb)} full_proxy={full_proxy}")

    # 载体
    peers = {}
    for slot, inc in (("nasdaq", "159941"), ("spx", "513500")):
        group = []
        for s in meta["stats"]:
            if s["slot"] == slot and s["family"] == "same":
                q = load_px(s["code"], "qfq")
                if q is None or premium_series(s["code"]) is None:
                    continue
                group.append((s["code"], rets_of(q)))
        # 现任可能已在 stats 里
        if inc not in [c for c, _ in group]:
            q = load_px(inc, "qfq")
            if q is not None and premium_series(inc) is not None:
                group.append((inc, rets_of(q)))
        peers[slot] = group
        print(f"[载体池] {slot} {[c for c,_ in group]}")

    def apply_carrier(which, force=None):
        Rp = R.copy()
        diag = {}
        for slot in ("nasdaq", "spx"):
            if which != "both" and which != slot:
                continue
            inc = INCUMBENT[slot]
            book, dg = carrier_book(R.index, R[inc], peers[slot], inc, shift=2,
                                    force=(inc if force == "incumbent" else force))
            Rp[inc] = book
            diag[slot] = dg
        return Rp, diag

    carrier_jobs = {
        "carrier:both": ("both", None),
        "carrier:nasdaq": ("nasdaq", None),
        "carrier:spx": ("spx", None),
        "carrier:control": ("both", "incumbent"),
    }
    for name, (which, force) in carrier_jobs.items():
        Rp, dg = apply_carrier(which, force)
        mp, resp = run_dyn(R, Rp, cfg, PROXY_START, bond)
        mr, _ = run_dyn(Rr if False else Rr, _align_real_carrier(Rr, R, Rp), cfg, REAL_START, bond)
        mo, _ = run_dyn(R, Rp, cfg, OOS_START, bond)
        # 自有样本：两袖都至少有一只行情的第一天。用控制序列没有拼接问题，取 proxy 全样本。
        rows[name] = {"slot": "carrier", "code": name, "kind": "carrier", "which": which,
                      "proxy": mp, "real": mr, "oos": mo, "diag": dg,
                      "full_real": True, "full_proxy": True,
                      "splice": {"spliced": False, "gap_ok": True, "first": PROXY_START},
                      "own": mp, "own_base": base["proxy"], "own_start": PROXY_START, "own_n": mp["n"],
                      "proxy_rets": resp["rets"]}
        print(f"[载体] {name} {cell(mp)} real {cell(mr)} oos {calf(mo['cal'])}")

    # 评判
    verdicts = {}
    for name, rec in rows.items():
        if rec.get("error"):
            verdicts[name] = {"pass": False, "why": rec["error"]}
            continue
        bproxy = base["proxy"]
        breal = base["real"]
        boos = base["oos"]["cal"]
        if rec["kind"] == "cash":
            bproxy = rec["proxy_base"]
            # 现金的 real 基线是全窗口 v37；缩短 proxy 的基线是同一起点
        g = gate_nums({"proxy": bproxy, "real": breal}, rec, rec.get("full_real", False), rec.get("full_proxy", False))
        if rec["kind"] == "cash":
            g["ok"] = False  # 盖不住 2014 就不能过 (a)，即便数字上同向
            g["why_cash"] = "现金不拼接，proxy 不是 2014 起的全窗口"
        oos_ok = rec["oos"]["cal"] >= boos - 1e-12 if rec.get("full_proxy") else False
        if rec["kind"] == "cash":
            oos_ok = False
        dp_own = rec["own"]["cal"] - rec["own_base"]["cal"]
        dp_full = None if rec["proxy"] is None or bproxy is None else rec["proxy"]["cal"] - bproxy["cal"]
        same_sign = (dp_own > 0 and dp_full is not None and dp_full > 0) or (dp_own < 0 and dp_full is not None and dp_full < 0)
        identified = bool(rec["own_n"] >= 252 and same_sign and rec["splice"].get("gap_ok", False))
        if rec["kind"] != "cash" and not rec["splice"].get("gap_ok", False):
            identified = False
        verdicts[name] = {"gate": g, "oos_ok": bool(oos_ok), "identified": identified,
                          "dp_own": dp_own, "pass_alone": bool(g["ok"] and oos_ok and identified)}

    # (b) 槽内至少两只同向为正
    slot_pass = {}
    for slot, codes in SHORTLIST.items():
        names = [f"{slot}:{c}" for c in codes]
        good_dir = []
        for n in names:
            g = verdicts.get(n, {}).get("gate") or {}
            if g.get("full") and g.get("dp") is not None and g.get("dr") is not None and g["dp"] > 0 and g["dr"] > 0:
                good_dir.append(n)
        alone = [n for n in names if verdicts.get(n, {}).get("pass_alone")]
        robust = len(good_dir) >= 2
        winner = None
        if robust and alone:
            winner = max(alone, key=lambda n: rows[n]["proxy"]["cal"])
        slot_pass[slot] = {"robust": robust, "good_dir": good_dir, "alone": alone, "winner": winner}

    # 载体额外条件
    ctrl = rows.get("carrier:control")
    both = rows.get("carrier:both")
    carrier_pass = False
    carrier_why = []
    if both and ctrl and verdicts["carrier:both"]["pass_alone"]:
        if both["proxy"]["cal"] <= ctrl["proxy"]["cal"]:
            carrier_why.append("没有高于始终现任场内价")
        else:
            signs = []
            for n in ("carrier:both", "carrier:nasdaq", "carrier:spx"):
                dp = verdicts[n]["gate"]["dp"]
                signs.append(dp)
            # 反号：一正一负且绝对差都 >= 0.02
            pos = [s for s in signs if s is not None and s >= 0.02]
            neg = [s for s in signs if s is not None and s <= -0.02]
            if pos and neg:
                carrier_why.append("单袖与合计反号")
            else:
                share = 0.0
                modal = None
                for sl, dg in both["diag"].items():
                    if dg["modal_share"] > share:
                        share = dg["modal_share"]
                        modal = dg["modal"]
                if share > 0.85:
                    carrier_why.append(f"决策日 {modal} 占 {share:.0%}，静态等价")
                else:
                    carrier_pass = True
    else:
        carrier_why.append("联合臂没有过 (a)(c) 与识别")
    if not carrier_pass and not carrier_why:
        carrier_why.append("未过")

    # 组合
    winners = {slot: slot_pass[slot]["winner"] for slot in SHORTLIST if slot_pass[slot]["winner"]}
    combo = None
    combo_pass = False
    if len(winners) >= 1:
        Rp = R.copy()
        Rr2 = Rr.copy()
        own_starts = []
        gap_ok = True
        for slot, name in winners.items():
            code = rows[name]["code"]
            col = INCUMBENT[slot]
            own = own_return(slot, code)
            pnl, ginfo = gap_fill(R[col], own, R.index)
            Rp[col] = pnl
            gap_ok = gap_ok and ginfo["gap_ok"]
            own_starts.append(pd.Timestamp(ginfo["first"]))
            seg = own.reindex(Rr.index[Rr.index >= REAL_START]).fillna(0.0)
            Rr2[col] = seg.reindex(Rr.index).combine_first(Rr[col])
        mp, resp = run_dyn(R, Rp, cfg, PROXY_START, bond)
        mr, _ = run_dyn(Rr, Rr2, cfg, REAL_START, bond)
        mo, _ = run_dyn(R, Rp, cfg, OOS_START, bond)
        own_start = str(max(own_starts + [pd.Timestamp(PROXY_START)]).date())
        if pd.Timestamp(own_start) <= pd.Timestamp(PROXY_START):
            m_own, m_own_b = mp, base["proxy"]
        else:
            m_own, _ = run_dyn(R, Rp, cfg, own_start, bond)
            m_own_b, _ = run_dyn(R, R, cfg, own_start, bond)
        combo = {"proxy": mp, "real": mr, "oos": mo, "own": m_own, "own_base": m_own_b,
                 "own_start": own_start, "own_n": m_own["n"], "rets": resp["rets"], "winners": winners}
        g = gate_nums(base, combo, True, True)
        oos_ok = mo["cal"] >= base["oos"]["cal"] - 1e-12
        dp_own = m_own["cal"] - m_own_b["cal"]
        dp_full = mp["cal"] - base["proxy"]["cal"]
        same_sign = (dp_own > 0 and dp_full > 0) or (dp_own < 0 and dp_full < 0)
        identified = m_own["n"] >= 252 and same_sign and gap_ok
        combo_pass = bool(g["ok"] and oos_ok and identified)
        combo["gate"] = g
        combo["oos_ok"] = oos_ok
        combo["identified"] = identified
        combo["pass"] = combo_pass
        print(f"[组合] {cell(mp)} real {cell(mr)} pass={combo_pass}")

    cfg_path = None
    if combo_pass:
        out_cfg = copy.deepcopy(cfg)
        swaps = {INCUMBENT[slot]: rows[name]["code"] for slot, name in winners.items()}
        out_cfg["version"] = "v39"
        out_cfg["instrument_swaps"] = swaps
        out_cfg["_note"] = (
            "v39 提议、未激活。默认仍是 final_cfg_v37.json。引擎不读 instrument_swaps，"
            "把本文件交给 run_backtest.py 不会换券。只记录同时过关的替换。"
            f" proxy {combo['proxy']['cagr']*100:.2f}%/{combo['proxy']['mdd']*100:.2f}%/Cal{combo['proxy']['cal']:.2f}"
            f" real {combo['real']['cagr']*100:.2f}%/{combo['real']['mdd']*100:.2f}%/Cal{combo['real']['cal']:.2f}"
            f" OOS Cal {combo['oos']['cal']:.4f}。"
        )
        cfg_path = os.path.join(REF, "final_cfg_v39.json")
        with open(cfg_path, "w") as f:
            json.dump(out_cfg, f, ensure_ascii=False, indent=1)
            f.write("\n")

    # YTD
    ytd = {"v37_proxy": ytd_of(base_rets["proxy"], common),
           "v37_real": ytd_of(base_rets["real"], common),
           "anchor_proxy": ytd_of(anchor.get("_rets_p", base_rets["proxy"]), common)}
    # 锚的路径没有存。补一次旧债 YTD 需要重跑。上面 anchor 没留 rets。重跑旧债一次取 YTD。
    _m, res_old = run_dyn(R, R, cfg, PROXY_START, bond_old)
    _mr, res_old_r = run_dyn(Rr, Rr, cfg, REAL_START, bond_old)
    ytd["old_proxy"] = ytd_of(res_old["rets"], min(common, res_old["rets"].index.max()))
    ytd["old_real"] = ytd_of(res_old_r["rets"], min(common, res_old_r["rets"].index.max()))
    ytd["v37_proxy"] = ytd_of(base_rets["proxy"], common)
    ytd["v37_real"] = ytd_of(base_rets["real"], common)
    for name, rec in rows.items():
        if rec.get("proxy_rets") is not None and rec.get("full_proxy"):
            rec["ytd_proxy"] = ytd_of(rec["proxy_rets"], common)
        elif rec.get("proxy_rets") is not None:
            rec["ytd_proxy"] = ytd_of(rec["proxy_rets"], min(common, rec["proxy_rets"].index.max()))

    report = render(digest, meta, info_bond, anchor, anchor_bad, base, rows, verdicts,
                    slot_pass, carrier_pass, carrier_why, combo, ytd, common, cfg_path)
    rep_path = os.path.join(REF, "exp_v39_universe_report.md")
    with open(rep_path, "w") as f:
        f.write(report)
    # json 不能存 Series
    dump_rows = {}
    for k, rec in rows.items():
        dump_rows[k] = {kk: vv for kk, vv in rec.items() if kk not in ("proxy_rets",)}
    dump = {
        "digest": digest, "anchor_bad": anchor_bad, "bond": info_bond,
        "common": str(common.date()), "base": base, "anchor": {k: {kk: vv for kk, vv in m.items()} for k, m in anchor.items()},
        "verdicts": verdicts, "slot_pass": slot_pass, "carrier_pass": carrier_pass,
        "carrier_why": carrier_why, "combo_pass": combo_pass,
        "combo": None if combo is None else {kk: vv for kk, vv in combo.items() if kk != "rets"},
        "ytd": ytd, "cfg": cfg_path,
    }
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (pd.Timestamp,)):
            return str(o.date())
        return o
    with open(os.path.join(REF, "exp_v39_universe_results.json"), "w") as f:
        json.dump(conv(dump), f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("[报告]", rep_path)


def _align_real_carrier(Rr, R, Rp):
    """real 窗口用 proxy 上已经按规则记好的载体收益（同一条场内价路径）。"""
    out = Rr.copy()
    for col in ("159941", "513500"):
        if not Rp[col].equals(R[col]):
            out[col] = Rp[col].reindex(Rr.index).combine_first(Rr[col])
    return out


def render(digest, meta, info_bond, anchor, anchor_bad, base, rows, verdicts,
           slot_pass, carrier_pass, carrier_why, combo, ytd, common, cfg_path):
    L = []
    A = L.append
    A("# v39 纸面实验：六个槽位的全市场换券")
    A("")
    A(f"预注册指纹 `{digest}`。")
    A(f"共同末日（五槽与 511260 都有原始打印）{common.date()}。")
    A("第 1 节是跑数之前写死的规则。第 2 节才是结果。结论在第 3 节，规则没有按结果回改。")
    A("默认配置仍是 `references/final_cfg_v37.json`。README 当前持仓段和 SKILL.md 没有改。")
    A("")
    A(PRE_TEXT.strip())
    A("")
    A("### 1.8 枚举到的数目（事实，来自已提交的 meta.json）")
    A("")
    A(f"名称筛到 {meta['screened']} 只场内 ETF。资格内 {len(meta['stats'])} 只有行情。")
    dropped = meta.get("dropped_reason_counts") or {}
    if dropped:
        bits = "、".join(f"{k} {v}" for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]))
        A(f"筛掉的原因计数：{bits}。")
    A("")
    A("短名单与紧挨着没进前三的事实：")
    A("")
    A("| 槽 | 代码 | 家族 | 指数 | 成立 | 行情 | 规模(亿) | 费率 | 近120日成交额(亿) | 溢价中位数 | 与现任前复权跟踪差 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    by = {s["code"]: s for s in meta["stats"]}
    for slot, codes in list(SHORTLIST.items()) + []:
        for code in codes:
            s = by.get(code)
            if not s:
                A(f"| {slot} | {code} | — | 无 meta | | | | | | | |")
                continue
            aum = "—" if not s.get("aum") else f"{s['aum']/1e8:.1f}"
            fee = "—" if s.get("fee") is None else f"{s['fee']*100:.2f}%"
            amt = "—" if not s.get("amt_120") else f"{s['amt_120']/1e8:.2f}"
            prem = "—" if not s.get("prem") else f"{s['prem']['median']*100:.2f}%"
            te = "—" if s.get("te_vs_incumbent") is None else f"{s['te_vs_incumbent']*100:.2f}%"
            A(f"| {slot} | {code} | {s['family']} | {s['index_name']} | {s['estab']} | {s['px_start']}~{s['px_end']} | {aum} | {fee} | {amt} | {prem} | {te} |")
    A("")
    A("东财 TRKERROR 原字段（不换算单位）：")
    A("")
    for slot, codes in SHORTLIST.items():
        bits = []
        for code in codes:
            s = by.get(code) or {}
            bits.append(f"{code}={s.get('trk_raw')}")
        A(f"- {slot}：" + "，".join(bits))
    A("")
    A("现金里看到但没进短名单的：511060（上证5年期地方政府债，成交额排第 4）、511160（中证1-3年国债，短于事先的 5 年下限）。")
    A("159967 的指数名是创业板动量成长，不是创业板成长。创业板成长指数的 159597 成交额更低，没进前三。")
    A("511100 的指数名是上证基准做市国债指数，不是十年期国债。用户举例时把它和十年债放在一起，数据对不上，按指数名写。")
    A("513100 的净值文件里有空点，已跳过空点后重算溢价。全历史溢价中位数约 0.48%，2026 年中位数明显高于此，末日溢价与 159941 同在两位数。排序用的是全历史中位数，这是事先的公式。")
    A("")
    A("## 2. 结果")
    A("")
    if anchor_bad:
        A("身份锚没有对上：" + "；".join(anchor_bad) + "。下面的候选不作数。")
    else:
        A("身份锚对上公布的 v37（旧 511260 文件，补日期之前）。")
    if info_bond.get("ok"):
        ov = info_bond.get("overlap_max_abs")
        ov_s = "无" if ov is None else f"{ov:.3e}"
        A(f"511260 从 {info_bond['old_end']} 追加 {info_bond['appended']} 日，新末日 {info_bond.get('new_end')}。"
          f"2017-08-24 之后重叠 {info_bond.get('overlap_n')} 日，日收益最大绝对差 {ov_s}。历史行没有改写。")
    else:
        A(f"511260 没有补上：{info_bond.get('reason')}。")
    A("")
    A(f"**刷新后的 v37**：proxy {cell(base['proxy'])}；real {cell(base['real'])}；OOS Cal {base['oos']['cal']:.4f}。")
    A(f"公布对照：proxy 12.04% / -9.56% / 1.26，real 24.32% / -3.74% / 6.50，OOS Cal 1.472。")
    A("ΔCal 相对刷新后的这一条，不是相对四舍五入的公布值。")
    A("")
    A("| 臂 | 拼接/起点 | proxy | ΔCal | real | ΔCal | OOS Cal | (a) | 识别 | 单过 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    A(f"| v37 | — | {cell(base['proxy'])} | — | {cell(base['real'])} | — | {base['oos']['cal']:.4f} | — | — | — |")
    for slot, codes in SHORTLIST.items():
        for code in codes:
            name = f"{slot}:{code}"
            rec = rows.get(name) or {}
            v = verdicts.get(name) or {}
            g = v.get("gate") or {}
            if rec.get("error"):
                A(f"| {name} | 无数据 | — | — | — | — | — | 否 | 否 | 否 |")
                continue
            sp = rec.get("splice") or {}
            start = sp.get("first", "")
            tag = "拼接自 " + start if sp.get("spliced") else "自身起 " + start
            if not rec.get("full_proxy", True):
                tag = "缩短 " + start
            dp = dlt(g.get("dp"))
            dr = dlt(g.get("dr")) if rec.get("full_real") else "非全窗"
            real_cell = cell(rec.get("real"))
            if rec["kind"] == "cash":
                real_cell = cell(rec.get("real")) + ("" if rec.get("full_real") else "")
                dp = dlt(None if rec.get("proxy") is None else rec["proxy"]["cal"] - rec["proxy_base"]["cal"])
                if rec.get("full_real") and rec.get("real") is not None:
                    dr = dlt(rec["real"]["cal"] - base["real"]["cal"])
            A(f"| {name} | {tag} | {cell(rec.get('proxy'))} | {dp} | {real_cell} | {dr} | {calf((rec.get('oos') or {}).get('cal'))} | "
              f"{'是' if g.get('ok') else '否'} | {'是' if v.get('identified') else '否'} | {'是' if v.get('pass_alone') else '否'} |")
    for name in ("carrier:both", "carrier:nasdaq", "carrier:spx", "carrier:control"):
        rec = rows.get(name) or {}
        v = verdicts.get(name) or {}
        g = v.get("gate") or {}
        if not rec:
            continue
        A(f"| {name} | 场内价 | {cell(rec.get('proxy'))} | {dlt(g.get('dp'))} | {cell(rec.get('real'))} | {dlt(g.get('dr'))} | "
          f"{calf((rec.get('oos') or {}).get('cal'))} | {'是' if g.get('ok') else '否'} | {'是' if v.get('identified') else '否'} | "
          f"{'是' if v.get('pass_alone') else '否'} |")
    A("")
    A("### 2.1 槽位结论")
    A("")
    for slot, info in slot_pass.items():
        A(f"- {slot}：同向为正 {info['good_dir'] or '无'}；单过 {info['alone'] or '无'}；"
          f"稳健 {'是' if info['robust'] else '否'}；晋升 {info['winner'] or '无'}。")
    A(f"- 载体规则：{'晋升' if carrier_pass else '不晋升'}。{('；'.join(carrier_why)) if carrier_why else ''}")
    if rows.get("carrier:both"):
        for sl, dg in rows["carrier:both"]["diag"].items():
            A(f"  - {sl} 决策 {dg['n_decisions']} 次，最多的是 {dg['modal']}（{dg['modal_share']:.0%}）。分布 {dg['counts']}。")
    if rows.get("carrier:control"):
        A("载体对照是「两个 QDII 槽都改记现任的场内价」，权重规则不动。它和 PR#5 臂 D（不同指数之间搬至多 2 个百分点底仓）不是同一条规则。")
    A("")
    A("### 2.2 2026 年以来")
    A("")
    A(f"截到 {common.date()}。累计收益，不是年化。现金臂的 YTD 来自上市日后才起步的缩短路径，状态没有从 2014 带入，不能跟 v37 的 YTD 直接相减。")
    A("")
    A("| 路径 | proxy YTD | proxy 段内 MDD | real YTD | real 段内 MDD |")
    A("|---|---|---|---|---|")
    def ycell(y):
        if not y:
            return "—", "—"
        return pct(y["total"]), pct(y["mdd"])
    op, omp = ycell(ytd.get("old_proxy"))
    orr, orm = ycell(ytd.get("old_real"))
    np_, nmp = ycell(ytd.get("v37_proxy"))
    nr, nrm = ycell(ytd.get("v37_real"))
    A(f"| v37 旧 511260（停在 09-18，其后引擎会把最后一笔日收益向前填） | {op} | {omp} | {orr} | {orm} |")
    A(f"| v37 511260 已追加 | {np_} | {nmp} | {nr} | {nrm} |")
    for slot, codes in SHORTLIST.items():
        for code in codes:
            rec = rows.get(f"{slot}:{code}") or {}
            y = rec.get("ytd_proxy")
            a, b = ycell(y)
            A(f"| {slot}:{code} | {a} | {b} | — | — |")
    if combo and combo.get("rets") is not None:
        a, b = ycell(ytd_of(combo["rets"], common))
        A(f"| 组合 | {a} | {b} | — | — |")
    A("")
    A("## 3. 结论")
    A("")
    if anchor_bad:
        A("锚失败，不解释候选，不写 v39。")
    else:
        any_slot = any(v["winner"] for v in slot_pass.values())
        if not any_slot and not carrier_pass:
            A("六个槽没有一只换券同时过 (a)(b)(c) 和识别。载体规则也不晋升。不写 `final_cfg_v39.json`。默认仍是 v37。")
        elif winners := {s: i["winner"] for s, i in slot_pass.items() if i["winner"]} :
            A("单槽晋升：" + "、".join(f"{s}→{rows[n]['code']}" for s, n in winners.items()) + "。")
            if combo_pass:
                A(f"组合再过关，已写提议 `{os.path.basename(cfg_path)}`，未接入默认路径。")
            else:
                A("组合没有再过关，不写配置文件。")
        elif carrier_pass:
            A("只有载体规则过关。它不改变配置文件里的代码，本实验不为此写 `final_cfg_v39.json`（引擎也没有这条规则的开关）。")
        else:
            A("没有可写的提议。默认仍是 v37。")
    A("")
    A("511260 的追加只影响现金层在 2026-09-18 之后的几天。权益换券的差异来自工具本身，不来自这条补数据。")
    A("样本外如果大部分日子仍是拼接，OOS 数字会靠近基线。识别条款要求自有样本至少 252 日且方向一致，就是为了不把拼接段算成证据。")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
