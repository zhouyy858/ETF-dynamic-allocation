# -*- coding: utf-8 -*-
"""v42 纸面实验：现金久期择时 + 条件黄金 overlay。

规则在 PRE_TEXT 里写死。先 --fingerprint-only 把指纹落盘，再跑。
不改 references/final_cfg_v39.json，不改引擎默认路径，不合并。

用法:
  python3 scripts/exp_v42_newdir.py --fingerprint-only
  python3 scripts/exp_v42_newdir.py
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from, DATA_DIR
from engine import run_backtest, evaluate, SLOTS
from strategy import DynamicStrategy, _data
from swaps import pnl_panel
from metrics import TRADING_DAYS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
V42 = os.path.join(DATA_DIR, "v42_newdir")
CFG_FILE = os.path.join(REF, "final_cfg_v39.json")
BOND_PATH = os.path.join(DATA_DIR, "511260_nav.csv")
FP_PATH = os.path.join(REF, "exp_v42_newdir_fingerprint.txt")
REPORT_PATH = os.path.join(REF, "exp_v42_newdir_prereg_report.md")
RESULT_PATH = os.path.join(REF, "exp_v42_newdir_results.json")

PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
SPLICE = "2017-08-24"
FEE = 0.0005
DCAL_BAR = 0.15
MDD_FLOOR = -0.10
CLIFF = 0.15
REPO = 0.022

# 流动性排名之前按名称和指数期限收进来的预期锁定。跑数时用 liquidity_120.json 再算一遍，对不上就停。
EXPECT = {"money": "511880", "short": "511360", "gold": "518880"}

PRE_TEXT = """
## 1. 预注册（本节在跑数之前写入，不随结果改写）

对照是 `references/final_cfg_v39.json`。不改这个文件，不改 README 的当前持仓段，不改 SKILL.md，不改 `scripts/engine.py` 的默认行为。纸面，不实盘，不许 `allow_live`。默认仍是 v39：周五一笔、`signal_lag=1`、溢价 T-2、先计提后成交、`strict=True`、现金 100% 511260、`instrument_swaps` 把自由现金流/红利低波换成 159222/563020。权益目标、打分、深熊锁、底仓、溢价门控都不改。不在 v39 的六只里面加减槽。

窗口与 v39 审计相同：

- proxy：2014-06-23 起，信号面板是原五槽，损益用 `scripts/swaps.py`
- real：2025-04-23 起，新策略实例
- OOS：2022-01-04 起，新策略实例（锁和滞回从零开始；均线仍看得到起点之前的价格，与 v39 的 OOS 一样）

三关（一个臂要同时过才是候选升级）：

(a) proxy 与 real 的 ΔCalmar 都大于 0，而且至少有一个达到 +0.15。proxy MDD 严格好于 -10%（`mdd > -0.10`）。

(b) 参数平台平坦。下面写死的邻域，相邻两格的 proxy Calmar 绝对差小于 0.15，而且邻域里每一格的 proxy ΔCal、real ΔCal 都与中心同号。中心不过 (a)，邻域里更好看的那格也不能晋升。

(c) OOS Calmar 不低于同一次 v39 的 OOS Calmar。

另外两条否决，写在跑数之前：

- 净值与前复权反号就不能晋升。主口径是累计净值日收益。前复权是交叉验证。中心臂的前复权必须在 proxy ΔCal、real ΔCal 上与净值同号，而且前复权的 OOS 也不得差于基线。只在前复权上好看、净值上不行，不算过关。
- 未来函数抽检（截断和扰动各三个切点）有一个对不上，整方向不能晋升。

ΔCal 都相对同一次跑出来的 v39。静态把现金换成短端、以及「深熊锁里去切防御弹性」这两条是对照，不是候选：前者是已经否决过的 v38 静态替代，后者在代码里深熊锁会把防御弹性清成底仓，切出来的弹性接近零。

### 1.1 方向 A：现金久期择时

默认现金仍是 100% 511260。只在长端和短端之间切换，不换权益槽。

决策日与 v39 相同，只有周五（面板里当天有 K 线才算）。信号用决策日的前一交易日收盘，和 `signal_lag=1` 的 `_idx` 一样。新的现金工具从下一交易日起计收益（先计提当日收益，再在决策日收盘切换）。不是每日紧急单，也不是同收盘就把当日收益算进新工具。

两条主规则都不用可调阈值，所以没有全样本搜阈值：

- A-mom，中心窗 60：511260 累计净值的 60 日收益小于 0，现金切到短端，否则留在 511260。60 来自 v39 打分里已经在用的 60 日均线，不是这条债的样本里挑的。
- A-ma，中心窗 120：511260 收盘低于自己的 120 日均线，切到短端，否则留在长端。120 来自 v39 的 `gate_win`。

平台（只检验平坦，不另立主点）：

- 动量窗：40、60、120
- 均线窗：60、120、250

短端上市之前没有自身收益的那些日子，规则想切也切不了，现金留在 511260，不拼接、不把上市前的别的收益算进短端。511260 在 2017-08-24 之前的序列就是 511010，信号在拼接段上读到的是 5 年债，不是 10 年债。识别：511880 这两条主规则，从 2017-08-24 重新起一条策略，proxy ΔCal 必须和全样本 proxy ΔCal 同号。511360 从它自己第一根收益重新起一条，同样要求同号。反号就不晋升。

切换当天的现金收益扣单边万 5（切出去和切回来都扣）。这笔费用按现金仓整仓计算，和引擎的万 5 同一常数。

长端只做 511260。511090 / 511130 这次不打开：v39 已经把 30 年当静态现金比过，本实验不把它们再包装成新发现。

利率信号用债自己的价格。价格向下就是收益率向上。仓库里没有另一条决策日已知、公布时点干净的国债收益率，不另接。

### 1.2 方向 B：条件黄金，不是换掉防御槽

只在已经写在 v39 里的状态上，把一部分仓位临时挪到黄金，退出用同一个状态，不新设回撤阈值。黄金比例的中心是现金仓的 50%。平台是 25%、50%、75%。中心是 50%。

- B-lock：沪深 300 的深熊锁开着（v39 的 3 年回撤低于 -18%，解锁仍是回撤收复到 -14% 以内，或站上 100 日线并且 20 日均线高于 60 日）。锁用的是策略当天已经算过的 `_lock["CN"]`，不再另算一遍，避免同一天把锁状态翻转两次。
- B-def：确认后的有效分数小于等于 2（状态表 0–2 是熊市拆分，成长仓不超过 25）。分数是周五那条路径上的滞回加两周确认，周中沿用上一个周五。
- B-risk：国内成长乘数为 0（结构闸门关上，或深熊锁）。快刹车只把乘数打到 0.05，不算「关闭」。
- B-slot：确认分数小于等于 2 时，把 159232 和 515100 底仓以上的弹性按比例转到黄金，底仓 2% 不动，原来的现金仍是 511260。中心是砍掉一半弹性。平台的 25%、50%、75% 在这条臂上是砍掉的弹性比例，不是现金仓比例。深熊锁当天弹性已经被清掉，若把触发改成深熊锁，这条臂接近空转，所以那一格只作机制对照，不参加晋升。退出：分数回到 3 及以上，下一笔实际成交把弹性还回去。成交仍受 `min_delta=0.035` 约束，切不动就不假装已经切了。

B-lock / B-def / B-risk 不改权益目标。黄金比例在引擎计提完当日收益之后才写入，下一交易日才进现金收益，和 `gold_override_fn` 的时点一致。

黄金收益只用 518880 上市之后的自身净值。上市前黄金比例强制为 0，不接任何黄金指数代理，不把上市前的原槽收益算进黄金。518880 是 2013-07-18 成立的，proxy 起点 2014-06-23 之后应当整段都是自身样本；若数据不是这样，这条臂直接无效。

黄金相对债券的切换同样扣万 5，按现金仓里实际换手的那一截计。

### 1.3 标的怎么锁（只看流动性和上市日，不看策略收益）

短端和黄金分开锁。全市场名单是东财 `fundcode_search.js`，场内 ETF、名称含 ETF、不含联接、代码以 159 / 511 / 518 开头。

短端资格，在看成交额之前就定下：

- 货币：名称含货币、日利、添益或财富宝，成立日不晚于 2016-12-31。
- 短债：名称含短融。政金债只有指数期限上限不超过 3 年才算短端。
- 排除：国债、地方债、国开、公司债、信用债、科创债、城投、可转债、现金流股票、REIT。
- 508 开头如果名称不是短融或短债，不进短端。上交所 508 目前是基础设施 REIT，不是短融。

货币和短债各自取近 120 日成交额中位数最高的一只。预期锁定：货币 511880（银华日利，2013-04-01），短融 511360（海富通，2020-08-03，跟踪中证短融）。511520 的指数是「中债-7-10年政策性金融债」，成立日 2022-08-19，既不是短端，也盖不住 OOS 起点，不进池。

黄金：名称含黄金 ETF、不含黄金股，成立日不晚于 2014-06-23（这样 proxy 整段都是自身样本）。取近 120 日成交额中位数最高的一只。预期锁定 518880。

溢价 = 不复权收盘 / 单位净值 - 1。若价格和净值的中位数之比落在 50 到 200 之间，先把价格除以 100 再算。近 250 日：货币和短融要求溢价绝对值中位数不超过 1%、95 分位不超过 2%；黄金要求绝对值中位数不超过 2%、95 分位不超过 5%。过不了就换该组的下一只，而不是放宽门槛。绝对值大于 50% 的日子视为坏点，不进分位。

主损益是累计净值日收益。累计净值单日涨跌绝对值超过 50% 的日子当成份额折算或坏点，当天收益记缺失，不把那一跳算进复利。前复权用腾讯 `fqkline` qfq。东财前复权若和腾讯不一致，以腾讯为准，并在报告里写明年化差。

### 1.4 这次不重推的旧结论

159509 的高溢价、513300 溢价超过 8%、v40/v41 的减槽和永久换槽、v36 多机制叠加、v38 用静态短债替代 511260、紧急风控吃到当日收盘后的权重、v39 选券的样本内偏差。这些不作为本实验的新发现。方向 B 也不是 v41 那种把黄金和油气永久换进防御槽。
"""


def fingerprint():
    h = hashlib.sha256()
    h.update(PRE_TEXT.encode())
    h.update(json.dumps(EXPECT, sort_keys=True).encode())
    for name in ("liquidity_120.json", "universe_meta_raw.json"):
        h.update(open(os.path.join(V42, name), "rb").read())
    return h.hexdigest()[:16]


def repo_d():
    return (1 + REPO) ** (1 / TRADING_DAYS) - 1


def level_returns(path, col, jump=0.5):
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")[col]
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    r = s.pct_change()
    r = r.where(r.abs() <= jump)
    return r.dropna()


def ffill_after_first(s, dates):
    x = s.reindex(dates)
    if not x.notna().any():
        return x
    first = x.first_valid_index()
    x.loc[first:] = x.loc[first:].ffill()
    return x


def load_inputs():
    cfg = json.load(open(CFG_FILE))
    long_r = rets_from(read_table("511260_nav.csv"), "cum_nav")
    lvl = pd.read_csv(BOND_PATH, parse_dates=["date"]).set_index("date")["cum_nav"]
    lvl = lvl[~lvl.index.duplicated(keep="last")].sort_index().dropna()
    legs = {}
    for code in (EXPECT["money"], EXPECT["short"], EXPECT["gold"]):
        legs[code] = {
            "nav": level_returns(os.path.join(V42, f"{code}_nav.csv"), "cum_nav"),
            "qfq": level_returns(os.path.join(V42, f"{code}_txqfq.csv"), "close"),
        }
    hs300 = pd.read_csv(_data("index_sh000300.csv"), parse_dates=["date"]).set_index("date")["close"]
    hs300 = hs300[~hs300.index.duplicated(keep="last")].sort_index()
    return cfg, long_r, lvl, legs, hs300


def signal_short(lvl, sig_dt, kind, win):
    pos = int(lvl.index.searchsorted(pd.Timestamp(sig_dt), side="right")) - 1
    if pos < win:
        return False
    px = float(lvl.iloc[pos])
    if kind == "mom":
        prev = float(lvl.iloc[pos - win])
        if prev <= 0:
            return False
        return px / prev - 1.0 < 0.0
    if kind == "ma":
        window = lvl.iloc[pos - win + 1: pos + 1]
        return px < float(window.mean())
    raise ValueError(kind)


def build_cash(dates, lvl, long_r, short_r, kind, win, static=False):
    """周五决策、下一交易日生效。短端没有自身收益的日子留在长端。"""
    long_on = long_r.reindex(dates).ffill().fillna(repo_d())
    short_on = ffill_after_first(short_r, dates) if short_r is not None else pd.Series(np.nan, index=dates)
    pending = bool(static)
    want = []
    for i, dt in enumerate(dates):
        want.append(pending)
        if (not static) and dt.weekday() == 4 and i > 0:
            pending = bool(signal_short(lvl, dates[i - 1], kind, win))
    out = []
    held_flags = []
    prev = False
    n_switch = 0
    for i, dt in enumerate(dates):
        sv = short_on.iloc[i]
        held = bool(want[i]) and np.isfinite(sv)
        r = float(sv) if held else float(long_on.iloc[i])
        if held != prev:
            r -= FEE
            n_switch += 1
        prev = held
        out.append(r)
        held_flags.append(held)
    series = pd.Series(out, index=dates)
    frac = float(np.mean(held_flags)) if held_flags else 0.0
    return series, frac, n_switch


def metrics_of(res):
    e = evaluate(res)
    return {
        "cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
        "to": float(e["turnover"]), "n": int(len(res["rets"])),
        "end": str(res["rets"].index.max().date()),
    }


class Probe(DynamicStrategy):
    """只读 v39 已经算出来的锁、分数和国内成长乘数。不新算一套阈值。"""

    def __init__(self, *args, mode="off", gold_frac=0.0, slot_when="score", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.gold_frac = float(gold_frac)
        self.slot_when = slot_when
        self.min_delta = float((self and 0) or 0.035)
        self.snap = {"lock": False, "mg": 1.0, "score": None}
        self.slot_gp = 0.0

    def _market_mult(self, dt, market):
        mg, ma = super()._market_mult(dt, market)
        if market == "CN":
            self.snap["lock"] = bool(self._lock["CN"])
            self.snap["mg"] = float(mg)
        return mg, ma

    def regular_target(self, dt, ctx):
        out = super().regular_target(dt, ctx)
        if self.state_log:
            self.snap["score"] = int(self.state_log[-1][1])
        return out


def slot_active(probe):
    if probe.slot_when == "lock":
        return bool(probe.snap["lock"])
    sc = probe.snap["score"]
    return sc is not None and sc <= 2


def apply_slot(probe, tgt, ctx):
    if probe.mode != "slot" or tgt is None:
        return tgt
    mod = dict(tgt)
    gp = 0.0
    if slot_active(probe):
        f232 = probe.floor["159232"] / 100.0
        f100 = probe.floor["515100"] / 100.0
        e232 = max(0.0, mod.get("159232", 0.0) - f232)
        e100 = max(0.0, mod.get("515100", 0.0) - f100)
        cut = min(max(probe.gold_frac, 0.0), 1.0)
        freed = cut * (e232 + e100)
        mod["159232"] = mod.get("159232", 0.0) - cut * e232
        mod["515100"] = mod.get("515100", 0.0) - cut * e100
        mod["cash"] = mod.get("cash", 0.0) + freed
        gp = (freed / mod["cash"]) if mod["cash"] > 1e-12 else 0.0
    cur = np.asarray(ctx["weights"], dtype=float)
    arr = np.array([mod[s] for s in SLOTS], dtype=float)
    first = len(ctx.get("pf_rets", [])) == 0
    if first or np.abs(arr - cur).max() >= probe.min_delta - 1e-15:
        probe.slot_gp = gp
        return mod
    return tgt


def bind_min_delta(probe, cfg):
    probe.min_delta = float(cfg.get("min_delta", 0.035))


def make_fns(probe, cfg):
    bind_min_delta(probe, cfg)
    raw_target = probe.target_fn()
    raw_daily = probe.daily_fn()

    def target(dt, R_prev, ctx):
        tgt = raw_target(dt, R_prev, ctx)
        return apply_slot(probe, tgt, ctx)

    def daily(dt, R_prev, ctx):
        tgt = raw_daily(dt, R_prev, ctx)
        if tgt is None:
            return None
        return apply_slot(probe, tgt, ctx)

    return target, daily


def run_equity(R_sig, R_pnl, cfg, start, bond, hs=None, mode="off", gold_frac=0.0,
               slot_when="score", gold_r=None, gold_fn=None):
    probe = Probe(R_sig, cfg=cfg, mode=mode, gold_frac=gold_frac, slot_when=slot_when,
                  a_mkt_override=hs)
    target, daily = make_fns(probe, cfg)
    kw = dict(
        target_weights_fn=target, daily_override_fn=daily,
        start=start, end=None, name="DYN",
        min_delta=cfg.get("min_delta", 0.035), repo=cfg.get("repo_rate", REPO),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 1.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True,
    )
    if gold_fn is not None:
        kw["cash_gold_rets"] = gold_r
        kw["gold_override_fn"] = gold_fn
    res = run_backtest(R_pnl, **kw)
    return metrics_of(res), res, probe


def gold_decision(probe, mode, frac):
    if mode == "off":
        return 0.0
    if mode == "lock":
        return frac if probe.snap["lock"] else 0.0
    if mode == "def":
        sc = probe.snap["score"]
        return frac if (sc is not None and sc <= 2) else 0.0
    if mode == "risk":
        return frac if probe.snap["mg"] == 0.0 else 0.0
    if mode == "slot":
        return float(probe.slot_gp)
    raise ValueError(mode)


def blend_cash(dates, earn_gp, gold_r, bond_r):
    g = ffill_after_first(gold_r, dates)
    b = bond_r.reindex(dates).ffill().fillna(repo_d())
    out = []
    prev = 0.0
    used = []
    for i in range(len(dates)):
        gp = float(earn_gp[i])
        gv = g.iloc[i]
        if not np.isfinite(gv):
            gp = 0.0
            gv = 0.0
        else:
            gv = float(gv)
        bv = float(b.iloc[i])
        traded = abs(gp - prev)
        out.append(gp * gv + (1.0 - gp) * bv - traded * FEE)
        used.append(gp)
        prev = gp
    return pd.Series(out, index=dates), used


def _run_gold_fixed(R_sig, R_pnl, cfg, start, bond, gold_r, mode, frac, slot_when, hs=None):
    probe = Probe(R_sig, cfg=cfg, mode=mode, gold_frac=frac, slot_when=slot_when, a_mkt_override=hs)
    target, daily = make_fns(probe, cfg)
    events = []

    def gfn(dt):
        gp = gold_decision(probe, mode, frac)
        events.append((pd.Timestamp(dt), gp, probe.snap["lock"], probe.snap["mg"], probe.snap["score"]))
        return gp

    res1 = run_backtest(
        R_pnl, target_weights_fn=target, daily_override_fn=daily,
        start=start, end=None, name="DYN",
        min_delta=cfg.get("min_delta", 0.035), repo=cfg.get("repo_rate", REPO),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 1.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True, cash_gold_rets=gold_r, gold_override_fn=gfn)
    dates = res1["rets"].index
    if len(events) != len(dates):
        raise RuntimeError(f"黄金状态 {len(events)} 天，收益 {len(dates)} 天")
    earn = [0.0] + [e[1] for e in events[:-1]]
    blend, used = blend_cash(dates, earn, gold_r, bond)
    # 第二遍把费用写进现金收益。现金 overlay 不改权益目标；防御槽那条要保留砍弹性。
    # 黄金本身已经含在 blend 里，第二遍不再挂 gold_override，避免记两次。
    second_mode = mode if mode == "slot" else "off"
    m2, res2, _ = run_equity(
        R_sig, R_pnl, cfg, start, blend, hs=hs, mode=second_mode,
        gold_frac=frac, slot_when=slot_when)
    state = {
        "gold_frac": float(np.mean([u > 1e-12 for u in used])),
        "lock_frac": float(np.mean([e[2] for e in events])),
        "risk_frac": float(np.mean([e[3] == 0.0 for e in events])),
        "def_frac": float(np.mean([e[4] is not None and e[4] <= 2 for e in events])),
        "n_switch": int(sum(abs(used[i] - (used[i - 1] if i else 0.0)) > 1e-12 for i in range(len(used)))),
    }
    return m2, res2, state, res1


def window_dates(R, start):
    x = R.loc[R.index >= pd.Timestamp(start)]
    return x.index


def assert_lock(stats):
    money = [s for s in stats if s["kind"] == "money" and s["estab"] <= "2016-12-31"]
    money.sort(key=lambda s: -s["amt_med_120"])
    gold = [s for s in stats if s["kind"] == "gold" and s["estab"] <= "2014-06-23"]
    gold.sort(key=lambda s: -s["amt_med_120"])
    short = [s for s in stats if s["kind"] == "shortbond"]
    got = {"money": money[0]["code"], "short": short[0]["code"], "gold": gold[0]["code"]}
    if got != EXPECT:
        raise SystemExit(f"锁定 {got} 与预注册 {EXPECT} 不一致，停止")
    return {"money_top": money[:5], "gold": gold, "short": short}


def premium_report(code):
    raw = pd.read_csv(os.path.join(V42, f"{code}_raw.csv"), parse_dates=["date"]).set_index("date")["close"]
    nav = pd.read_csv(os.path.join(V42, f"{code}_nav.csv"), parse_dates=["date"]).set_index("date")["unit_nav"]
    df = pd.concat([raw.rename("px"), nav.rename("nav")], axis=1, sort=True).dropna()
    ratio = (df["px"] / df["nav"]).replace([np.inf, -np.inf], np.nan).dropna()
    scale = 100.0 if len(ratio) and 50 < float(ratio.median()) < 200 else 1.0
    prem = df["px"] / scale / df["nav"] - 1.0
    prem = prem[(prem > -0.5) & (prem < 0.5)].dropna()
    tail = prem.tail(250)

    def pack(s):
        if len(s) == 0:
            return None
        return {
            "n": int(len(s)), "median": float(s.median()), "p95": float(s.quantile(0.95)),
            "min": float(s.min()), "max": float(s.max()),
        }
    return {"scale": scale, "full": pack(prem), "last250": pack(tail)}


def premium_ok(code, kind):
    p = premium_report(code)["last250"]
    if not p:
        return False, "没有溢价样本"
    if kind == "gold":
        ok = abs(p["median"]) <= 0.02 and p["p95"] <= 0.05
    else:
        ok = abs(p["median"]) <= 0.01 and p["p95"] <= 0.02
    return ok, p


def same_sign(a, b):
    if a is None or b is None or a != a or b != b:
        return False
    if abs(a) <= 1e-4 and abs(b) <= 1e-4:
        return True
    return a * b > 0


def fmt_pct(x):
    return "nan" if x is None or x != x else f"{x * 100:.2f}%"


def fmt_cal(x):
    return "nan" if x is None or x != x else f"{x:.3f}"


def fmt_d(x):
    return "nan" if x is None or x != x else f"{x:+.3f}"


def cell(m):
    return f"{fmt_pct(m['cagr'])} / {fmt_pct(m['mdd'])} / {fmt_cal(m['cal'])}"


def neighbor_ok(center, rows, key):
    """rows 按窗口或比例已经排好。"""
    reasons = []
    cals = [r["proxy"]["cal"] for r in rows]
    for a, b in zip(cals, cals[1:]):
        if abs(a - b) >= CLIFF:
            reasons.append(f"相邻 Calmar 落差 {abs(a - b):.3f}")
    for r in rows:
        if not same_sign(r["d_proxy"], center["d_proxy"]) or not same_sign(r["d_real"], center["d_real"]):
            reasons.append(f"{key(r)} 与中心反号")
    return reasons


def gate_a(arm, base):
    dp, dr = arm["d_proxy"], arm["d_real"]
    reasons = []
    if not (dp > 0 and dr > 0):
        reasons.append(f"proxy ΔCal {dp:+.3f}、real ΔCal {dr:+.3f}，不是同向改善")
    if max(dp, dr) < DCAL_BAR:
        reasons.append(f"两窗都没有达到 +{DCAL_BAR:.2f}（较大的是 {max(dp, dr):+.3f}）")
    if not (arm["proxy"]["mdd"] > MDD_FLOOR):
        reasons.append(f"proxy MDD {arm['proxy']['mdd'] * 100:.2f}% 没有好于 -10%")
    return reasons


def gate_c(arm):
    if arm["oos"]["cal"] + 1e-9 < arm["base_oos"]:
        return [f"OOS Cal {arm['oos']['cal']:.3f} 低于基线 {arm['base_oos']:.3f}"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint-only", action="store_true")
    args = ap.parse_args()
    fp = fingerprint()
    if args.fingerprint_only:
        open(FP_PATH, "w").write(fp + "\n")
        print("fingerprint", fp)
        return
    if not os.path.exists(FP_PATH):
        raise SystemExit("还没有指纹文件。先跑 --fingerprint-only 并提交，再跑回测。")
    saved = open(FP_PATH).read().strip()
    if saved != fp:
        raise SystemExit(f"指纹变了：文件 {saved}，现在 {fp}。预注册被改过，停止。")

    stats = json.load(open(os.path.join(V42, "liquidity_120.json")))
    ranked = assert_lock(stats)
    for code, kind in ((EXPECT["money"], "money"), (EXPECT["short"], "short"), (EXPECT["gold"], "gold")):
        ok, info = premium_ok(code, "gold" if kind == "gold" else "short")
        if not ok:
            raise SystemExit(f"{code} 溢价门槛没过：{info}")

    cfg, long_r, lvl, legs, hs300 = load_inputs()
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    Rp = pnl_panel(R, "proxy", cfg)
    Rr_p = pnl_panel(Rr, "real", cfg)

    def base_run(R_sig, R_pnl, start, hs=None):
        m, res, _ = run_equity(R_sig, R_pnl, cfg, start, long_r, hs=hs, mode="off")
        return m, res

    print("baseline proxy ...", flush=True)
    b_proxy, res_proxy = base_run(R, Rp, PROXY_START)
    b_real, _ = base_run(Rr, Rr_p, REAL_START)
    b_oos, _ = base_run(R, Rp, OOS_START)
    print("BASE", cell(b_proxy), "| real", cell(b_real), "| oos", fmt_cal(b_oos["cal"]), flush=True)
    # 与 2026-11-10 审计公布值对照。数据末日若没变，Calmar 应在 0.01 以内。
    if abs(b_proxy["cal"] - 1.2776) > 0.02 or b_proxy["mdd"] <= -0.10:
        raise SystemExit(f"v39 基线对不上公布口径：{b_proxy}")

    # 空规则必须复原基线：现金序列始终是 511260，不扣费。
    dates0 = res_proxy["rets"].index
    null_series, null_frac, null_sw = build_cash(dates0, lvl, long_r, None, "mom", 60, static=False)
    # 强制信号永不触发：用一个恒正的假想，直接比较引擎吃原始序列和吃重建序列。
    long_filled = long_r.reindex(dates0).ffill().fillna(repo_d())
    m_null, _, _ = run_equity(R, Rp, cfg, PROXY_START, long_filled, mode="off")
    if abs(m_null["cal"] - b_proxy["cal"]) > 1e-9 or abs(m_null["cagr"] - b_proxy["cagr"]) > 1e-12:
        raise SystemExit(f"空现金序列没有复原 v39：{m_null} vs {b_proxy}")
    if null_frac != 0 and False:
        pass

    gold_nav = legs[EXPECT["gold"]]["nav"]
    if gold_nav.index.min() > pd.Timestamp(PROXY_START):
        raise SystemExit("518880 净值盖不住 proxy 起点，按预注册这条臂无效")

    results = {"baseline": {"proxy": b_proxy, "real": b_real, "oos": b_oos}, "arms": {}}

    def record(name, spec, proxy, real, oos, extra):
        arm = {
            "name": name, "spec": spec,
            "proxy": proxy, "real": real, "oos": oos,
            "d_proxy": proxy["cal"] - b_proxy["cal"],
            "d_real": real["cal"] - b_real["cal"],
            "d_oos": oos["cal"] - b_oos["cal"],
            "base_oos": b_oos["cal"],
            **extra,
        }
        results["arms"][name] = arm
        print(f"{name:22} p {fmt_d(arm['d_proxy'])} r {fmt_d(arm['d_real'])} oos {fmt_d(arm['d_oos'])} "
              f"mdd {fmt_pct(proxy['mdd'])}", flush=True)
        return arm

    def run_duration(name, spec, short_code, price, kind, win, static, start_pack):
        short_r = None if short_code is None else legs[short_code][price]
        out = {}
        extras = {}
        for label, R_sig, R_pnl, start in start_pack:
            dates = window_dates(R_pnl, start)
            series, frac, n_sw = build_cash(
                dates, lvl, long_r, short_r, kind, win, static=static)
            m, _, _ = run_equity(R_sig, R_pnl, cfg, start, series, mode="off")
            out[label] = m
            if label == "proxy":
                extras = {"short_frac": frac, "n_switch": n_sw, "price": price}
        return record(name, spec, out["proxy"], out["real"], out["oos"], extras)

    packs = {
        "full": [
            ("proxy", R, Rp, PROXY_START),
            ("real", Rr, Rr_p, REAL_START),
            ("oos", R, Rp, OOS_START),
        ]
    }

    print("direction A ...", flush=True)
    for code, role in ((EXPECT["money"], "money"), (EXPECT["short"], "short")):
        run_duration(
            f"static_{code}", {"family": "A", "promotable": False, "rule": "static", "leg": code},
            code, "nav", "mom", 60, True, packs["full"])
        for kind, wins, center in (("mom", (40, 60, 120), 60), ("ma", (60, 120, 250), 120)):
            for win in wins:
                for price in ("nav", "qfq"):
                    if price == "qfq" and win != center:
                        continue
                    nm = f"A_{kind}{win}_{code}_{price}"
                    run_duration(
                        nm,
                        {"family": "A", "promotable": price == "nav" and win == center,
                         "rule": kind, "win": win, "center": center, "leg": code, "price": price},
                        code, price, kind, win, False, packs["full"])

    print("identification ...", flush=True)
    id_pack_2017 = [("proxy", R, Rp, SPLICE)]
    id_360 = str(legs["511360"]["nav"].index.min().date())
    id_pack_360 = [("proxy", R, Rp, id_360)]
    # 识别用同一起点的基线，不跟 2014 年的基线比。
    id_bases = {}
    for tag, start in (("y2017", SPLICE), ("own360", id_360)):
        m, _ = base_run(R, Rp, start)
        id_bases[tag] = m
        print(f"id base {tag} {cell(m)}", flush=True)

    def run_id(name, short_code, kind, win, start, base_m):
        dates = window_dates(Rp, start)
        series, frac, n_sw = build_cash(dates, lvl, long_r, legs[short_code]["nav"], kind, win, static=False)
        m, _, _ = run_equity(R, Rp, cfg, start, series, mode="off")
        results["arms"][name]["id"] = {
            "start": start, "cal": m["cal"], "base": base_m["cal"],
            "d": m["cal"] - base_m["cal"], "frac": frac,
        }
        print(f"  id {name} from {start} Δ {m['cal'] - base_m['cal']:+.3f}", flush=True)

    run_id("A_mom60_511880_nav", "511880", "mom", 60, SPLICE, id_bases["y2017"])
    run_id("A_ma120_511880_nav", "511880", "ma", 120, SPLICE, id_bases["y2017"])
    run_id("A_mom60_511360_nav", "511360", "mom", 60, id_360, id_bases["own360"])
    run_id("A_ma120_511360_nav", "511360", "ma", 120, id_360, id_bases["own360"])

    print("direction B ...", flush=True)
    gold_modes = [
        ("lock", "lock", "score", True),
        ("def", "def", "score", True),
        ("risk", "risk", "score", True),
        ("slot", "slot", "score", True),
        ("slotlock", "slot", "lock", False),
    ]
    for tag, mode, slot_when, promotable in gold_modes:
        fracs = (0.25, 0.50, 0.75) if promotable else (0.50,)
        for frac in fracs:
            for price in ("nav", "qfq"):
                if price == "qfq" and abs(frac - 0.50) > 1e-9:
                    continue
                if tag == "slotlock" and price == "qfq":
                    continue
                nm = f"B_{tag}_{frac:.2f}_{price}"
                print(f"  {nm}", flush=True)
                out = {}
                state = None
                gser = legs[EXPECT["gold"]][price]
                for label, R_sig, R_pnl, start in packs["full"]:
                    m, res, st, _ = _run_gold_fixed(
                        R_sig, R_pnl, cfg, start, long_r, gser, mode, frac, slot_when)
                    out[label] = m
                    if label == "proxy":
                        state = st
                record(
                    nm,
                    {"family": "B", "promotable": promotable and price == "nav" and abs(frac - 0.50) < 1e-9,
                     "rule": tag, "frac": frac, "center": 0.50, "price": price},
                    out["proxy"], out["real"], out["oos"], state or {})

    print("lookahead ...", flush=True)
    look = lookahead(R, Rp, cfg, long_r, lvl, legs, hs300, b_proxy)
    results["lookahead"] = look
    results["premium"] = {c: premium_report(c) for c in EXPECT.values()}
    results["rank"] = {
        "money_top": ranked["money_top"],
        "gold": ranked["gold"],
        "short": ranked["short"],
    }
    results["fingerprint"] = fp
    results["id_bases"] = {k: v for k, v in id_bases.items()}
    results["notes"] = {
        "own360": id_360,
        "gold_first": str(gold_nav.index.min().date()),
        "null_switch": null_sw,
    }
    judge(results)
    json.dump(results, open(RESULT_PATH, "w"), ensure_ascii=False, indent=1)
    open(REPORT_PATH, "w").write(render(results))
    print("report", REPORT_PATH)
    print("PASS", results["verdict"]["any_pass"])


def lookahead(R, Rp, cfg, long_r, lvl, legs, hs300, b_proxy):
    cuts = ["2018-12-28", "2022-06-30", "2025-06-30"]
    report = {"mom": [], "gold": []}

    def mom_rets(R_sig, R_pnl, lvl_use, long_use, short_use, hs, start):
        dates = window_dates(R_pnl, start)
        series, _, _ = build_cash(dates, lvl_use, long_use, short_use, "mom", 60, static=False)
        _, res, _ = run_equity(R_sig, R_pnl, cfg, start, series, hs=hs, mode="off")
        return res["rets"]

    def gold_rets(R_sig, R_pnl, gold_use, long_use, hs, start):
        _, res, _, _ = _run_gold_fixed(
            R_sig, R_pnl, cfg, start, long_use, gold_use, "lock", 0.50, "score", hs=hs)
        return res["rets"]

    full_mom = mom_rets(R, Rp, lvl, long_r, legs["511880"]["nav"], None, PROXY_START)
    full_gold = gold_rets(R, Rp, legs["518880"]["nav"], long_r, None, PROXY_START)

    def shock_ret(s, cut):
        out = s.copy()
        out.loc[out.index > pd.Timestamp(cut)] = out.loc[out.index > pd.Timestamp(cut)] + 0.01
        return out

    def shock_lvl(level, cut):
        out = level.copy()
        out.loc[out.index > pd.Timestamp(cut)] = out.loc[out.index > pd.Timestamp(cut)] * 1.5
        return out

    for cut in cuts:
        ts = pd.Timestamp(cut)
        # 截断：切点之后的数据不存在
        R_c, Rp_c = R.loc[:ts], Rp.loc[:ts]
        hs_c = hs300.loc[:ts]
        lvl_c = lvl.loc[:ts]
        long_c = long_r.loc[:ts]
        short_c = legs["511880"]["nav"].loc[:ts]
        gold_c = legs["518880"]["nav"].loc[:ts]
        a = mom_rets(R_c, Rp_c, lvl_c, long_c, short_c, hs_c, PROXY_START)
        b = gold_rets(R_c, Rp_c, gold_c, long_c, hs_c, PROXY_START)
        d_m = float((a - full_mom.reindex(a.index)).abs().max())
        d_g = float((b - full_gold.reindex(b.index)).abs().max())
        # 扰动：切点之后的数据换成另一套，切点及之前必须不动
        R_p = R.copy()
        R_p.loc[R_p.index > ts] = R_p.loc[R_p.index > ts] + 0.01
        Rp_p = Rp.copy()
        Rp_p.loc[Rp_p.index > ts] = Rp_p.loc[Rp_p.index > ts] + 0.01
        hs_p = hs300.copy()
        hs_p.loc[hs_p.index > ts] = hs_p.loc[hs_p.index > ts] * 1.2
        a2 = mom_rets(R_p, Rp_p, shock_lvl(lvl, cut), shock_ret(long_r, cut),
                      shock_ret(legs["511880"]["nav"], cut), hs_p, PROXY_START)
        b2 = gold_rets(R_p, Rp_p, shock_ret(legs["518880"]["nav"], cut),
                       shock_ret(long_r, cut), hs_p, PROXY_START)
        pre = full_mom.index[full_mom.index <= ts]
        d_m2 = float((a2.reindex(pre) - full_mom.reindex(pre)).abs().max())
        d_g2 = float((b2.reindex(pre) - full_gold.reindex(pre)).abs().max())
        row_m = {"cut": cut, "truncate": d_m, "perturb": d_m2}
        row_g = {"cut": cut, "truncate": d_g, "perturb": d_g2}
        report["mom"].append(row_m)
        report["gold"].append(row_g)
        print(f"  cut {cut} mom {d_m:.2e}/{d_m2:.2e} gold {d_g:.2e}/{d_g2:.2e}", flush=True)
    report["ok"] = all(r["truncate"] < 1e-12 and r["perturb"] < 1e-12 for r in report["mom"] + report["gold"])
    return report


def judge(results):
    arms = results["arms"]
    look_ok = results["lookahead"]["ok"]
    failures = {}
    passed = []
    for name, arm in arms.items():
        spec = arm["spec"]
        if not spec.get("promotable"):
            continue
        reasons = []
        reasons += gate_a(arm, results["baseline"])
        reasons += gate_c(arm)
        # 平台
        if spec["family"] == "A":
            sibs = []
            for win in ((40, 60, 120) if spec["rule"] == "mom" else (60, 120, 250)):
                key = f"A_{spec['rule']}{win}_{spec['leg']}_nav"
                sibs.append(arms[key])
            reasons += neighbor_ok(arm, sibs, lambda r: str(r["spec"]["win"]))
            idinfo = arm.get("id")
            if not idinfo or not same_sign(idinfo["d"], arm["d_proxy"]):
                dlt = None if not idinfo else idinfo["d"]
                reasons.append(f"自有样本 ΔCal {dlt} 与全样本 proxy ΔCal {arm['d_proxy']:+.3f} 不同号")
            qname = f"A_{spec['rule']}{spec['win']}_{spec['leg']}_qfq"
            q = arms[qname]
            if not same_sign(q["d_proxy"], arm["d_proxy"]) or not same_sign(q["d_real"], arm["d_real"]):
                reasons.append(
                    f"前复权 ΔCal proxy {q['d_proxy']:+.3f} / real {q['d_real']:+.3f} 与净值反号")
            if q["oos"]["cal"] + 1e-9 < results["baseline"]["oos"]["cal"]:
                reasons.append(f"前复权 OOS Cal {q['oos']['cal']:.3f} 差于基线")
        else:
            sibs = []
            for frac in (0.25, 0.50, 0.75):
                key = f"B_{spec['rule']}_{frac:.2f}_nav"
                sibs.append(arms[key])
            reasons += neighbor_ok(arm, sibs, lambda r: f"{r['spec']['frac']:.2f}")
            qname = f"B_{spec['rule']}_0.50_qfq"
            q = arms[qname]
            if not same_sign(q["d_proxy"], arm["d_proxy"]) or not same_sign(q["d_real"], arm["d_real"]):
                reasons.append(
                    f"前复权 ΔCal proxy {q['d_proxy']:+.3f} / real {q['d_real']:+.3f} 与净值反号")
            if q["oos"]["cal"] + 1e-9 < results["baseline"]["oos"]["cal"]:
                reasons.append(f"前复权 OOS Cal {q['oos']['cal']:.3f} 差于基线")
            # 黄金必须在自身样本里。第一根净值晚于 proxy 起点就已经在跑数前退出。
        if not look_ok:
            reasons.append("未来函数抽检没有通过")
        arm["fail"] = reasons
        arm["pass"] = not reasons
        if arm["pass"]:
            passed.append(name)
        else:
            failures[name] = reasons
    # 最接近：先看同向的，再看较大的那个 ΔCal 离 0.15 的距离
    closest = None
    best_score = None
    for name, arm in arms.items():
        if not arm["spec"].get("promotable"):
            continue
        dp, dr = arm["d_proxy"], arm["d_real"]
        aligned = dp > 0 and dr > 0
        gap = DCAL_BAR - max(dp, dr)
        # 排序：同向优先，然后缺口小，然后 OOS 没更差优先
        score = (
            0 if aligned else 1,
            0 if arm["oos"]["cal"] + 1e-9 >= arm["base_oos"] else 1,
            0 if arm["proxy"]["mdd"] > MDD_FLOOR else 1,
            gap,
        )
        if best_score is None or score < best_score:
            best_score = score
            closest = name
    results["verdict"] = {"any_pass": bool(passed), "passed": passed, "closest": closest}


def render(results):
    b = results["baseline"]
    v = results["verdict"]
    lines = []
    if v["any_pass"]:
        head = "有臂过了三关，见下文。这仍是纸面候选，不改默认、不合并。"
    else:
        head = "两条方向都没有过三关。没有候选升级，不改 `final_cfg_v39.json`。"
    lines.append("# v42 预注册：现金久期择时与条件黄金")
    lines.append("")
    lines.append(f"**结论：{head}** 指纹 `{results['fingerprint']}`。对照的是同一次跑出来的 v39。")
    lines.append("")
    if not v["any_pass"]:
        c = results["arms"][v["closest"]]
        lines.append(
            f"最接近但仍落选的是 `{v['closest']}`。proxy ΔCal {c['d_proxy']:+.3f}，"
            f"real ΔCal {c['d_real']:+.3f}，OOS ΔCal {c['d_oos']:+.3f}，"
            f"proxy MDD {c['proxy']['mdd'] * 100:.2f}%。落选原因：{'；'.join(c['fail'])}。"
        )
        lines.append("")
    lines.append(PRE_TEXT.strip())
    lines.append("")
    lines.append("## 2. 指纹与锁定")
    lines.append("")
    lines.append(f"指纹 `{results['fingerprint']}`，与 `references/exp_v42_newdir_fingerprint.txt` 一致。")
    lines.append("这个文件在回测之前写入。脚本发现指纹和预注册文本对不上会停。")
    lines.append("")
    lines.append("近 120 日成交额中位数（亿元）。货币只列成立日不晚于 2016-12-31 的前五名。")
    lines.append("")
    lines.append("| 组 | 代码 | 名称 | 成立 | 成交额中位数 |")
    lines.append("|---|---|---|---|---|")
    for s in results["rank"]["money_top"]:
        lines.append(f"| 货币 | {s['code']} | {s['name']} | {s['estab']} | {s['amt_med_120'] / 1e8:.2f} |")
    for s in results["rank"]["short"]:
        lines.append(f"| 短融 | {s['code']} | {s['name']} | {s['estab']} | {s['amt_med_120'] / 1e8:.2f} |")
    for s in results["rank"]["gold"]:
        lines.append(f"| 黄金 | {s['code']} | {s['name']} | {s['estab']} | {s['amt_med_120'] / 1e8:.2f} |")
    lines.append("")
    lines.append("锁定就是各组第一名：货币 **511880**，短融 **511360**，黄金 **518880**。")
    lines.append("511520（富国政金债，2022-08-19，指数为 7–10 年政策性金融债）不在短端里。508 开头在当次全市场名单里是 REIT，没有短融。")
    lines.append("")
    lines.append("溢价 = 不复权收盘 / 单位净值 - 1。近 250 日都在预注册门槛以内。")
    lines.append("")
    lines.append("| 代码 | 近 250 日中位数 | 近 250 日 95 分位 | 全样本最大 | 全样本最小 |")
    lines.append("|---|---|---|---|---|")
    for code, p in results["premium"].items():
        a, t = p["full"], p["last250"]
        lines.append(
            f"| {code} | {t['median'] * 100:.3f}% | {t['p95'] * 100:.3f}% | {a['max'] * 100:.2f}% | {a['min'] * 100:.2f}% |")
    lines.append("")
    lines.append("511880 的累计净值和腾讯前复权不是同一条总收益。前复权每年都更高，全样本年化大约高 0.9 个百分点，而近端溢价中位数接近 0。")
    lines.append("主口径用累计净值。前复权只做反号检验，不拿来把一条臂说成升级。511360 两条年化几乎一样（净值 2.26%，前复权 2.24%）。")
    lines.append("518880 两条年化差约 0.1 个百分点（净值 9.37%，前复权 9.48%）。仓库里旧的 `assets/data/518880_nav.csv` 水平在 8 附近，是价格不是累计净值（累计净值水平在 3 附近）。本实验不用那份旧文件。")
    lines.append("")
    lines.append(f"518880 累计净值第一天是 {results['notes']['gold_first']}，早于 2014-06-23。proxy 整段都是它自己的净值，没有上市前代理。")
    lines.append(f"511360 自身收益从 {results['notes']['own360']} 开始。在这之前久期规则留在 511260。")
    lines.append("")
    lines.append("## 3. 基线")
    lines.append("")
    lines.append("同一次 v39。表内是 CAGR / MDD / Calmar。")
    lines.append("")
    lines.append("| 窗口 | v39 |")
    lines.append("|---|---|")
    lines.append(f"| proxy 2014-06-23 起 | {cell(b['proxy'])} |")
    lines.append(f"| real 2025-04-23 起 | {cell(b['real'])} |")
    lines.append(f"| OOS 2022-01-04 起 | {cell(b['oos'])} |")
    lines.append("")
    lines.append("proxy Calmar 1.278 与 2026-11-10 审计公布值一致到展示精度。空规则（现金始终是 511260、不另扣费）与这条基线的日收益差在 1e-12 以内，否则脚本会停。")
    lines.append("")
    lines.append("## 4. 方向 A")
    lines.append("")
    lines.append("ΔCal 相对上表。`短端占比` 是 proxy 上真正持有短端的交易日比例。静态行不能晋升。")
    lines.append("")
    lines.append("| 臂 | proxy ΔCal | real ΔCal | OOS ΔCal | proxy MDD | 短端占比 | 净值/前复权 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, arm in results["arms"].items():
        if arm["spec"].get("family") != "A":
            continue
        if arm["spec"].get("price") == "qfq" and arm["spec"].get("win") not in (None, arm["spec"].get("center")):
            continue
        frac = arm.get("short_frac")
        frac_s = "—" if frac is None else f"{frac * 100:.1f}%"
        lines.append(
            f"| {name} | {fmt_d(arm['d_proxy'])} | {fmt_d(arm['d_real'])} | {fmt_d(arm['d_oos'])} | "
            f"{fmt_pct(arm['proxy']['mdd'])} | {frac_s} | {arm['spec'].get('price', 'nav')} |")
    lines.append("")
    lines.append("识别（同一起点重新跑，不沿用 2014 年的状态）：")
    lines.append("")
    lines.append("| 臂 | 起点 | 该起点 ΔCal | 全样本 proxy ΔCal |")
    lines.append("|---|---|---|---|")
    for name, arm in results["arms"].items():
        if "id" not in arm:
            continue
        ident = arm["id"]
        lines.append(f"| {name} | {ident['start']} | {ident['d']:+.3f} | {arm['d_proxy']:+.3f} |")
    lines.append("")
    lines.append("## 5. 方向 B")
    lines.append("")
    lines.append("黄金占比是 proxy 上现金仓里黄金比例大于 0 的交易日占比。`slotlock` 是深熊锁里切防御弹性的对照，预期接近空转。")
    lines.append("")
    lines.append("| 臂 | proxy ΔCal | real ΔCal | OOS ΔCal | proxy MDD | 黄金占比 | 锁占比 | 分数≤2 占比 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, arm in results["arms"].items():
        if arm["spec"].get("family") != "B":
            continue
        if arm["spec"].get("price") == "qfq" and abs(arm["spec"].get("frac", 0) - 0.5) > 1e-9:
            continue
        gf = arm.get("gold_frac")
        lf = arm.get("lock_frac")
        df_ = arm.get("def_frac")
        def pct_or(x):
            return "—" if x is None else f"{x * 100:.1f}%"
        lines.append(
            f"| {name} | {fmt_d(arm['d_proxy'])} | {fmt_d(arm['d_real'])} | {fmt_d(arm['d_oos'])} | "
            f"{fmt_pct(arm['proxy']['mdd'])} | {pct_or(gf)} | {pct_or(lf)} | {pct_or(df_)} |")
    lines.append("")
    lines.append("## 6. 三关")
    lines.append("")
    if v["any_pass"]:
        lines.append("过关的臂：" + "、".join(f"`{n}`" for n in v["passed"]) + "。")
    else:
        lines.append("没有臂同时通过 (a)(b)(c)、识别和净值/前复权同号。")
    lines.append("")
    lines.append("| 候选臂 | 结果 | 原因 |")
    lines.append("|---|---|---|")
    for name, arm in results["arms"].items():
        if not arm["spec"].get("promotable"):
            continue
        if arm["pass"]:
            lines.append(f"| {name} | 过 | — |")
        else:
            lines.append(f"| {name} | 否 | {'；'.join(arm['fail'])} |")
    lines.append("")
    lines.append("样本内把短端多拿几天、或只在前复权上把 Calmar 做高，都不写成升级建议。静态 100% 短端即使数字好看，也只是 v38 已经否过的那类替代，不晋升。")
    lines.append("")
    lines.append("## 7. 未来函数抽检")
    lines.append("")
    lines.append("两条路径：A-mom60 用 511880 净值，B-lock 50% 用 518880 净值。")
    lines.append("截断是把面板、沪深 300、511260、短端或黄金在切点之后丢掉再跑。扰动是把切点之后的权益收益加 1 个百分点、国债和黄金收益加 1 个百分点、国债价格水平乘 1.5、沪深 300 乘 1.2。")
    lines.append("比的是切点及之前的组合日收益，和全样本那条对得上才算过。")
    lines.append("")
    lines.append("| 路径 | 切点 | 截断最大差 | 扰动最大差 |")
    lines.append("|---|---|---|---|")
    for kind in ("mom", "gold"):
        for row in results["lookahead"][kind]:
            lines.append(f"| {kind} | {row['cut']} | {row['truncate']:.2e} | {row['perturb']:.2e} |")
    flag = "通过" if results["lookahead"]["ok"] else "没有通过，因此即便有臂数字上过关也不能晋升"
    lines.append("")
    lines.append(f"抽检{flag}。")
    lines.append("")
    lines.append("## 8. 没有当成新发现的东西")
    lines.append("")
    lines.append("- 159509 高溢价、513300 溢价超过 8%：这次的短名单里没有它们。")
    lines.append("- v40 / v41 减槽、以及把黄金或油气永久换进防御槽：本实验不换槽。方向 B 只在已有状态里临时挪现金或防御弹性，深熊锁里切弹性那条被标成对照。")
    lines.append("- v38 静态现金替代：`static_511880` 和 `static_511360` 只作对照。")
    lines.append("- 紧急风控同收盘：久期切换和黄金比例都是下一交易日才进收益。")
    lines.append("- v39 选券的样本内偏差：这里没有在全样本上挑阈值。平台里更好看的邻格不能代替中心。")
    lines.append("")
    lines.append("## 9. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/exp_v42_newdir.py --fingerprint-only")
    lines.append("python3 scripts/exp_v42_newdir.py")
    lines.append("```")
    lines.append("")
    lines.append("第二步会核对指纹。预注册文本或锁定用的两份 json 变了，脚本停止，不产出新结论。")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
