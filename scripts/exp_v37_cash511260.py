# -*- coding: utf-8 -*-
"""v37 纸面实验：现金层全部放入 511260（2026-11-09）

规则在 PRE / JOBS / ANCHORS / PRE_REG_MD 里写死。先提交指纹，再跑数。
结果不得回改这些对象。不改 final_cfg_v30.json、README、SKILL.md。
不改 scripts/engine.py。三关全过才写 references/final_cfg_v37.json，并标明未激活。

用法:
  python3 scripts/exp_v37_cash511260.py --fingerprint-only
  python3 scripts/exp_v37_cash511260.py
"""
import sys, os, json, copy, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from metrics import TRADING_DAYS, annualized_ret, max_drawdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
CFG_FILE = os.path.join(REF, "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
SPLICE = "2017-08-24"
OOS_FLOOR = 1.26

# v35 在 001ea76 上的生产路径锚。对不上就中止，不解释候选。
PIN_PROXY = {"cagr": 0.11111021816342403, "mdd": -0.09910849911812059,
             "cal": 1.1210967692185454}

# v35 臂 A 主点 A_p75 的公布全精度。N75_rv30 必须落在 1e-8 以内。
V35_A75 = {
    "proxy": {"cagr": 0.1161119422, "mdd": -0.0991084991, "cal": 1.1715639243},
    "real": {"cagr": 0.2415356134, "mdd": -0.0374611143, "cal": 6.4476355716},
    "oos": {"cal": 1.4155074201},
}

PRE = {
    "dcal_pos": 0.0,          # 双窗都必须严格变好
    "dcal_one": 0.15,         # 至少一边达到
    "proxy_mdd": -0.10,       # 严格大于 -10%
    "cliff": 0.15,            # 相邻 proxy Calmar
    "mono_tol": 0.02,         # 允许的反向回撤
    "oos_floor": OOS_FLOOR,
    "replicate_tol": 1e-8,
    "blend_tol": 1e-12,
    "identity_tol": 1e-12,
    "splice_tol": 1e-8,
    "stress_mdd_worse": 0.01, # 1.00 个百分点，只陈列
}

ANCHORS = {"pin_proxy": PIN_PROXY, "v35_a75": V35_A75, "splice": SPLICE,
           "oos_floor": OOS_FLOOR}

# 权重 (w260, w010, wrepo)。A100 是两条轴的 100% 端点，不重复成第三个任务。
JOBS = [
    {"name": "A100", "role": "primary", "w260": 1.0, "w010": 0.0, "wrepo": 0.0},
    {"name": "A_all", "role": "lock_check", "w260": 1.0, "w010": 0.0, "wrepo": 0.0},
    {"name": "N50_r0", "role": "neighbor", "w260": 0.50, "w010": 0.50, "wrepo": 0.0},
    {"name": "N75_r0", "role": "neighbor", "w260": 0.75, "w010": 0.25, "wrepo": 0.0},
    {"name": "N50_rv30", "role": "neighbor", "w260": 0.50, "w010": 0.25, "wrepo": 0.25},
    {"name": "N75_rv30", "role": "replicate", "w260": 0.75, "w010": 0.0, "wrepo": 0.25},
]
AXIS = {
    "repo0": ["N50_r0", "N75_r0", "A100"],
    "v30repo": ["N50_rv30", "N75_rv30", "A100"],
}

STRESS = [
    ("2016Q4-2017", "2016-10-01", "2017-12-31"),
    ("2016Q4-2017_拼接段", "2016-10-01", "2017-08-23"),
    ("2017-08-24至年末", "2017-08-24", "2017-12-31"),
    ("2020-05至10", "2020-05-01", "2020-10-31"),
    ("2022Q4", "2022-10-01", "2022-12-31"),
]

PRE_REG_MD = """\
## 1. 预注册（本节在跑数之前写入脚本，不随结果改写）

对照是 `references/final_cfg_v30.json`。不改这个文件，不改 README 的当前持仓段，不改 SKILL.md，不把任何新配置设成默认。纸面，不实盘。

引擎、费用（单边万 5）、信号（`signal_lag=1`、溢价 T-2、先计提后成交、`strict=True`）、窗口与 v35 / `scripts/exp_opt31.py` 相同：

- proxy 全历史从 2014-06-23 起
- real 从 2025-04-23 起
- OOS 是 proxy 上的独立起点 2022-01-04（新策略实例，锁状态不从 2014 带入）

### 1.1 这次在问什么

v35 臂 A（PR #5）把现金层里那 75% 的 511010 换成 511260，25% 逆回购没动。公布结果：proxy ΔCal +0.050，real ΔCal +0.128，proxy MDD 仍是 -9.91%，不够 +0.15。本轮只做一件事：把剩下的现金也放进 511260。不新搜阈值，不改权益目标，不改深熊锁的股票规则，不动底仓（前提④，锁里仍留 2% CN 红利）。

### 1.2 511260 用哪条序列

沿用 `assets/data/511260_nav.csv`，与 v35 相同：`rets_from(..., "cum_nav")`。这条文件由 `scripts/fetch_cash_bond.py` 生成，腾讯前复权（含分红再投资）。**2017-08-24 上市之前，净值路径等于 511010，不是另一条 10Y 代理。** 本实验不外推、不另接中债指数、不补 2013 钱荒。

跑数时核对（对不上就判本实验无效，不解释候选）：

- 2017-08-24 之前，511260 与 511010 的重叠日收益绝对差必须全部 ≤ 1e-8
- 第一条绝对差 > 1e-8 的日期必须落在 2017-08-24 到 2017-09-15 之间

2016Q4–2017 必须拆成两段陈列：2016-10-01～2017-08-23（拼接段，差异只可能来自「逆回购换成 511010」），以及 2017-08-24～2017-12-31（才是 10Y）。拼接段不得写成 10Y 战胜或输给 5Y。

### 1.3 现金三元组

闲置现金的日收益是 `w_repo * 逆回购 + w_010 * 511010 + w_260 * 511260`，三项权重和为 1。逆回购利率用配置里的 `repo_rate`（0.022），日化与引擎一致：`(1+r)**(1/252)-1`。缺日先按引擎的方式 `reindex + ffill + 缺了填逆回购`，再加权。合成后的整条现金收益以 `cash_bond_pct=1` 喂进现有引擎。**不改 `scripts/engine.py`。**

v30 基线不走合成路径，仍是 `cash_bond_code=511010`、`cash_bond_pct=0.75`。合成复原 `(0, 0.75, 0.25)` 必须与这条基线的 proxy 日收益最大绝对差 ≤ 1e-12，否则无效。

身份锚（v35 在同一提交 `001ea76` 上的生产路径）：proxy CAGR 0.11111021816342403、MDD -0.09910849911812059、Calmar 1.1210967692185454。对不上就中止，不解释候选。

### 1.4 臂

权重写成 (511260, 511010, 逆回购)。

| 名字 | 角色 | 权重 |
|---|---|---|
| A100 | 唯一晋升点 | (1, 0, 0)。整层现金 100% 511260，没有逆回购，没有 511010 |
| A_all | 锁内核对，不单独晋升 | (1, 0, 0)。与 A100 同一个现金工具 |
| N50_r0 | repo0 邻域 | (0.50, 0.50, 0)。其余全部 511010 |
| N75_r0 | repo0 邻域 | (0.75, 0.25, 0) |
| N50_rv30 | v30 逆回购邻域 | (0.50, 0.25, 0.25)。511260 占现金一半，逆回购维持 25%，剩 25% 给 511010 |
| N75_rv30 | v35 臂 A 复现 | (0.75, 0, 0.25)。必须对上 v35 公布的 A_p75 |

100% 这一点无法再留 25% 逆回购，所以两条邻域轴的右端都是 A100，不重复跑第三遍。

repo0 轴：N50_r0、N75_r0、A100。v30repo 轴：N50_rv30、N75_rv30、A100。

**A_all 事先写明的代码事实，不是回测结论：** v30 的非权益只有残差现金。深熊锁把 CN 弹性清零之后，腾出的权重进的是同一笔现金，引擎对现金没有第二套工具。因此 A_all 与 A100 的权益目标相同、现金工具相同，proxy 日收益最大绝对差预期为 0。不为了做出差别去卖掉 2% 红利底仓。锁内要报告的是：锁了多少日、锁内现金权重、锁内日收益连乘相对 v30、最长的几段锁的组合回撤。锁日历来自市场信号（`_market_mult`），不看组合净值；v30 与 A100 的锁日历必须一致，否则无效。

**v35 复现锚（A_p75，同一数据）：** proxy CAGR 0.1161119422 / MDD -0.0991084991 / Cal 1.1715639243；real CAGR 0.2415356134 / MDD -0.0374611143 / Cal 6.4476355716；OOS Cal 1.4155074201。N75_rv30 与这组数的绝对差，CAGR、MDD、Calmar 都要 ≤ 1e-8。对不上就无效，后面的臂不作数。

### 1.5 三关（只评 A100）

1. **同向并且至少一边够 +0.15**：proxy 与 real 的 ΔCalmar 都 > 0；两者里至少有一个 ≥ +0.15；proxy MDD **严格好于 -10%**（`mdd > -0.10`，等于 -10% 也不过）。不另加 v35 的 real MDD -4.1% 线。
2. **平台**：两条轴都要平坦。点按 511260 占比从低到高。
   - 每个点双窗 ΔCalmar 都 > 0，proxy MDD 都 > -10%
   - 相邻两点的 proxy Calmar 绝对差 ≤ 0.15（悬崖线；这条轴是凸组合，0.50 的旧悬崖线太松，事先收紧）
   - 允许单调爬坡。只禁止反向：后一点的 proxy Calmar 不得低于前一点超过 0.02
   - 邻域点即使比 A100 更高，也不升格、不改主点
3. **OOS**：A100 的 OOS Calmar ≥ **1.26**（绝对数，用未四舍五入的值）。v35 档案里 v30 的 OOS 是 1.258688，显示成 1.26。这条比「不差于 v30」严大约 0.001。不跟本次重跑的基线漂。

换手只陈列，不设否决。债熊三段只陈列，不设否决。若 A100 在任一段的组合 MDD 比 v30 深过 1.00 个百分点（`mdd_A - mdd_v30 < -0.01`），结论必须写「这一段债熊更深」，不得把提议说成无条件。

三关都过、并且 1.2–1.4 的有效性核对都过，才写 `references/final_cfg_v37.json`。文件头注明提议、未激活。默认路径仍是 v30。三关不过就不写这个文件。

### 1.6 债熊与现金层自身回撤

组合与现金层都报，窗是闭区间：

- 2016Q4–2017：2016-10-01～2017-12-31，并按 1.2 拆开
- 2020 年 5–10 月：2020-05-01～2020-10-31
- 2022Q4：2022-10-01～2022-12-31

现金层自身：用上面的现金日收益（不是组合收益）做最大回撤。至少给出 v30 混合、纯 511010、纯拼接 511260、A100。另给拼接 511260 从 2017-08-24 起（去掉拼接段）的最大回撤，避免把 5Y 的路径算进 10Y。

### 1.7 事先不做的事

不重跑 v35 的 B–F，不重开日频再平衡，不扫更密的比例网格，不把红利底仓换成债券，不改 README / SKILL / `final_cfg_v30.json`。
"""


class LockProbe(DynamicStrategy):
    """只记录 CN 锁。不改目标。关记录时必须与父类逐日相同。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lock_at = {}

    def _mark(self, dt):
        self.lock_at[pd.Timestamp(dt)] = bool(self._lock["CN"])

    def target_fn(self):
        inner = super().target_fn()

        def fn(dt, R_prev, ctx):
            out = inner(dt, R_prev, ctx)
            self._mark(dt)
            return out
        return fn

    def daily_fn(self):
        inner = super().daily_fn()

        def fn(dt, R_prev, ctx):
            out = inner(dt, R_prev, ctx)
            self._mark(dt)
            return out
        return fn


def fingerprint():
    blob = json.dumps(
        {"PRE": PRE, "JOBS": JOBS, "AXIS": AXIS, "STRESS": STRESS,
         "ANCHORS": ANCHORS, "md": PRE_REG_MD},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def weights_ok(job):
    s = job["w260"] + job["w010"] + job["wrepo"]
    if abs(s - 1.0) > 1e-12:
        raise SystemExit(f"权重和不为 1: {job['name']} {s}")


def pack(e):
    return {"cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
            "sharpe": float(e["sharpe"]), "to": float(e["turnover"]),
            "cash": float(e["avg_cash"])}


def repo_daily(repo):
    return (1 + repo) ** (1 / TRADING_DAYS) - 1


def align_ret(series, dates, repo_d):
    return series.reindex(dates).ffill().fillna(repo_d)


def cash_on(dates, r010, r260, w260, w010, wrepo, repo):
    repo_d = repo_daily(repo)
    a010 = align_ret(r010, dates, repo_d)
    a260 = align_ret(r260, dates, repo_d)
    return wrepo * repo_d + w010 * a010 + w260 * a260


def execute(R, cfg, start, bond, pct, end=None, cls=DynamicStrategy):
    ds = cls(R, cfg=cfg)
    res = run_backtest(
        R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="DYN", min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=pct,
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    return pack(evaluate(res)), res, ds


def run_job(R, cfg, start, r010, r260, job, end=None, cls=DynamicStrategy):
    """合成现金收益，以 pct=1 喂给引擎。日期与引擎的 start/end 切片一致。"""
    weights_ok(job)
    sub = R
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]
    dates = sub.index
    cr = cash_on(dates, r010, r260, job["w260"], job["w010"], job["wrepo"],
                 cfg.get("repo_rate", 0.022))
    return execute(R, cfg, start, cr, 1.0, end=end, cls=cls)


def window_stats(rets, start, end):
    sub = rets[(rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))]
    if len(sub) < 5:
        return {"n": int(len(sub)), "cagr": None, "mdd": None, "total": None}
    w = (1 + sub).cumprod()
    mdd, pk, tr = max_drawdown(w)
    ann = float(annualized_ret(sub))
    return {"n": int(len(sub)), "cagr": ann, "mdd": float(mdd),
            "total": float(w.iloc[-1] - 1),
            "cal": float(ann / abs(mdd)) if mdd < 0 else None,
            "peak": None if pk is None else str(pk.date()),
            "trough": None if tr is None else str(tr.date())}


def series_stats(r):
    r = r.dropna()
    w = (1 + r).cumprod()
    mdd, pk, tr = max_drawdown(w)
    return {"n": int(len(r)), "start": str(r.index.min().date()), "end": str(r.index.max().date()),
            "cagr": float(annualized_ret(r)), "mdd": float(mdd),
            "total": float(w.iloc[-1] - 1),
            "peak": None if pk is None else str(pk.date()),
            "trough": None if tr is None else str(tr.date())}


def splice_check(r010, r260):
    both = pd.concat([r010.rename("a"), r260.rename("b")], axis=1).dropna()
    pre = both[both.index < pd.Timestamp(SPLICE)]
    pre_max = float((pre["a"] - pre["b"]).abs().max()) if len(pre) else float("nan")
    gap = (both["a"] - both["b"]).abs()
    hit = gap[gap > PRE["splice_tol"]]
    first = None if hit.empty else str(hit.index.min().date())
    ok = (pre_max <= PRE["splice_tol"] and first is not None
          and "2017-08-24" <= first <= "2017-09-15")
    return {"n_pre": int(len(pre)), "pre_max_abs": pre_max, "first_diff": first,
            "ok": bool(ok)}


def near(got, exp, keys, tol):
    bad = []
    for k in keys:
        if abs(float(got[k]) - float(exp[k])) > tol:
            bad.append(f"{k}:{got[k]:.10f}!={exp[k]:.10f}")
    return bad


def deltas(row, base):
    return {"dcal_p": row["proxy"]["cal"] - base["proxy"]["cal"],
            "dcal_r": row["real"]["cal"] - base["real"]["cal"],
            "dcal_o": row["oos"]["cal"] - base["oos"]["cal"],
            "to_ratio": row["proxy"]["to"] / base["proxy"]["to"]}


def gate_a(row, base):
    d = deltas(row, base)
    same = d["dcal_p"] > PRE["dcal_pos"] and d["dcal_r"] > PRE["dcal_pos"]
    one = max(d["dcal_p"], d["dcal_r"]) >= PRE["dcal_one"]
    mdd_ok = row["proxy"]["mdd"] > PRE["proxy_mdd"]
    return {"same": bool(same), "one": bool(one), "mdd": bool(mdd_ok),
            "ok": bool(same and one and mdd_ok), "deltas": d}


def platform_axis(names, rows, base):
    reasons = []
    cals = []
    for n in names:
        row = rows[n]
        d = deltas(row, base)
        cals.append(row["proxy"]["cal"])
        if not (d["dcal_p"] > 0 and d["dcal_r"] > 0):
            reasons.append(f"{n} 双窗未同向({d['dcal_p']:+.3f}/{d['dcal_r']:+.3f})")
        if row["proxy"]["mdd"] <= PRE["proxy_mdd"]:
            reasons.append(f"{n} proxy MDD {row['proxy']['mdd']*100:.2f}% 不优于 -10%")
    for i in range(1, len(cals)):
        gap = abs(cals[i] - cals[i - 1])
        if gap > PRE["cliff"]:
            reasons.append(f"{names[i-1]}→{names[i]} 相邻 Calmar 差 {gap:.3f} > {PRE['cliff']}")
        if cals[i] < cals[i - 1] - PRE["mono_tol"]:
            reasons.append(f"{names[i]} 比 {names[i-1]} 低 {cals[i-1]-cals[i]:.3f}，反向超过 {PRE['mono_tol']}")
    return {"ok": len(reasons) == 0, "reasons": reasons, "cals": cals}


def lock_flags(ds, index):
    """决策日上的锁标记向后保持到下一次决策。首日之前视为未锁。"""
    marks = pd.Series(ds.lock_at).sort_index()
    flag = pd.Series(index=index, dtype="boolean")
    if len(marks):
        flag.loc[flag.index.intersection(marks.index)] = marks.reindex(flag.index.intersection(marks.index)).astype(bool)
    flag = flag.ffill().fillna(False).astype(bool)
    return flag


def spells_of(flag):
    out, start, prev = [], None, None
    for dt, on in flag.items():
        if on and start is None:
            start = prev = dt
        elif on:
            prev = dt
        elif start is not None:
            out.append((start, prev))
            start = None
    if start is not None:
        out.append((start, prev))
    return out


def compounded(rets, mask):
    sub = rets[mask]
    if len(sub) == 0:
        return {"n": 0, "total": None}
    return {"n": int(len(sub)), "total": float((1 + sub).prod() - 1)}


def pct(x):
    if x is None or x != x:
        return "nan"
    return f"{x * 100:.2f}%"


def calf(x):
    if x is None or x != x:
        return "nan"
    return f"{x:.2f}"


def dlt(x):
    return f"{x:+.3f}"


def cell(m):
    return f"{pct(m['cagr'])} / {pct(m['mdd'])} / {calf(m['cal'])} / {m['to']:.2f}"


def stress_cell(s):
    if not s or s.get("total") is None:
        return "n/a"
    return f"{pct(s['total'])} / {pct(s['mdd'])}"


def build_report(digest, meta, rows, base, gates, plat, valid, stress, cash_mdd, lock, promoted):
    L = []
    A = L.append
    A("# v37 纸面实验：现金全部放入 511260")
    A("")
    A(f"预注册指纹 `{digest}`。数据 proxy 至 {meta['proxy_end']}，real 至 {meta['real_end']}。")
    A("第 1 节是跑数之前写死的规则。第 2 节才是结果。结论在第 3 节，规则没有按结果回改。")
    A("默认配置仍是 `references/final_cfg_v30.json`。README 当前持仓段和 SKILL.md 没有改。")
    A("")
    A(PRE_REG_MD.rstrip())
    A("")
    A("## 2. 结果")
    A("")
    if not valid["pin_ok"]:
        A("身份锚没有对上，候选全部不作数。")
        A("")
        A("```")
        A(json.dumps(valid, ensure_ascii=False, indent=1))
        A("```")
        return "\n".join(L) + "\n"
    A(f"身份校验：生产路径与 v35 锚一致（CAGR {meta['pin']['cagr']:.6f} / "
      f"MDD {meta['pin']['mdd']:.6f} / Cal {meta['pin']['cal']:.6f}）。")
    A(f"合成复原 v30（0% 511260 + 75% 511010 + 25% 逆回购）相对生产路径的最大日收益差 "
      f"{valid['blend_gap']:.1e}。")
    sp = valid["splice"]
    A(f"拼接核对：2017-08-24 之前重叠 {sp['n_pre']} 日，最大绝对差 {sp['pre_max_abs']:.2e}；"
      f"第一条超过 1e-8 的日期是 {sp['first_diff']}。{'通过' if sp['ok'] else '未通过'}。")
    A(f"v35 臂 A 复现（N75_rv30）：{'通过' if valid['replicate_ok'] else '未通过'}。"
      f"差 {valid['replicate_bad'] or '无'}。")
    A(f"A_all 相对 A100 的 proxy 最大日收益差 {valid['a_all_gap']:.1e}。"
      f"锁日历{'一致' if valid['lock_same'] else '不一致'}。")
    A("")
    A(f"**基线 v30**：proxy {cell(base['proxy'])}；real {cell(base['real'])}；"
      f"OOS Cal {base['oos']['cal']:.4f}（显示 {calf(base['oos']['cal'])}）。")
    A("公布对照：proxy 11.11% / -9.91% / 1.12，real 23.70% / -3.75% / 6.32。")
    A("换手是引擎里单边权重变动的累加，不是年化。")
    A("")
    A("| 点 | 角色 | proxy CAGR/MDD/Cal/换手 | ΔCal | real CAGR/MDD/Cal/换手 | ΔCal | OOS Cal | ΔOOS | 换手比 |")
    A("|---|---|---|---|---|---|---|---|---|")
    order = ["v30"] + [j["name"] for j in JOBS]
    roles = {"v30": "基线"}
    for j in JOBS:
        roles[j["name"]] = j["role"]
    for name in order:
        row = base if name == "v30" else rows[name]
        if name == "v30":
            A(f"| v30 | 基线 | {cell(row['proxy'])} | — | {cell(row['real'])} | — | "
              f"{row['oos']['cal']:.4f} | — | 1 |")
            continue
        d = deltas(row, base)
        A(f"| {name} | {roles[name]} | {cell(row['proxy'])} | {dlt(d['dcal_p'])} | "
          f"{cell(row['real'])} | {dlt(d['dcal_r'])} | {row['oos']['cal']:.4f} | "
          f"{dlt(d['dcal_o'])} | {d['to_ratio']:.3f} |")
    A("")
    A("### 2.1 三关")
    A("")
    g = gates
    A(f"- 同向：proxy ΔCal {dlt(g['deltas']['dcal_p'])}，real ΔCal {dlt(g['deltas']['dcal_r'])}。"
      f"{'两边都为正' if g['same'] else '没有两边都为正'}。")
    A(f"- 至少一边 ≥ +0.15：{'达到' if g['one'] else '未达到'}。")
    A(f"- proxy MDD {pct(rows['A100']['proxy']['mdd'])}：{'严格好于 -10%' if g['mdd'] else '没有好于 -10%'}。")
    A(f"- 一关合计：{'通过' if g['ok'] else '未通过'}。")
    for axis, info in plat.items():
        text = "通过" if info["ok"] else "未通过（" + "；".join(info["reasons"]) + "）"
        A(f"- 平台 {axis}：{text}。proxy Calmar 依次是 " +
          "、".join(f"{c:.3f}" for c in info["cals"]) + "。")
    plat_ok = all(p["ok"] for p in plat.values())
    A(f"- 平台合计：{'通过' if plat_ok else '未通过'}。")
    oos_v = rows["A100"]["oos"]["cal"]
    A(f"- OOS Calmar {oos_v:.4f} 对比绝对门槛 1.26：{'不低于' if oos_v >= OOS_FLOOR else '低于'}。")
    A(f"- 有效性（锚、拼接、复现、A_all 同路径、锁日历）：{'通过' if valid['ok'] else '未通过'}。"
      + ("" if valid["ok"] else "原因：" + "；".join(valid["reasons"]) + "。"))
    A(f"- 晋升：{'写提议配置' if promoted else '不写 final_cfg_v37.json'}。")
    A("")
    A("### 2.2 债熊窗口（组合总收益 / 窗口内 MDD）")
    A("")
    A("拼接段的差异来自逆回购换成 511010（这段 511260 序列就是 511010）。2017-08-24 之后才是 10Y。")
    A("")
    A("| 窗 | 天数 | v30 | A100 | 总收益差 | MDD 差 |")
    A("|---|---|---|---|---|---|")
    for name, _, _ in STRESS:
        sv, sa = stress[name]["v30"], stress[name]["A100"]
        if sv.get("total") is None or sa.get("total") is None:
            A(f"| {name} | {sv.get('n')} | n/a | n/a | n/a | n/a |")
            continue
        d_tot = sa["total"] - sv["total"]
        d_mdd = (sa["mdd"] - sv["mdd"]) * 100
        worse = "，更深" if sa["mdd"] - sv["mdd"] < -PRE["stress_mdd_worse"] else ""
        A(f"| {name} | {sv['n']} | {stress_cell(sv)} | {stress_cell(sa)} | {d_tot*100:+.2f}pp | {d_mdd:+.2f}pp{worse} |")
    A("")
    A("### 2.3 现金层自身")
    A("")
    A("这是现金工具的日收益，不是组合收益。全样本从 2014-06-23 的 proxy 交易日起。")
    A("")
    A("| 序列 | 全样本 CAGR/MDD | 谷底 | 2017-08-24 起 MDD |")
    A("|---|---|---|---|")
    for name, st in cash_mdd["full"].items():
        tail = cash_mdd["from_splice"].get(name)
        tail_s = "—" if tail is None else pct(tail["mdd"])
        A(f"| {name} | {pct(st['cagr'])} / {pct(st['mdd'])} | {st['trough']} | {tail_s} |")
    A("")
    A("债熊窗里的现金层总收益 / MDD：")
    A("")
    A("| 窗 | v30 混合 | 纯 511010 | 拼接 511260 | A100 |")
    A("|---|---|---|---|---|")
    for name, _, _ in STRESS:
        cells = []
        for key in ("v30", "511010", "511260_spliced", "A100"):
            cells.append(stress_cell(cash_mdd["stress"][name][key]))
        A(f"| {name} | " + " | ".join(cells) + " |")
    A("")
    A("### 2.4 CN 深熊锁")
    A("")
    lk = lock
    A(f"锁内 {lk['n_days']} 个交易日，共 {lk['n_spells']} 段。")
    A(f"v30 锁内平均现金权重 {pct(lk['cash_v30'])}，锁外 {pct(lk['cash_v30_off'])}。")
    A(f"A100 锁内平均现金权重 {pct(lk['cash_a100'])}，锁外 {pct(lk['cash_a100_off'])}。")
    A(f"锁内日收益连乘（非连续持有）：v30 {pct(lk['lock_total_v30'])}，A100 {pct(lk['lock_total_a100'])}，"
      f"差 {(lk['lock_total_a100'] - lk['lock_total_v30'])*100:+.2f}pp。")
    A(f"锁外连乘差 {(lk['off_total_a100'] - lk['off_total_v30'])*100:+.2f}pp。")
    A("下表是按天数排序的前 8 段。窗口内 MDD 是这段连续交易日上的组合回撤。")
    A("")
    A("| 起 | 止 | 天数 | v30 总收益/MDD | A100 总收益/MDD |")
    A("|---|---|---|---|---|")
    for sp in lk["longest"]:
        A(f"| {sp['start']} | {sp['end']} | {sp['n']} | {stress_cell(sp['v30'])} | {stress_cell(sp['a100'])} |")
    A("")
    A("## 3. 结论")
    A("")
    A(meta["verdict"])
    A("")
    if meta.get("cfg_path"):
        A(f"提议文件：`{meta['cfg_path']}`。没有接入默认路径。")
        A("")
    worse = [k for k, v in stress.items()
             if v["A100"].get("mdd") is not None and v["v30"].get("mdd") is not None
             and v["A100"]["mdd"] - v["v30"]["mdd"] < -PRE["stress_mdd_worse"]]
    if worse:
        A("债熊更深的窗：" + "、".join(worse) + "。")
        A("")
    A("合并说明：生产默认路径没有切换。`final_cfg_v30.json` 仍是基线。引擎没有改。")
    A("")
    return "\n".join(L) + "\n"


def verdict_text(promoted, gates, plat, valid, rows, base):
    if not valid["pin_ok"]:
        return "身份锚失败，本实验无效。默认仍是 v30。"
    if not valid["ok"]:
        return ("有效性核对没有过（" + "；".join(valid["reasons"]) + "）。"
                "候选不作数。默认仍是 v30，不写提议配置。")
    d = gates["deltas"]
    bits = []
    if gates["ok"] and all(p["ok"] for p in plat.values()) and rows["A100"]["oos"]["cal"] >= OOS_FLOOR:
        bits.append(
            f"A100 三关在数字上通过：proxy ΔCal {dlt(d['dcal_p'])}，real ΔCal {dlt(d['dcal_r'])}，"
            f"proxy MDD {pct(rows['A100']['proxy']['mdd'])}，OOS Cal {rows['A100']['oos']['cal']:.4f}。"
            "已写入 `references/final_cfg_v37.json`，这是提议，不是默认。"
        )
    else:
        missed = []
        if not gates["same"]:
            missed.append("双窗 Calmar 没有同向变好")
        if not gates["one"]:
            missed.append("没有一边达到 +0.15")
        if not gates["mdd"]:
            missed.append("proxy MDD 没有好于 -10%")
        if not all(p["ok"] for p in plat.values()):
            missed.append("平台不平坦")
        if rows["A100"]["oos"]["cal"] < OOS_FLOOR:
            missed.append(f"OOS Cal {rows['A100']['oos']['cal']:.4f} 低于 1.26")
        bits.append(
            f"A100 未过三关（{'；'.join(missed)}）。"
            f"proxy ΔCal {dlt(d['dcal_p'])}，real ΔCal {dlt(d['dcal_r'])}，"
            f"proxy MDD {pct(rows['A100']['proxy']['mdd'])}，"
            f"OOS Cal {rows['A100']['oos']['cal']:.4f}。"
            "不写 `final_cfg_v37.json`。默认仍是 v30。"
        )
    bits.append(
        "多出来的收益如果存在，来自把逆回购和 5Y 换成更长的久期，不是新的择时。"
        "2017-08-24 之前的 511260 就是 511010，样本里没有 2013 钱荒。"
        "回撤控制只看股票端，现金层利率回撤不会触发减仓。"
    )
    bits.append(
        f"v30 基线本次测得 proxy {pct(base['proxy']['cagr'])}/{pct(base['proxy']['mdd'])}/{calf(base['proxy']['cal'])}，"
        f"real {pct(base['real']['cagr'])}/{pct(base['real']['mdd'])}/{calf(base['real']['cal'])}。"
    )
    return "".join(bits)


def maybe_write_cfg(base_cfg, rows, base):
    cfg = copy.deepcopy(base_cfg)
    p, r, o = rows["A100"]["proxy"], rows["A100"]["real"], rows["A100"]["oos"]
    bp, br = base["proxy"], base["real"]
    cfg["version"] = "v37"
    cfg["cash_bond_pct"] = 1.0
    cfg["cash_bond_code"] = "511260"
    cfg["_note"] = (
        "v37 提议、未激活（2026-11-09 纸面）。默认仍是 final_cfg_v30.json。"
        "相对 v30：现金层从 25% 逆回购 + 75% 511010 改为 100% 511260，权益目标与深熊锁不动。"
        f"本次 proxy {p['cagr']*100:.2f}%/{p['mdd']*100:.2f}%/Cal{p['cal']:.2f}"
        f"（v30 {bp['cagr']*100:.2f}%/{bp['mdd']*100:.2f}%/Cal{bp['cal']:.2f}），"
        f"real {r['cagr']*100:.2f}%/{r['mdd']*100:.2f}%/Cal{r['cal']:.2f}"
        f"（v30 {br['cagr']*100:.2f}%/{br['mdd']*100:.2f}%/Cal{br['cal']:.2f}），"
        f"OOS Cal {o['cal']:.4f}。"
        "511260 用 assets/data/511260_nav.csv 的腾讯前复权拼接：2017-08-24 之前等于 511010。"
        "超额是久期，不是择时。样本内没有 2013 钱荒。回撤控制只看股票端。"
        "repo_rate 在 cash_bond_pct=1 时不被引擎使用。"
    )
    path = os.path.join(REF, "final_cfg_v37.json")
    with open(path, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return "references/final_cfg_v37.json"


def main():
    digest = fingerprint()
    print(f"[预注册指纹] {digest}")
    if "--fingerprint-only" in sys.argv:
        return

    base_cfg = json.load(open(CFG_FILE))
    r010 = rets_from(read_table("511010_nav.csv"), "cum_nav")
    r260 = rets_from(read_table("511260_nav.csv"), "cum_nav")
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    print(f"[数据] proxy {R.index.min().date()}~{R.index.max().date()} n={len(R)} | "
          f"real {Rr.index.min().date()}~{Rr.index.max().date()} n={len(Rr)}")

    reasons = []
    sp = splice_check(r010, r260)
    print(f"[拼接] pre_max={sp['pre_max_abs']:.2e} first_diff={sp['first_diff']} ok={sp['ok']}")
    if not sp["ok"]:
        reasons.append(f"拼接核对失败 first_diff={sp['first_diff']} pre_max={sp['pre_max_abs']}")

    m_pin, res_pin, _ = execute(R, base_cfg, PROXY_START, r010, base_cfg.get("cash_bond_pct", 0.75))
    pin_bad = near(m_pin, PIN_PROXY, ("cagr", "mdd", "cal"), 1e-9)
    pin_ok = len(pin_bad) == 0
    print(f"[身份] {m_pin['cagr']:.6f} {m_pin['mdd']:.6f} {m_pin['cal']:.6f} ok={pin_ok}")
    if not pin_ok:
        reasons.append("身份锚失败 " + ",".join(pin_bad))

    m_probe, res_probe, ds_probe = execute(
        R, base_cfg, PROXY_START, r010, base_cfg.get("cash_bond_pct", 0.75), cls=LockProbe)
    probe_gap = float((res_pin["rets"] - res_probe["rets"]).abs().max())
    if probe_gap > PRE["identity_tol"]:
        reasons.append(f"LockProbe 改变了 v30 日收益 {probe_gap}")

    v30_job = {"name": "v30_blend", "w260": 0.0, "w010": 0.75, "wrepo": 0.25}
    m_blend, res_blend, _ = run_job(R, base_cfg, PROXY_START, r010, r260, v30_job)
    blend_gap = float((res_pin["rets"] - res_blend["rets"]).abs().max())
    print(f"[复原] blend_gap={blend_gap:.1e} probe_gap={probe_gap:.1e}")
    if blend_gap > PRE["blend_tol"]:
        reasons.append(f"合成复原与生产路径差 {blend_gap}")

    base = {"proxy": m_pin}
    base_rets = {"proxy": res_pin["rets"]}
    base_w = {"proxy": res_pin["weights"]}
    for win, panel, start in (("real", Rr, REAL_START), ("oos", R, OOS_START)):
        m, res, _ = execute(panel, base_cfg, start, r010, base_cfg.get("cash_bond_pct", 0.75))
        base[win] = m
        base_rets[win] = res["rets"]
        print(f"[基线] {win} {cell(m)}")

    if not pin_ok:
        valid = {"ok": False, "pin_ok": False, "reasons": reasons, "splice": sp,
                 "blend_gap": blend_gap, "replicate_ok": False, "replicate_bad": pin_bad,
                 "a_all_gap": None, "lock_same": False}
        meta = {"proxy_end": str(R.index.max().date()), "real_end": str(Rr.index.max().date()),
                "pin": m_pin, "verdict": verdict_text(False, {}, {}, valid, {}, base),
                "cfg_path": None}
        text = build_report(digest, meta, {}, base, {}, {}, valid, {}, {"full": {}, "from_splice": {}, "stress": {}},
                            {"n_days": 0, "n_spells": 0, "cash_v30": None, "cash_v30_off": None,
                             "cash_a100": None, "cash_a100_off": None, "lock_total_v30": 0,
                             "lock_total_a100": 0, "off_total_v30": 0, "off_total_a100": 0, "longest": []},
                            False)
        path = os.path.join(REF, "exp_v37_cash511260_report.md")
        with open(path, "w") as f:
            f.write(text)
        print(f"[写] {path}")
        return

    rows, rets_keep, w_keep = {}, {}, {}
    for job in JOBS:
        if job["name"] == "A_all":
            continue
        row = {}
        for win, panel, start in (("proxy", R, PROXY_START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
            m, res, _ = run_job(panel, base_cfg, start, r010, r260, job)
            row[win] = m
            if win == "proxy":
                rets_keep[job["name"]] = res["rets"]
                w_keep[job["name"]] = res["weights"]
        rows[job["name"]] = row
        d = deltas(row, base)
        print(f"{job['name']:<10} dP {dlt(d['dcal_p'])} dR {dlt(d['dcal_r'])} dO {dlt(d['dcal_o'])} "
              f"to×{d['to_ratio']:.3f} | {cell(row['proxy'])}")

    m_all, res_all, ds_all = run_job(R, base_cfg, PROXY_START, r010, r260, JOBS[1], cls=LockProbe)
    a_all_gap = float((rets_keep["A100"] - res_all["rets"]).abs().max())
    print(f"[A_all] gap={a_all_gap:.1e}")
    if a_all_gap <= PRE["identity_tol"]:
        rows["A_all"] = copy.deepcopy(rows["A100"])
    else:
        reasons.append(f"A_all 与 A100 日收益差 {a_all_gap}")
        rows["A_all"] = {"proxy": m_all}
        for win, panel, start in (("real", Rr, REAL_START), ("oos", R, OOS_START)):
            m, _, _ = run_job(panel, base_cfg, start, r010, r260, JOBS[1])
            rows["A_all"][win] = m

    rep_bad = []
    for win, keys in (("proxy", ("cagr", "mdd", "cal")), ("real", ("cagr", "mdd", "cal")), ("oos", ("cal",))):
        rep_bad += [f"{win}.{x}" for x in near(rows["N75_rv30"][win], V35_A75[win], keys, PRE["replicate_tol"])]
    replicate_ok = len(rep_bad) == 0
    if not replicate_ok:
        reasons.append("N75_rv30 没有对上 v35 A_p75：" + ",".join(rep_bad))

    flag_v30 = lock_flags(ds_probe, res_pin["rets"].index)
    # A100 的锁：再跑一只探针，确认日历不随现金工具变
    _, _, ds_a = run_job(R, base_cfg, PROXY_START, r010, r260,
                         {"name": "A100", "w260": 1.0, "w010": 0.0, "wrepo": 0.0}, cls=LockProbe)
    flag_a = lock_flags(ds_a, rets_keep["A100"].index)
    common = flag_v30.index.intersection(flag_a.index)
    lock_same = bool((flag_v30.loc[common] == flag_a.loc[common]).all())
    if not lock_same:
        reasons.append("v30 与 A100 的 CN 锁日历不一致")

    valid = {"ok": len(reasons) == 0, "pin_ok": True, "reasons": reasons, "splice": sp,
             "blend_gap": blend_gap, "replicate_ok": replicate_ok, "replicate_bad": rep_bad,
             "a_all_gap": a_all_gap, "lock_same": lock_same, "probe_gap": probe_gap}

    g = gate_a(rows["A100"], base)
    plat = {axis: platform_axis(names, rows, base) for axis, names in AXIS.items()}
    oos_ok = rows["A100"]["oos"]["cal"] >= OOS_FLOOR
    promoted = bool(valid["ok"] and g["ok"] and all(p["ok"] for p in plat.values()) and oos_ok)
    print(f"[门] A={g['ok']} plat={ {k: v['ok'] for k, v in plat.items()} } OOS={oos_ok} "
          f"valid={valid['ok']} promoted={promoted}")

    # 债熊：组合
    stress = {}
    for name, s, e in STRESS:
        stress[name] = {"v30": window_stats(base_rets["proxy"], s, e),
                        "A100": window_stats(rets_keep["A100"], s, e)}

    # 现金层
    dates = res_pin["rets"].index
    repo = base_cfg.get("repo_rate", 0.022)
    cash_series = {
        "v30 混合": cash_on(dates, r010, r260, 0.0, 0.75, 0.25, repo),
        "纯 511010": cash_on(dates, r010, r260, 0.0, 1.0, 0.0, repo),
        "拼接 511260": cash_on(dates, r010, r260, 1.0, 0.0, 0.0, repo),
        "A100": cash_on(dates, r010, r260, 1.0, 0.0, 0.0, repo),
    }
    # A100 与拼接 511260 是同一条。表里两行都留，避免读者找不着臂的名字。
    full = {k: series_stats(v) for k, v in cash_series.items()}
    from_splice = {}
    for k, v in cash_series.items():
        tail = v[v.index >= pd.Timestamp(SPLICE)]
        from_splice[k] = series_stats(tail)
    cash_stress = {}
    for name, s, e in STRESS:
        cash_stress[name] = {
            "v30": window_stats(cash_series["v30 混合"], s, e),
            "511010": window_stats(cash_series["纯 511010"], s, e),
            "511260_spliced": window_stats(cash_series["拼接 511260"], s, e),
            "A100": window_stats(cash_series["A100"], s, e),
        }
    cash_mdd = {"full": full, "from_splice": from_splice, "stress": cash_stress}

    # 锁
    locked = flag_v30.astype(bool)
    wv, wa = base_w["proxy"]["cash"], w_keep["A100"]["cash"]
    spells = spells_of(locked)
    spell_rows = []
    for s, e in spells:
        sv = window_stats(base_rets["proxy"], s, e)
        sa = window_stats(rets_keep["A100"], s, e)
        spell_rows.append({"start": str(pd.Timestamp(s).date()), "end": str(pd.Timestamp(e).date()),
                           "n": int(locked.loc[s:e].sum()), "v30": sv, "a100": sa})
    spell_rows.sort(key=lambda z: z["n"], reverse=True)
    off = ~locked
    lock = {
        "n_days": int(locked.sum()),
        "n_spells": len(spells),
        "cash_v30": float(wv[locked].mean()) if locked.any() else None,
        "cash_v30_off": float(wv[off].mean()) if off.any() else None,
        "cash_a100": float(wa[locked].mean()) if locked.any() else None,
        "cash_a100_off": float(wa[off].mean()) if off.any() else None,
        "lock_total_v30": compounded(base_rets["proxy"], locked)["total"],
        "lock_total_a100": compounded(rets_keep["A100"], locked)["total"],
        "off_total_v30": compounded(base_rets["proxy"], off)["total"],
        "off_total_a100": compounded(rets_keep["A100"], off)["total"],
        "longest": spell_rows[:8],
    }

    text_verdict = verdict_text(promoted, g, plat, valid, rows, base)
    cfg_path = maybe_write_cfg(base_cfg, rows, base) if promoted else None
    meta = {"proxy_end": str(R.index.max().date()), "real_end": str(Rr.index.max().date()),
            "pin": m_pin, "verdict": text_verdict, "cfg_path": cfg_path}
    report = build_report(digest, meta, rows, base, g, plat, valid, stress, cash_mdd, lock, promoted)
    report_path = os.path.join(REF, "exp_v37_cash511260_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    payload = {
        "digest": digest, "promoted": promoted, "valid": valid, "gate_a": g, "platform": plat,
        "oos_ok": oos_ok, "baseline": base, "rows": rows, "stress": stress,
        "cash_layer": {"full": full, "from_splice": from_splice, "stress": cash_stress},
        "lock": lock, "data": meta["proxy_end"] and {
            "proxy_end": meta["proxy_end"], "real_end": meta["real_end"]},
        "cfg": cfg_path, "verdict": text_verdict,
    }
    json_path = os.path.join(REF, "exp_v37_cash511260_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
        f.write("\n")
    print(f"[写] {report_path}")
    print(f"[写] {json_path}")
    print(text_verdict)


if __name__ == "__main__":
    main()
