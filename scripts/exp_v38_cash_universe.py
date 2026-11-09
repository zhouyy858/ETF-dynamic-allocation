# -*- coding: utf-8 -*-
"""v38 纸面实验：现金层工具对照（2026-11-09）

规则在 PRE / PURE / JOBS / PROMO / STRESS / ANCHORS / FROZEN / PRE_REG_MD 里写死。
先提交指纹，再跑数。结果不得回改这些对象。
不改 final_cfg_v30.json、README、SKILL.md、scripts/engine.py。
不写 final_cfg_v38.json，不设新的默认配置。

用法:
  python3 scripts/exp_v38_cash_universe.py --fingerprint-only
  python3 scripts/exp_v38_cash_universe.py
"""
import sys, os, json, hashlib, time, urllib.request
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from metrics import TRADING_DAYS, annualized_ret, max_drawdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
DATA = os.path.join(ROOT, "assets", "data", "v38_cash")
CFG_FILE = os.path.join(REF, "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
SPLICE = "2017-08-24"
OOS_FLOOR = 1.26
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}

# 信号只跟面板走，跟现金序列无关。同一面板复用一次，避免每个臂重算 RSRS。
_SIG = {}
_ORIG_SIG = DynamicStrategy.SIG_CLASS


def _sig_cached(R, a_mkt_override=None, lag=0, gate_win=120):
    key = (id(R), None if a_mkt_override is None else id(a_mkt_override), int(lag), int(gate_win))
    hit = _SIG.get(key)
    if hit is None:
        hit = _ORIG_SIG(R, a_mkt_override=a_mkt_override, lag=lag, gate_win=gate_win)
        _SIG[key] = hit
    return hit


DynamicStrategy.SIG_CLASS = _sig_cached

# v37 结果文件里的生产路径与 A100。容差 1e-8。对不上就中止。
PIN_V30 = {
    "proxy": {"cagr": 0.11111021816342403, "mdd": -0.09910849911812059, "cal": 1.1210967692185454},
    "real": {"cagr": 0.23703303024906486, "mdd": -0.03751029681180773, "cal": 6.319145685204232},
    "oos": {"cagr": 0.09731964730125986, "mdd": -0.07731831259471256, "cal": 1.2586881947539437},
}
PIN_A100 = {
    "proxy": {"cagr": 0.12041732788095327, "mdd": -0.09563306611941624, "cal": 1.2591599617917626},
    "real": {"cagr": 0.24323726154202463, "mdd": -0.03744283692152839, "cal": 6.496229493822656},
    "oos": {"cagr": 0.10961349033808165, "mdd": -0.07446729341874136, "cal": 1.4719682333787487},
}

FROZEN = {
    "references/final_cfg_v30.json": "dcff7b40aae3b7b30c0a2561f94f8cb1c2e296645dcf8ceb2ac5bdd244a04630",
    "README.md": "36ab41bd6279ab719f1d491a476e1a9c45e173b00ce6c45454405fba0ea08b02",
    "SKILL.md": "37a52f2214835d646f9031acc08d271520a2dbc96a864d7440c6846d30b0b7bb",
    "scripts/engine.py": "4c70a4d2207f60ef33ffa062d546cbd943bb183776ef17ca3e05ef5ce5ff403a",
}

PRE = {
    "dcal_pos": 0.0,
    "dcal_one": 0.15,
    "proxy_mdd": -0.10,
    "cliff": 0.15,
    "mono_tol": 0.02,
    "oos_floor": OOS_FLOOR,
    "anchor_tol": 1e-8,
    "splice_tol": 1e-8,
    "source_warn": 1e-3,
    "hole_days": 5,
    "hole_frac": 0.02,
    "flat_span": 0.01,
    "mm_cagr_lo": 0.008,
    "mm_cagr_hi": 0.06,
    "own_mdd_proposal": -0.08,
    "stress_mdd_worse": 0.01,
    "selloff_mdd": -0.03,
    "thin_yuan": 1.0e8,
    "zero_amt_frac": 0.20,
    "list_gap_days": 15,
}

PURE = [
    "511090", "511130", "511520", "511020", "511030",
    "511360", "511380", "518880", "511880", "511990",
]
MONEY = ["511880", "511990"]
RISK_CODES = ["518880", "511380"]
FETCH_CODES = PURE + ["511260", "511010"]

ANCHORS = {"v30": PIN_V30, "a100": PIN_A100, "splice": SPLICE, "oos_floor": OOS_FLOOR}


def _w(**kw):
    return {k: float(v) for k, v in kw.items()}


def build_jobs():
    jobs = [{"id": "A100", "role": "baseline", "w": _w(**{"511260": 1.0})}]
    for c in PURE:
        jobs.append({"id": f"N50_{c}", "role": "neighbor", "w": _w(**{c: 0.50, "511260": 0.50})})
        jobs.append({"id": f"N75_{c}", "role": "neighbor", "w": _w(**{c: 0.75, "511260": 0.25})})
        jobs.append({"id": f"P100_{c}", "role": "primary", "w": _w(**{c: 1.0})})
    jobs.append({"id": "N25_511090", "role": "neighbor", "w": _w(**{"511090": 0.25, "511260": 0.75})})
    jobs.append({"id": "G15", "role": "neighbor", "w": _w(**{"518880": 0.15, "511260": 0.85})})
    jobs.append({"id": "G30", "role": "primary", "w": _w(**{"518880": 0.30, "511260": 0.70})})
    jobs.append({"id": "G45", "role": "neighbor", "w": _w(**{"518880": 0.45, "511260": 0.55})})
    t = 1.0 / 3.0
    jobs.append({"id": "LAD", "role": "primary", "w": _w(**{"511010": t, "511260": t, "511090": t})})
    jobs.append({"id": "LAD_lo", "role": "neighbor",
                 "w": _w(**{"511010": 0.50, "511260": 0.25, "511090": 0.25})})
    jobs.append({"id": "LAD_hi", "role": "neighbor",
                 "w": _w(**{"511010": 0.25, "511260": 0.25, "511090": 0.50})})
    return jobs


JOBS = build_jobs()
JOB = {j["id"]: j for j in JOBS}

PROMO = []
for _c in PURE:
    PROMO.append({
        "id": f"P_{_c}",
        "primary": f"P100_{_c}",
        "axis": [f"N50_{_c}", f"N75_{_c}", f"P100_{_c}"],
        "rule": "endpoint",
        "risk": _c in RISK_CODES,
    })
PROMO.append({
    "id": "M50", "primary": "N50_511090",
    "axis": ["N25_511090", "N50_511090", "N75_511090"],
    "rule": "interior", "risk": False,
})
PROMO.append({
    "id": "M70g", "primary": "G30",
    "axis": ["G15", "G30", "G45"],
    "rule": "interior", "risk": True,
})
PROMO.append({
    "id": "LAD", "primary": "LAD",
    "axis": ["LAD_lo", "LAD", "LAD_hi"],
    "rule": "interior", "risk": False,
})

STRESS = [
    ("2016Q4-2017", "2016-10-01", "2017-12-31", True),
    ("2016Q4-2017_拼接段", "2016-10-01", "2017-08-23", True),
    ("2017-08-24至年末", "2017-08-24", "2017-12-31", True),
    ("2020-05至10", "2020-05-01", "2020-10-31", True),
    ("2022Q4", "2022-10-01", "2022-12-31", True),
    ("2024自然年", "2024-01-01", "2024-12-31", False),
    ("2025自然年", "2025-01-01", "2025-12-31", False),
]

for _j in JOBS:
    _s = sum(_j["w"].values())
    if abs(_s - 1.0) > 1e-9:
        raise SystemExit(f"权重和不为 1: {_j['id']} {_s}")
if len(JOB) != len(JOBS):
    raise SystemExit("臂 id 重复")
if len(PROMO) != 13:
    raise SystemExit(f"主点应是 13 个，现在是 {len(PROMO)}")


PRE_REG_MD = """\
## 1. 预注册（本节在跑数之前写入脚本，不随结果改写）

对照两层。参考是 `references/final_cfg_v30.json`（生产路径，`cash_bond_code=511010`、`cash_bond_pct=0.75`）。新基线是 v37 的 A100：现金层 100% 511260。不改 v30，不改 README，不改 SKILL.md，不改 `scripts/engine.py`，不写新的默认配置，也不写 `final_cfg_v38.json`。纸面，不实盘。

引擎、费用（单边万 5）、信号（`signal_lag=1`、溢价 T-2、先计提后成交、`strict=True`）、窗口与 v37 相同：

- proxy 全历史从 2014-06-23 起
- real 从 2025-04-23 起
- OOS 是 proxy 上的独立起点 2022-01-04（新策略实例，状态不从 2014 带入）

### 1.1 这次在问什么

v37 的 A100（整层现金 100% 511260）过了三关：proxy 12.04% / -9.56% / 1.26，real 24.32% / -3.74% / 6.50，OOS Cal 1.472。本轮只换现金层里的工具，不新搜阈值，不改权益目标，不改深熊锁，不动底仓。问题是：有没有工具在同一套门槛下优于 511260。

名单在跑数前写死。纯工具各做现金层的 100%：511090、511130、511520、511020、511030、511360、511380、518880、511880、511990。三个混合也各做现金层的 100%：M50 = 50% 511260 + 50% 511090；M70g = 70% 511260 + 30% 518880；LAD = 511010、511260、511090 各 1/3。不另加名单外的代码。用户括号里的「政金债/公司债」只是提问时的分类。简称以交易所行情为准，对不上就照实写，不改代码。

### 1.2 序列从哪来，缺了怎么办

511010、511260 的回测收益只用仓库里的 `assets/data/511010_nav.csv` 与 `assets/data/511260_nav.csv`（`rets_from(..., "cum_nav")`）。511260 在 2017-08-24 之前等于 511010，这是 v37 已经写明的拼接，本轮继承，不把它换成另一条 10Y 代理。起点早于 2017-08-24 的结果，只要含有 511260，就必须写明这段是拼接。起点不早于 2017-08-24 的窗口里，511260 成分就是上市后的 10Y 本身。

其余代码的价格用东财日 K 前复权收盘（`push2his.eastmoney.com` 的 `kline/get`，`secid=1.{code}`，`klt=101`，`fqt=1`，收盘是第 3 字段，成交额是第 7 字段，单位元）。成交额只用于流动性，不参与收益。名称用东财行情简称（`f58`），上市日用 `f189`；取不到就写首根 K 线的日期，并注明字段缺失。`f189` 与首根 K 线相差超过 15 个自然日时，两个都写上。

不把短历史品种接到 511010、511260 或逆回购上。不外推，不编指数。上市前的收益保持缺失，不用逆回购填。引擎原有的「缺日填逆回购」只允许出现在 511010 / 511260 这两条仓库序列上（与 v37 的 `align_ret` 相同），以便 A100 能对上锚。新品种的回测窗口一旦含有缺失日，该窗口不作数。

内部缺口：在该品种首个收益日到面板末日之间，若缺失交易日的最长连续段不超过 5 天且缺失占比不超过 2%，这些日子的收益记 0 并写明填了几天。超过就判数据有洞，该代码不进回测。

货币基金只有 511880 与 511990。若前复权收盘的全样本最高价相对最低价的幅度小于 1%，判定价格不含分红，经济收益不可用，不回测，也不用累计净值或一个假设收益率去补。幅度够了之后，这条价格收益的 CAGR 还必须落在 0.8% 到 6% 之间，否则同样判不可用。

下载写入 `assets/data/v38_cash/{code}.csv`。文件已在就不再下载。

### 1.3 有效性（对不上就中止，不解释候选）

- v30 生产路径必须对上 v37 公布的全精度。proxy：CAGR 0.11111021816342403、MDD -0.09910849911812059、Calmar 1.1210967692185454。real 与 OOS 的 CAGR、MDD、Calmar 对上 v37 结果里的基线。容差 1e-8。
- A100（100% 仓库 511260）必须对上 v37 的 A100 全精度，proxy / real / OOS 的 CAGR、MDD、Calmar 容差 1e-8。公布对照是 proxy 12.04% / -9.56% / 1.26，real 24.32% / -3.74% / 6.50，OOS Cal 1.472。
- 511260 与 511010 的拼接核对与 v37 相同：2017-08-24 之前重叠日收益绝对差全部不超过 1e-8，第一条更大的差异落在 2017-08-24 到 2017-09-15。
- v30 与 A100 的 CN 锁日历必须一致。锁来自市场信号，不看组合净值。
- `references/final_cfg_v30.json`、`README.md`、`SKILL.md`、`scripts/engine.py` 的 sha256 必须等于脚本里的 FROZEN。

东财 511260 与仓库拼接文件在 2017-08-24 之后的日收益差、东财与仓库里已有的 518880 / 511380 / 511010 文件的差，只陈列，不替换回测序列。差的绝对值最大超过 1e-3 时，结论里要点明来源不一致。

### 1.4 全窗口才有资格进门槛

一个臂在某个标准窗口上「全窗口可评」，当且仅当现金收益从该窗口的标准起点到面板末日每个交易日都有定义。三个标准窗口都可评，才评门槛。否则该窗口改从各成分都有自身收益的第一天起跑，v30 和 A100 用同一个新起点重新建实例。缩短窗口的数字单独列表，不能过门槛，也不能叫做优于 511260。

real 窗如果被品种的上市日盖住（上市早于 2025-04-23），用的就是标准 real 窗，可以和 A100 的 real 直接比。这一条仍然不够，还要 proxy 全历史。

### 1.5 邻域

纯工具 C 的轴，按 C 的权重从低到高：50% C + 50% 511260，75% C + 25% 511260，100% C。主点是 100%。规则与 v37 的平台相同：每个点相对 v30 的 proxy 与 real ΔCalmar 都大于 0，proxy MDD 都严格好于 -10%（`mdd > -0.10`）；相邻两点的 proxy Calmar 绝对差不超过 0.15；沿权重升高的方向，后一点的 proxy Calmar 不得低于前一点超过 0.02。邻域点即使更高也不升格。

M50 的主点是 50% 511090 + 50% 511260。邻域是 25% 与 75% 的 511090，其余为 511260。这是内部点，不要求单调爬到 100%。平台改为：三点都满足上面的点态条件，并且两个邻域点的 proxy Calmar 与主点的绝对差都不超过 0.15。

M70g 的主点是 30% 518880 + 70% 511260。邻域是 15% 与 45% 黄金，其余为 511260。同样用内部点规则。

LAD 的主点是三等分。邻域：511010 / 511260 / 511090 = 0.50 / 0.25 / 0.25，以及 0.25 / 0.25 / 0.50。同样用内部点规则。

全窗口不可评的轴，仍按同一公式在缩短窗口上做诊断。诊断不过或过了都不产生「优于」的标签。诊断用的 v30 和 A100 与该轴同一个起点。

### 1.6 标签「优于 511260」

只评 13 个主点（10 个纯工具 + M50 + M70g + LAD），不评邻域。必须同时：

1. 三个标准窗口都可评，并且 1.3 的有效性通过。
2. 相对 v30：proxy 与 real 的 ΔCalmar 都大于 0，至少一个达到 +0.15；proxy MDD 严格好于 -10%。
3. 该主点的平台通过（1.5）。
4. OOS Calmar 不低于 1.26（绝对数，用未四舍五入的值；不跟本次 v30 重跑漂）。
5. proxy Calmar 严格高于 A100，并且 real Calmar 严格高于 A100。

换手只陈列，不否决。债熊只陈列，不否决。任一段组合 MDD 比同一路径上的 v30 深过 1.00 个百分点（`mdd_arm - mdd_v30 < -0.01`），结论必须写「这一段更深」。这条必写句覆盖 2016Q4–2017、2020 年 5–10 月、2022Q4，以及 1.8 的机械抛售窗。窗口被上市日截断、组合路径盖不住整段时，不拿残段做这句比较，改报现金层。2024 与 2025 两个自然年只陈列。

多个主点同时拿到标签时，提议文本里只点名 proxy Calmar 最高的那一个。标签本身可以同时有多条。

### 1.7 提议和标签分开

本实验不写配置文件。提议只出现在结论里。

518880 与 511380 事先点名为风险资产，不是现金。含有二者任一正权重的主点，即使拿到数字标签，提议仍然是维持 A100，不把黄金或可转债写进现金层。

其余主点还要看现金层自身的最大回撤（各成分按权重合成的日收益，从该序列第一天到 proxy 面板末日）。自身 MDD 深于 -8% 的，不提议替换现金层。-8% 比 v37 记录的拼接 511260 自身 MDD -5.90% 多留了约 2 个百分点，写在跑数之前。

没有任何主点既拿到标签、又通过上面两条提议约束时，提议就是：不替换 511260，默认仍是 v30。

缩短窗口上更好，或者只有 real 窗更好，都写进结果，但是不能变成提议。

### 1.8 债熊、自身回撤、流动性

组合与现金层都报，闭区间：

- 2016-10-01～2017-12-31，并拆成 2016-10-01～2017-08-23（拼接段）与 2017-08-24～2017-12-31
- 2020-05-01～2020-10-31
- 2022-10-01～2022-12-31
- 2024 自然年、2025 自然年（陈列）

2024-2025 长债抛售窗，跑数时按这个规则定，不定死日期：在 511090 的自身前复权收益上，取 2024-01-01～2025-12-31（数据先结束就截到数据末日）的峰到谷。回撤深于 -3% 才成立，窗口就是峰值日到谷值日。511090 不可用时改 511130，再不可用改东财 511260（上市后的序列，不用仓库拼接）。都不深于 -3% 就写「未出现」，不另造一段。

流动性：每个代码的日均成交额等于成交额字段里大于 0 的那些日子的算术平均。全样本和最近 252 个交易日各报一次，另报中位数。成交额字段不是成交量。最近 252 日日均低于 1 亿元的，结论里要点名流动性薄，这一条不否决。零成交额的日子占比超过 20% 就写明字段不可靠。

现金层自身回撤用现金工具的日收益，不是组合收益。至少给出 v30 混合（25% 逆回购 + 75% 511010）、纯 511010、拼接 511260、A100，以及每个收益可用的候选和三个混合。拼接 511260 另给从 2017-08-24 起的 MDD。

### 1.9 事先不做的事

不改权益侧，不重开日频，不扫名单外的 ETF，不把短历史拼成 2014 起的伪长样本，不写 `final_cfg_v38.json`，不改 README / SKILL / v30 / 引擎。13 个主点是一次固定名单上的比较，不是单臂确认。结论里写明看过几条。
"""


def fingerprint():
    blob = json.dumps(
        {"PRE": PRE, "PURE": PURE, "MONEY": MONEY, "RISK_CODES": RISK_CODES,
         "JOBS": JOBS, "PROMO": PROMO, "STRESS": STRESS, "ANCHORS": ANCHORS,
         "FROZEN": FROZEN, "md": PRE_REG_MD},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class LockProbe(DynamicStrategy):
    """只记录 CN 锁。不改目标。"""

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


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_get_retry(url, tries=4):
    err = None
    for i in range(tries):
        try:
            return http_get(url)
        except Exception as e:
            err = e
            time.sleep(1.5 * (2 ** i))
    raise err


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def frozen_ok():
    bad = []
    for rel, exp in FROZEN.items():
        got = sha256_file(os.path.join(ROOT, rel))
        if got != exp:
            bad.append(rel)
    return bad


def repo_daily(repo):
    return (1 + repo) ** (1 / TRADING_DAYS) - 1


def align_ret(series, dates, repo_d):
    return series.reindex(dates).ffill().fillna(repo_d)


def pack(e):
    return {"cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
            "sharpe": float(e["sharpe"]), "to": float(e["turnover"]),
            "cash": float(e["avg_cash"]), "n": int(e.get("n", 0)) if "n" in e else None}


def execute(R, cfg, start, bond, pct, end=None, cls=DynamicStrategy):
    ds = cls(R, cfg=cfg)
    res = run_backtest(
        R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="DYN", min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=pct,
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    ev = evaluate(res)
    ev["n"] = int(len(res["rets"]))
    out = pack(ev)
    out["n"] = int(len(res["rets"]))
    return out, res, ds


def near(got, exp, keys, tol):
    bad = []
    for k in keys:
        if abs(float(got[k]) - float(exp[k])) > tol:
            bad.append(f"{k}:{got[k]:.10f}!={exp[k]:.10f}")
    return bad


def splice_check(r010, r260):
    both = pd.concat([r010.rename("a"), r260.rename("b")], axis=1).dropna()
    pre = both[both.index < pd.Timestamp(SPLICE)]
    pre_max = float((pre["a"] - pre["b"]).abs().max()) if len(pre) else float("nan")
    gap = (both["a"] - both["b"]).abs()
    hit = gap[gap > PRE["splice_tol"]]
    first = None if hit.empty else str(hit.index.min().date())
    ok = (pre_max <= PRE["splice_tol"] and first is not None
          and "2017-08-24" <= first <= "2017-09-15")
    return {"n_pre": int(len(pre)), "pre_max_abs": pre_max, "first_diff": first, "ok": bool(ok)}


def max_run(mask):
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def window_stats(rets, start, end):
    sub = rets[(rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))].dropna()
    if len(sub) < 5:
        return {"n": int(len(sub)), "cagr": None, "mdd": None, "total": None}
    w = (1 + sub).cumprod()
    mdd, pk, tr = max_drawdown(w)
    ann = float(annualized_ret(sub))
    return {"n": int(len(sub)), "cagr": ann, "mdd": float(mdd),
            "total": float(w.iloc[-1] - 1),
            "cal": float(ann / abs(mdd)) if mdd < 0 else None,
            "peak": None if pk is None else str(pd.Timestamp(pk).date()),
            "trough": None if tr is None else str(pd.Timestamp(tr).date())}


def series_stats(r):
    r = r.dropna()
    if len(r) < 5:
        return {"n": int(len(r)), "cagr": None, "mdd": None}
    w = (1 + r).cumprod()
    mdd, pk, tr = max_drawdown(w)
    return {"n": int(len(r)), "start": str(r.index.min().date()), "end": str(r.index.max().date()),
            "cagr": float(annualized_ret(r)), "mdd": float(mdd),
            "total": float(w.iloc[-1] - 1),
            "peak": None if pk is None else str(pd.Timestamp(pk).date()),
            "trough": None if tr is None else str(pd.Timestamp(tr).date())}


def pct(x):
    if x is None or x != x:
        return "nan"
    return f"{x * 100:.2f}%"


def calf(x):
    if x is None or x != x:
        return "nan"
    return f"{x:.2f}"


def dlt(x):
    if x is None or x != x:
        return "nan"
    return f"{x:+.3f}"


def cell(m):
    if m is None:
        return "n/a"
    return f"{pct(m['cagr'])} / {pct(m['mdd'])} / {calf(m['cal'])} / {m['to']:.2f}"


def stress_cell(s):
    if not s or s.get("total") is None:
        return "无数据"
    return f"{pct(s['total'])} / {pct(s['mdd'])} / {s['n']}日"


def wtxt(w):
    keys = sorted(w, key=lambda k: (-w[k], k))
    parts = []
    for k in keys:
        x = w[k]
        if abs(x * 3 - 1) < 1e-9:
            parts.append(f"{k} 1/3")
        else:
            parts.append(f"{k} {x * 100:.0f}%")
    return " + ".join(parts)


def yi(x):
    if x is None or x != x:
        return "nan"
    return f"{x / 1e8:.2f}亿"


def fetch_kline(code):
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid=1.{code}&klt=101&fqt=1&beg=20000101&end=20261231&lmt=100000"
        "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57"
    )
    raw = http_get_retry(url)
    j = json.loads(raw.decode("utf-8", "replace"))
    data = j.get("data") or {}
    rows = []
    for line in data.get("klines") or []:
        p = line.split(",")
        if len(p) < 7:
            continue
        rows.append({"date": p[0], "close": float(p[2]), "amount": float(p[6])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, data.get("name")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").sort_values("date")
    return df, data.get("name")


def fetch_quote(code):
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=1.{code}&fields=f57,f58,f189"
    j = json.loads(http_get_retry(url).decode("utf-8", "replace"))
    d = j.get("data") or {}
    listed = d.get("f189")
    listed_s = None
    if listed:
        s = str(int(listed))
        if len(s) == 8:
            listed_s = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return {"name": d.get("f58"), "listed": listed_s}


def load_or_fetch(code):
    path = os.path.join(DATA, f"{code}.csv")
    name = None
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df, name = fetch_kline(code)
        if df.empty:
            raise RuntimeError(f"{code} 日 K 为空")
        os.makedirs(DATA, exist_ok=True)
        out = df.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        time.sleep(0.35)
    quote = {"name": name, "listed": None}
    try:
        quote = fetch_quote(code)
        time.sleep(0.2)
    except Exception as e:
        quote["error"] = str(e)[:160]
    if quote.get("name") is None and name:
        quote["name"] = name
    return df, quote


def prepare_price(df):
    s = df.set_index("date")["close"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    amt = df.set_index("date")["amount"].astype(float).sort_index()
    amt = amt[~amt.index.duplicated(keep="last")]
    rets = s.pct_change().dropna()
    return s, amt, rets


def liquidity(amt):
    amt = amt.dropna()
    if amt.empty:
        return {"n": 0, "zero_frac": None, "mean": None, "median": None,
                "mean_252": None, "median_252": None, "unreliable": True}
    zero = float((amt <= 0).mean())
    pos = amt[amt > 0]
    tail = pos.iloc[-252:] if len(pos) else pos
    return {
        "n": int(len(amt)),
        "n_pos": int(len(pos)),
        "zero_frac": zero,
        "mean": None if pos.empty else float(pos.mean()),
        "median": None if pos.empty else float(pos.median()),
        "mean_252": None if tail.empty else float(tail.mean()),
        "median_252": None if tail.empty else float(tail.median()),
        "unreliable": bool(zero > PRE["zero_amt_frac"] or pos.empty),
        "start": str(amt.index.min().date()),
        "end": str(amt.index.max().date()),
    }


def place_on_calendar(rets, calendar):
    """把自身收益放到面板日历上。前缀缺失保留。内部小洞记 0。"""
    aligned = rets.reindex(calendar)
    if aligned.dropna().empty:
        return aligned, {"ok": False, "reason": "没有收益", "filled": 0, "max_gap": 0}
    first = aligned.first_valid_index()
    last_own = aligned.last_valid_index()
    end = calendar.max()
    # 尾部只允许面板比行情多出最多 hole_days 天
    after = calendar[calendar > last_own]
    if len(after) > PRE["hole_days"]:
        return aligned, {"ok": False, "reason": f"行情在 {last_own.date()} 结束，面板还往后 {len(after)} 天",
                         "filled": 0, "max_gap": int(len(after))}
    sl = aligned.loc[first:end]
    miss = sl.isna()
    gap = max_run(miss.tolist())
    frac = float(miss.mean()) if len(sl) else 1.0
    if gap > PRE["hole_days"] or frac > PRE["hole_frac"]:
        return aligned, {"ok": False, "reason": f"内部缺口 最长{gap}日 占比{frac:.2%}",
                         "filled": int(miss.sum()), "max_gap": gap}
    filled = aligned.copy()
    filled.loc[first:end] = filled.loc[first:end].fillna(0.0)
    return filled, {"ok": True, "reason": "", "filled": int(miss.sum()), "max_gap": gap,
                    "first": str(first.date()), "last": str(end.date())}


def money_check(code, close, rets):
    if code not in MONEY:
        return {"ok": True, "reason": ""}
    span = float((close.max() - close.min()) / close.min()) if close.min() else 0.0
    if span < PRE["flat_span"]:
        return {"ok": False, "span": span,
                "reason": f"前复权价幅度 {span:.3%}，价格不含分红，经济收益不可用"}
    st = series_stats(rets)
    cagr = st.get("cagr")
    if cagr is None or not (PRE["mm_cagr_lo"] <= cagr <= PRE["mm_cagr_hi"]):
        return {"ok": False, "span": span, "cagr": cagr,
                "reason": f"货币基金前复权 CAGR {cagr} 不在 0.8%–6%，不采用"}
    return {"ok": True, "span": span, "cagr": cagr, "reason": ""}


def mix_on(dates, weights, r010, r260, ready, repo_d):
    acc = pd.Series(0.0, index=dates)
    nan = pd.Series(False, index=dates)
    for code, w in weights.items():
        if code == "511260":
            leg = align_ret(r260, dates, repo_d)
        elif code == "511010":
            leg = align_ret(r010, dates, repo_d)
        else:
            leg = ready[code].reindex(dates)
            nan = nan | leg.isna()
        acc = acc + w * leg.fillna(0.0)
    acc[nan] = np.nan
    return acc


def usable_start(cash, window_start):
    sl = cash[cash.index >= pd.Timestamp(window_start)]
    if sl.empty:
        return None, False
    finite = sl.notna().to_numpy()
    if finite.all():
        return str(pd.Timestamp(window_start).date()), True
    if not finite.any():
        return None, False
    first = int(np.argmax(finite))
    if finite[first:].all():
        return str(sl.index[first].date()), False
    return None, False


def run_v30(panel, cfg, r010, start, cls=DynamicStrategy):
    pct = cfg.get("cash_bond_pct", 0.75)
    return execute(panel, cfg, start, r010, pct, cls=cls)


def run_mix(panel, cfg, r010, r260, ready, weights, start, repo_d, cls=DynamicStrategy):
    sub = panel[panel.index >= pd.Timestamp(start)]
    dates = sub.index
    cash = mix_on(dates, weights, r010, r260, ready, repo_d)
    if cash.isna().any():
        raise RuntimeError(f"现金收益在 {start} 之后仍有缺失 {int(cash.isna().sum())}")
    return execute(panel, cfg, start, cash, 1.0, cls=cls)


def lock_flags(ds, index):
    marks = pd.Series(ds.lock_at).sort_index()
    flag = pd.Series(index=index, dtype="boolean")
    if len(marks):
        inter = flag.index.intersection(marks.index)
        flag.loc[inter] = marks.reindex(inter).astype(bool)
    return flag.ffill().fillna(False).astype(bool)


def pointwise(row_p, row_r, base_p, base_r):
    reasons = []
    if row_p is None or row_r is None:
        return {"ok": False, "reasons": ["缺窗口"]}
    d_p = row_p["cal"] - base_p["cal"]
    d_r = row_r["cal"] - base_r["cal"]
    if not (d_p > PRE["dcal_pos"] and d_r > PRE["dcal_pos"]):
        reasons.append(f"双窗未同向({d_p:+.3f}/{d_r:+.3f})")
    if not (row_p["mdd"] > PRE["proxy_mdd"]):
        reasons.append(f"proxy MDD {row_p['mdd']*100:.2f}% 不优于 -10%")
    return {"ok": not reasons, "reasons": reasons, "dcal_p": d_p, "dcal_r": d_r}


def platform_endpoint(ids, mets, bases):
    """mets/bases: id -> {proxy, real} 已经是同一套起点上的指标。"""
    reasons = []
    cals = []
    for i in ids:
        pw = pointwise(mets[i]["proxy"], mets[i]["real"], bases["v30"]["proxy"], bases["v30"]["real"])
        cals.append(mets[i]["proxy"]["cal"])
        for r in pw["reasons"]:
            reasons.append(f"{i} {r}")
    for i in range(1, len(cals)):
        gap = abs(cals[i] - cals[i - 1])
        if gap > PRE["cliff"]:
            reasons.append(f"{ids[i-1]}→{ids[i]} 相邻 Calmar 差 {gap:.3f} > {PRE['cliff']}")
        if cals[i] < cals[i - 1] - PRE["mono_tol"]:
            reasons.append(
                f"{ids[i]} 比 {ids[i-1]} 低 {cals[i-1]-cals[i]:.3f}，反向超过 {PRE['mono_tol']}")
    return {"ok": not reasons, "reasons": reasons, "cals": cals}


def platform_interior(ids, primary, mets, bases):
    reasons = []
    for i in ids:
        pw = pointwise(mets[i]["proxy"], mets[i]["real"], bases["v30"]["proxy"], bases["v30"]["real"])
        for r in pw["reasons"]:
            reasons.append(f"{i} {r}")
    pcal = mets[primary]["proxy"]["cal"]
    for i in ids:
        if i == primary:
            continue
        gap = abs(mets[i]["proxy"]["cal"] - pcal)
        if gap > PRE["cliff"]:
            reasons.append(f"{i} 与主点 proxy Calmar 差 {gap:.3f} > {PRE['cliff']}")
    return {"ok": not reasons, "reasons": reasons,
            "cals": [mets[i]["proxy"]["cal"] for i in ids]}


def contains_511260_before_splice(start):
    return str(start) < SPLICE


def deeper(arm, base):
    if not arm or not base or arm.get("mdd") is None or base.get("mdd") is None:
        return None
    return float(arm["mdd"] - base["mdd"])


def fmt_deeper(delta):
    if delta is None:
        return "n/a"
    if delta < -PRE["stress_mdd_worse"]:
        return f"{delta*100:+.2f}pp，更深"
    return f"{delta*100:+.2f}pp"


def ret_diff(a, b):
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if both.empty:
        return {"n": 0, "max_abs": None}
    return {"n": int(len(both)), "max_abs": float((both["a"] - both["b"]).abs().max()),
            "start": str(both.index.min().date()), "end": str(both.index.max().date())}


def selloff_window(ready_map, status):
    """按预注册的优先序找 2024-2025 峰谷。返回 dict。"""
    order = ["511090", "511130", "511260_em"]
    for key in order:
        if key == "511260_em":
            series = ready_map.get("511260_em")
            label = "东财511260"
        else:
            if not status.get(key, {}).get("usable"):
                continue
            series = ready_map.get(key)
            label = key
        if series is None:
            continue
        sub = series[(series.index >= "2024-01-01") & (series.index <= "2025-12-31")].dropna()
        if len(sub) < 20:
            continue
        st = series_stats(sub)
        fired = st["mdd"] is not None and st["mdd"] < PRE["selloff_mdd"]
        return {"fired": bool(fired), "source": label, "n": st["n"],
                "mdd": st["mdd"], "peak": st.get("peak"), "trough": st.get("trough"),
                "start": st.get("peak") if fired else None,
                "end": st.get("trough") if fired else None}
    return {"fired": False, "source": None, "reason": "三条序列都不够 20 个交易日或不可用"}


def main():
    digest = fingerprint()
    print(f"[预注册指纹] {digest}", flush=True)
    if "--fingerprint-only" in sys.argv:
        return

    bad_frozen = frozen_ok()
    if bad_frozen:
        raise SystemExit("冻结文件被改动: " + ",".join(bad_frozen))

    base_cfg = json.load(open(CFG_FILE))
    repo = float(base_cfg.get("repo_rate", 0.022))
    repo_d = repo_daily(repo)
    r010 = rets_from(read_table("511010_nav.csv"), "cum_nav")
    r260 = rets_from(read_table("511260_nav.csv"), "cum_nav")
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    print(f"[数据] proxy {R.index.min().date()}~{R.index.max().date()} n={len(R)} | "
          f"real {Rr.index.min().date()}~{Rr.index.max().date()} n={len(Rr)}", flush=True)

    reasons = []
    sp = splice_check(r010, r260)
    print(f"[拼接] pre_max={sp['pre_max_abs']:.2e} first_diff={sp['first_diff']} ok={sp['ok']}", flush=True)
    if not sp["ok"]:
        reasons.append(f"拼接核对失败 first_diff={sp['first_diff']} pre_max={sp['pre_max_abs']}")

    panels = {"proxy": R, "real": Rr}
    windows = {"proxy": (R, PROXY_START), "real": (Rr, REAL_START), "oos": (R, OOS_START)}

    # 锚：v30 与 A100 的三个标准窗口。A100 在下载之前就能跑。
    anchor_runs = {"v30": {}, "A100": {}}
    anchor_rets = {}
    m, res, ds_v30 = run_v30(R, base_cfg, r010, PROXY_START, cls=LockProbe)
    anchor_runs["v30"]["proxy"] = m
    anchor_rets["v30"] = res["rets"]
    pin_bad = near(m, PIN_V30["proxy"], ("cagr", "mdd", "cal"), PRE["anchor_tol"])
    print(f"[身份 v30 proxy] {m['cagr']:.6f} {m['mdd']:.6f} {m['cal']:.6f} ok={not pin_bad}", flush=True)
    if pin_bad:
        reasons.append("v30 proxy 锚失败 " + ",".join(pin_bad))
    m_a, res_a, ds_a = run_mix(R, base_cfg, r010, r260, {}, {"511260": 1.0}, PROXY_START, repo_d, cls=LockProbe)
    anchor_runs["A100"]["proxy"] = m_a
    anchor_rets["A100"] = res_a["rets"]
    a_bad = near(m_a, PIN_A100["proxy"], ("cagr", "mdd", "cal"), PRE["anchor_tol"])
    print(f"[身份 A100 proxy] {m_a['cagr']:.6f} {m_a['mdd']:.6f} {m_a['cal']:.6f} ok={not a_bad}", flush=True)
    if a_bad:
        reasons.append("A100 proxy 锚失败 " + ",".join(a_bad))
    flag_v = lock_flags(ds_v30, res["rets"].index)
    flag_a = lock_flags(ds_a, res_a["rets"].index)
    common = flag_v.index.intersection(flag_a.index)
    lock_same = bool((flag_v.loc[common] == flag_a.loc[common]).all())
    if not lock_same:
        reasons.append("v30 与 A100 的 CN 锁日历不一致")

    for win, panel, start in (("real", Rr, REAL_START), ("oos", R, OOS_START)):
        mv, _, _ = run_v30(panel, base_cfg, r010, start)
        ma, _, _ = run_mix(panel, base_cfg, r010, r260, {}, {"511260": 1.0}, start, repo_d)
        anchor_runs["v30"][win] = mv
        anchor_runs["A100"][win] = ma
        vb = near(mv, PIN_V30[win], ("cagr", "mdd", "cal"), PRE["anchor_tol"])
        ab = near(ma, PIN_A100[win], ("cagr", "mdd", "cal"), PRE["anchor_tol"])
        print(f"[锚] {win} v30 {cell(mv)} ok={not vb} | A100 {cell(ma)} ok={not ab}", flush=True)
        if vb:
            reasons.append(f"v30 {win} 锚失败 " + ",".join(vb))
        if ab:
            reasons.append(f"A100 {win} 锚失败 " + ",".join(ab))

    valid = {"ok": not reasons, "reasons": reasons, "splice": sp, "lock_same": lock_same}
    if not valid["ok"]:
        text = render(digest, valid, None)
        path = os.path.join(REF, "exp_v38_cash_universe_report.md")
        with open(path, "w") as f:
            f.write(text)
        print(f"[写] {path}（锚失败，候选不作数）", flush=True)
        return

    calendar = R.index.union(Rr.index).sort_values()
    os.makedirs(DATA, exist_ok=True)
    books = {}
    ready = {}
    em_rets = {}
    for code in FETCH_CODES:
        print(f"[取] {code}", flush=True)
        try:
            df, quote = load_or_fetch(code)
        except Exception as e:
            books[code] = {"code": code, "usable": False, "reason": f"下载失败 {e}"[:200]}
            print(f"  失败 {e}", flush=True)
            continue
        close, amt, rets = prepare_price(df)
        placed, hole = place_on_calendar(rets, calendar)
        mm = money_check(code, close, rets)
        listed = quote.get("listed")
        first_bar = str(close.index.min().date())
        last_bar = str(close.index.max().date())
        gap_days = None
        if listed:
            gap_days = abs((pd.Timestamp(listed) - pd.Timestamp(first_bar)).days)
        usable = bool(hole.get("ok") and mm.get("ok"))
        reason = hole.get("reason") or mm.get("reason") or ""
        spike = None
        if len(rets):
            ix = rets.abs().idxmax()
            spike = {"date": str(ix.date()), "ret": float(rets.loc[ix])}
        books[code] = {
            "code": code, "name": quote.get("name"), "listed": listed, "first_bar": first_bar,
            "last_bar": last_bar, "n_bar": int(len(close)), "list_gap_days": gap_days,
            "quote_error": quote.get("error"), "usable": usable, "reason": reason,
            "hole": hole, "money": mm, "liq": liquidity(amt), "spike": spike,
            "close0": float(close.iloc[0]), "close1": float(close.iloc[-1]),
        }
        em_rets[code] = rets
        if usable and code not in ("511010", "511260"):
            ready[code] = placed
        elif code == "511260":
            # 抛售检测用上市后的东财序列，不进回测。
            ready_tail = placed[placed.index >= SPLICE]
            books[code]["detector_series"] = True
        print(f"  {books[code].get('name')} listed={listed} bar={first_bar}~{last_bar} "
              f"n={len(close)} usable={usable} {reason}", flush=True)

    # 东财 511260 上市后的收益，供抛售检测。不替换仓库序列。
    if "511260" in em_rets:
        em260 = em_rets["511260"]
        em260 = em260[em260.index >= SPLICE]
        placed260, _ = place_on_calendar(em260, calendar)
        ready["511260_em"] = placed260

    em260 = em_rets["511260"] if "511260" in em_rets else pd.Series(dtype=float)
    crosses = {
        "511260_after_splice": ret_diff(em260[em260.index >= SPLICE], r260[r260.index >= SPLICE]),
        "511010": ret_diff(em_rets.get("511010", pd.Series(dtype=float)), r010),
    }
    for code, fn in (("518880", "518880_nav.csv"), ("511380", "511380_nav.csv")):
        if code not in em_rets:
            crosses[code] = {"n": 0, "max_abs": None, "missing_file": True}
            continue
        try:
            local = rets_from(read_table(fn), "cum_nav")
            crosses[code] = ret_diff(em_rets[code], local)
        except Exception as e:
            crosses[code] = {"n": 0, "max_abs": None, "error": str(e)[:160]}

    # 每个臂的覆盖。511010/511260 视为从面板起点就有（仓库序列 + v37 填充）。
    def code_first(code):
        if code in ("511010", "511260"):
            return pd.Timestamp(PROXY_START)
        if code not in ready:
            return None
        s = ready[code].dropna()
        if s.empty:
            return None
        return s.index.min()

    def job_coverage(job):
        firsts = []
        for code in job["w"]:
            if code not in ("511010", "511260") and not books.get(code, {}).get("usable", False):
                return None, f"{code} 收益不可用：{books.get(code, {}).get('reason', '缺失')}"
            f = code_first(code)
            if f is None:
                return None, f"{code} 没有可用收益"
            firsts.append(f)
        return max(firsts), ""

    coverage = {}
    for job in JOBS:
        cov, why = job_coverage(job)
        coverage[job["id"]] = {"first": None if cov is None else str(cov.date()), "why": why, "ok": cov is not None}

    # 需要跑的 (kind, window, start)。kind 是 v30 或 job id。
    cache = {}

    def remember(kind, win, start, pack_m, rets):
        cache[(kind, win, start)] = {"m": pack_m, "rets": rets}

    # 锚已经跑过的标准窗口放进缓存。
    remember("v30", "proxy", PROXY_START, anchor_runs["v30"]["proxy"], anchor_rets["v30"])
    remember("A100", "proxy", PROXY_START, anchor_runs["A100"]["proxy"], anchor_rets["A100"])
    for win in ("real", "oos"):
        remember("v30", win, windows[win][1], anchor_runs["v30"][win], None)
        remember("A100", win, windows[win][1], anchor_runs["A100"][win], None)

    def ensure(kind, win, start):
        key = (kind, win, start)
        if key in cache and cache[key]["m"] is not None:
            return cache[key]
        panel, _ = windows[win]
        if kind == "v30":
            m, res, _ = run_v30(panel, base_cfg, r010, start)
        else:
            m, res, _ = run_mix(panel, base_cfg, r010, r260, ready, JOB[kind]["w"], start, repo_d)
        remember(kind, win, start, m, res["rets"] if win == "proxy" else None)
        return cache[key]

    runs = {}
    for job in JOBS:
        info = coverage[job["id"]]
        runs[job["id"]] = {"coverage": info, "windows": {}}
        if not info["ok"]:
            continue
        first = pd.Timestamp(info["first"])
        for win, (panel, wstart) in windows.items():
            start_use = max(pd.Timestamp(wstart), first)
            # 在该面板上确认从 start_use 起现金完整
            dates = panel[panel.index >= start_use].index
            if len(dates) < 5:
                runs[job["id"]]["windows"][win] = {"ok": False, "reason": "样本太短"}
                continue
            cash = mix_on(dates, job["w"], r010, r260, ready, repo_d)
            if cash.isna().any() or not np.isfinite(cash.to_numpy()).all():
                runs[job["id"]]["windows"][win] = {
                    "ok": False, "reason": f"起点 {start_use.date()} 之后仍有缺失"}
                continue
            full = str(start_use.date()) == wstart
            got = ensure(job["id"], win, str(start_use.date()))
            runs[job["id"]]["windows"][win] = {
                "ok": True, "full": full, "start": str(start_use.date()),
                "m": got["m"], "n": got["m"]["n"],
            }
            print(f"  {job['id']:<14} {win:<5} {str(start_use.date())} full={full} {cell(got['m'])}", flush=True)

    # 缩短起点上的 v30 / A100。标准起点已在缓存里。
    needed_starts = set()
    for job in JOBS:
        for win, rec in runs[job["id"]]["windows"].items():
            if rec.get("ok") and not rec.get("full"):
                needed_starts.add((win, rec["start"]))
    for win, start in sorted(needed_starts):
        ensure("v30", win, start)
        ensure("A100", win, start)
        print(f"[对照起点] {win} {start} v30 {cell(cache[('v30', win, start)]['m'])} "
              f"A100 {cell(cache[('A100', win, start)]['m'])}", flush=True)

    def metrics_of(kind, win, start):
        return cache[(kind, win, start)]["m"]

    # 门槛
    gate_rows = []
    for promo in PROMO:
        pid = promo["id"]
        primary = promo["primary"]
        recs = runs[primary]["windows"]
        full = all(recs.get(w, {}).get("full") for w in ("proxy", "real", "oos"))
        entry = {"id": pid, "primary": primary, "weights": JOB[primary]["w"],
                 "risk": promo["risk"], "full": bool(full and coverage[primary]["ok"]),
                 "coverage_why": coverage[primary]["why"], "label": False, "propose_ok": False}
        # 描述性：各窗口相对同一起点的 v30 / A100
        desc = {}
        for win in ("proxy", "real", "oos"):
            rec = recs.get(win) or {}
            if not rec.get("ok"):
                desc[win] = {"ok": False, "reason": rec.get("reason") or coverage[primary]["why"] or "未跑"}
                continue
            start = rec["start"]
            base = metrics_of("v30", win, start)
            a100 = metrics_of("A100", win, start)
            desc[win] = {"ok": True, "full": rec["full"], "start": start, "n": rec["n"],
                         "m": rec["m"],
                         "dcal_v30": rec["m"]["cal"] - base["cal"],
                         "dcal_a100": rec["m"]["cal"] - a100["cal"],
                         "dmdd_v30": rec["m"]["mdd"] - base["mdd"],
                         "dmdd_a100": rec["m"]["mdd"] - a100["mdd"]}
        entry["desc"] = desc

        # 平台：全窗口用标准起点；否则若三点 proxy 起点一致，做缩短诊断。
        axis_ok_full = full and all(
            runs[i]["windows"].get("proxy", {}).get("full") and runs[i]["windows"].get("real", {}).get("full")
            for i in promo["axis"])
        diag = None
        if axis_ok_full:
            mets = {i: {"proxy": runs[i]["windows"]["proxy"]["m"],
                        "real": runs[i]["windows"]["real"]["m"]} for i in promo["axis"]}
            bases = {"v30": {"proxy": metrics_of("v30", "proxy", PROXY_START),
                             "real": metrics_of("v30", "real", REAL_START)}}
            if promo["rule"] == "endpoint":
                plat = platform_endpoint(promo["axis"], mets, bases)
            else:
                plat = platform_interior(promo["axis"], primary, mets, bases)
            plat["kind"] = "full"
        else:
            plat = {"ok": False, "reasons": ["不是三个标准窗口都可评"], "cals": [], "kind": "ineligible"}
            starts = []
            can = True
            for i in promo["axis"]:
                pr = runs[i]["windows"].get("proxy") or {}
                rr = runs[i]["windows"].get("real") or {}
                if not (pr.get("ok") and rr.get("ok")):
                    can = False
                    break
                starts.append((pr["start"], pr.get("full"), rr["start"], rr.get("full")))
            if can and len(set(s[0] for s in starts)) == 1 and len(set(s[2] for s in starts)) == 1:
                ps, rs = starts[0][0], starts[0][2]
                mets = {i: {"proxy": runs[i]["windows"]["proxy"]["m"],
                            "real": runs[i]["windows"]["real"]["m"]} for i in promo["axis"]}
                bases = {"v30": {"proxy": metrics_of("v30", "proxy", ps),
                                 "real": metrics_of("v30", "real", rs)}}
                if promo["rule"] == "endpoint":
                    diag = platform_endpoint(promo["axis"], mets, bases)
                else:
                    diag = platform_interior(promo["axis"], primary, mets, bases)
                diag["proxy_start"] = ps
                diag["real_start"] = rs
                diag["kind"] = "short"
        entry["platform"] = plat
        entry["platform_diag"] = diag

        if entry["full"] and valid["ok"]:
            d_p = desc["proxy"]["dcal_v30"]
            d_r = desc["real"]["dcal_v30"]
            same = d_p > PRE["dcal_pos"] and d_r > PRE["dcal_pos"]
            one = max(d_p, d_r) >= PRE["dcal_one"]
            mdd_ok = desc["proxy"]["m"]["mdd"] > PRE["proxy_mdd"]
            oos_ok = desc["oos"]["m"]["cal"] >= PRE["oos_floor"]
            beat = desc["proxy"]["dcal_a100"] > 0 and desc["real"]["dcal_a100"] > 0
            entry["checks"] = {"same": bool(same), "one": bool(one), "mdd": bool(mdd_ok),
                               "oos": bool(oos_ok), "platform": bool(plat["ok"]), "beat": bool(beat)}
            entry["label"] = bool(same and one and mdd_ok and oos_ok and plat["ok"] and beat)
        else:
            entry["checks"] = None
            entry["label"] = False
        gate_rows.append(entry)
        print(f"[门] {pid} full={entry['full']} label={entry['label']}", flush=True)

    # 现金层序列（proxy 日历）
    proxy_dates = R[R.index >= PROXY_START].index
    cash_book = {
        "v30混合": 0.25 * repo_d + 0.75 * align_ret(r010, proxy_dates, repo_d),
        "511010": align_ret(r010, proxy_dates, repo_d),
        "511260拼接": align_ret(r260, proxy_dates, repo_d),
    }
    for code, series in ready.items():
        if code == "511260_em":
            continue
        cash_book[code] = series.reindex(proxy_dates)
    for job_id in [p["primary"] for p in PROMO]:
        if not coverage[job_id]["ok"]:
            continue
        cash_book["mix:" + job_id] = mix_on(proxy_dates, JOB[job_id]["w"], r010, r260, ready, repo_d)

    cash_stats = {k: series_stats(v.dropna()) for k, v in cash_book.items()}
    cash_from_splice = {}
    for k, v in cash_book.items():
        tail = v[v.index >= SPLICE].dropna()
        cash_from_splice[k] = series_stats(tail)

    detector = selloff_window(ready, books)
    stress_specs = [(n, a, b, must) for n, a, b, must in STRESS]
    if detector.get("fired"):
        stress_specs.append((f"2024-2025长债抛售({detector['peak']}~{detector['trough']})",
                             detector["peak"], detector["trough"], True))
        print(f"[抛售] {detector['source']} MDD {detector['mdd']:.2%} {detector['peak']}~{detector['trough']}", flush=True)
    else:
        print(f"[抛售] 未触发 {detector}", flush=True)

    # 组合压力：全窗口主点用标准 proxy 路径；v30 / A100 用锚路径。
    port_rets = {
        "v30": anchor_rets["v30"],
        "A100": anchor_rets["A100"],
    }
    for promo in PROMO:
        pid = promo["primary"]
        rec = runs[pid]["windows"].get("proxy") or {}
        if rec.get("ok") and rec.get("full"):
            port_rets[promo["id"]] = cache[(pid, "proxy", rec["start"])]["rets"]
        elif rec.get("ok"):
            port_rets[promo["id"]] = cache[(pid, "proxy", rec["start"])]["rets"]

    port_stress = {}
    cash_stress = {}
    for name, a, b, must in stress_specs:
        port_stress[name] = {"must": must, "v30_n": window_stats(port_rets["v30"], a, b)["n"]}
        for k, series in port_rets.items():
            st = window_stats(series, a, b)
            v30n = port_stress[name]["v30_n"] or 0
            complete = st["n"] is not None and v30n and st["n"] >= v30n - 2
            port_stress[name][k] = st
            port_stress[name][k]["complete"] = bool(complete)
        cash_stress[name] = {}
        for k, series in cash_book.items():
            cash_stress[name][k] = window_stats(series, a, b)

    # 自身 MDD 是否深于 -8%（提议约束）。用 mix 序列。
    for entry in gate_rows:
        key = "mix:" + entry["primary"]
        st = cash_stats.get(key) or {}
        entry["own_mdd"] = st.get("mdd")
        entry["own_stats"] = st
        risk = entry["risk"] or any(JOB[entry["primary"]]["w"].get(c, 0) > 0 for c in RISK_CODES)
        entry["risk"] = bool(risk)
        own_ok = st.get("mdd") is not None and st["mdd"] >= PRE["own_mdd_proposal"]
        entry["propose_ok"] = bool(entry["label"] and own_ok and not risk)

    labeled = [e for e in gate_rows if e["label"]]
    proposable = [e for e in gate_rows if e["propose_ok"]]
    proposable.sort(key=lambda e: e["desc"]["proxy"]["m"]["cal"], reverse=True)
    proposal = proposable[0]["id"] if proposable else None

    source_warns = []
    for k, d in crosses.items():
        mx = d.get("max_abs")
        if mx is not None and mx > PRE["source_warn"]:
            source_warns.append(f"{k} 最大日收益差 {mx:.3e}")

    ctx = {
        "digest": digest,
        "valid": valid,
        "meta": {"proxy_end": str(R.index.max().date()), "real_end": str(Rr.index.max().date()),
                 "proxy_n": int((R.index >= PROXY_START).sum()),
                 "real_n": int((Rr.index >= REAL_START).sum())},
        "books": books,
        "crosses": crosses,
        "source_warns": source_warns,
        "runs": runs,
        "gates": gate_rows,
        "labeled": [e["id"] for e in labeled],
        "proposal": proposal,
        "cash_stats": cash_stats,
        "cash_from_splice": cash_from_splice,
        "detector": detector,
        "stress_specs": [(n, a, b, must) for n, a, b, must in stress_specs],
        "port_stress": port_stress,
        "cash_stress": cash_stress,
        "anchors": {"v30": anchor_runs["v30"], "A100": anchor_runs["A100"]},
    }

    text = render(digest, valid, ctx)
    report_path = os.path.join(REF, "exp_v38_cash_universe_report.md")
    with open(report_path, "w") as f:
        f.write(text)
    json_path = os.path.join(REF, "exp_v38_cash_universe_results.json")
    dump = {
        "digest": digest,
        "valid": valid,
        "meta": ctx["meta"],
        "books": books,
        "crosses": crosses,
        "source_warns": source_warns,
        "anchors": ctx["anchors"],
        "gates": [{k: v for k, v in e.items()} for e in gate_rows],
        "labeled": ctx["labeled"],
        "proposal": proposal,
        "detector": detector,
        "cash_stats": cash_stats,
        "cash_from_splice": {k: v for k, v in cash_from_splice.items()
                             if k in ("v30混合", "511010", "511260拼接")},
        "port_stress": port_stress,
        "cash_stress": cash_stress,
    }
    with open(json_path, "w") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1, default=_json_default)
        f.write("\n")
    print(f"[写] {report_path}", flush=True)
    print(f"[写] {json_path}", flush=True)
    print(f"[提议] {proposal} 标签 {ctx['labeled']}", flush=True)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    raise TypeError(type(o))


def render(digest, valid, ctx):
    L = []
    A = L.append
    A("# v38 纸面实验：现金层还有没有比 511260 更好的工具")
    A("")
    A(f"预注册指纹 `{digest}`。")
    if ctx:
        A(f"数据 proxy 至 {ctx['meta']['proxy_end']}，real 至 {ctx['meta']['real_end']}。")
    A("第 1 节是跑数之前写死的规则。第 2 节才是结果。结论在第 3 节，规则没有按结果回改。")
    A("默认配置仍是 `references/final_cfg_v30.json`。README 当前持仓段和 SKILL.md 没有改。没有新的配置文件。")
    A("")
    A(PRE_REG_MD.rstrip())
    A("")
    A("## 2. 结果")
    A("")
    if not valid["ok"]:
        A("有效性没有通过，候选全部不作数。")
        A("")
        A("```")
        A(json.dumps(valid, ensure_ascii=False, indent=1, default=_json_default))
        A("```")
        A("")
        return "\n".join(L) + "\n"

    v30 = ctx["anchors"]["v30"]
    a100 = ctx["anchors"]["A100"]
    sp = valid["splice"]
    A(f"身份校验通过。v30 proxy {v30['proxy']['cagr']:.6f} / {v30['proxy']['mdd']:.6f} / {v30['proxy']['cal']:.6f}，"
      f"A100 proxy {a100['proxy']['cagr']:.6f} / {a100['proxy']['mdd']:.6f} / {a100['proxy']['cal']:.6f}。")
    A(f"拼接核对：2017-08-24 之前重叠 {sp['n_pre']} 日，最大绝对差 {sp['pre_max_abs']:.2e}；"
      f"第一条超过 1e-8 的日期是 {sp['first_diff']}。通过。")
    A(f"v30 与 A100 的 CN 锁日历{'一致' if valid['lock_same'] else '不一致'}。")
    A(f"**v30**：proxy {cell(v30['proxy'])}；real {cell(v30['real'])}；OOS Cal {v30['oos']['cal']:.4f}。")
    A(f"**A100（新基线，100% 511260）**：proxy {cell(a100['proxy'])}；real {cell(a100['real'])}；"
      f"OOS Cal {a100['oos']['cal']:.4f}。")
    A("公布对照：A100 proxy 12.04% / -9.56% / 1.26，real 24.32% / -3.74% / 6.50，OOS Cal 1.472。")
    A("换手是引擎里单边权重变动的累加，不是年化。ΔCal 是同一起点上的 Calmar 差。")
    A("")

    A("### 2.1 工具、上市日、序列")
    A("")
    A("回测用的 511010 / 511260 仍是仓库 csv。下表的行情来自这次下载的东财前复权，用来核对上市日、成交额，以及和仓库文件差多少。新候选的收益只用东财前复权，不接代理。")
    A("")
    A("| 代码 | 简称 | 上市日 f189 | 首根K线 | 末日 | 根数 | 收益 | 说明 |")
    A("|---|---|---|---|---|---|---|---|")
    for code in ["511260", "511010"] + PURE:
        b = ctx["books"].get(code) or {"usable": False, "reason": "没有记录"}
        name = b.get("name") or ""
        listed = b.get("listed") or "字段缺失"
        gap = b.get("list_gap_days")
        note = b.get("reason") or ""
        if gap is not None and gap > PRE["list_gap_days"]:
            note = (note + " " if note else "") + f"上市日与首根K线相差 {gap} 天"
        if code in ("511010", "511260"):
            use = "仓库 csv（拼接见上）" if code == "511260" else "仓库 csv"
        elif b.get("usable"):
            use = "东财前复权，可用"
            filled = (b.get("hole") or {}).get("filled") or 0
            if filled:
                note = (note + " " if note else "") + f"小洞填 0 共 {filled} 日"
        else:
            use = "不可用"
        A(f"| {code} | {name} | {listed} | {b.get('first_bar','')} | {b.get('last_bar','')} | "
          f"{b.get('n_bar','')} | {use} | {note} |")
    A("")
    b020 = ctx["books"].get("511020") or {}
    b030 = ctx["books"].get("511030") or {}
    b520 = ctx["books"].get("511520") or {}
    A(f"511020 的行情简称是「{b020.get('name') or '未知'}」。"
      f"511030 是「{b030.get('name') or '未知'}」。"
      f"511520 是「{b520.get('name') or '未知'}」。"
      "提问里把 511020/511030 写成政金债/公司债。政金债对得上的是 511520。511020 的简称不是政金债。")
    A("")
    b990 = ctx["books"].get("511990") or {}
    if not b990.get("usable"):
        A(f"511990（{b990.get('name') or '华宝添益'}）不进入回测。{b990.get('reason') or '收益不可用'}。"
          f"首末收盘 {b990.get('close0')} → {b990.get('close1')}。"
          "不把接近 0 的价格涨跌写成货币基金收益，也不用一个假设收益率去补。")
    A("含 511260、且窗口起点早于 2017-08-24 的臂（A100、M70g、以及全历史邻域里的 511260 腿）在上市前用的是 511010。起点不早于 2017-08-24 的缩短窗口里，这条腿是 10Y 本身。")
    A("")
    A("与仓库已有文件的日收益绝对差（不替换回测序列）：")
    A("")
    for k, d in ctx["crosses"].items():
        mx = d.get("max_abs")
        extra = ""
        if mx is not None and mx > PRE["source_warn"]:
            extra = "，超过 1e-3"
        A(f"- {k}：重叠 {d.get('n')} 日，最大绝对差 "
          f"{'无' if mx is None else f'{mx:.3e}'}{extra}。")
    A("")

    A("### 2.2 主点")
    A("")
    A("全窗口可评才进入门槛。缩短窗口的 proxy 与同一天重新起步的 A100 比，不能叫做优于 511260。real 只要上市早于 2025-04-23，就是标准 real 窗。")
    A("")
    A("| 主点 | 权重 | proxy 起点 | 全? | proxy CAGR/MDD/Cal/换手 | Δv30 | ΔA100 | real Δv30 | real ΔA100 | OOS Cal | 标签 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in ctx["gates"]:
        d = e["desc"]
        pr = d.get("proxy") or {}
        rr = d.get("real") or {}
        oo = d.get("oos") or {}
        if pr.get("ok"):
            pcell = cell(pr["m"])
            pstart = pr["start"] + ("" if pr["full"] else " 缩短")
            dv = dlt(pr["dcal_v30"])
            da = dlt(pr["dcal_a100"])
        else:
            pcell, pstart, dv, da = "不可评", "—", "—", "—"
        if rr.get("ok"):
            rv, ra = dlt(rr["dcal_v30"]), dlt(rr["dcal_a100"])
        else:
            rv = ra = "—"
        if oo.get("ok"):
            oc = f"{oo['m']['cal']:.4f}" + ("" if oo["full"] else " 缩短")
        else:
            oc = "不可评"
        tag = "优于511260" if e["label"] else "否"
        A(f"| {e['id']} | {wtxt(e['weights'])} | {pstart} | {'是' if e['full'] else '否'} | {pcell} | "
          f"{dv} | {da} | {rv} | {ra} | {oc} | {tag} |")
    A("")

    A("### 2.3 门槛拆开")
    A("")
    A("只列全窗口可评的主点。其余主点的平台在缩短窗口上只作诊断，写在后面，不产生标签。")
    A("")
    any_full = False
    for e in ctx["gates"]:
        if not e["full"]:
            continue
        any_full = True
        c = e["checks"]
        A(f"**{e['id']}**（{wtxt(e['weights'])}）")
        A(f"- 同向：proxy ΔCal {dlt(e['desc']['proxy']['dcal_v30'])}，real ΔCal {dlt(e['desc']['real']['dcal_v30'])}。"
          f"{'通过' if c['same'] else '未通过'}。")
        A(f"- 至少一边 ≥ +0.15：{'达到' if c['one'] else '未达到'}。")
        A(f"- proxy MDD {pct(e['desc']['proxy']['m']['mdd'])}：{'严格好于 -10%' if c['mdd'] else '未过'}。")
        A(f"- 平台：{'通过' if c['platform'] else '未通过'}。"
          + ("；".join(e["platform"]["reasons"]) if e["platform"]["reasons"] else "proxy Calmar 依次是 "
             + "、".join(f"{x:.3f}" for x in e["platform"]["cals"]) + "。"))
        A(f"- OOS Cal {e['desc']['oos']['m']['cal']:.4f} 对比 1.26：{'不低于' if c['oos'] else '低于'}。")
        A(f"- 相对 A100：proxy ΔCal {dlt(e['desc']['proxy']['dcal_a100'])}，"
          f"real ΔCal {dlt(e['desc']['real']['dcal_a100'])}。{'两边都更高' if c['beat'] else '没有两边都更高'}。")
        A(f"- 标签：{'优于 511260' if e['label'] else '不叫优于 511260'}。")
        A("")
    if not any_full:
        A("没有主点三个标准窗口都可评。标签全部落空。")
        A("")

    A("缩短窗口的平台诊断（不过门槛）：")
    A("")
    for e in ctx["gates"]:
        diag = e.get("platform_diag")
        if not diag:
            why = e["coverage_why"] or "轴上有臂跑不起来"
            if not e["full"]:
                A(f"- {e['id']}：没有完整的缩短诊断。{why}")
            continue
        A(f"- {e['id']}：proxy 起点 {diag['proxy_start']}，real 起点 {diag['real_start']}。"
          f"{'诊断通过' if diag['ok'] else '诊断未通过'}。"
          + ("；".join(diag["reasons"]) if diag["reasons"] else
             "proxy Calmar 依次是 " + "、".join(f"{x:.3f}" for x in diag["cals"]) + "。"))
    A("")

    A("### 2.4 标准 real 窗")
    A("")
    A(f"real 从 {REAL_START} 到 {ctx['meta']['real_end']}，约 {ctx['meta']['real_n']} 个交易日。上市早于这一天的工具，这一窗和 A100 是同一段。它只有一年半，单独赢这一窗不算优于 511260。")
    A("")
    A("| 主点 | 天数 | CAGR/MDD/Cal | ΔCal v30 | ΔCal A100 | ΔMDD v30 |")
    A("|---|---|---|---|---|---|")
    A(f"| v30 | {v30['real']['n']} | {cell(v30['real'])} | — | — | — |")
    A(f"| A100 | {a100['real']['n']} | {cell(a100['real'])} | {dlt(a100['real']['cal']-v30['real']['cal'])} | — | "
      f"{(a100['real']['mdd']-v30['real']['mdd'])*100:+.2f}pp |")
    for e in ctx["gates"]:
        rr = e["desc"].get("real") or {}
        if not rr.get("ok"):
            A(f"| {e['id']} | — | 不可评 | — | — | — |")
            continue
        mark = "" if rr["full"] else " 缩短"
        A(f"| {e['id']}{mark} | {rr['n']} | {pct(rr['m']['cagr'])} / {pct(rr['m']['mdd'])} / {calf(rr['m']['cal'])} | "
          f"{dlt(rr['dcal_v30'])} | {dlt(rr['dcal_a100'])} | {rr['dmdd_v30']*100:+.2f}pp |")
    A("")

    A("### 2.5 债熊")
    A("")
    det = ctx["detector"]
    if det.get("fired"):
        A(f"长债抛售规则触发。序列是 {det['source']}，2024-01-01 至 2025-12-31 内 MDD {pct(det['mdd'])}，"
          f"峰值 {det['peak']}，谷值 {det['trough']}。下面用这一段。")
    else:
        A(f"长债抛售规则没有触发（深于 -3% 才算）。{det.get('reason') or ''} "
          f"来源 {det.get('source') or '无'}，MDD {pct(det.get('mdd')) if det.get('mdd') is not None else '无'}。不另造窗口。")
    A("组合表是策略净值在该窗里的总收益 / 窗内 MDD / 天数。天数明显少于 v30 的，这段组合比较不完整，必写句改看现金层。")
    A("")
    # 组合：v30、A100、全窗口主点，以及天数完整的缩短主点
    port_cols = ["v30", "A100"]
    for e in ctx["gates"]:
        if e["id"] in ctx["port_stress"].get(ctx["stress_specs"][0][0], {}):
            port_cols.append(e["id"])
    # 去重并保持顺序
    seen = []
    for c in port_cols:
        if c not in seen:
            seen.append(c)
    head = "| 窗 | " + " | ".join(seen) + " |"
    A(head)
    A("|" + "---|" * (len(seen) + 1))
    for name, a, b, must in ctx["stress_specs"]:
        row = ctx["port_stress"][name]
        cells = []
        for c in seen:
            st = row.get(c)
            if not st:
                cells.append("无路径")
            else:
                mark = "" if st.get("complete") else " 不完整"
                cells.append(stress_cell(st) + mark)
        A(f"| {name} | " + " | ".join(cells) + " |")
    A("")
    A("比 v30 深过 1.00 个百分点的完整组合窗（必写，不否决）：")
    A("")
    any_deep = False
    for name, a, b, must in ctx["stress_specs"]:
        if not must:
            continue
        row = ctx["port_stress"][name]
        base = row.get("v30")
        for c in seen:
            if c == "v30":
                continue
            st = row.get(c)
            if not st or not st.get("complete"):
                continue
            delta = deeper(st, base)
            if delta is not None and delta < -PRE["stress_mdd_worse"]:
                any_deep = True
                A(f"- {c} 在 {name}：组合 MDD 差 {fmt_deeper(delta)}。")
    if not any_deep:
        A("- 没有。")
    A("")

    A("现金层自身在这些窗里的总收益 / MDD / 天数。这是工具的日收益，不是组合。上市前是「无数据」，不是 0。")
    A("")
    cash_cols = ["v30混合", "511010", "511260拼接"]
    for code in PURE:
        if ctx["books"].get(code, {}).get("usable"):
            cash_cols.append(code)
    for e in ctx["gates"]:
        key = "mix:" + e["primary"]
        if key in ctx["cash_stats"] and ctx["cash_stats"][key].get("n"):
            cash_cols.append(key)
    # 太宽就拆成两张：单券，混合。
    single_cols = [c for c in cash_cols if not str(c).startswith("mix:")]
    mix_cols = ["v30混合", "511260拼接"] + [c for c in cash_cols if str(c).startswith("mix:")]

    def cash_table(cols):
        A("| 窗 | " + " | ".join(cols) + " |")
        A("|" + "---|" * (len(cols) + 1))
        for name, a, b, must in ctx["stress_specs"]:
            row = ctx["cash_stress"][name]
            cells = [stress_cell(row.get(c)) for c in cols]
            A(f"| {name} | " + " | ".join(cells) + " |")
        A("")

    A("单券：")
    A("")
    cash_table(single_cols)
    A("混合（mix 前缀是该主点的现金日收益）：")
    A("")
    cash_table(mix_cols)

    A("### 2.6 现金层自身回撤")
    A("")
    A("全样本从该序列在 proxy 日历上的第一天算到面板末日。拼接 511260 的峰如果落在 2017-08-24 之前，那段是 5Y。")
    A("")
    A("| 序列 | 样本 | CAGR | MDD | 峰值 | 谷值 | 2017-08-24 起 MDD |")
    A("|---|---|---|---|---|---|---|")
    show = ["v30混合", "511010", "511260拼接"] + [c for c in PURE if ctx["books"].get(c, {}).get("usable")]
    show += [c for c in cash_cols if str(c).startswith("mix:")]
    seen_c = []
    for k in show:
        if k in seen_c or k not in ctx["cash_stats"]:
            continue
        seen_c.append(k)
        st = ctx["cash_stats"][k]
        tail = ctx["cash_from_splice"].get(k) or {}
        if not st or st.get("cagr") is None:
            A(f"| {k} | 无 | | | | | |")
            continue
        A(f"| {k} | {st.get('start')}~{st.get('end')} / {st.get('n')}日 | {pct(st.get('cagr'))} | "
          f"{pct(st.get('mdd'))} | {st.get('peak')} | {st.get('trough')} | {pct(tail.get('mdd'))} |")
    A("")
    spliced = ctx["cash_stats"].get("511260拼接") or {}
    tail = ctx["cash_from_splice"].get("511260拼接") or {}
    if spliced.get("mdd") is not None:
        A(f"拼接 511260 全样本 MDD {pct(spliced['mdd'])}，峰值 {spliced.get('peak')}，谷值 {spliced.get('trough')}。"
          f"从 2017-08-24 起 MDD {pct(tail.get('mdd'))}，峰值 {tail.get('peak')}，谷值 {tail.get('trough')}。")
        A("")

    A("### 2.7 流动性")
    A("")
    A("日均成交额是东财成交额（元）里大于 0 的交易日的算术平均，不是成交量，也不是策略换手。最近 252 日单独一列。低于 1 亿元会在结论里点名，不否决。")
    A("")
    A("| 代码 | 简称 | 全样本日均 | 全样本中位 | 近252日日均 | 近252日中位 | 零额占比 |")
    A("|---|---|---|---|---|---|---|")
    thin = []
    for code in ["511260", "511010"] + PURE:
        b = ctx["books"].get(code) or {}
        liq = b.get("liq") or {}
        if not liq:
            A(f"| {code} | {b.get('name') or ''} | 无 | | | | |")
            continue
        A(f"| {code} | {b.get('name') or ''} | {yi(liq.get('mean'))} | {yi(liq.get('median'))} | "
          f"{yi(liq.get('mean_252'))} | {yi(liq.get('median_252'))} | {pct(liq.get('zero_frac'))} |")
        m252 = liq.get("mean_252")
        if m252 is not None and m252 < PRE["thin_yuan"]:
            thin.append(f"{code}（{yi(m252)}）")
        if liq.get("unreliable"):
            A(f"|  |  | {code} 成交额字段不可靠 | | | | |")
    A("")
    if thin:
        A("近 252 日日均低于 1 亿元：" + "、".join(thin) + "。")
    else:
        A("近 252 日日均没有低于 1 亿元的代码。")
    A("")

    A("## 3. 结论")
    A("")
    for para in conclusion(ctx):
        A(para)
        A("")
    return "\n".join(L) + "\n"


def conclusion(ctx):
    out = []
    books = ctx["books"]
    gates = ctx["gates"]
    labeled = [e for e in gates if e["label"]]
    full_ids = [e["id"] for e in gates if e["full"]]
    out.append(
        f"看过 13 个主点。三个标准窗口都盖住的有 {len(full_ids)} 个"
        + ( "：" + "、".join(full_ids) if full_ids else "。")
        + ("" if full_ids else "")
        + "这是固定名单上的比较，不是事先指定的单臂。"
    )
    if not labeled:
        out.append("没有主点拿到「优于 511260」的标签。全窗口门槛和「两边都高于 A100」同时成立的，一条都没有。")
    else:
        bits = []
        for e in labeled:
            bits.append(
                f"{e['id']} proxy ΔCal 对 A100 {dlt(e['desc']['proxy']['dcal_a100'])}、"
                f"对 v30 {dlt(e['desc']['proxy']['dcal_v30'])}，"
                f"real ΔCal 对 A100 {dlt(e['desc']['real']['dcal_a100'])}"
            )
        out.append("拿到标签的是：" + "；".join(bits) + "。")

    # 全窗口未过的原因，每个一句
    for e in gates:
        if e["label"]:
            continue
        if not e["full"]:
            pr = e["desc"].get("proxy") or {}
            rr = e["desc"].get("real") or {}
            why = e["coverage_why"] or "历史盖不住 2014-06-23 起的 proxy"
            extra = ""
            if pr.get("ok"):
                extra += (f"缩短 proxy 从 {pr['start']} 起 {pr['n']} 日，"
                          f"Cal {calf(pr['m']['cal'])}，相对同一起点的 A100 {dlt(pr['dcal_a100'])}。")
            if rr.get("ok"):
                extra += (f"标准 real Cal {calf(rr['m']['cal'])}，相对 A100 {dlt(rr['dcal_a100'])}。")
            out.append(f"{e['id']} 不能称为优于 511260。{why}{extra}")
        else:
            c = e["checks"]
            failed = []
            if not c["same"] or not c["one"]:
                failed.append("相对 v30 的 ΔCal 不够")
            if not c["mdd"]:
                failed.append("proxy MDD 不优于 -10%")
            if not c["platform"]:
                failed.append("平台未过")
            if not c["oos"]:
                failed.append("OOS 低于 1.26")
            if not c["beat"]:
                failed.append("没有在 proxy 和 real 上都高于 A100")
            out.append(f"{e['id']} 全窗口可评，但是{'、'.join(failed)}。")

    # 债熊必写
    deep_lines = []
    for name, a, b, must in ctx["stress_specs"]:
        if not must:
            continue
        row = ctx["port_stress"][name]
        base = row.get("v30")
        for key, st in row.items():
            if key in ("v30", "must", "v30_n") or not isinstance(st, dict):
                continue
            if not st.get("complete"):
                continue
            delta = deeper(st, base)
            if delta is not None and delta < -PRE["stress_mdd_worse"]:
                deep_lines.append(f"{key} 在 {name} 的组合 MDD 比 v30 深 {-delta*100:.2f}pp")
    if deep_lines:
        out.append("债熊更深的完整窗口：" + "；".join(deep_lines) + "。这些只陈列，不否决，但是不能把提议说成无条件。")
    else:
        out.append("预注册点名的债熊窗里，完整组合路径没有一段比 v30 深过 1.00 个百分点。")

    # 现金层在抛售窗或 2024/2025 的自身回撤，点出最深的长债
    det = ctx["detector"]
    if det.get("fired"):
        name = f"2024-2025长债抛售({det['peak']}~{det['trough']})"
        row = ctx["cash_stress"].get(name) or {}
        bits = []
        for code in ("511090", "511130", "511520", "511260拼接", "518880"):
            st = row.get(code)
            if st and st.get("mdd") is not None:
                bits.append(f"{code} {pct(st['total'])} / MDD {pct(st['mdd'])}")
        if bits:
            out.append(f"抛售窗 {det['peak']} 至 {det['trough']}（由 {det['source']} 的自身回撤定出）现金层："
                       + "；".join(bits) + "。")

    # 自身 MDD 深的
    deep_own = []
    for e in gates:
        mdd = e.get("own_mdd")
        if mdd is not None and mdd < PRE["own_mdd_proposal"]:
            deep_own.append(f"{e['id']} {pct(mdd)}")
    if deep_own:
        out.append("现金层自身 MDD 深于 -8% 的主点：" + "、".join(deep_own) + "。这些即使数字过关也不提议替换现金层。")

    b020 = books.get("511020") or {}
    if b020.get("name") and "政金" not in str(b020.get("name")):
        out.append(f"511020 的简称是「{b020['name']}」，不是政金债。政金债是 511520"
                   f"（{(books.get('511520') or {}).get('name') or ''}）。")

    b990 = books.get("511990") or {}
    if not b990.get("usable"):
        out.append("511990 的场内前复权价几乎不动，分红不在价格里。没有可靠的总收益序列，这一臂不作比较。")

    if ctx["source_warns"]:
        out.append("来源差异超过 1e-3：" + "；".join(ctx["source_warns"]) + "。回测没有改用另一条序列。")

    thin = []
    for code in ["511260", "511010"] + PURE:
        liq = (books.get(code) or {}).get("liq") or {}
        m252 = liq.get("mean_252")
        if m252 is not None and m252 < PRE["thin_yuan"]:
            thin.append(f"{code} 近252日日均 {yi(m252)}")
    if thin:
        out.append("流动性薄：" + "；".join(thin) + "。")

    if ctx["proposal"]:
        e = next(x for x in gates if x["id"] == ctx["proposal"])
        out.append(
            f"提议（不是默认，没有配置文件）：现金层改为 {wtxt(e['weights'])}。"
            f"它在预注册的数字门槛上优于 511260，自身 MDD {pct(e.get('own_mdd'))}，"
            "并且不是事先点名的黄金或可转债。默认仍是 v30，A100 仍只是 v37 的未激活提议。"
        )
    else:
        if labeled:
            out.append(
                "有主点在数字上优于 511260，但是它们要么含有黄金或可转债，要么现金层自身 MDD 深于 -8%。"
                "按预注册，不把它们提议成现金层。提议维持 A100 作为相对 v30 的现金层候选，默认配置仍是 v30。"
            )
        else:
            out.append("提议：不替换 511260。不新写配置。默认仍是 v30。A100 仍是 v37 里那份未激活的提议，本轮没有更好的现金工具能接过它。")

    out.append("多出来或少掉的 Calmar 主要是久期、信用或商品 beta，不是新的择时。回撤控制只看股票端，现金层的利率回撤和黄金回撤都不会触发减仓。样本里没有 2013 年钱荒。2017-08-24 之前的 511260 就是 511010。")
    return out


if __name__ == "__main__":
    main()
