# -*- coding: utf-8 -*-
"""v36 预注册（纸面，对照 v30）。规则在 PRE_REG_MD，先于任何候选回测。

用法:
  python3 scripts/exp_v36_prereg.py --self-check
  python3 scripts/exp_v36_prereg.py --prereg-only
  python3 scripts/exp_v36_prereg.py --identity
  python3 scripts/exp_v36_prereg.py

默认路径会拒绝运行，除非 HEAD 上的 references/exp_v36_prereg_report.md
已经包含本节 PRE_REG_MD 与同一内容指纹。不改 final_cfg_v30.json。
"""
import sys, os, json, copy, hashlib, subprocess
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate, SLOTS
from strategy import DynamicStrategy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
OUT_DIR = os.path.join(ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)
CFG_FILE = os.path.join(REF, "final_cfg_v30.json")
REPORT = os.path.join(REF, "exp_v36_prereg_report.md")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
# v35 同一数据锚。对不上就中止，不改锚、不解释候选。
PIN_PROXY = {"cagr": 0.11111021816342403, "mdd": -0.09910849911812059,
             "cal": 1.1210967692185454}

PRE = {
    "dcal": 0.15,
    "cal_eps": 1e-4,          # |ΔCal|<=1e-4 视为没变
    "proxy_mdd": -0.10,       # 必须严格好于 -10%
    "cliff": 0.50,
    "spike": 0.15,            # 主点 proxy Cal 比任一邻点高出超过 0.15 = 尖峰，平台失败
    "vol_win": 60,
    "vol_ann": 252,
    "reentry_sma": 60,
    "reentry_lookback": 20,
    "prem_span": 0.06,
    "min_delta": 0.035,       # 冻结，本轮不扫
    "trade_eps": 1e-8,
}

JOBS = [
    {"name": "A_k1", "arm": "A", "role": "neighbor", "overlay": {"friday_max_legs": 1}},
    {"name": "A_k2", "arm": "A", "role": "primary", "overlay": {"friday_max_legs": 2}},
    {"name": "A_k3", "arm": "A", "role": "neighbor", "overlay": {"friday_max_legs": 3}},
    {"name": "B_v08", "arm": "B", "role": "neighbor", "overlay": {"eq_vol_target": 0.08}},
    {"name": "B_v10", "arm": "B", "role": "primary", "overlay": {"eq_vol_target": 0.10}},
    {"name": "B_v12", "arm": "B", "role": "neighbor", "overlay": {"eq_vol_target": 0.12}},
    {"name": "C_g04", "arm": "C", "role": "neighbor", "overlay": {"derisk_gap": 0.04}},
    {"name": "C_g05", "arm": "C", "role": "primary", "overlay": {"derisk_gap": 0.05}},
    {"name": "C_g06", "arm": "C", "role": "neighbor", "overlay": {"derisk_gap": 0.06}},
    {"name": "D_a25", "arm": "D", "role": "neighbor", "overlay": {"us_rp_alpha": 0.25}},
    {"name": "D_a50", "arm": "D", "role": "primary", "overlay": {"us_rp_alpha": 0.50}},
    {"name": "D_a75", "arm": "D", "role": "neighbor", "overlay": {"us_rp_alpha": 0.75}},
    {"name": "E_s01", "arm": "E", "role": "neighbor", "overlay": {"reentry_slope": 0.01}},
    {"name": "E_s02", "arm": "E", "role": "primary", "overlay": {"reentry_slope": 0.02}},
    {"name": "E_s03", "arm": "E", "role": "neighbor", "overlay": {"reentry_slope": 0.03}},
    {"name": "F_t02", "arm": "F", "role": "neighbor", "overlay": {"prem_scale_thr": 0.02}},
    {"name": "F_t03", "arm": "F", "role": "primary", "overlay": {"prem_scale_thr": 0.03}},
    {"name": "F_t04", "arm": "F", "role": "neighbor", "overlay": {"prem_scale_thr": 0.04}},
]
PRIMARY = {j["arm"]: j["name"] for j in JOBS if j["role"] == "primary"}
ARM_OF = {}
for j in JOBS:
    ARM_OF.setdefault(j["arm"], []).append(j["name"])

PRE_REG_MD = """\
## 1. 预注册（本节在跑数之前写入，不随结果改写）

对照是 `references/final_cfg_v30.json`。费用单边万 5，`signal_lag=1`，溢价 T-2，先计提后成交，`strict=True`。窗口与 `scripts/exp_opt31.py`、v35 相同：

- proxy 全历史从 2014-06-23 起
- real 从 2025-04-23 起
- OOS 是 proxy 上的独立起点 2022-01-04。新开一个策略实例，深熊锁从「未锁」起步，不把 2014 年以来的锁状态带进来。信号本身仍用起点之前已经发生的价格（滞后 1 日），这不是未来函数
- 全历史 proxy 回撤必须 **严格好于 -10%**（等于 -10% 也不过）

公布基线（v35 同一锚，本脚本对不上就中止）：proxy 11.11% / -9.91% / Cal 1.12；real 23.70% / -3.75% / Cal 6.32；OOS Cal 1.26。OOS 门槛用本次配对跑出来的 v30 OOS，不把 1.26 这个四舍五入数再当第二套门槛。

每个臂只有一个被扫描的参数，主点是有序三点的中间那个。邻点更高也不升为主点。六臂不叠加。

### 1.1 三关（晋升线，六臂共用）

用户给定的三关，全部要过，才允许写 `references/final_cfg_v36.json`。那个文件是提议，不是默认。`final_cfg_v30.json`、README 当前持仓段、SKILL 都不改。

1. **双窗同向**。记 Δ 为候选 Calmar 减 v30 Calmar。|Δ| ≤ 1e-4 视为没变（不算变好，也不算变差）。两条窗口都不得变差；至少一条 Δ ≥ +0.15；proxy MDD > -10%。real MDD 只陈列，**不是**否决线（本轮不沿用 v35 的 real MDD > -4.1%，也不沿用共振熊 0.50pp 附加条款）。
2. **平台**。三点按参数从小到大排列。相邻两点的 proxy Calmar 绝对差 > 0.50 记为悬崖。主点 proxy Calmar 比任一邻点高出超过 0.15 记为尖峰。两个邻点的 proxy MDD 都要 > -10%。有悬崖、有尖峰、或邻点 MDD 不守住，平台失败。邻点不需要自己达到 +0.15。
3. **OOS**。主点 OOS Calmar ≥ 本次 v30 的 OOS Calmar − 1e-4。

弱标签不能晋升：一关通过、同向不足、单窗、双窗未改善、MDD破线。主点没过一关也照样算平台和 OOS，数字写进表，不因此改规则。

多于一个主点走完三关时，只提议 `min(ΔCal_proxy, ΔCal_real)` 更大的那一个。

换手是引擎的单边权重 L1 之和，年化 = 累计 × 252 / 交易日数。成交日 = `turnover_series > 1e-8` 的天数，trades/yr = 成交日 × 252 / 交易日数。同一天里两只标的成交只算 1 个成交日。引擎多返回这条序列，撮合公式不动。

### 1.2 本轮明确不重做的事

v35 已经否掉的，阈值不再加密：511260 现金层、CN 深熊锁内红利阶梯、触发日债券袖套、溢价差换载体、解锁后再等 10 个交易日、小数位舍入、`min_delta=0.010`、100 股一手。PR#3 幽灵交易员、PR#4 六 ETF 20 日动量轮动也不重开。v31 的 `growth_iv`（三只成长一起按倒数波动替换并 clamp）、`breadth_gate`（宽度差就砍弹性）、已删除的 `vol_target` 键（15%–22% 上从不触发的空转）都不是本轮的扫描对象。`min_delta` 保持 v30 的 0.035。

### 1.3 Arm A — 周五最多两腿

**假设**：v30 在周五只要有一只标的的 |目标−持仓| ≥ 0.035，就把五只一起拨到目标，没过门槛的也跟着动。本臂改成按标的独立门槛，并且一周常规单最多动 K 只。

**规则**（只作用在常规调仓日，含回测首日；日频紧急覆盖仍走 v30，可在同一天盖过本臂）：

- 候选 = |Δ| ≥ 0.035 的标的，按 |Δ| 从大到小，同额时按 `159232, 515100, 159941, 513500, 159952` 的固定顺序
- 只把前 K 只设成 v30 目标，其余标的的目标改回当时持仓
- 未入选标的锁定在当时持仓。若入选标的的目标加上锁定仓位会超过 100%，只缩小入选标的、保持它们之间的比例，给锁定仓位腾地方
- 缩小之后如果最大 |Δ| 又掉到 0.035 以下，这一天常规单不成交
- 现金是残差。不改 `min_delta`，不改周五以外的紧急单
- 单名缺口若始终小于 3.5%（风险偏好高时两只红利底仓往往各只有 1%–3%），这只标的可以整段都不被常规单碰到。这是规则本身，不是实现疏漏

**参数**：K = 1 / **2** / 3。主点是 2。

### 1.4 Arm B — 60 日实现波动率目标

**假设**：已删除的 `vol_target` 把目标放在 15%–22%，审计结论是组合波动从未摸到，所以是空转。本臂把目标放到会真正约束仓位的水平，并且只许减仓、不许加杠杆。

**规则**：

- σ = 决策时点之前已经入账的组合日收益，最近 60 个交易日，样本标准差（ddof=1）× √252
- 不足 60 日、σ 非有限或 σ ≤ 0：倍数 = 1
- 倍数 = min(1, 目标波动 / σ)。五只权益权重一起乘这个倍数，现金吃下差额。底仓也乘，不单砍弹性
- 常规单和日频紧急单都乘。倍数永远 ≤ 1，不会加到 v30 权重之上

**参数**：目标波动 = 8% / **10%** / 12%。窗口 60 日冻结。

### 1.5 Arm C — 周中只减不加

**假设**：v30 的日频覆盖在「目标权益比当前低 4 个百分点」时，周中就把整本换成原始分数下的目标，里面可以带买入。本臂把周中改成：只有风控层已经打开，才允许减仓；加仓一律留到周五。

**周中（周一到周四）规则**：

- 先照 v30 把日频函数跑完，锁和计数器的状态与 v30 同一天对齐。然后丢掉它返回的整本目标
- 参考目标用与 v30 日频覆盖相同的原始分数（不是周三确认后的分数），避免在非周五推进 `_eff_score`
- 样本前 40 个组合收益日不交易，与 v30 日频函数的早退一致
- 层（任一为真才允许周中减仓）：CN 深熊锁、US 深熊锁、CN 快刹车、US 快刹车、CN 结构闸门关闭（收盘不高于 gate_win 均线）、US 结构闸门关闭、两只 QDII 的 T-2 溢价取高者 ≥ 5%、相关性风控乘数 < 1
- 触发后，每只标的的新目标 = min(当时持仓, 参考目标)。只减不加
- 单名最大减仓 < gap 则这天不交易。gap 都 ≥ 0.04，高于冻结的 0.035，所以过了 gap 的单不会被引擎门槛吃掉

**周五**：日频函数原样返回 v30 的结果（含那条 4 个百分点的整本替换）。加仓仍只在周五的常规单里发生。

**参数**：gap = 0.04 / **0.05** / 0.06。层的定义不扫描。

### 1.6 Arm D — 美股弹性内部的风险平价

**假设**：风险平价只动两只美股弹性之间的相对权重。创业板份额（前提②）和四只底仓（前提①）保持 v30 算出的数字。这不是 v31b 的 `growth_iv`（那条把创业板、纳指、标普三只一起换掉，还套了 clamp）。

**规则**（在 v30 的 `_base_target` 全部做完之后，百分数权重上）：

- 弹性 = max(0, 权重 − 底仓)，只取 159941 与 513500。弹性合计为 0 则不动
- 各自 σ = 滞后信号日及以前 60 个交易日收益的样本标准差
- 倒数波动份额 ∝ 1/σ。混合份额 = (1−α) × v30 弹性份额 + α × 倒数波动份额
- 混合后写回，弹性合计与底仓都不变。创业板、两只红利不动。窗口 60 日冻结，不再套 growth_clamp

**参数**：α = 0.25 / **0.50** / 0.75。

### 1.7 Arm E — 解锁后按均线斜率再进创业板

**假设**：v35 的 E 是解锁后再等固定交易日，B 是锁内把红利阶梯还回去。两条都已否掉。本臂不在锁内加仓，也不数天数。CN 锁按原条件打开之后，创业板弹性暂时乘一个 0–1 的斜率系数；红利弹性在解锁当天就回到 v30（前提④）。US 锁不改。

**规则**：

- 只包 CN 的 `_market_mult`。锁上时系数状态清掉，弹性仍是 0
- 锁从开到关的那一次调用，开始再进状态
- 斜率 = SMA60(沪深300) 相对 20 个交易日前的变化率，用滞后后的信号下标。样本不够则不缩放
- 系数 = clip(斜率 / θ, 0, 1)。创业板弹性乘数再乘这个系数。防御乘数 `m_all` 不乘
- 系数达到 1 就结束这次再进。之后即使斜率重新变小，也不再罚，除非再次锁上再解锁。结构闸门、快刹车仍按 v30 自己的乘数生效

**参数**：θ = 1% / **2%** / 3%。SMA 窗口 60、回看 20 日冻结。

### 1.8 Arm F — 溢价线性缩仓，减下来的进现金

**假设**：v35 的 D 是溢价差超过阈值时，在纳指和标普之间搬仓。本臂不搬。每只 QDII 按自己的 T-2 溢价，把底仓以上的弹性线性打掉，差额留在现金，不加给另一只 QDII，也不加给创业板。v30 的阶梯门控（5/8/12%）和相对溢价倾斜先做完，本臂在其后再乘一层，不替换那两层。

**规则**：

- 跨度固定 6 个百分点
- 溢价 ≤ 阈值：系数 1
- 溢价 ≥ 阈值 + 6pp：系数 0
- 中间线性。缺溢价的那只系数 = 1
- 底仓不乘（前提①）

**参数**：阈值 = 2% / **3%** / 4%。3% 落在阶梯门控的第一档 5% 之下，测的是阶梯还没动手的那一段，不是把 5/8/12 再扫一遍。

### 1.9 诊断（不参与晋升）

每臂主点额外记下：与 v30 的最大日收益差、以及该臂自己的计数（周五入选腿数、波动倍数是否小于 1、周中减仓天数、再进天数与平均系数、溢价系数均值）。计数只用来判断「是不是空转」，不能拿来改主点。
"""


def digest_of():
    blob = json.dumps({"PRE": PRE, "JOBS": JOBS, "md": PRE_REG_MD},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def apply_legs(tgt, w, max_legs, min_delta):
    """tgt/w: slot -> weight. 返回 (new_tgt_with_cash, n_legs)。"""
    order = {s: i for i, s in enumerate(SLOTS)}
    cands = []
    for s in SLOTS:
        gap = abs(float(tgt[s]) - float(w[s]))
        if gap >= min_delta - 1e-15:
            cands.append((gap, order[s], s))
    cands.sort(key=lambda t: (-t[0], t[1]))
    chosen = [t[2] for t in cands[:int(max_legs)]]
    out = {s: (float(tgt[s]) if s in chosen else float(w[s])) for s in SLOTS}
    if not chosen:
        out["cash"] = 1.0 - sum(out[s] for s in SLOTS)
        return out, 0
    held = sum(out[s] for s in SLOTS if s not in chosen)
    chosen_sum = sum(out[s] for s in chosen)
    room = max(0.0, 1.0 - held)
    if chosen_sum > room + 1e-12 and chosen_sum > 0:
        k = room / chosen_sum
        for s in chosen:
            out[s] *= k
    if max(abs(out[s] - float(w[s])) for s in SLOTS) < min_delta - 1e-15:
        out = {s: float(w[s]) for s in SLOTS}
        chosen = []
    out["cash"] = 1.0 - sum(out[s] for s in SLOTS)
    return out, len(chosen)


def vol_scale(sigma, target):
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return 1.0
    return float(min(1.0, target / sigma))


def prem_factor(prem, thr, span):
    if prem is None or not np.isfinite(prem):
        return 1.0
    if prem <= thr:
        return 1.0
    if prem >= thr + span:
        return 0.0
    return float(1.0 - (prem - thr) / span)


def slope_factor(slope, theta):
    if slope is None or not np.isfinite(slope) or theta <= 0:
        return None
    return float(min(1.0, max(0.0, slope / theta)))


def self_check():
    w = {s: 0.0 for s in SLOTS}
    tgt = {"159232": 0.02, "515100": 0.02, "159941": 0.40, "513500": 0.10, "159952": 0.30}
    out, n = apply_legs(tgt, w, 2, 0.035)
    assert n == 2, n
    assert abs(out["159941"] - 0.40) < 1e-12
    assert abs(out["159952"] - 0.30) < 1e-12
    assert out["513500"] == 0.0 and out["159232"] == 0.0
    out1, n1 = apply_legs(tgt, w, 1, 0.035)
    assert n1 == 1 and abs(out1["159941"] - 0.40) < 1e-12 and out1["159952"] == 0.0
    # 只有一只过门槛：K=2 也只动这一只
    tgt2 = dict(tgt)
    tgt2["159952"] = 0.02
    out2, n2 = apply_legs(tgt2, w, 2, 0.035)
    assert n2 == 2  # 159941=0.40, 513500=0.10, both >= 0.035; 159952 now 0.02
    assert abs(out2["159941"] - 0.40) < 1e-12 and abs(out2["513500"] - 0.10) < 1e-12
    assert out2["159952"] == 0.0
    # 缩小后掉到门槛下 -> 不成交
    w3 = {s: 0.0 for s in SLOTS}
    w3["159232"] = 0.50
    w3["515100"] = 0.48
    tgt3 = {s: w3[s] for s in SLOTS}
    tgt3["159941"] = 0.40
    out3, n3 = apply_legs(tgt3, w3, 1, 0.035)
    assert n3 == 0, (n3, out3)
    assert abs(vol_scale(0.05, 0.10) - 1.0) < 1e-12
    assert abs(vol_scale(0.20, 0.10) - 0.5) < 1e-12
    assert vol_scale(float("nan"), 0.10) == 1.0
    assert prem_factor(0.03, 0.03, 0.06) == 1.0
    assert abs(prem_factor(0.06, 0.03, 0.06) - 0.5) < 1e-12
    assert prem_factor(0.09, 0.03, 0.06) == 0.0
    assert prem_factor(None, 0.03, 0.06) == 1.0
    assert abs(slope_factor(0.01, 0.02) - 0.5) < 1e-12
    assert slope_factor(-0.01, 0.02) == 0.0
    assert slope_factor(0.05, 0.02) == 1.0
    assert slope_factor(None, 0.02) is None
    print("[self-check] ok", digest_of())


class OptStrategy(DynamicStrategy):
    """实验子类。开关缺省时走父类，v30 配置必须与 DynamicStrategy 的日收益一致。"""

    def __init__(self, R_full, cfg=None, a_mkt_override=None):
        super().__init__(R_full, cfg=cfg, a_mkt_override=a_mkt_override)
        cfg = cfg or {}
        self.max_legs = int(cfg.get("friday_max_legs", 0) or 0)
        self.vol_tgt = float(cfg.get("eq_vol_target", 0.0) or 0.0)
        self.derisk_gap = float(cfg.get("derisk_gap", 0.0) or 0.0)
        self.rp_alpha = float(cfg.get("us_rp_alpha", 0.0) or 0.0)
        self.reentry_theta = float(cfg.get("reentry_slope", 0.0) or 0.0)
        thr = cfg.get("prem_scale_thr", None)
        self.prem_thr = None if thr is None else float(thr)
        self.prem_span = float(cfg.get("prem_scale_span", PRE["prem_span"]))
        self.min_delta_frozen = float(cfg.get("min_delta", PRE["min_delta"]))
        self._reentry = False
        self.diag = {"leg_days": 0, "leg_sum": 0, "rebal_days": 0,
                     "vol_days": 0, "vol_bind": 0, "vol_scale_sum": 0.0,
                     "derisk_days": 0, "derisk_seen": {},
                     "reentry_days": 0, "reentry_s_sum": 0.0,
                     "prem_days": 0, "prem_f_sum": 0.0,
                     "rp_days": 0, "rp_nq_sum": 0.0}

    def _once(self, key, dt):
        last = self.diag.get(key)
        if last == dt:
            return False
        self.diag[key] = dt
        return True

    def _sigma60(self, ctx):
        pf = ctx.get("pf_rets", pd.Series(dtype=float))
        if pf is None or len(pf) < PRE["vol_win"]:
            return None
        sig = float(pf.iloc[-PRE["vol_win"]:].std(ddof=1)) * np.sqrt(PRE["vol_ann"])
        return sig

    def _apply_vol(self, tgt, ctx, dt):
        if tgt is None or self.vol_tgt <= 0:
            return tgt
        sig = self._sigma60(ctx)
        scale = vol_scale(sig, self.vol_tgt)
        if self._once("_vol_dt", dt):
            self.diag["vol_days"] += 1
            self.diag["vol_scale_sum"] += scale
            if scale < 1.0 - 1e-12:
                self.diag["vol_bind"] += 1
        if scale >= 1.0 - 1e-15:
            return tgt
        out = {s: float(tgt[s]) * scale for s in SLOTS}
        out["cash"] = 1.0 - sum(out.values())
        return out

    def _apply_legs(self, tgt, ctx, dt):
        if tgt is None or self.max_legs <= 0:
            return tgt
        w = ctx.get("weights")
        if w is None:
            return tgt
        cur = {s: float(w[i]) for i, s in enumerate(SLOTS)}
        out, n = apply_legs(tgt, cur, self.max_legs, self.min_delta_frozen)
        self.diag["rebal_days"] += 1
        self.diag["leg_days"] += int(n > 0)
        self.diag["leg_sum"] += n
        return out

    def target_fn(self):
        base = super().target_fn()
        if self.max_legs <= 0 and self.vol_tgt <= 0:
            return base

        def fn(dt, R_prev, ctx):
            tgt = base(dt, R_prev, ctx)
            tgt = self._apply_vol(tgt, ctx, dt)
            tgt = self._apply_legs(tgt, ctx, dt)
            return tgt
        return fn

    def _layers(self, dt):
        flags = {}
        for mkt in ("CN", "US"):
            dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, mkt, self.gate_win)
            fb_thr = self.market_dd[mkt][0]
            flags[mkt + "_lock"] = bool(self._lock[mkt])
            flags[mkt + "_fast"] = bool(dd < -fb_thr and not (ok20 and ok20_60))
            flags[mkt + "_gate"] = not bool(ok120)
        pm = self._premium_at(dt) if self._premium is not None else None
        flags["premium"] = bool(pm is not None and pm >= self.premium_thr[0])
        flags["corr"] = bool(self._corr_mult(dt) < 1.0 - 1e-12)
        return flags

    def _raw_final(self, dt):
        sc = self.sig.score(dt)
        return self._finalize(self._base_target(dt, sc))

    def _sell_only(self, dt, ctx):
        tgt = self._raw_final(dt)
        w = ctx.get("weights")
        if w is None:
            return None
        flags = self._layers(dt)
        if not any(flags.values()):
            return None
        out = {}
        max_cut = 0.0
        for i, s in enumerate(SLOTS):
            cur = float(w[i])
            want = float(tgt[s])
            if want < cur:
                out[s] = want
                max_cut = max(max_cut, cur - want)
            else:
                out[s] = cur
        if max_cut + 1e-15 < self.derisk_gap:
            return None
        out["cash"] = 1.0 - sum(out[s] for s in SLOTS)
        if self._once("_de_dt", dt):
            self.diag["derisk_days"] += 1
            for k, v in flags.items():
                if v:
                    self.diag["derisk_seen"][k] = self.diag["derisk_seen"].get(k, 0) + 1
        return out

    def daily_fn(self):
        raw = super().daily_fn()
        if self.derisk_gap <= 0 and self.vol_tgt <= 0:
            return raw

        def fn(dt, R_prev, ctx):
            parent = raw(dt, R_prev, ctx)
            if self.derisk_gap > 0 and dt.weekday() != 4:
                pf = ctx.get("pf_rets", pd.Series(dtype=float))
                tgt = None if pf is None or len(pf) < 40 else self._sell_only(dt, ctx)
            else:
                tgt = parent
            return self._apply_vol(tgt, ctx, dt)
        return fn

    def _cn_ma_slope(self, dt):
        i = self.sig._idx(dt)
        sma = self.sig.sma["a_mkt"][PRE["reentry_sma"]]
        look = PRE["reentry_lookback"]
        if i < PRE["reentry_sma"] + look or i >= len(sma):
            return None
        a = float(sma.iloc[i])
        b = float(sma.iloc[i - look])
        if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
            return None
        return a / b - 1.0

    def _market_mult(self, dt, market):
        if self.reentry_theta <= 0 or market != "CN":
            return super()._market_mult(dt, market)
        prev = self._lock[market]
        m_g, m_all = super()._market_mult(dt, market)
        if self._lock[market]:
            self._reentry = False
            return m_g, m_all
        if prev and not self._lock[market]:
            self._reentry = True
        if not self._reentry:
            return m_g, m_all
        s = slope_factor(self._cn_ma_slope(dt), self.reentry_theta)
        if s is None:
            return m_g, m_all
        if self._once("_re_dt", dt):
            self.diag["reentry_days"] += 1
            self.diag["reentry_s_sum"] += s
        if s >= 1.0 - 1e-12:
            self._reentry = False
            return m_g, m_all
        return m_g * s, m_all

    def _apply_rp(self, dt, out):
        codes = ["159941", "513500"]
        elas = {c: max(0.0, float(out[c]) - float(self.floor[c])) for c in codes}
        tot = sum(elas.values())
        if tot <= 1e-9:
            return out
        i = self.sig._idx(dt)
        if i < PRE["vol_win"]:
            return out
        inv = {}
        for c in codes:
            seg = self.R[c].iloc[i - PRE["vol_win"] + 1: i + 1].fillna(0.0)
            sd = float(seg.std(ddof=1))
            if not np.isfinite(sd) or sd <= 1e-12:
                sd = 1e-12
            inv[c] = 1.0 / sd
        sinv = sum(inv.values())
        out = dict(out)
        for c in codes:
            base_share = elas[c] / tot
            iv_share = inv[c] / sinv
            share = (1.0 - self.rp_alpha) * base_share + self.rp_alpha * iv_share
            out[c] = float(self.floor[c]) + tot * share
        if self._once("_rp_dt", dt):
            share_nq = ((1.0 - self.rp_alpha) * (elas["159941"] / tot)
                        + self.rp_alpha * (inv["159941"] / sinv))
            self.diag["rp_days"] += 1
            self.diag["rp_nq_sum"] += share_nq
        return out

    def _name_prem(self, dt, code):
        if self._premium is None or dt not in self._premium.index:
            return None
        row = self._premium.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        v = row.get(code) if hasattr(row, "get") else None
        if v is None or not np.isfinite(v):
            return None
        return float(v)

    def _apply_prem(self, dt, out):
        out = dict(out)
        fs = []
        for c in ("159941", "513500"):
            f = prem_factor(self._name_prem(dt, c), self.prem_thr, self.prem_span)
            elas = max(0.0, float(out[c]) - float(self.floor[c]))
            out[c] = float(self.floor[c]) + elas * f
            fs.append(f)
        if self._once("_prem_dt", dt) and fs:
            self.diag["prem_days"] += 1
            self.diag["prem_f_sum"] += float(np.mean(fs))
        return out

    def _base_target(self, dt, sc):
        out = super()._base_target(dt, sc)
        if self.rp_alpha > 0:
            out = self._apply_rp(dt, out)
        if self.prem_thr is not None:
            out = self._apply_prem(dt, out)
        return out


def cfg_of(base, job):
    cfg = copy.deepcopy(base)
    cfg.update(job["overlay"])
    if "prem_scale_thr" in job["overlay"]:
        cfg["prem_scale_span"] = PRE["prem_span"]
    return cfg


def pack_metrics(e, res):
    ts = res["turnover_series"]
    n = max(len(ts), 1)
    return {"cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
            "to": float(e["turnover"]), "to_ann": float(ts.sum()) * PRE["vol_ann"] / n,
            "trades_yr": float((ts > PRE["trade_eps"]).sum()) * PRE["vol_ann"] / n,
            "n": int(n)}


def execute(cfg, R, start, bond, cls=DynamicStrategy):
    ds = cls(R, cfg=cfg)
    res = run_backtest(
        R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    e = evaluate(res)
    diag = getattr(ds, "diag", {})
    return pack_metrics(e, res), diag, res["rets"]


def cell(m):
    return f"{m['cagr'] * 100:.2f}% / {m['mdd'] * 100:.2f}% / {m['cal']:.2f}"


def dlt(x):
    return f"{x:+.3f}"


def deltas(row, base):
    return {"dcal_p": row["proxy"]["cal"] - base["proxy"]["cal"],
            "dcal_r": row["real"]["cal"] - base["real"]["cal"],
            "dcal_o": row["oos"]["cal"] - base["oos"]["cal"]}


def changed(x):
    return abs(x) > PRE["cal_eps"]


def worse(x):
    return x < -PRE["cal_eps"]


def gate_a(row, base):
    d = deltas(row, base)
    no_worse = (not worse(d["dcal_p"])) and (not worse(d["dcal_r"]))
    hit = d["dcal_p"] >= PRE["dcal"] - 1e-12 or d["dcal_r"] >= PRE["dcal"] - 1e-12
    mdd_ok = row["proxy"]["mdd"] > PRE["proxy_mdd"]
    return bool(no_worse and hit and mdd_ok), d


def platform_of(names, rows):
    cals = [rows[n]["proxy"]["cal"] for n in names]
    cliff = any(abs(cals[i + 1] - cals[i]) > PRE["cliff"] for i in range(len(cals) - 1))
    spike = (cals[1] - cals[0] > PRE["spike"]) or (cals[1] - cals[2] > PRE["spike"])
    mdd_ok = all(rows[n]["proxy"]["mdd"] > PRE["proxy_mdd"] for n in names)
    return (not cliff) and (not spike) and mdd_ok, cliff, spike, mdd_ok


def gate_c(row, base):
    return row["oos"]["cal"] >= base["oos"]["cal"] - PRE["cal_eps"]


def label_of(row, base):
    d = deltas(row, base)
    pw, rw = not worse(d["dcal_p"]), not worse(d["dcal_r"])
    pb, rb = d["dcal_p"] > PRE["cal_eps"], d["dcal_r"] > PRE["cal_eps"]
    mdd_bad = row["proxy"]["mdd"] <= PRE["proxy_mdd"]
    tags = []
    if pw and rw and max(d["dcal_p"], d["dcal_r"]) >= PRE["dcal"] - 1e-12:
        tags.append("MDD破线" if mdd_bad else "一关通过")
    elif pw and rw and (pb or rb):
        tags.append("同向不足")
    elif (pb and not rw) or (rb and not pw):
        tags.append("单窗")
    else:
        tags.append("双窗未改善")
    if mdd_bad and "MDD破线" not in tags:
        tags.append("MDD破线")
    return "+".join(tags), d


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if not np.isfinite(x) else x
    if isinstance(o, (np.integer, int)) and not isinstance(o, bool):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def diag_brief(arm, dg):
    if not dg:
        return ""
    if arm == "A":
        n = dg.get("rebal_days") or 0
        avg = (dg.get("leg_sum") or 0) / n if n else 0
        return f"常规调仓日 {n}，其中成交 {dg.get('leg_days', 0)}，平均入选腿 {avg:.2f}"
    if arm == "B":
        n = dg.get("vol_days") or 0
        avg = (dg.get("vol_scale_sum") or 0) / n if n else 0
        return f"波动评估 {n} 日，其中倍数<1 的 {dg.get('vol_bind', 0)} 日，平均倍数 {avg:.3f}"
    if arm == "C":
        seen = dg.get("derisk_seen") or {}
        bits = ",".join(f"{k}:{v}" for k, v in sorted(seen.items())) or "无"
        return f"周中减仓 {dg.get('derisk_days', 0)} 日；层计数 {bits}"
    if arm == "D":
        n = dg.get("rp_days") or 0
        avg = (dg.get("rp_nq_sum") or 0) / n if n else 0
        return f"重分配 {n} 日，混合后纳指弹性份额均值 {avg:.3f}"
    if arm == "E":
        n = dg.get("reentry_days") or 0
        avg = (dg.get("reentry_s_sum") or 0) / n if n else 0
        return f"再进状态 {n} 日，平均系数 {avg:.3f}"
    if arm == "F":
        n = dg.get("prem_days") or 0
        avg = (dg.get("prem_f_sum") or 0) / n if n else 0
        return f"溢价评估 {n} 日，两只系数均值 {avg:.3f}"
    return ""


def render(p, digest, commit):
    b = p["baseline"]
    L = []
    A = L.append
    A("# v36 预注册优化报告")
    A("")
    A(f"预注册指纹 `{digest}`。预注册提交 `{commit}`。")
    A(f"数据 proxy 至 {p['data']['proxy_end']}，real 至 {p['data']['real_end']}。")
    A("第 1 节是跑数之前写死的规则。第 2 节才是结果。结论在第 3 节，规则没有按结果回改。")
    A("")
    A(PRE_REG_MD.rstrip())
    A("")
    A("## 2. 结果")
    A("")
    A(f"身份校验：生产路径与 v35 锚一致（CAGR {PIN_PROXY['cagr']:.6f} / "
      f"MDD {PIN_PROXY['mdd']:.6f} / Cal {PIN_PROXY['cal']:.6f}）。"
      f"关开关的实验子类与父类最大日收益差 {p['identity']['subclass_gap']:.1e}。"
      f"v30 proxy 累计换手 {b['proxy']['to']:.2f}，年化换手 {b['proxy']['to_ann']:.2f}，"
      f"成交 {b['proxy']['trades_yr']:.2f} 次/年。")
    A("")
    A(f"**基线 v30**：proxy {cell(b['proxy'])}；real {cell(b['real'])}；"
      f"OOS {cell(b['oos'])}。")
    A("")
    A("| 点 | 角色 | proxy CAGR/MDD/Cal | ΔCal | real CAGR/MDD/Cal | ΔCal | OOS Cal | ΔOOS | 年化换手 | 成交/年 | 标签 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    A(f"| v30 | 基线 | {cell(b['proxy'])} | — | {cell(b['real'])} | — | {b['oos']['cal']:.2f} | — | "
      f"{b['proxy']['to_ann']:.1f} | {b['proxy']['trades_yr']:.1f} | 基线 |")
    for job in JOBS:
        row = p["rows"][job["name"]]
        tag, d = label_of(row, b)
        A(f"| {job['name']} | {job['arm']}/{job['role']} | {cell(row['proxy'])} | {dlt(d['dcal_p'])} | "
          f"{cell(row['real'])} | {dlt(d['dcal_r'])} | {row['oos']['cal']:.2f} | {dlt(d['dcal_o'])} | "
          f"{row['proxy']['to_ann']:.1f} | {row['proxy']['trades_yr']:.1f} | {tag} |")
    A("")
    A("诊断（proxy，不参与挑选）：")
    A("")
    for job in JOBS:
        if job["role"] != "primary":
            continue
        gap = p["gaps"].get(job["name"], {}).get("proxy")
        brief = diag_brief(job["arm"], p["diags"].get(job["name"]))
        gap_s = "n/a" if gap is None else f"{gap:.3e}"
        A(f"- {job['name']}: 与 v30 最大日收益差 {gap_s}。{brief}")
    A("")
    A("### 2.1 主点门控")
    A("")
    A("| 臂 | 主点 | 一关 | 平台 | 悬崖 | 尖峰 | 邻点MDD | OOS | 晋升 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for arm in "ABCDEF":
        v = p["verdicts"][arm]
        A(f"| {arm} | {v['primary']} | {v['gate_a']} | {v['platform']} | {v['cliff']} | {v['spike']} | "
          f"{v['plat_mdd']} | {v['oos']} | {v['promote']} |")
    A("")
    A("## 3. 结论")
    A("")
    if p["winner"]:
        w = p["winner"]
        v = p["verdicts"][w]
        d = v["deltas"]
        A(f"**一句话**：臂 {w} 的主点 {v['primary']} 走完三关"
          f"（proxy ΔCal {d['dcal_p']:+.3f}，real ΔCal {d['dcal_r']:+.3f}，"
          f"OOS ΔCal {d['dcal_o']:+.3f}）。提议 `references/final_cfg_v36.json`，默认仍是 v30，配置没有接进实盘入口。")
    else:
        A("**一句话**：六条机制都没有走完三关，默认继续用 v30，没有 `final_cfg_v36.json`。")
    A("")
    for arm in "ABCDEF":
        v = p["verdicts"][arm]
        d = v["deltas"]
        row = p["rows"][v["primary"]]
        why = []
        if not v["gate_a"]:
            why.append("一关没过")
        if not v["platform"]:
            bits = []
            if v["cliff"]:
                bits.append("悬崖")
            if v["spike"]:
                bits.append("尖峰")
            if not v["plat_mdd"]:
                bits.append("邻点 MDD")
            why.append("平台没过" + ("（" + "/".join(bits) + "）" if bits else ""))
        if not v["oos"]:
            why.append("OOS 差于 v30")
        if v["promote"]:
            why.append("三关都过")
        A(f"- **{arm} {v['primary']}**：proxy ΔCal {d['dcal_p']:+.3f}（MDD {row['proxy']['mdd']*100:.2f}%），"
          f"real ΔCal {d['dcal_r']:+.3f}（MDD {row['real']['mdd']*100:.2f}%），"
          f"OOS Cal {row['oos']['cal']:.2f}（Δ {d['dcal_o']:+.3f}）。"
          f"年化换手 {row['proxy']['to_ann']:.1f}（v30 {b['proxy']['to_ann']:.1f}），"
          f"成交 {row['proxy']['trades_yr']:.1f}/年（v30 {b['proxy']['trades_yr']:.1f}）。"
          + "；".join(why) + f"。标签 {v['label']}。")
    A("")
    if p["winner"]:
        A("提议文件只是候选。`run_backtest.py` 仍读 `final_cfg_v30.json`。新键只在 `scripts/exp_v36_prereg.py` 的实验子类里生效。")
    else:
        A("**下一步**：维持 v30。不要在这六个参数上再做网格。下一轮如果还做，必须换机制，并在跑数之前重新写死规则。")
    A("")
    return "\n".join(L) + "\n"


def write_proposal(base, job, digest, commit):
    cfg = cfg_of(base, job)
    cfg["version"] = "v36-proposal"
    cfg["_proposal_only"] = True
    cfg["_prereg_arm"] = job["name"]
    cfg["_prereg_digest"] = digest
    cfg["_prereg_commit"] = commit
    cfg["_note"] = (
        "v36 提议，不是默认。由 scripts/exp_v36_prereg.py 在三关全过之后写入。"
        f" 臂 {job['arm']} 主点 {job['name']}。README/SKILL/final_cfg_v30.json 未改。"
        " 新键只被实验子类读取，run_backtest.py 的默认入口不会加载本文件。"
    )
    path = os.path.join(REF, "final_cfg_v36.json")
    json.dump(cfg, open(path, "w"), ensure_ascii=False, indent=1)
    return path


def prereg_header(digest):
    return (
        "# v36 预注册优化报告\n\n"
        f"预注册指纹 `{digest}`。\n"
        "第 1 节在跑数之前写入。指纹只覆盖第 1 节和脚本里的 PRE/JOBS，不覆盖结果。\n"
        "提交号在跑完后写入第 2 节开头。结果出来之前，本节以下不得改写。\n\n"
        + PRE_REG_MD
    )


def assert_committed_prereg(digest):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    text = subprocess.check_output(
        ["git", "show", f"{head}:references/exp_v36_prereg_report.md"], cwd=ROOT, text=True)
    if PRE_REG_MD not in text:
        raise SystemExit("中止：HEAD 上的报告没有本脚本的预注册正文。先提交预注册再跑数。")
    if f"`{digest}`" not in text:
        raise SystemExit(f"中止：HEAD 上的指纹与当前 PRE/JOBS 不一致（现在是 {digest}）。")
    # 预注册提交 = 报告文件历史里第一次出现该指纹的提交
    log = subprocess.check_output(
        ["git", "log", "--format=%H", "--", "references/exp_v36_prereg_report.md"],
        cwd=ROOT, text=True).split()
    first = None
    for rev in reversed(log):
        blob = subprocess.check_output(
            ["git", "show", f"{rev}:references/exp_v36_prereg_report.md"], cwd=ROOT, text=True)
        if f"`{digest}`" in blob and PRE_REG_MD in blob:
            first = rev
            break
    if first is None:
        raise SystemExit("中止：找不到含该指纹的预注册提交。")
    return first


def load_inputs():
    base = json.load(open(CFG_FILE))
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    return base, bond, R, Rr


def run_identity(base, bond, R):
    m_pin, _, r_pin = execute(base, R, PROXY_START, bond, cls=DynamicStrategy)
    for k, v in PIN_PROXY.items():
        if abs(m_pin[k] - v) > 1e-9:
            raise SystemExit(f"身份失败: 生产路径 {k}={m_pin[k]} 与锚 {v} 不一致，中止")
    m_opt, _, r_opt = execute(base, R, PROXY_START, bond, cls=OptStrategy)
    gap = float((r_pin - r_opt).abs().max())
    if gap > 1e-12:
        raise SystemExit(f"身份失败: 关开关的子类与父类最大日收益差 {gap}")
    print(f"[身份] 锚通过 | 子类差 {gap:.1e} | proxy {cell(m_pin)} | "
          f"换手累计 {m_pin['to']:.2f} 年化 {m_pin['to_ann']:.2f} 成交/年 {m_pin['trades_yr']:.2f}")
    return m_pin, r_pin, gap


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--run"
    digest = digest_of()
    if mode == "--self-check":
        self_check()
        return
    if mode == "--prereg-only":
        open(REPORT, "w").write(prereg_header(digest))
        print(f"[预注册] 写入 {REPORT}")
        print(f"[预注册指纹] {digest}")
        return
    if mode not in ("--identity", "--run"):
        raise SystemExit("用法: --self-check | --prereg-only | --identity | --run")

    commit = None
    if mode == "--run":
        commit = assert_committed_prereg(digest)
        print(f"[预注册提交] {commit}  指纹 {digest}")

    base, bond, R, Rr = load_inputs()
    print(f"[数据] proxy {R.index.min().date()}~{R.index.max().date()} n={len(R)} | "
          f"real {Rr.index.min().date()}~{Rr.index.max().date()} n={len(Rr)}")
    m_pin, r_pin, gap = run_identity(base, bond, R)
    if mode == "--identity":
        return

    rows = {"v30": {"proxy": m_pin}}
    rets_b = {"proxy": r_pin}
    for win, panel, start in (("real", Rr, REAL_START), ("oos", R, OOS_START)):
        m, _, rets = execute(base, panel, start, bond, cls=DynamicStrategy)
        rows["v30"][win] = m
        rets_b[win] = rets
        print(f"[基线] {win:5} {cell(m)} OOS口径外也打印")
    print(f"[基线] proxy {cell(m_pin)}")

    diags, gaps = {}, {}
    for job in JOBS:
        cfg = cfg_of(base, job)
        row = {}
        gaps[job["name"]] = {}
        diag = None
        for win, panel, start in (("proxy", R, PROXY_START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
            m, dg, rets = execute(cfg, panel, start, bond, cls=OptStrategy)
            row[win] = m
            # 对齐到共同日期再比（两边起点相同，索引应一致）
            both = rets.align(rets_b[win], join="inner")
            gaps[job["name"]][win] = float((both[0] - both[1]).abs().max())
            if win == "proxy":
                diag = {k: v for k, v in dg.items() if not str(k).startswith("_")}
        rows[job["name"]] = row
        diags[job["name"]] = diag
        tag, d = label_of(row, rows["v30"])
        print(f"{job['name']:<8} {tag:<16} dP {dlt(d['dcal_p'])} dR {dlt(d['dcal_r'])} "
              f"dO {dlt(d['dcal_o'])} | proxy {cell(row['proxy'])} | "
              f"tr {row['proxy']['trades_yr']:.1f}")

    verdicts = {}
    promoted = []
    for arm, pname in PRIMARY.items():
        ok_a, d = gate_a(rows[pname], rows["v30"])
        plat, cliff, spike, mdd_ok = platform_of(ARM_OF[arm], rows)
        ook = gate_c(rows[pname], rows["v30"])
        full = bool(ok_a and plat and ook)
        verdicts[arm] = {"primary": pname, "gate_a": ok_a, "platform": plat, "cliff": cliff,
                         "spike": spike, "plat_mdd": mdd_ok, "oos": ook, "promote": full,
                         "deltas": d, "label": label_of(rows[pname], rows["v30"])[0]}
        if full:
            promoted.append(arm)
        print(f"[门] {arm} {pname} 一关={ok_a} 平台={plat} cliff={cliff} spike={spike} "
              f"mdd={mdd_ok} OOS={ook} promote={full}")

    winner = None
    if promoted:
        winner = max(promoted, key=lambda a: min(verdicts[a]["deltas"]["dcal_p"],
                                                 verdicts[a]["deltas"]["dcal_r"]))
    proposal = None
    if winner:
        proposal = write_proposal(base, next(j for j in JOBS if j["name"] == PRIMARY[winner]),
                                  digest, commit)
        print(f"[提议] {proposal} （默认仍是 v30）")

    payload = {
        "digest": digest,
        "prereg_commit": commit,
        "pin_proxy": PIN_PROXY,
        "identity": {"subclass_gap": gap},
        "data": {"proxy_end": str(R.index.max().date()), "real_end": str(Rr.index.max().date()),
                 "proxy_n": int(len(R)), "real_n": int(len(Rr))},
        "baseline": rows["v30"],
        "rows": rows,
        "diags": diags,
        "gaps": gaps,
        "verdicts": verdicts,
        "promoted": promoted,
        "winner": winner,
        "proposal": proposal,
    }
    payload = clean(payload)
    json.dump(payload, open(os.path.join(REF, "exp_v36_prereg_results.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(payload, open(os.path.join(OUT_DIR, "exp_v36_prereg.json"), "w"),
              ensure_ascii=False, indent=1)
    open(REPORT, "w").write(render(payload, digest, commit))
    print("[ok]", REPORT)


if __name__ == "__main__":
    main()
