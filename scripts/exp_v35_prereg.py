# -*- coding: utf-8 -*-
"""v35 预注册评估（2026-11-09，数据至面板末日）

规则在本文件 PRE_REG / PRE_REG_MD 里写死，先于任何候选回测。
结果出来之后不得回改这些阈值、主点或通过线。端点即使更好也不升为主点。
不叠加各臂。不改 README/SKILL 默认。三关全过才写 references/final_cfg_v35.json，
并在报告里标明「提议、不是默认」。

用法: python3 scripts/exp_v35_prereg.py
输出: references/exp_v35_prereg_report.md , references/exp_v35_prereg_results.json
"""
import sys, os, json, copy, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from stress_test import synthetic_resonance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "out")
REF = os.path.join(ROOT, "references")
os.makedirs(OUT_DIR, exist_ok=True)
CFG_FILE = os.path.join(REF, "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
# 引擎改动前、本脚本编写前，同一数据上 v30 proxy 的锚（身份校验，不是候选结果）
PIN_PROXY = {"cagr": 0.11111021816342403, "max_dd": -0.09910849911812059,
             "calmar": 1.1210967692185454}

# 三关与 exp_opt31.py 同一组常数。平台必须三点都过，主点取邻域中点。
PRE = {
    "dcal": 0.15,
    "proxy_mdd": -0.10,       # 严格大于 -10%
    "real_mdd": -0.041,
    "plat_win": 0.10,
    "cliff": 0.50,
    "min_ok": 3,
    "oos_bad": 0.05,
    "resonance_worse": 0.005, # 0.50 个百分点
    "churn_turn_ratio": 1.05,
    "churn_cal_eps": 0.01,
    "trigger_cut": 0.22,      # 溢价门控中档：max(T-2 溢价)>=8% 才算「海外被砍」
    "lot_nav0": 1_000_000.0,
    "lot_size": 100,
}

PRE_REG_MD = """\
## 1. 预注册（本节在跑数之前写入脚本，不随结果改写）

对照是 `references/final_cfg_v30.json`。引擎、费用（单边万 5）、信号（`signal_lag=1`、溢价 T-2、先计提后成交、`strict=True`）、窗口与 `scripts/exp_opt31.py` 相同：

- proxy 全历史从 2014-06-23 起
- real 从 2025-04-23 起
- OOS 是 proxy 上的独立起点 2022-01-04（锁状态不从 2014 带入，与 exp_opt31 一致）
- 全历史 proxy 回撤必须 **严格好于 -10%**（等于 -10% 也不过）

### 1.1 三关（晋升线，六臂共用）

1. **双窗同向**：proxy 与 real 的 Calmar 都比 v30 高至少 **+0.15**，且 proxy MDD > -10%，real MDD > -4.1%。
2. **平台**：该臂预注册的 3 个点，沿事先定好的轴排列，相邻 proxy Calmar 落差不超过 0.50；三点的 proxy MDD 都 > -10%；三点相对 v30 的双窗 ΔCalmar 都 **> +0.10**。主点是中点。端点更高也不改主点、不晋升端点。
3. **OOS**：主点 OOS Calmar ≥ v30 OOS − 0.05。

附加（事先写明，不是跑完再加）：主点若要晋升，两段共振熊（窗截止 2026-07-31，以及窗截止到样本末日）的 MDD 相对 v30 恶化不得超过 0.50 个百分点。其余压力情景只陈列，不另设否决线。共振熊历史本来就在 -10% 以外，不要求共振窗本身回到 -10% 以内。

报告用的弱标签（不能晋升）：

- **接近**：双窗 ΔCalmar 都 > 0，两条 MDD 都守住，OOS 没有差过 0.05，但没走完三关。
- **同向但 OOS 超限**：双窗为正且 MDD 守住，OOS 差过 0.05。
- **单窗**：只有一个窗口 Calmar 变好。直接否决。

不叠加各臂。若多于一个主点走完三关，只提议 `min(ΔCal_proxy, ΔCal_real)` 更大的那一个。

### 1.2 权重现在在哪里被量化（Arm F 第 1 条，读代码，不是回测结论）

| 位置 | 做什么 | 粒度 |
|---|---|---|
| `state_map` | 配置里的成长/防御是整数百分数 | 参数网格 1%，不是 `round()` |
| `_eff_score` | `int(round(平滑分))` | 状态 0–9，不量化持仓 |
| `hyst_up/down` | 0.66 / 0.24 | 只平滑分数 |
| `score_confirm_weeks` | 2，周三确认 | 加仓确认，不量化权重 |
| `_finalize` | 百分数权重 ÷ 100 | float64，无取整 |
| `engine` 持仓 | float64 权重 | 没有手数、没有百分数网格 |
| `min_delta` | v30 = **0.035** | 单资产最大缺口 < 3.5% 就不成交 |
| 调仓日程 | 周五、`tranche_weights=[1.0]` | 一周最多一笔常规单，另加日频紧急覆盖 |
| dashboard | 展示 `round(..., 1)` | 只影响显示 |

因此：回测路径已经是连续浮点。F1 若只把目标舍入到 1bp，而 `min_delta` 仍是 350bp、一周仍只调一次，预期与 v30 几乎逐位相同。F 真正能改变成交的是 F2（放宽成交门槛）和 F3（按 100 股一手把已经决定的成交再落成手数）。F **不改** `state_map`、不改滞回、不改周频。日频再平衡以前已经证伪，本轮不重开。`min_delta=0.04` 的旧扫描是往更粗的方向、且只有 proxy 单窗，本轮不重测 0.04，只测更细的三档。

手数登记（跑策略之前，用净值/场内价的中位数，100 万初始资金）：A 股 ETF 用 `unit_nav`，纳指/标普用场内 `close`。一手约 **1.2–1.8bp** 的组合权重，比 3.5% 的 `min_delta` 小两个数量级。159232 的净值只从 2025-04-23 起有，更早的代理日不按手数取整（缺价就保持连续权重）。价格只向前填充 5 个交易日，不回填。

### 1.3 Arm A — 复核 v33 现金层 511260

**假设**：把现金层债券从 511010（5Y）换成 511260（10Y），超额仍主要是久期 beta。最新数据上数字可能还在，但这不等于能当默认。

**主点**：`cash_bond_pct=0.75`、`cash_bond_code=511260`（即现成的 v33，不是新搜的）。邻域只动比例：0.70 / **0.75** / 0.80，债券都是 511260。

**候选保留线**（比三关松，仍然不能当默认）：双窗 Calmar 都不低于 v30，两条 MDD 守住，OOS 不差过 0.05，两段共振熊恶化 ≤ 0.50pp。过了这条也只写「仍为候选」。不过则写「不再维持候选」。

**即使三关全过**：2017-08-24 之前用 511010 拼接、样本里没有 2013 债熊，这个结构缺陷事先写死。可以提议 `final_cfg_v35.json`，默认路径仍是 v30，直到使用者明确接受久期风险。

### 1.4 Arm B — CN 深熊锁内的红利再入阶梯

**假设**：锁是开关。解锁线（回撤收回 -14%，或站上 SMA120 且 20/60 多头）之前，创业板继续为 0（前提②），只把红利弹性（前提④，慢周期、不是动量轮动）按回撤收窄的程度还回一部分。不碰 US 锁（前提①）。

锁内、且原解锁条件尚未成立时，用市场回撤 `dd`（不是 3 年区间位置 `rec`。`rec` 可能在刚锁上时就已经很高）：

- `dd ≤ -22%`：弹性乘数 0（比锁线 -18% 更深）
- `-22% < dd ≤ -18%`：防御弹性 × **0.35**，成长弹性 × 0
- `-18% < dd ≤ -14%`：防御弹性 × **0.60**，成长弹性 × 0（这是现成滞回带，满解锁仍走原条件）
- 原解锁条件成立：立刻满解锁，之后结构闸门和快刹车照旧

低分时防御桶大约 30%，×0.60 ≈ 18 个百分点，加上 2% 底仓，CN 红利最高大约 20%，创业板仍是 0。这是有上限的恢复。若把 proxy MDD 推到 -10% 或更差，直接失败。

邻域（更保守 → 主点 → 更松），只为平台，不另立假设：

- 紧：-23%/-19%，×0.25/×0.50
- **中：-22%/-18%，×0.35/×0.60**
- 松：-21%/-17%，×0.45/×0.70

### 1.5 Arm C — 双杀时的现金混合，只在邻域里动

**假设**：CN 深熊锁开着，并且 QDII 溢价门控已经把海外成长砍到中档或更狠（T-2 的两只 QDII 取较高溢价 ≥ 8%，即 `prem_cut ≤ 0.22`）时，组合主要是现金。这时只微调现金内部，不加股票、不碰创业板。触发条件不扫描。

- **C1** 触发时把 511010 比例从 0.75 调到邻域 0.80 / **0.85** / 0.90，其余时间回到 0.75。次日才计提（与黄金袖套同一时点）。
- **C2** 触发时债券袖套改用 511260，比例仍是 0.75。邻域是袖套里 10Y 的占比 0.50 / **0.75** / 1.00。未触发时回到 100% 511010。

C1、C2 是同一条假设下的两个预注册子动作，不是第七个机制。各用各的三点平台。

### 1.6 Arm D — 美股底仓内部的溢价换载体（不是动量轮动）

**假设**：前提①要求美股长拿，前提③的纳指超配主要在成长桶（bull 63%），不在 5%+5% 的底仓。底仓目前不吃溢价倾斜（倾斜只动底仓以上的弹性）。当 T-2 的纳指溢价比标普高过 **5 个百分点**，把纳指权重里最多 **2 个百分点**换到标普，纳指权重不低于 3%；反向对称。美股合计不变，创业板不动，不用动量。与已有的弹性倾斜叠加：先倾斜，再动这 2 个百分点。

邻域只动阈值：4pp / **5pp** / 6pp。换仓幅度固定 2pp，下限固定 3。

### 1.7 Arm E — 解锁时间滞回，不是失败的 1% 价格带

**为什么这次 OOS 有可能不同**：`gate_band=1%` 改的是每天都会碰到的 SMA120 结构闸门，2022–2024 的 OOS 正好是均线来回穿。本臂不碰结构闸门。它只在 CN 深熊锁已经锁上之后，要求原解锁条件连续成立 **10 个交易日** 才解锁。锁本身很少出现。US 解锁不改。

邻域：5 / **10** / 15 个交易日。没有阶梯。10 日是中点（大约两周），不是跑完挑的。

### 1.8 Arm F — 持仓精度（本轮使用者追加，与 A–E 同一份预注册）

**假设**：更粗的舍入会留下对不齐目标的缺口并漏掉收益；更细的小数权重可能提高 CAGR，同时不把 proxy MDD 弄到 -10% 或更差。读完代码之后的诚实版本：瓶颈若是 `min_delta` 和一周一笔，而不是显示用的百分数，F1 就不会有增益；F2 才是在测成交门槛；F3 是在连续权重上再加一手约束，按 100 万资金大约只有 1–2bp，预期抬不动 Calmar。

- **F0**：v30 原样（基线，不是候选）。
- **F1**：目标组合比例舍入到 10bp / **1bp（0.0001，四位小数）** / 0.1bp。`min_delta`、滞回、周频都不动。舍入用 numpy 的 `round`（银行家舍入）。
- **F2**：不舍入。只把 `min_delta` 调细到 0.020 / **0.010** / 0.002（200bp / 100bp / 20bp）。0.002 是引擎函数签名里的旧默认，作为端点，不是主点。仍是周五一笔。
- **F3**：不在百分数上取整。初始资金 **100 万**（不扫描资金规模），一手 100 股，按最近 / 向下 / 向上取整到手数后再换回权重。主点是**最近**。`min_delta` 仍是 0.035。缺价的资产保持连续权重。

**换手条款**：proxy 换手比 v30 高过 5%，且双窗 ΔCalmar 都 ≤ +0.01，标成「只换手不增益」，不得晋升。换手升了但 Calmar 没到 +0.15，标成「换手上升、未过线」，同样不得晋升。三关本身已经要求 +0.15；这条是把「只换手」写明。

### 1.9 事先声明不会做的事

不重跑已经证伪的 LC×动量、GEM-CN+RB、幽灵交易员、六资产 20 日+MA28、v31 全套、`gate_band=1%`、max-Calmar、速度刹车、波动率目标、`dd_eq_cap`、黄金危机袖套、宽度/ADX。不把本轮阈值再加密网格。默认配置文件保持 v30。
"""

# 平台轴顺序 = 列表顺序。role=primary 才允许晋升。
JOBS = [
    {"name": "A_p70", "arm": "A", "role": "neighbor", "overlay": {"cash_bond_code": "511260", "cash_bond_pct": 0.70}},
    {"name": "A_p75", "arm": "A", "role": "primary", "overlay": {"cash_bond_code": "511260", "cash_bond_pct": 0.75}},
    {"name": "A_p80", "arm": "A", "role": "neighbor", "overlay": {"cash_bond_code": "511260", "cash_bond_pct": 0.80}},
    {"name": "B_tight", "arm": "B", "role": "neighbor", "overlay": {"cn_unlock_ladder": [-0.23, -0.19, 0.25, 0.50]}},
    {"name": "B_mid", "arm": "B", "role": "primary", "overlay": {"cn_unlock_ladder": [-0.22, -0.18, 0.35, 0.60]}},
    {"name": "B_loose", "arm": "B", "role": "neighbor", "overlay": {"cn_unlock_ladder": [-0.21, -0.17, 0.45, 0.70]}},
    {"name": "C_p80", "arm": "C1", "role": "neighbor", "overlay": {}, "sleeve": {"pct": 0.80, "alt": 0.0}},
    {"name": "C_p85", "arm": "C1", "role": "primary", "overlay": {}, "sleeve": {"pct": 0.85, "alt": 0.0}},
    {"name": "C_p90", "arm": "C1", "role": "neighbor", "overlay": {}, "sleeve": {"pct": 0.90, "alt": 0.0}},
    {"name": "Cdur_a50", "arm": "C2", "role": "neighbor", "overlay": {}, "sleeve": {"pct": 0.75, "alt": 0.50}},
    {"name": "Cdur_a75", "arm": "C2", "role": "primary", "overlay": {}, "sleeve": {"pct": 0.75, "alt": 0.75}},
    {"name": "Cdur_a100", "arm": "C2", "role": "neighbor", "overlay": {}, "sleeve": {"pct": 0.75, "alt": 1.00}},
    {"name": "D_t04", "arm": "D", "role": "neighbor", "overlay": {"floor_prem_shift": 2.0, "floor_prem_thr": 0.04, "floor_prem_min": 3.0}},
    {"name": "D_t05", "arm": "D", "role": "primary", "overlay": {"floor_prem_shift": 2.0, "floor_prem_thr": 0.05, "floor_prem_min": 3.0}},
    {"name": "D_t06", "arm": "D", "role": "neighbor", "overlay": {"floor_prem_shift": 2.0, "floor_prem_thr": 0.06, "floor_prem_min": 3.0}},
    {"name": "E_d05", "arm": "E", "role": "neighbor", "overlay": {"unlock_confirm_days": 5}},
    {"name": "E_d10", "arm": "E", "role": "primary", "overlay": {"unlock_confirm_days": 10}},
    {"name": "E_d15", "arm": "E", "role": "neighbor", "overlay": {"unlock_confirm_days": 15}},
    {"name": "F1_q10bp", "arm": "F1", "role": "neighbor", "overlay": {}, "quantum": 0.001},
    {"name": "F1_q1bp", "arm": "F1", "role": "primary", "overlay": {}, "quantum": 0.0001},
    {"name": "F1_q01bp", "arm": "F1", "role": "neighbor", "overlay": {}, "quantum": 1e-5},
    {"name": "F2_d200", "arm": "F2", "role": "neighbor", "overlay": {"min_delta": 0.020}},
    {"name": "F2_d100", "arm": "F2", "role": "primary", "overlay": {"min_delta": 0.010}},
    {"name": "F2_d020", "arm": "F2", "role": "neighbor", "overlay": {"min_delta": 0.002}},
    {"name": "F3_floor", "arm": "F3", "role": "neighbor", "overlay": {}, "lot_mode": "floor"},
    {"name": "F3_nearest", "arm": "F3", "role": "primary", "overlay": {}, "lot_mode": "nearest"},
    {"name": "F3_ceil", "arm": "F3", "role": "neighbor", "overlay": {}, "lot_mode": "ceil"},
]
PRIMARY = {j["arm"]: j["name"] for j in JOBS if j["role"] == "primary"}
ARM_OF = {}
for j in JOBS:
    ARM_OF.setdefault(j["arm"], []).append(j["name"])


class OptStrategy(DynamicStrategy):
    """实验子类。新开关缺省时走父类，v30 配置必须与 DynamicStrategy 逐位一致。"""

    def __init__(self, R_full, cfg=None, a_mkt_override=None):
        super().__init__(R_full, cfg=cfg, a_mkt_override=a_mkt_override)
        cfg = cfg or {}
        lad = cfg.get("cn_unlock_ladder")
        self.ladder = [float(x) for x in lad] if lad else None
        self.unlock_confirm_days = int(cfg.get("unlock_confirm_days", 0))
        self.floor_prem_shift = float(cfg.get("floor_prem_shift", 0.0))
        self.floor_prem_thr = float(cfg.get("floor_prem_thr", 0.05))
        self.floor_prem_min = float(cfg.get("floor_prem_min", 3.0))
        self.diag = {"ladder_on": 0, "locked_calls": 0, "floor_shift": 0,
                     "cash_on": 0, "cash_calls": 0}

    def _unlock_ready(self, dt, market):
        dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, market, self.gate_win)
        recov = self.market_dd[market][3]
        n = self.unlock_confirm_days
        if n <= 0:
            return dd > -recov or (ok120 and ok20_60)
        idx = self.R.index
        lag = max(1, int(self.sig.lag))
        i_sig = self.sig._idx(dt)
        for k in range(n):
            j = i_sig - k
            pos = j + lag
            if j < 0 or pos >= len(idx):
                return False
            dd_k, ok20_k, ok2060_k, rec_k, ok120_k = self.sig.mkt_info(idx[pos], market, self.gate_win)
            if not (dd_k > -recov or (ok120_k and ok2060_k)):
                return False
        return True

    def _market_mult(self, dt, market):
        custom = market == "CN" and (self.ladder is not None or self.unlock_confirm_days > 0)
        if not custom:
            return super()._market_mult(dt, market)
        dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, market, self.gate_win)
        fb_thr, fb_cut, deep_thr, recov = self.market_dd[market]
        m_g, m_all = 1.0, 1.0
        if self._lock[market]:
            if self._unlock_ready(dt, market):
                self._lock[market] = False
            else:
                self.diag["locked_calls"] += 1
                if self.ladder is None:
                    return 0.0, 0.0
                deep_cut, mid_cut, lo_mult, hi_mult = self.ladder
                if dd <= deep_cut:
                    mult = 0.0
                elif dd <= mid_cut:
                    mult = lo_mult
                else:
                    mult = hi_mult
                if mult > 0:
                    self.diag["ladder_on"] += 1
                return 0.0, mult
        if dd < -deep_thr:
            self._lock[market] = True
            return 0.0, 0.0
        gst = self._gate_state(dt, market) if self.gate_band > 0 else ok120
        if not gst:
            m_g = 0.0
        if dd < -fb_thr and not (ok20 and ok20_60):
            m_g = min(m_g, fb_cut)
        return m_g, m_all

    def _base_target(self, dt, sc):
        out = super()._base_target(dt, sc)
        if self.floor_prem_shift <= 0 or self._premium is None or dt not in self._premium.index:
            return out
        row = self._premium.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        p941, p500 = row.get("159941"), row.get("513500")
        if pd.isna(p941) or pd.isna(p500):
            return out
        diff = float(p941) - float(p500)
        if diff > self.floor_prem_thr:
            src, dst = "159941", "513500"
        elif diff < -self.floor_prem_thr:
            src, dst = "513500", "159941"
        else:
            return out
        room = float(out[src]) - self.floor_prem_min
        mv = min(self.floor_prem_shift, max(0.0, room))
        if mv > 1e-9:
            out[src] -= mv
            out[dst] += mv
            self.diag["floor_shift"] += 1
        return out


def load_prices():
    def col(fn, c):
        s = read_table(fn)[c]
        return s[~s.index.duplicated(keep="last")].sort_index()
    px = pd.DataFrame({
        "159232": col("159232_nav.csv", "unit_nav"),
        "515100": col("515100_nav.csv", "unit_nav"),
        "159952": col("159952_nav.csv", "unit_nav"),
        "159941": col("qdii_price_159941.csv", "close"),
        "513500": col("qdii_price_513500.csv", "close"),
    })
    return px[~px.index.duplicated(keep="last")].sort_index()


def pack(e):
    return {"cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
            "sharpe": float(e["sharpe"]), "to": float(e["turnover"]), "cash": float(e["avg_cash"])}


def make_sleeve(ds, spec, base_pct):
    pct_on, alt_on = float(spec["pct"]), float(spec["alt"])
    cut_line = PRE["trigger_cut"]

    def fn(dt):
        locked = bool(ds._lock.get("CN", False))
        pm = ds._premium_at(dt) if ds._premium is not None else None
        cut = 1.0
        if pm is not None and pm > 0:
            for thr, c in zip(ds.premium_thr, ds.premium_cut):
                if pm >= thr:
                    cut = min(cut, c)
        ds.diag["cash_calls"] += 1
        if locked and cut <= cut_line:
            ds.diag["cash_on"] += 1
            return {"pct": pct_on, "alt_frac": alt_on}
        return {"pct": base_pct, "alt_frac": 0.0}
    return fn


def execute(cfg, R, start, bonds, prices, end=None, a_mkt=None, sleeve=None,
            quantum=None, lot_mode=None, cls=OptStrategy):
    ds = cls(R, cfg=cfg, a_mkt_override=a_mkt)
    code = cfg.get("cash_bond_code", "511010")
    bond = bonds[code]
    alt = bonds["511260"] if sleeve is not None else None
    fn = make_sleeve(ds, sleeve, float(cfg.get("cash_bond_pct", 0.75))) if sleeve else None
    kw = {}
    if quantum is not None:
        kw["weight_quantum"] = quantum
    if lot_mode is not None:
        kw["lot_prices"] = prices
        kw["lot_mode"] = lot_mode
        kw["lot_nav0"] = PRE["lot_nav0"]
        kw["lot_size"] = PRE["lot_size"]
    res = run_backtest(
        R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="DYN", min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True, cash_sleeve_fn=fn, cash_bond_alt_rets=alt, **kw)
    return pack(evaluate(res)), getattr(ds, "diag", {}), res["rets"]


def cfg_of(base, job):
    cfg = copy.deepcopy(base)
    cfg.update(job.get("overlay") or {})
    return cfg


def deltas(row, base):
    to_b = base["proxy"]["to"]
    return {
        "dcal_p": row["proxy"]["cal"] - base["proxy"]["cal"],
        "dcal_r": row["real"]["cal"] - base["real"]["cal"],
        "dcal_o": row["oos"]["cal"] - base["oos"]["cal"],
        "to_ratio": (row["proxy"]["to"] / to_b) if to_b else float("nan"),
    }


def gate1(row, base):
    d = deltas(row, base)
    ok = (d["dcal_p"] >= PRE["dcal"] and d["dcal_r"] >= PRE["dcal"]
          and row["proxy"]["mdd"] > PRE["proxy_mdd"] and row["real"]["mdd"] > PRE["real_mdd"]
          and np.isfinite(d["dcal_p"]) and np.isfinite(d["dcal_r"]))
    return ok, d


def platform_ok(names, rows, base):
    pts = [rows[n] for n in names]
    cals = [p["proxy"]["cal"] for p in pts]
    cliff = any(abs(cals[i] - cals[i - 1]) > PRE["cliff"] for i in range(1, len(cals)))
    n_ok, mdd_ok = 0, True
    for p in pts:
        if p["proxy"]["mdd"] <= PRE["proxy_mdd"]:
            mdd_ok = False
        if (p["proxy"]["cal"] - base["proxy"]["cal"] > PRE["plat_win"]
                and p["real"]["cal"] - base["real"]["cal"] > PRE["plat_win"]):
            n_ok += 1
    return (not cliff) and mdd_ok and n_ok >= PRE["min_ok"], n_ok, cliff, mdd_ok


def oos_ok(row, base):
    return row["oos"]["cal"] >= base["oos"]["cal"] - PRE["oos_bad"]


def churn_only(d):
    return d["to_ratio"] > PRE["churn_turn_ratio"] and d["dcal_p"] <= PRE["churn_cal_eps"] and d["dcal_r"] <= PRE["churn_cal_eps"]


def label_of(row, base):
    d = deltas(row, base)
    g1, _ = gate1(row, base)
    same = d["dcal_p"] > 0 and d["dcal_r"] > 0
    mdd = row["proxy"]["mdd"] > PRE["proxy_mdd"] and row["real"]["mdd"] > PRE["real_mdd"]
    oos = oos_ok(row, base)
    if churn_only(d):
        tag = "只换手不增益"
    elif same and mdd and oos and not g1:
        tag = "接近"
    elif same and mdd and not oos:
        tag = "同向但OOS超限"
    elif (d["dcal_p"] > 0) ^ (d["dcal_r"] > 0):
        tag = "单窗"
    elif g1:
        tag = "过一关"
    elif d["dcal_p"] <= 0 and d["dcal_r"] <= 0:
        tag = "双窗未改善"
    else:
        tag = "未过"
    if row["proxy"]["mdd"] <= PRE["proxy_mdd"]:
        tag += "+MDD破线"
    return tag, d


def reso_ok(st_job, st_base):
    for k in ("共振熊_至2026-07-31", "共振熊_至样本末日"):
        if st_job[k]["mdd"] < st_base[k]["mdd"] - PRE["resonance_worse"]:
            return False
    return True


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


def pct(x):
    return f"{x * 100:.2f}%"


def cell(m):
    return f"{m['cagr'] * 100:.2f}% / {m['mdd'] * 100:.2f}% / {m['cal']:.2f}"


def dlt(x):
    return f"{x:+.3f}"


def scenario_list(Rr, synth_c, idx_c, synth_e, idx_e):
    end_real = str(Rr.index.max().date())
    return [
        ("牛市_2019-2021", "2019-01-04", "2021-02-18", "proxy", None),
        ("牛市_真实窗口", REAL_START, end_real, "real", None),
        ("震荡市_2023-2024", "2023-01-03", "2024-08-30", "proxy", None),
        ("熊市_2015股灾", "2015-06-15", "2016-02-29", "proxy", None),
        ("熊市_2018", "2018-01-02", "2019-01-03", "proxy", None),
        ("熊市_2021-2022", "2021-02-19", "2022-10-31", "proxy", None),
        ("共振熊_至2026-07-31", None, None, "synth", (synth_c, idx_c)),
        ("共振熊_至样本末日", None, None, "synth", (synth_e, idx_e)),
    ]


def main():
    pre_blob = json.dumps({"PRE": PRE, "JOBS": JOBS, "md": PRE_REG_MD}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(pre_blob.encode("utf-8")).hexdigest()[:16]
    print(f"[预注册指纹] {digest}  （阈值已冻结，后面只跑数）")
    base_cfg = json.load(open(CFG_FILE))
    bonds = {
        "511010": rets_from(read_table("511010_nav.csv"), "cum_nav"),
        "511260": rets_from(read_table("511260_nav.csv"), "cum_nav"),
    }
    prices = load_prices()
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    print(f"[数据] proxy {R.index.min().date()}~{R.index.max().date()} n={len(R)} | "
          f"real {Rr.index.min().date()}~{Rr.index.max().date()} n={len(Rr)}")

    # ---- 身份：生产路径不得因钩子改变；子类关开关必须等于父类 ----
    m_pin, _, r_pin = execute(base_cfg, R, PROXY_START, bonds, prices, cls=DynamicStrategy)
    for k, v in PIN_PROXY.items():
        if abs(m_pin[k] - v) > 1e-9:
            raise SystemExit(f"身份失败: 生产路径 {k}={m_pin[k]} 与锚 {v} 不一致，中止，不解释候选")
    m_opt, _, r_opt = execute(base_cfg, R, PROXY_START, bonds, prices, cls=OptStrategy)
    gap = float((r_pin - r_opt).abs().max())
    if gap > 1e-12:
        raise SystemExit(f"身份失败: OptStrategy 关开关与父类最大日收益差 {gap}")
    m_sl, _, r_sl = execute(base_cfg, R, PROXY_START, bonds, prices,
                            sleeve={"pct": 0.75, "alt": 0.0}, cls=OptStrategy)
    gap_sl = float((r_pin - r_sl).abs().max())
    print(f"[身份] 生产路径对上锚 | 子类差 {gap:.1e} | 静态袖套差 {gap_sl:.1e}")
    if gap_sl > 1e-8:
        raise SystemExit("身份失败: 袖套保持 0.75/0 时与 v30 日收益差超过 1e-8")

    rows, diags = {}, {}
    base_row = {"proxy": m_pin}
    print("[基线] proxy", cell(m_pin))
    for win, panel, start in (("real", Rr, REAL_START), ("oos", R, OOS_START)):
        m, _, _ = execute(base_cfg, panel, start, bonds, prices, cls=DynamicStrategy)
        base_row[win] = m
        print(f"[基线] {win:5} {cell(m)}")
    rows["F0_v30"] = base_row

    for job in JOBS:
        cfg = cfg_of(base_cfg, job)
        row, diag = {}, None
        for win, panel, start in (("proxy", R, PROXY_START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
            m, dg, rets = execute(cfg, panel, start, bonds, prices, sleeve=job.get("sleeve"),
                                  quantum=job.get("quantum"), lot_mode=job.get("lot_mode"))
            row[win] = m
            if win == "proxy":
                diag = dg
                if job["name"] == "F1_q1bp":
                    rows["_f1_gap"] = float((rets - r_pin).abs().max())
        rows[job["name"]] = row
        diags[job["name"]] = diag
        tag, d = label_of(row, base_row)
        print(f"{job['name']:<12} {tag:<10} dP {dlt(d['dcal_p'])} dR {dlt(d['dcal_r'])} "
              f"dO {dlt(d['dcal_o'])} to×{d['to_ratio']:.3f} | proxy {cell(row['proxy'])}")

    # ---- 三关，只评主点；压力：A 必跑，过了一关的主点也跑 ----
    need_stress = ["A_p75"]
    verdicts = {}
    for arm, pname in PRIMARY.items():
        ok1, d = gate1(rows[pname], base_row)
        plat, n_ok, cliff, mdd_ok = platform_ok(ARM_OF[arm], rows, base_row)
        ook = oos_ok(rows[pname], base_row)
        verdicts[arm] = {"primary": pname, "gate1": ok1, "platform": plat, "plat_n": n_ok,
                         "cliff": cliff, "plat_mdd": mdd_ok, "oos": ook, "deltas": d,
                         "churn": churn_only(d), "label": label_of(rows[pname], base_row)[0]}
        if ok1 and pname not in need_stress:
            need_stress.append(pname)
        print(f"[门] {arm} {pname} 一关={ok1} 平台={plat}({n_ok}, cliff={cliff}, mdd={mdd_ok}) OOS={ook}")

    end_real = str(Rr.index.max().date())
    print("[压力] 合成共振…")
    synth_c, idx_c = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
    synth_e, idx_e = synthetic_resonance(Rr, ("2024-09-02", end_real), -0.30, -0.20, -0.25)
    scens = scenario_list(Rr, synth_c, idx_c, synth_e, idx_e)
    panels = {"proxy": R, "real": Rr}

    def run_stress(job):
        cfg = base_cfg if job is None else cfg_of(base_cfg, job)
        sleeve = None if job is None else job.get("sleeve")
        quantum = None if job is None else job.get("quantum")
        lot_mode = None if job is None else job.get("lot_mode")
        out = {}
        for name, ps, pe, kind, extra in scens:
            if kind == "synth":
                panel, a_mkt = extra
                m, _, _ = execute(cfg, panel, None, bonds, prices, a_mkt=a_mkt,
                                  sleeve=sleeve, quantum=quantum, lot_mode=lot_mode)
            else:
                m, _, _ = execute(cfg, panels[kind], ps, bonds, prices, end=pe,
                                  sleeve=sleeve, quantum=quantum, lot_mode=lot_mode)
            out[name] = {"cagr": m["cagr"], "mdd": m["mdd"], "cal": m["cal"]}
            print(f"  [压力] {('v30' if job is None else job['name'])} {name}: {m['cagr']*100:.2f}% / {m['mdd']*100:.2f}%")
        return out

    stress = {"F0_v30": run_stress(None)}
    job_by = {j["name"]: j for j in JOBS}
    for nm in need_stress:
        stress[nm] = run_stress(job_by[nm])

    promoted = []
    for arm, v in verdicts.items():
        pname = v["primary"]
        v["resonance"] = None
        if pname in stress:
            v["resonance"] = reso_ok(stress[pname], stress["F0_v30"])
        # 晋升必须压力也跑了并且共振通过。A 总会有压力。
        full = v["gate1"] and v["platform"] and v["oos"] and (v["resonance"] is True) and not v["churn"]
        v["promote"] = full
        if full:
            promoted.append(arm)
        print(f"[判决] {arm} promote={full} resonance={v['resonance']}")

    winner = None
    if promoted:
        winner = max(promoted, key=lambda a: min(verdicts[a]["deltas"]["dcal_p"], verdicts[a]["deltas"]["dcal_r"]))

    # A 的候选保留线（不能当默认）
    a_row = rows["A_p75"]
    a_d = deltas(a_row, base_row)
    a_candidate = (a_d["dcal_p"] >= 0 and a_d["dcal_r"] >= 0
                   and a_row["proxy"]["mdd"] > PRE["proxy_mdd"] and a_row["real"]["mdd"] > PRE["real_mdd"]
                   and a_d["dcal_o"] >= -PRE["oos_bad"]
                   and reso_ok(stress["A_p75"], stress["F0_v30"]))

    v35_path = None
    if winner:
        job = job_by[PRIMARY[winner]]
        cfg = cfg_of(base_cfg, job)
        cfg["version"] = "v35"
        cfg["_proposal_only"] = True
        cfg["_prereg_arm"] = job["name"]
        cfg["_prereg_digest"] = digest
        cfg["_note"] = (
            "v35 提议，不是默认。由 scripts/exp_v35_prereg.py 在三关全过之后写入。"
            f" 臂 {winner} 主点 {job['name']}。README/SKILL 仍指向 final_cfg_v30.json。"
            " 若本臂依赖实验子类或引擎钩子，实盘入口未改，合并前必须由人确认。"
            " Arm A 即使中选也保留久期样本缺陷（2017-08 前拼接、无 2013 债熊）。"
        )
        # 引擎钩子写进配置，便于复核；run_backtest.py 默认仍读 v30，不会自动生效。
        if job.get("quantum") is not None:
            cfg["weight_quantum"] = job["quantum"]
        if job.get("lot_mode") is not None:
            cfg["lot_mode"] = job["lot_mode"]
            cfg["lot_nav0"] = PRE["lot_nav0"]
            cfg["lot_size"] = PRE["lot_size"]
        if job.get("sleeve") is not None:
            cfg["cash_sleeve"] = job["sleeve"]
            cfg["cash_sleeve_trigger_cut"] = PRE["trigger_cut"]
        v35_path = os.path.join(REF, "final_cfg_v35.json")
        json.dump(cfg, open(v35_path, "w"), ensure_ascii=False, indent=1)
        print(f"[提议] 写入 {v35_path} （默认仍是 v30）")

    payload = {
        "digest": digest,
        "pin_proxy": PIN_PROXY,
        "identity": {"subclass_gap": gap, "static_sleeve_gap": gap_sl, "f1_primary_max_abs_ret": rows.get("_f1_gap")},
        "data": {"proxy_end": str(R.index.max().date()), "real_end": str(Rr.index.max().date())},
        "baseline": rows["F0_v30"],
        "rows": {k: v for k, v in rows.items() if not k.startswith("_")},
        "diags": diags,
        "verdicts": verdicts,
        "stress": stress,
        "a_still_candidate": a_candidate,
        "promoted": promoted,
        "winner": winner,
        "v35": v35_path,
    }
    payload = clean(payload)
    json.dump(payload, open(os.path.join(REF, "exp_v35_prereg_results.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(payload, open(os.path.join(OUT_DIR, "exp_v35_prereg.json"), "w"), ensure_ascii=False, indent=1)
    report = render(payload, digest)
    open(os.path.join(REF, "exp_v35_prereg_report.md"), "w").write(report)
    print("[ok] references/exp_v35_prereg_report.md")
    return payload


def render(p, digest):
    b = p["baseline"]
    L = []
    A = L.append
    A("# v35 预注册优化报告")
    A("")
    A(f"预注册指纹 `{digest}`。数据 proxy 至 {p['data']['proxy_end']}，real 至 {p['data']['real_end']}。")
    A("第 1 节是跑数之前写死的规则。第 2 节才是结果。结论在第 3 节，规则没有按结果回改。")
    A("")
    A(PRE_REG_MD)
    A("## 2. 结果")
    A("")
    A(f"身份校验：生产路径与脚本编写前的 v30 proxy 锚一致"
      f"（CAGR {PIN_PROXY['cagr']:.6f} / MDD {PIN_PROXY['max_dd']:.6f} / Cal {PIN_PROXY['calmar']:.6f}）。"
      f"关开关的实验子类日收益差 {p['identity']['subclass_gap']:.1e}。"
      f"袖套锁定在 75% 511010 时日收益差 {p['identity']['static_sleeve_gap']:.1e}。")
    f1g = p["identity"].get("f1_primary_max_abs_ret")
    if f1g is not None:
        A(f"F1 主点（1bp 舍入）相对 v30 的最大日收益差 {f1g:.3e}。"
          + ("这就是「已经是浮点，小数位不是瓶颈」。" if f1g < 1e-8 else "差不是尘埃，见 F1 行。"))
    A("")
    A(f"**基线 v30**：proxy {cell(b['proxy'])}；real {cell(b['real'])}；OOS {cell(b['oos'])}；"
      f"proxy 换手 {b['proxy']['to']:.2f}。")
    A("")
    A("| 点 | 角色 | proxy CAGR/MDD/Cal | ΔCal | real CAGR/MDD/Cal | ΔCal | OOS Cal | ΔOOS | 换手比 | 标签 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    A(f"| F0_v30 | 基线 | {cell(b['proxy'])} | — | {cell(b['real'])} | — | {b['oos']['cal']:.2f} | — | 1 | 基线 |")
    labels = {}
    for job in JOBS:
        row = p["rows"][job["name"]]
        tag, d = label_of(row, b)
        labels[job["name"]] = tag
        A(f"| {job['name']} | {job['arm']}/{job['role']} | {cell(row['proxy'])} | {dlt(d['dcal_p'])} | "
          f"{cell(row['real'])} | {dlt(d['dcal_r'])} | {row['oos']['cal']:.2f} | {dlt(d['dcal_o'])} | "
          f"{d['to_ratio']:.3f} | {tag} |")
    A("")
    A("诊断（proxy 窗口，描述用，不参与挑选）：")
    A("")
    for name, dg in p["diags"].items():
        if not dg:
            continue
        bits = []
        if dg.get("locked_calls"):
            bits.append(f"锁内调用 {dg['locked_calls']}，其中阶梯>0 的 {dg['ladder_on']}")
        if dg.get("cash_calls"):
            bits.append(f"现金触发 {dg['cash_on']}/{dg['cash_calls']} 日")
        if dg.get("floor_shift"):
            bits.append(f"底仓溢价换载体决策 {dg['floor_shift']} 次")
        if bits:
            A(f"- {name}: " + "；".join(bits))
    A("")
    A("### 2.1 主点门控")
    A("")
    A("| 臂 | 主点 | 一关 | 平台 | 平台点数 | 悬崖 | 平台MDD | OOS | 共振 | 只换手 | 晋升 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, v in p["verdicts"].items():
        A(f"| {arm} | {v['primary']} | {v['gate1']} | {v['platform']} | {v['plat_n']} | {v['cliff']} | "
          f"{v['plat_mdd']} | {v['oos']} | {v['resonance']} | {v['churn']} | {v['promote']} |")
    A("")
    A("### 2.2 压力（v30 与 Arm A；另加任何过了一关的主点）")
    A("")
    names = list(p["stress"].keys())
    A("| 情景 | " + " | ".join(names) + " |")
    A("|---|" + "|".join(["---"] * len(names)) + "|")
    sc_names = list(next(iter(p["stress"].values())).keys())
    for sc in sc_names:
        cells = []
        for nm in names:
            s = p["stress"][nm][sc]
            cells.append(f"{s['cagr']*100:.2f}% / {s['mdd']*100:.2f}%")
        A(f"| {sc} | " + " | ".join(cells) + " |")
    A("")
    A("## 3. 结论")
    A("")
    if p["winner"]:
        one = (f"三关通过的是 {p['winner']}（主点 {p['verdicts'][p['winner']]['primary']}）。"
               f"已写 `final_cfg_v35.json` 作为提议，默认仍是 v30，README/SKILL 没改。")
    else:
        one = "六条机制都没有走完三关，默认继续用 v30，没有新的默认配置。"
    A("**一句话**： " + one)
    A("")
    A(f"Arm A（511260，75%）{'仍只是候选，不能当默认' if p['a_still_candidate'] else '连候选保留线都没过，不再维持候选'}。"
      "久期样本缺陷（2017-08 前拼接、没有 2013 债熊）不因数字好看而消失。")
    A("")
    almost = [n for n, t in labels.items() if t.startswith("接近")]
    oos_bad = [n for n, t in labels.items() if "OOS" in t]
    one_win = [n for n, t in labels.items() if t.startswith("单窗")]
    churns = [n for n, t in labels.items() if "只换手" in t]
    mdd_bad = [n for n, t in labels.items() if "MDD" in t]
    A("**接近但未过三关**： " + ("、".join(almost) if almost else "无"))
    A("")
    A("**同向但 OOS 超限**： " + ("、".join(oos_bad) if oos_bad else "无"))
    A("")
    A("**单窗（否决）**： " + ("、".join(one_win) if one_win else "无"))
    A("")
    A("**只换手不增益**： " + ("、".join(churns) if churns else "无"))
    A("")
    A("**proxy MDD 破 -10%**： " + ("、".join(mdd_bad) if mdd_bad else "无"))
    A("")
    if p["winner"]:
        A("**下一步**：看 `final_cfg_v35.json` 的 `_note`。人确认之前不要把 README、SKILL 或实盘入口改到 v35。"
          "本轮没有把通过的臂和其他臂叠在一起。")
    else:
        A("**下一步**：维持 v30。不要在这批阈值上再做网格。下一轮如果还做，必须换一个机制，并在跑数之前重新写死规则。"
          "F 若已经证明小数位没有信息，就不要再调显示精度；剩下的约束是一周一笔，而日频再平衡不在本轮重开。")
    A("")
    A("合并说明：生产默认路径没有切换。`final_cfg_v30.json` 仍是基线。"
      "引擎新增的袖套、1bp 舍入、手数取整都要显式传入才生效；不传时与改动前的 v30 proxy 日收益逐位相同。")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
