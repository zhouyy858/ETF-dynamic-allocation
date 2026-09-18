# -*- coding: utf-8 -*-
"""动态策略v21(无未来函数): 结构闸门 + 快刹车(成长仓) + 深熊锁定(全部CN/US弹性清零, 仅留底仓)
- 打分9维 -> 状态(成长g/防御d) -> 权重骨架; SignalSet(lag=1): 所有信号只用截至决策日前一交易日收盘数据
- CN弹性(159952成长 + 515100/159232防御弹性) × CN市场乘数; US弹性(159941/513500额外) × US市场乘数
- 快刹车: 市场回撤>7%(CN)/10%(US)且跌破SMA20 -> 成长弹性×0.08/0.12
- 深熊锁定: CN回撤>22%/US回撤>24% -> 该市场全部弹性归零(仅留底仓), 恢复条件: 回撤收窄或SMA120+双均线确认
- 组合深度回撤 -> 权益上限(趋势弱时); 5%国内+5%海外底仓固定(floor_pct配置); 每周三决策、分3周三笔(引擎tranche_weights);
- 日度风控触发(组合回撤上限/市场快刹车/深熊锁/闸门导致目标权益显著降低)当日生效并取代未完成计划
- QDII溢价门控用T-2口径(premium_shift=2, QDII净值T+1晚间发布)"""

import os, numpy as np, pandas as pd
from data_prep import DATA_DIR

def _data(fn):
    return os.path.join(DATA_DIR, fn)

SLOTS = ["159232", "515100", "159941", "513500", "159952"]
FLOOR = {"159232": 7.5, "515100": 7.5, "159941": 7.5, "513500": 7.5, "159952": 0.0}
FLOOR_EQ = 30.0

STATE_MAP = {
    9: (78, 4), 8: (74, 7), 7: (69, 10), 6: (62, 13), 5: (55, 16),
    4: (47, 18), 3: (39, 21), 2: (31, 24), 1: (23, 27), 0: (16, 30),
}
GROWTH_SPLIT_BULL = {"159952": 0.34, "159941": 0.48, "513500": 0.20}
GROWTH_SPLIT_BEAR = {"159952": 0.44, "159941": 0.36, "513500": 0.20}
DEFENSE_SPLIT = {"159232": 0.45, "515100": 0.55}
DEFENSE_MOMENTUM_WIN = 60
DEFENSE_MOMENTUM_T = 3.0
DEFENSE_CLAMP = (0.35, 0.65)
US_SPLIT_CLAMP = (0.55, 0.85)
VALUATION_WIN_DAYS = 2520
# QDII 溢价门控: 场内价/净值-1(>阈值削减海外成长弹性), 数据为前一日溢价(无未来函数)
PREMIUM_FILES = {"159941": "qdii_price_159941.csv", "513500": "qdii_price_513500.csv"}
PREMIUM_SPLIT_DROP = {"159941": "2022-07-04"}  # 4:1份额折算过渡日剔除
PREMIUM_CLIP = (-0.10, 0.15)
PREMIUM_THR = [0.03, 0.05, 0.08]
PREMIUM_CUT = [0.5, 0.25, 0.1]

MIN_CASH = 0.05
MAX_EQ = 0.97
# 市场刹车: (快刹车触发DD%, 快刹车成长削减, 深熊锁定DD%, 解锁DD%)
MARKET_DD = {"CN": (0.07, 0.08, 0.20, 0.10), "US": (0.12, 0.18, 0.26, 0.12)}
HYST_UP, HYST_DOWN = 0.54, 0.17

class SignalSet:
    def __init__(self, R, a_mkt_override=None, lag=0, gate_win=120):
        if a_mkt_override is not None:
            hs300 = a_mkt_override
        else:
            hs300 = pd.read_csv(_data("index_sh000300.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
        lvl = {}
        for s in SLOTS:
            lvl[s] = (1 + R[s].dropna()).cumprod()
        self.a_mkt = hs300.reindex(R.index).ffill()
        self.a_g = lvl["159952"].reindex(R.index).ffill()
        self.u_g = lvl["159941"].reindex(R.index).ffill()
        self.u_m = lvl["513500"].reindex(R.index).ffill()
        self.R = R
        self.rsrs = self._rsrs_panel(lvl, win=18)
        self.ind = self._ind_panel(lvl)
        self.lag = int(lag)  # 信号时点: 0=当日收盘, 1=前一日收盘(严格无未来, 实盘T日尾盘用T-1信号)
        self.gate_win = int(gate_win)
        self.sma = {}
        _win = sorted(set((20, 60, 120, self.gate_win)))
        for nm, x in [("a_mkt", self.a_mkt), ("a_g", self.a_g), ("u_g", self.u_g), ("u_m", self.u_m)]:
            self.sma[nm] = {w: x.rolling(w, min_periods=10).mean() for w in _win}
        # 性能缓存: 3年(756交易日)滚动窗口 max/min, 与 mkt_info 的 iloc[max(0,i-755):i+1] 逐点等价
        self.dd_max, self.dd_min = {}, {}
        for nm, x in (("a_mkt", self.a_mkt), ("u_m", self.u_m)):
            self.dd_max[nm] = x.rolling(756, min_periods=1).max()
            self.dd_min[nm] = x.rolling(756, min_periods=1).min()

    def _rsrs_panel(self, lvl, win=18):
        """RSRS(close-only版, 94fsckbzfd V8.7口径): 18日 价格-时间 滚动相关系数 r, r^3/std_x
        阈值0.05 ~ r≈0.64(18日强趋势); 与high/low无关, 纯close序列可算"""
        std_x = float(np.std(np.arange(win)))
        out = {}
        for s, l in lvl.items():
            l = l.dropna()
            x = pd.Series(np.arange(len(l)), index=l.index)
            c = l.rolling(win).corr(x)
            v = (c ** 3) / std_x
            v = v.where(np.isfinite(v), 0.0).reindex(self.R.index).ffill().fillna(0.0)
            out[s] = v
        return out

    def _ind_panel(self, lvl):
        """常用技术指标面板(close-only, 无high/low/volume数据; 全部归一化为强度0~1, 越大越强;
        中性值0.5; 早期样本fillna(0.5)避免误触发). 指标与价格均为滞后序列, 由_idx结构防线保证无未来"""
        out = {}
        for s, l in lvl.items():
            c = l.dropna()
            d = {}
            # MACD(12,26,9): DIF-DEA柱状图, 以120日std归一化后tanh映射
            e12 = c.ewm(span=12, adjust=False).mean()
            e26 = c.ewm(span=26, adjust=False).mean()
            dif = e12 - e26
            dea = dif.ewm(span=9, adjust=False).mean()
            hist = (dif - dea) * 2.0
            hsd = hist.rolling(120, min_periods=30).std()
            d["macd"] = 0.5 + 0.5 * np.tanh(hist / (2.0 * hsd + 1e-12))
            # RSI(14): 标准Wilder平滑
            delta = c.diff()
            ru = delta.clip(lower=0.0).ewm(alpha=1 / 14.0, adjust=False).mean()
            rd = (-delta).clip(lower=0.0).ewm(alpha=1 / 14.0, adjust=False).mean()
            rsi = 100.0 - 100.0 / (1.0 + ru / (rd + 1e-12))
            d["rsi14"] = (rsi / 100.0)
            # KDJ(9,3,3) close近似: RSV用close的HHV/LLV
            hhv = c.rolling(9, min_periods=3).max()
            llv = c.rolling(9, min_periods=3).min()
            rsv = ((c - llv) / (hhv - llv + 1e-12)).clip(0, 1) * 100.0
            k = rsv.ewm(alpha=1 / 3.0, adjust=False).mean()
            d["kdj"] = (k / 100.0)
            # BOLL %B(20,2)
            mid = c.rolling(20, min_periods=10).mean()
            sd = c.rolling(20, min_periods=10).std()
            d["boll_pctb"] = ((c - mid) / (2.0 * sd + 1e-12)).clip(0, 1)
            # CCI(14) close近似: TP=C
            sma14 = c.rolling(14, min_periods=7).mean()
            md = (c - sma14).abs().rolling(14, min_periods=7).mean()
            cci = (c - sma14) / (0.015 * md + 1e-12)
            d["cci14"] = 0.5 + 0.5 * np.tanh(cci / 200.0)
            # TRIX(12): 三重EMA变化率
            tr = c.ewm(span=12, adjust=False).mean()
            tr = tr.ewm(span=12, adjust=False).mean()
            tr = tr.ewm(span=12, adjust=False).mean()
            trix = tr.pct_change()
            d["trix12"] = 0.5 + 0.5 * np.tanh(trix * 100.0 / 2.0)
            # BIAS(24): 乖离率±10%映射
            ma24 = c.rolling(24, min_periods=12).mean()
            bias = (c - ma24) / (ma24 + 1e-12)
            d["bias24"] = 0.5 + 0.5 * np.tanh(bias / 0.10)
            # WILLR(14) close近似
            h14 = c.rolling(14, min_periods=7).max()
            l14 = c.rolling(14, min_periods=7).min()
            d["willr14"] = ((h14 - c) / (h14 - l14 + 1e-12)).clip(0, 1)
            # MOM20: 20日动量(参照系, 与绝对动量门控重叠度检验)
            d["mom20"] = 0.5 + 0.5 * np.tanh(c.pct_change(20) * 100.0 / 10.0)
            # ZSCORE60: 60日z-score(与市场刹车/布林带重叠度检验)
            ma60 = c.rolling(60, min_periods=30).mean()
            sd60 = c.rolling(60, min_periods=30).std()
            d["zscore60"] = 0.5 + 0.5 * np.tanh((c - ma60) / (sd60 + 1e-12) / 2.0)
            df = pd.DataFrame(d).reindex(self.R.index).ffill().fillna(0.5)
            out[s] = df
        return out

    def _idx(self, dt):
        # 结构防线(不可配置): 信号最小滞后1个交易日 —— 决策日当天收盘数据在盘中不可得,
        # 即使误配 lag=0 也无法读取当日数据; 审计脚本用"数据平移法"(R.shift(-1))模拟旧版
        # 泄漏口径做历史对照, 生产策略代码本身不存在泄漏路径
        # 性能: searchsorted(side="right")-1 与 get_indexer(method="ffill") 语义等价(<=dt 的末位)
        i = int(self.R.index.searchsorted(dt, side="right")) - 1
        return max(0, i - max(1, int(self.lag)))

    def score(self, dt):
        i = self._idx(dt)
        def gt(x, nm, win):
            return 1.0 if (i < len(x) and x.iloc[i] > self.sma[nm][win].iloc[i]) else 0.0
        comps = [gt(self.a_mkt, "a_mkt", 120), gt(self.a_mkt, "a_mkt", 60), gt(self.a_mkt, "a_mkt", 20),
                 gt(self.a_g, "a_g", 120), gt(self.a_g, "a_g", 60), gt(self.a_g, "a_g", 20),
                 gt(self.u_g, "u_g", 120), gt(self.u_m, "u_m", 120), gt(self.u_g, "u_g", 60)]
        return int(sum(comps))

    def mkt_info(self, dt, market, gate_win=120):
        i = self._idx(dt)
        x = self.a_mkt if market == "CN" else self.u_m
        if i >= len(x):
            return 0.0, True, True, 1.0, True
        nm = "a_mkt" if market == "CN" else "u_m"
        px = x.iloc[i]
        # 市场回撤用3年滚动窗口(避免长历史早期峰值失真); 预计算缓存避免逐日重复 O(755) 切片
        peak = self.dd_max[nm].iloc[i]
        trough = self.dd_min[nm].iloc[i]
        dd = float(px / peak - 1.0) if peak > 0 else 0.0
        s20 = self.sma[nm][20].iloc[i]; s60 = self.sma[nm][60].iloc[i]; sg = self.sma[nm][gate_win].iloc[i]
        rec = float((px - trough) / (peak - trough)) if peak > trough else 1.0
        return dd, bool(px > s20), bool(s20 > s60), rec, bool(px > sg)

class DynamicStrategy:
    SIG_CLASS = SignalSet  # 审计脚本可用子类注入"泄漏口径"信号集(仅审计对照, 生产代码无泄漏路径)

    def __init__(self, R_full, cfg=None, a_mkt_override=None):
        self.sig = self.SIG_CLASS(R_full, a_mkt_override=a_mkt_override,
                                  lag=cfg.get("signal_lag", 1) if cfg else 1,
                                  gate_win=cfg.get("gate_win", 120) if cfg else 120)
        self.R = R_full
        cfg = cfg or {}
        sm = cfg.get("state_map", STATE_MAP)
        self.state_map = {int(k): tuple(v) for k, v in sm.items()}
        self.min_cash = cfg.get("min_cash", MIN_CASH)
        self.max_eq = cfg.get("max_eq", MAX_EQ)
        self.market_dd = cfg.get("market_dd", MARKET_DD)
        self.hyst_up = cfg.get("hyst_up", HYST_UP)
        self.hyst_down = cfg.get("hyst_down", HYST_DOWN)
        self.gate_win = cfg.get("gate_win", 120)
        self.growth_split_bull = dict(cfg.get("growth_split_bull", GROWTH_SPLIT_BULL))
        self.growth_split_bear = dict(cfg.get("growth_split_bear", GROWTH_SPLIT_BEAR))
        self.defense_momentum = bool(cfg.get("defense_momentum", False))
        self.defense_momentum_multi = bool(cfg.get("defense_momentum_multi", False))
        self.defense_momentum_win = int(cfg.get("defense_momentum_win", DEFENSE_MOMENTUM_WIN))
        self.defense_momentum_t = float(cfg.get("defense_momentum_t", DEFENSE_MOMENTUM_T))
        self.defense_clamp = tuple(cfg.get("defense_clamp", DEFENSE_CLAMP))
        self.us_rotation = bool(cfg.get("us_rotation", False))
        self.us_split_clamp = tuple(cfg.get("us_split_clamp", US_SPLIT_CLAMP))
        self.valuation_gate = bool(cfg.get("valuation_gate", False))
        self.premium_gate = bool(cfg.get("premium_gate", False))
        self.premium_rotate = bool(cfg.get("premium_rotate", False))
        self.premium_tilt = bool(cfg.get("premium_tilt", False))
        self.premium_tilt_thr = float(cfg.get("premium_tilt_thr", 0.02))
        self.premium_tilt_cap = float(cfg.get("premium_tilt_cap", 0.05))
        self.premium_tilt_max = float(cfg.get("premium_tilt_max", 0.5))
        self.corr_risk = bool(cfg.get("corr_risk", False))
        self.corr_conditional = bool(cfg.get("corr_conditional", False))
        self.corr_risk_win = int(cfg.get("corr_risk_win", 60))
        self.corr_risk_thr = [float(x) for x in cfg.get("corr_risk_thr", [0.5, 0.65])]
        self.corr_risk_cut = [float(x) for x in cfg.get("corr_risk_cut", [0.85, 0.65])]
        self.recovery_ramp = bool(cfg.get("recovery_ramp", False))
        self.recovery_ramp_min = float(cfg.get("recovery_ramp_min", 0.5))
        self.score_confirm = int(cfg.get("score_confirm_weeks", 0))
        self.confirm_weekday = int(cfg.get("confirm_weekday", 2))  # 确认采样日默认周三(A/B: 周三优于周五, 与调仓日解耦)
        self._last_confirmed = None
        self.rsrs_gate = bool(cfg.get("rsrs_gate", False))
        self.rsrs_thr = float(cfg.get("rsrs_thr", 0.05))
        self.rsrs_cut = float(cfg.get("rsrs_cut", 0.5))
        self.rsrs_gate_all = bool(cfg.get("rsrs_gate_all", False))
        self.rsrs_defense = bool(cfg.get("rsrs_defense", False))
        self.rsrs_defense_t = float(cfg.get("rsrs_defense_t", 4.0))
        self.rsrs_defense_mix = float(cfg.get("rsrs_defense_mix", 0.5))
        self.ind_gate = bool(cfg.get("ind_gate", False))
        self.ind_gate_all = bool(cfg.get("ind_gate_all", False))
        self.ind_name = str(cfg.get("ind_name", "macd"))
        self.ind_thr = float(cfg.get("ind_thr", 0.5))
        self.ind_cut = float(cfg.get("ind_cut", 0.5))
        self.ind_defense = bool(cfg.get("ind_defense", False))
        self.ind_defense_t = float(cfg.get("ind_defense_t", 4.0))
        self.ind_defense_mix = float(cfg.get("ind_defense_mix", 0.5))
        self.hh_stop = bool(cfg.get("hh_stop", False))
        self.hh_win = int(cfg.get("hh_win", 20))
        self.hh_thr = float(cfg.get("hh_thr", 0.08))
        self.hh_cut = float(cfg.get("hh_cut", 0.5))
        self.cool_off_weeks = int(cfg.get("cool_off_weeks", 0))
        self.reentry_step = float(cfg.get("reentry_step", 0.0))  # 每周重新加仓权益上限增量(pp/week, 0=关闭)
        self.asset_sb = bool(cfg.get("asset_sb", False))  # 单资产速度刹车(某标的5日急跌->当日削该标的弹性)
        self.asset_sb_win = int(cfg.get("asset_sb_win", 5))
        self.asset_sb_thr = float(cfg.get("asset_sb_thr", 0.08))
        self.asset_sb_cut = float(cfg.get("asset_sb_cut", 0.5))
        self._cool_from = {}
        self._prev_target = {}
        self.speed_brake = bool(cfg.get("speed_brake", False))
        self.speed_brake_win = int(cfg.get("speed_brake_win", 5))
        self.speed_brake_thr = float(cfg.get("speed_brake_thr", -0.04))
        self.speed_brake_cut = float(cfg.get("speed_brake_cut", 0.6))
        self.speed_brake_recover = int(cfg.get("speed_brake_recover", 8))
        self.speed_brake_dd_thr = float(cfg.get("speed_brake_dd_thr", -0.03))
        self.speed_brake_mkt = bool(cfg.get("speed_brake_mkt", False))  # 分市场速度刹车(仅砍下跌市场, 0=关闭)
        self.speed_brake_mkt_thr = float(cfg.get("speed_brake_mkt_thr", -0.04))
        self.speed_brake_mkt_cut = float(cfg.get("speed_brake_mkt_cut", 0.5))
        self._sb_mkt_on = {}
        self._speed_brake_on = 0
        self.growth_rotation = bool(cfg.get("growth_rotation", False))
        self.growth_rotation_win = int(cfg.get("growth_rotation_win", 60))
        self.growth_rotation_t = float(cfg.get("growth_rotation_t", 2.0))
        self.growth_clamp = [float(x) for x in cfg.get("growth_clamp", [0.08, 0.60])]
        self.growth_iv = bool(cfg.get("growth_iv", False))
        self.growth_iv_win = int(cfg.get("growth_iv_win", 60))
        self.growth_iv_t = float(cfg.get("growth_iv_t", 1.0))
        self.vol_gate = bool(cfg.get("vol_gate", False))
        self.vol_gate_win = int(cfg.get("vol_gate_win", 20))
        self.vol_gate_bands = [float(x) for x in cfg.get("vol_gate_bands", [0.30, 0.40, 0.50])]
        self.vol_gate_cuts = [float(x) for x in cfg.get("vol_gate_cuts", [0.7, 0.4, 0.1])]
        self.adx_gate = bool(cfg.get("adx_gate", False))
        self.adx_win = int(cfg.get("adx_win", 14))
        self.adx_bands = [float(x) for x in cfg.get("adx_bands", [10.0, 15.0, 25.0])]
        self.adx_cuts = [float(x) for x in cfg.get("adx_cuts", [0.7, 0.85, 1.0])]
        self.breadth_gate = bool(cfg.get("breadth_gate", False))
        self.breadth_win = int(cfg.get("breadth_win", 20))
        self.breadth_thr = float(cfg.get("breadth_thr", 0.5))
        self.breadth_cut = float(cfg.get("breadth_cut", 0.7))
        self.reversal_filter = bool(cfg.get("reversal_filter", False))
        self.rev_thr = float(cfg.get("rev_thr", 0.08))
        self.rev_span = float(cfg.get("rev_span", 0.10))
        self.rev_min_k = float(cfg.get("rev_min_k", 0.4))
        self.gold_crisis = bool(cfg.get("gold_crisis", False))
        self.gold_dd = float(cfg.get("gold_dd", -0.05))
        self.gold_rec = float(cfg.get("gold_rec", -0.03))
        self.gold_pct = float(cfg.get("gold_pct", 0.5))
        self._adx = None
        self._breadth = None
        self._gold_on = False
        self.premium_thr = [float(x) for x in cfg.get("premium_thr", PREMIUM_THR)]
        self.premium_cut = [float(x) for x in cfg.get("premium_cut", PREMIUM_CUT)]
        self.premium_shift = int(cfg.get("premium_shift", 1))
        self._premium = self._load_premium_panel(R_full, self.premium_shift) if self.premium_gate else None
        self.valuation_win = int(cfg.get("valuation_win_days", VALUATION_WIN_DAYS))
        self.valuation_thr = [float(x) for x in cfg.get("valuation_thr", [0.95, 0.98])]
        self.valuation_cut = [float(x) for x in cfg.get("valuation_cut", [0.6, 0.35])]
        self._prev_eff = None
        self.am_gate = bool(cfg.get("am_gate", False))
        self.am_win = int(cfg.get("am_win", 120))
        self.am_cut = float(cfg.get("am_cut", 0.5))
        self.regime_gate = bool(cfg.get("regime_gate", False))
        self.regime_win = int(cfg.get("regime_win", 250))
        self.regime_hist = int(cfg.get("regime_hist", 250))
        self.regime_bands = [float(x) for x in cfg.get("regime_bands", [0.60, 0.75, 0.90])]
        self.regime_cuts = [float(x) for x in cfg.get("regime_cuts", [1.0, 0.8, 0.5, 0.25])]
        self._regime_pct = None
        if self.regime_gate:
            try:
                a = self.sig.a_mkt.pct_change().rolling(self.regime_win).std() * np.sqrt(252)
                self._regime_pct = a.rolling(self.regime_hist, min_periods=60).apply(
                    lambda x: float((x[-1] >= x).mean()), raw=True)
            except Exception:
                self._regime_pct = None
        fp = cfg.get("floor_pct", {"cn": 15.0, "us": 15.0})
        cn_f, us_f = float(fp["cn"]), float(fp["us"])
        self.floor = {"159232": cn_f / 2.0, "515100": cn_f / 2.0,
                      "159941": us_f / 2.0, "513500": us_f / 2.0, "159952": 0.0}
        self.exclude = set(cfg.get("exclude", []))
        for s in self.exclude:
            self.floor[s] = 0.0
        cn_f = sum(self.floor[s] for s in ["159232", "515100"])
        us_f = sum(self.floor[s] for s in ["159941", "513500"])
        self.floor_eq = cn_f + us_f
        # 成长拆分剔除被排除标的并归一化
        for nm in ("growth_split_bull", "growth_split_bear"):
            d = {k: v for k, v in getattr(self, nm).items() if k not in self.exclude}
            tot = sum(d.values())
            if tot > 0:
                d = {k: v / tot for k, v in d.items()}
            setattr(self, nm, d)
        ds = {k: v for k, v in DEFENSE_SPLIT.items() if k not in self.exclude}
        tot = sum(ds.values())
        self.defense_split_base = {k: v / tot for k, v in ds.items()} if tot > 0 else {}
        self._lock = {"CN": False, "US": False}
        self.state_log = []
        self.risk_log = []
        # ---- 性能缓存(仅加速, 与逐日重算数值等价; 配置为关时留空, 走原路径) ----
        self._corr_cache = None
        if self.corr_risk:
            _cn = self.R[["159232", "515100", "159952"]].fillna(0.0).mean(axis=1)
            _us = self.R[["159941", "513500"]].fillna(0.0).mean(axis=1)
            self._corr_cache = {"cn": _cn, "us": _us,
                                "corr": _cn.rolling(self.corr_risk_win).corr(_us),
                                "cn_std": _cn.rolling(self.corr_risk_win).std(),
                                "us_std": _us.rolling(self.corr_risk_win).std()}
        if self.adx_gate:
            _c = self.sig.a_mkt
            _d = _c.diff()
            _tr = _d.abs()
            _pdm = _d.clip(lower=0.0); _ndm = (-_d).clip(lower=0.0)
            def _wilder(s):
                return s.ewm(alpha=1.0 / self.adx_win, adjust=False).mean()
            _atr = _wilder(_tr)
            _pdi = 100.0 * _wilder(_pdm) / _atr.replace(0, np.nan)
            _ndi = 100.0 * _wilder(_ndm) / _atr.replace(0, np.nan)
            _dx = ((_pdi - _ndi).abs() / (_pdi + _ndi).replace(0, np.nan) * 100.0).fillna(0.0)
            self._adx = _wilder(_dx).reindex(self.R.index).ffill().fillna(0.0)
        if self.breadth_gate:
            _idx = ["index_sh000016.csv", "index_sh000300.csv", "index_sh000905.csv",
                    "index_sz399006.csv", "index_000922.csv", "index_930955.csv", "index_932365.csv"]
            _px = []
            for _f in _idx:
                try:
                    _df = pd.read_csv(_data(_f), parse_dates=["date"]).set_index("date")
                    _c = _df["close"] if "close" in _df.columns else _df["level"]
                    _px.append(_c.reindex(self.R.index).ffill())
                except Exception:
                    pass
            if len(_px) >= 3:
                _pxf = pd.concat(_px, axis=1)
                _sma = _pxf.rolling(self.breadth_win, min_periods=10).mean()
                self._breadth = (_pxf > _sma).astype(float).mean(axis=1)
        self._mom_cache = {}
        if self.defense_momentum:
            for _s in ("159232", "515100"):
                self._mom_cache[_s] = (1 + self.R[_s].fillna(0.0)).rolling(
                    self.defense_momentum_win).apply(np.prod, raw=True)
        self._pf_n = None; self._pf_wv = 0.0; self._pf_cmax = 1.0; self._pf_dd = 0.0

    @staticmethod
    def _load_premium_panel(R, shift=1):
        """QDII溢价=场内收盘/单位净值-1; 剔除份额折算过渡日, 裁剪极端值。
        shift=1: 决策日用前一日(T-1)溢价(场内价T-1已知; QDII单位净值T+1晚间才发布, 严格口径
        应取shift=2 -> 决策日可用T-2溢价, 两者均晚于T-1收盘价, 无未来函数)"""
        cols = {}
        for code, fn in PREMIUM_FILES.items():
            px = pd.read_csv(_data(fn), parse_dates=["date"]).set_index("date")["close"].sort_index()
            nav = pd.read_csv(_data(f"{code}_nav.csv"), parse_dates=["date"]).set_index("date")
            nav = nav[~nav.index.duplicated(keep="last")].sort_index()["unit_nav"]
            df = pd.concat([px.rename("px"), nav.rename("nav")], axis=1).dropna()
            p = df["px"] / df["nav"] - 1
            if code in PREMIUM_SPLIT_DROP:
                p = p.drop(pd.Timestamp(PREMIUM_SPLIT_DROP[code]), errors="ignore")
            cols[code] = p.clip(*PREMIUM_CLIP).reindex(R.index).ffill()
        out = pd.DataFrame(cols)
        return out.shift(shift)

    def _premium_at(self, dt):
        """返回决策日可用的前一日溢价(取两只QDII较高者, 保守); 无数据返回None"""
        if self._premium is None or dt not in self._premium.index:
            return None
        row = self._premium.loc[dt]
        vals = row.dropna()
        if len(vals) == 0:
            return None
        return float(vals.max())

    def _growth_split(self, dt, base):
        """论坛方向: 跨市场成长相对动量轮动(创业板/纳指/标普)
        按60日趋势强度(收盘/SMA20-1)的t次幂分配成长弹性, 单只clamp防止过度集中
        替代固定 growth_split_bull/bear"""
        if not self.growth_rotation:
            return base
        i = self.sig._idx(dt)
        if i < self.growth_rotation_win or i >= len(self.R):
            return base
        codes = [c for c in ["159952", "159941", "513500"] if c not in self.exclude]
        seg = self.R.iloc[i - self.growth_rotation_win + 1: i + 1]
        lvl = {c: float((1 + seg[c].fillna(0.0)).prod()) for c in codes}
        strengths = {}
        for c in codes:
            s = (1 + self.R[c].iloc[max(0, i-19): i + 1].fillna(0.0)).prod()
            strengths[c] = (s - 1.0) * 100.0  # 20日动量%
        w = {}
        tot = 0.0
        for c in codes:
            k = max(lvl[c], 1e-9) ** self.growth_rotation_t * max(1.0 + strengths[c] / 5.0, 0.1)
            w[c] = k
            tot += k
        lo, hi = self.growth_clamp
        out = {}
        for c in codes:
            out[c] = min(max(w[c] / tot, lo), hi)
        # 归一化到和 base 总成长弹性一致
        base_total = sum(base.get(c, 0.0) for c in codes)
        s = sum(out.values())
        if s > 0:
            for c in codes:
                out[c] = out[c] / s * base_total
        return out

    def _growth_split_iv(self, dt, base):
        """候选v31b-收益端: 成长桶内波动率倒数加权(替代固定growth_split)
        权重∝(1/σ)^t, σ=过去growth_iv_win日收益标准差(滚动, 决策日及以前, 无未来);
        保持base总成长弹性不变, clamp限制单只; 与前提权重冲突风险见探索记录"""
        i = self.sig._idx(dt)
        if i < self.growth_iv_win or i >= len(self.R):
            return base
        codes = [c for c in ["159952", "159941", "513500"] if c not in self.exclude]
        w, tot = {}, 0.0
        for c in codes:
            seg = self.R[c].iloc[i - self.growth_iv_win + 1: i + 1].fillna(0.0)
            sd = float(seg.std())
            if not np.isfinite(sd) or sd <= 1e-12:
                sd = 1e-12
            k = (1.0 / sd) ** self.growth_iv_t
            w[c] = k
            tot += k
        lo, hi = self.growth_clamp
        out = {}
        for c in codes:
            out[c] = min(max(w[c] / tot, lo), hi)
        base_total = sum(base.get(c, 0.0) for c in codes)
        s = sum(out.values())
        if s > 0:
            for c in codes:
                out[c] = out[c] / s * base_total
        return out

    def _adx_mult(self, dt):
        """候选v31c-回撤控制端: 市场趋势强度(close-only ADX, 沪深300)分档调弹性
        低ADX=震荡/无趋势 -> 降弹性, 高ADX=趋势 -> 维持/小幅加仓; 决策日及以前数据, 无未来"""
        if not self.adx_gate or self._adx is None:
            return 1.0
        i = self.sig._idx(dt)
        if i < 0 or i >= len(self._adx):
            return 1.0
        a = float(self._adx.iloc[i])
        if not np.isfinite(a):
            return 1.0
        for b, c in zip(self.adx_bands, self.adx_cuts):
            if a <= b:
                return c
        return self.adx_cuts[-1]

    def _breadth_mult(self, dt):
        """候选v31d-回撤控制端: 市场宽度(7个A股指数站上SMA占比<阈值 -> CN成长弹性降)
        学术依据: Momentum+Breadth+Correlation in TAA; 决策日及以前数据, 无未来"""
        if not self.breadth_gate or self._breadth is None:
            return 1.0
        i = self.sig._idx(dt)
        if i < 0 or i >= len(self._breadth):
            return 1.0
        b = float(self._breadth.iloc[i])
        if not np.isfinite(b):
            return 1.0
        return self.breadth_cut if b < self.breadth_thr else 1.0

    def _reversal_filter(self, dt, split_g):
        """候选v31c-收益端: 短期过度延伸降权(20日涨幅>rev_thr后线性惩罚至rev_min_k)
        仅作用于成长弹性并归一化保持总弹性; 决策日及以前数据, 无未来"""
        i = self.sig._idx(dt)
        if i < 20 or i >= len(self.R):
            return split_g
        codes = [c for c in ("159952", "159941", "513500") if c in split_g]
        k = {}
        for c in codes:
            r20 = float((1 + self.R[c].iloc[i - 19: i + 1].fillna(0.0)).prod() - 1.0)
            if r20 > self.rev_thr:
                k[c] = max(self.rev_min_k, 1.0 - (r20 - self.rev_thr) / self.rev_span)
            else:
                k[c] = 1.0
        out = {c: split_g[c] * k.get(c, 1.0) for c in codes}
        tot0 = sum(split_g.get(c, 0.0) for c in codes)
        tot1 = sum(out.values())
        if tot1 > 1e-12 and tot0 > 0:
            for c in codes:
                out[c] = out[c] / tot1 * tot0
        for c in split_g:
            out.setdefault(c, split_g[c])
        return out

    def gold_pct_fn(self):
        """候选v31c-现金层: 沪深300一年窗回撤<gold_dd时现金层gold_pct转518880黄金,
        收复gold_rec转回; 返回engine可调用的 现金层黄金占比 函数(无未来, 状态滞回防抖)
        1年窗回撤与策略3年窗市场回撤口径不同: 3年窗在中长趋势尾端长期<阈值, 不满足"危机"语义"""
        if not self.gold_crisis:
            return None
        def f(dt):
            try:
                i = self.sig._idx(dt)
                x = self.sig.a_mkt
                seg = x.iloc[max(0, i - 251): i + 1]
                px = float(x.iloc[i])
                dd = float(px / seg.max() - 1.0) if seg.max() > 0 else 0.0
            except Exception:
                return 0.0
            if dd is None or not np.isfinite(dd):
                return 0.0
            if dd < self.gold_dd:
                self._gold_on = True
            elif dd >= self.gold_rec:
                self._gold_on = False
            return self.gold_pct if self._gold_on else 0.0
        return f

    def _corr_mult(self, dt):
        """跨市场相关性风控: CN(三只A股等权) vs US(两只QDII等权) 60日滚动相关
        相关性越高 -> 跨市场共振下跌风险越大, 削减成长弹性(防御仓不动)"""
        if not self.corr_risk:
            return 1.0
        i = self.sig._idx(dt)
        if i < self.corr_risk_win or i >= len(self.R):
            return 1.0
        cc = self._corr_cache
        cn_std = float(cc["cn_std"].iloc[i])
        us_std = float(cc["us_std"].iloc[i])
        if cn_std < 1e-12 or us_std < 1e-12:
            return 1.0
        c = float(cc["corr"].iloc[i])
        if not np.isfinite(c):
            return 1.0
        # 条件化: 双市场均强势(收盘>60日均线)时豁免相关性折扣, 只在弱势/同跌风险期启用
        if self.corr_conditional:
            try:
                ok_cn = bool(self.sig.a_mkt.iloc[i] > self.sig.sma["a_mkt"][60].iloc[i])
                ok_us = bool(self.sig.u_m.iloc[i] > self.sig.sma["u_m"][60].iloc[i])
                if ok_cn and ok_us:
                    return 1.0
            except (IndexError, KeyError):
                pass
        cut = 1.0
        for thr, cc in zip(self.corr_risk_thr, self.corr_risk_cut):
            if c >= thr:
                cut = min(cut, cc)
        return cut

    def _recovery_mult(self, dt, market):
        """恢复度渐进: 深度回撤后按收复比例渐进恢复弹性, 避免V型反转过早满仓
        仅在回撤中(-20%~-5%)生效: 收复比例rec低 -> 弹性受限; 创新高(rec≈1) -> 完全恢复"""
        if not self.recovery_ramp:
            return 1.0
        dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, market, self.gate_win)
        if dd >= -0.05:
            return 1.0
        if dd <= -0.20:
            return 1.0  # 深跌区由快刹车/深熊锁处理
        return max(self.recovery_ramp_min, min(1.0, rec / 0.5))

    def _defense_split(self, dt):
        """国内防御双持轮动: 159232(自由现金流) vs 515100(红利低波100) 按相对动量分配防御弹性
        multi模式(Antonacci): 3/6/12月(63/126/252日)加权动量 0.3/0.3/0.4, 文献标准多窗口"""
        if not self.defense_momentum:
            return dict(self.defense_split_base)
        i = self.sig._idx(dt)
        if i < self.defense_momentum_win or i >= len(self.R):
            return dict(DEFENSE_SPLIT)
        if self.defense_momentum_multi:
            def mm(s):
                vals = {}
                for win in (63, 126, 252):
                    seg2 = self.R.iloc[max(0, i - win + 1): i + 1]
                    vals[win] = float((1 + seg2[s].fillna(0.0)).prod()) - 1.0
                return 0.3 * vals[63] + 0.3 * vals[126] + 0.4 * vals[252]
            k232 = max(1.0 + mm("159232"), 0.01) ** self.defense_momentum_t
            k100 = max(1.0 + mm("515100"), 0.01) ** self.defense_momentum_t
            w232 = k232 / (k232 + k100)
        elif self.rsrs_defense:
            rs232 = float(self.sig.rsrs["159232"].iloc[i])
            rs100 = float(self.sig.rsrs["515100"].iloc[i])
            if self.rsrs_defense_mix >= 1.0:
                k232 = max(rs232, 1e-4) ** self.rsrs_defense_t
                k100 = max(rs100, 1e-4) ** self.rsrs_defense_t
            else:
                seg = self.R.iloc[max(0, i - self.defense_momentum_win + 1): i + 1]
                g232 = float((1 + seg["159232"].fillna(0.0)).prod())
                g100 = float((1 + seg["515100"].fillna(0.0)).prod())
                mom232 = g232 ** self.defense_momentum_t
                mom100 = g100 ** self.defense_momentum_t
                k232 = (1 - self.rsrs_defense_mix) * mom232 + self.rsrs_defense_mix * max(rs232, 1e-4) ** self.rsrs_defense_t
                k100 = (1 - self.rsrs_defense_mix) * mom100 + self.rsrs_defense_mix * max(rs100, 1e-4) ** self.rsrs_defense_t
            w232 = k232 / (k232 + k100)
        elif self.ind_defense:
            ind232 = float(self.sig.ind["159232"][self.ind_name].iloc[i])
            ind100 = float(self.sig.ind["515100"][self.ind_name].iloc[i])
            if self.ind_defense_mix >= 1.0:
                k232 = max(ind232, 1e-4) ** self.ind_defense_t
                k100 = max(ind100, 1e-4) ** self.ind_defense_t
            else:
                seg = self.R.iloc[max(0, i - self.defense_momentum_win + 1): i + 1]
                g232 = float((1 + seg["159232"].fillna(0.0)).prod())
                g100 = float((1 + seg["515100"].fillna(0.0)).prod())
                mom232 = g232 ** self.defense_momentum_t
                mom100 = g100 ** self.defense_momentum_t
                k232 = (1 - self.ind_defense_mix) * mom232 + self.ind_defense_mix * max(ind232, 1e-4) ** self.ind_defense_t
                k100 = (1 - self.ind_defense_mix) * mom100 + self.ind_defense_mix * max(ind100, 1e-4) ** self.ind_defense_t
            w232 = k232 / (k232 + k100)
        else:
            # 60/80日滚动乘积缓存(与 iloc 窗口逐点等价)
            g232 = float(self._mom_cache["159232"].iloc[i])
            g100 = float(self._mom_cache["515100"].iloc[i])
            if g232 <= 0 or g100 <= 0:
                return dict(DEFENSE_SPLIT)
            w232 = g232 ** self.defense_momentum_t / (g232 ** self.defense_momentum_t + g100 ** self.defense_momentum_t)
        lo, hi = self.defense_clamp
        w232 = min(max(float(w232), lo), hi)
        return {"159232": w232, "515100": 1.0 - w232}

    def _momentum_ratio(self, dt, a, b):
        i = self.sig._idx(dt)
        if i < self.defense_momentum_win or i >= len(self.R):
            return None
        seg = self.R.iloc[i - self.defense_momentum_win + 1: i + 1]
        ga = float((1 + seg[a].fillna(0.0)).prod())
        gb = float((1 + seg[b].fillna(0.0)).prod())
        if ga <= 0 or gb <= 0:
            return None
        return ga ** self.defense_momentum_t / (ga ** self.defense_momentum_t + gb ** self.defense_momentum_t)

    def _us_split(self, dt):
        """海外腿内部轮动: 纳指(159941) vs 标普(513500) 按相对动量分配海外成长弹性"""
        if not self.us_rotation:
            return None
        if "159941" in self.exclude or "513500" in self.exclude:
            return None
        r = self._momentum_ratio(dt, "159941", "513500")
        if r is None:
            return None
        lo, hi = self.us_split_clamp
        r = min(max(float(r), lo), hi)
        return {"159941": r, "513500": 1.0 - r}

    def _valuation_gate(self, dt, market):
        """估值分位门控(价格分位代理): 标的分位>阈值时削减对应市场弹性"""
        if not self.valuation_gate:
            return 1.0
        code = "159941" if market == "US" else "159952"
        i = self.sig._idx(dt)
        if i < 260:
            return 1.0
        x = self.R[code].iloc[max(0, i - self.valuation_win + 1): i + 1].dropna()
        if len(x) < 500:
            return 1.0
        lvl = (1 + x).cumprod()
        pct = float((lvl <= lvl.iloc[-1]).mean())
        cut = 1.0
        for thr, c in zip(self.valuation_thr, self.valuation_cut):
            if pct >= thr:
                cut = min(cut, c)
        return cut

    def target_fn(self):
        def fn(dt, R_prev, ctx):
            return self.regular_target(dt, ctx)
        return fn

    def _pf_stats(self, pf):
        """组合净值回撤增量缓存: 引擎逐日追加1个收益, 与全量 cumprod/cummax 逐点等价;
        长度不连续(策略跨回测复用/异常调用)时自动回退全量重算"""
        if len(pf) == 0:
            return 0.0, 1.0
        n = len(pf)
        if self._pf_n is not None:
            if n == self._pf_n:
                return self._pf_dd, self._pf_wv
            if n == self._pf_n + 1:
                wv = self._pf_wv * (1.0 + float(pf.iloc[-1]))
                cm = self._pf_cmax if wv <= self._pf_cmax else wv
                self._pf_n, self._pf_wv, self._pf_cmax = n, wv, cm
                dd = wv / cm - 1.0 if cm > 0 else 0.0
                self._pf_dd = dd
                return dd, wv
        wv_s = (1 + pf).cumprod()
        wv = float(wv_s.iloc[-1])
        cm = float(wv_s.cummax().iloc[-1])
        dd = wv / cm - 1.0 if cm > 0 else 0.0
        self._pf_n, self._pf_wv, self._pf_cmax, self._pf_dd = n, wv, cm, dd
        return dd, wv

    def daily_fn(self):
        def fn(dt, R_prev, ctx):
            pf = ctx.get("pf_rets", pd.Series(dtype=float))
            if len(pf) < 40:
                return None
            dd, _ = self._pf_stats(pf)
            sc = self.sig.score(dt)
            if self.asset_sb and len(pf) >= 40:
                i = self.sig._idx(dt)
                if i >= self.asset_sb_win:
                    base = self._base_target(dt, sc)
                    mod = False
                    for s in SLOTS:
                        seg = self.R.iloc[i - self.asset_sb_win + 1: i + 1][s]
                        if len(seg.dropna()) >= self.asset_sb_win:
                            r5 = float((1 + seg.fillna(0.0)).prod() - 1.0)
                            if r5 < -self.asset_sb_thr:
                                base[s] = self.floor[s] + (base[s] - self.floor[s]) * self.asset_sb_cut
                                mod = True
                    if mod:
                        return self._finalize(base)
            # 分市场速度刹车: 仅对5日急跌且回撤中的市场削减弹性(不误伤未跌市场), 恢复期同组合刹车
            if self.speed_brake_mkt:
                for mkt, codes in (("CN", ["159232", "515100", "159952"]), ("US", ["159941", "513500"])):
                    i = self.sig._idx(dt)
                    if i >= self.speed_brake_win and i < len(self.R):
                        seg = self.R.iloc[i - self.speed_brake_win + 1: i + 1][codes]
                        rn_m = float((1 + seg.fillna(0.0)).mean(axis=1).prod() - 1.0)
                        dd_m, _, _, _, _ = self.sig.mkt_info(dt, mkt, self.gate_win)
                        if rn_m < self.speed_brake_mkt_thr and dd_m < self.speed_brake_dd_thr:
                            self._sb_mkt_on[mkt] = self.speed_brake_recover
                        if self._sb_mkt_on.get(mkt, 0) > 0:
                            self._sb_mkt_on[mkt] -= 1
                            return self.target_with_mkt_cut(dt, ctx, mkt, self.speed_brake_mkt_cut)
            # 速度刹车: 组合急跌(近N日)且处于回撤中 -> 日度减成长弹性, 反弹或N日后自动恢复
            if self.speed_brake and len(pf) >= self.speed_brake_win:
                rn = float((1 + pf.iloc[-self.speed_brake_win:]).prod() - 1.0)
                if rn < self.speed_brake_thr and dd < self.speed_brake_dd_thr:
                    self._speed_brake_on = self.speed_brake_recover
                if self._speed_brake_on > 0:
                    self._speed_brake_on -= 1
                    base = self._base_target(dt, sc)
                    cur = float(ctx.get("equity", 1.0))
                    eq_cap = cur * self.speed_brake_cut
                    return self.target_with_eq_cap(dt, ctx, eq_cap * 100)
            # 市场级风控: 目标权益比当前显著低(深熊锁/快刹车/结构闸门生效) -> 日度触发, 仍分3周三笔
            base = self._base_target(dt, sc)
            eq_tgt = sum(base.values()) / 100.0
            cur = float(ctx.get("equity", 1.0))
            if eq_tgt <= cur - 0.04:
                return self._finalize(base)
            return None
        return fn

    def _eff_score(self, dt):
        sc = self.sig.score(dt)
        if self._prev_eff is None:
            self._prev_eff = float(sc)
        else:
            w = self.hyst_up if sc > self._prev_eff else self.hyst_down
            self._prev_eff = w * sc + (1 - w) * self._prev_eff
        return int(round(self._prev_eff))

    def _market_mult(self, dt, market):
        """返回 (成长弹性乘数, 全弹性乘数)"""
        dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, market, self.gate_win)
        fb_thr, fb_cut, deep_thr, recov = self.market_dd[market]
        m_g, m_all = 1.0, 1.0
        # 深熊锁定: 该市场全部弹性清零
        if self._lock[market]:
            if dd > -recov or (ok120 and ok20_60):
                self._lock[market] = False
            else:
                return 0.0, 0.0
        if dd < -deep_thr:
            self._lock[market] = True
            return 0.0, 0.0
        # 结构闸门
        if not ok120:
            m_g = 0.0
        # 快刹车
        if dd < -fb_thr and not (ok20 and ok20_60):
            m_g = min(m_g, fb_cut)
        return m_g, m_all

    def _confirm_score(self, dt, sc):
        """向上调仓需连续N个确认日确认, 向下立即生效(防假突破/降换手); 采样日默认周三
    与调仓日(周五)解耦 = 信号比调仓日早2个交易日, 等效额外保守滞后; 2026-08-04 A/B实测
    周三采样 proxy Cal1.57/real Cal6.44 优于周五采样 1.34/5.83, 保持周三"""
        raw = self.sig.score(dt)
        if self._last_confirmed is None:
            self._last_confirmed = sc
            return sc
        if sc <= self._last_confirmed:
            self._last_confirmed = min(self._last_confirmed, sc)
            return sc
        idx = self.R.index
        i = self.sig._idx(dt)
        ws = []
        j = i
        while len(ws) < self.score_confirm and j >= 0:
            if idx[j].weekday() == self.confirm_weekday:
                ws.append(self.sig.score(idx[j]))
            j -= 1
        L = sc
        while L > self._last_confirmed:
            if all(r >= L for r in ws):
                self._last_confirmed = L
                return L
            L -= 1
        return self._last_confirmed

    def _base_target(self, dt, sc):
        g, d = self.state_map.get(sc, self.state_map[min(max(sc, 0), 9)])
        split_g = dict(self.growth_split_bull if g >= 30 else self.growth_split_bear)
        split_d = self._defense_split(dt)
        for s in self.exclude:
            split_g.setdefault(s, 0.0)
            split_d.setdefault(s, 0.0)
        if self.growth_rotation:
            split_g = self._growth_split(dt, {s: split_g.get(s, 0.0) for s in ("159952", "159941", "513500")})
        if self.growth_iv:
            split_g = self._growth_split_iv(dt, {s: split_g.get(s, 0.0) for s in ("159952", "159941", "513500")})
        if self.reversal_filter:
            split_g = self._reversal_filter(dt, split_g)
        us_split = self._us_split(dt)
        vg_us = self._valuation_gate(dt, "US")
        vg_cn = self._valuation_gate(dt, "CN")
        prem_cut = 1.0
        if self._premium is not None:
            pm = self._premium_at(dt)
            if pm is not None and pm > 0:
                for thr, c in zip(self.premium_thr, self.premium_cut):
                    if pm >= thr:
                        prem_cut = min(prem_cut, c)
        corr_cut = self._corr_mult(dt)
        m_adx = self._adx_mult(dt)
        m_breadth = self._breadth_mult(dt)
        out = {
            "159232": self.floor["159232"] + split_d["159232"] * d,
            "515100": self.floor["515100"] + split_d["515100"] * d,
            "159941": self.floor["159941"] + split_g["159941"] * g,
            "513500": self.floor["513500"] + split_g["513500"] * g,
            "159952": self.floor["159952"] + split_g["159952"] * g,
        }
        m_cn_g, m_cn_all = self._market_mult(dt, "CN")
        m_us_g, m_us_all = self._market_mult(dt, "US")
        # CN弹性: 成长仓受快刹车+闸门; 防御弹性只受深熊锁定
        out["159952"] = self.floor["159952"] + (out["159952"] - self.floor["159952"]) * m_cn_g * m_cn_all * vg_cn * corr_cut * m_adx * m_breadth
        out["159232"] = self.floor["159232"] + (out["159232"] - self.floor["159232"]) * m_cn_all
        out["515100"] = self.floor["515100"] + (out["515100"] - self.floor["515100"]) * m_cn_all
        if us_split is not None:
            us_total = (out["159941"] - self.floor["159941"] + out["513500"] - self.floor["513500"]) * m_us_g * m_us_all * vg_us * prem_cut * corr_cut
            out["159941"] = self.floor["159941"] + us_total * us_split["159941"]
            out["513500"] = self.floor["513500"] + us_total * us_split["513500"]
        else:
            prem_freed = 0.0
            if self.premium_rotate and prem_cut < 1.0:
                prem_freed = (1.0 - prem_cut) * (out["159941"] - self.floor["159941"] + out["513500"] - self.floor["513500"]) * m_us_g * m_us_all * vg_us * corr_cut
            out["159941"] = self.floor["159941"] + (out["159941"] - self.floor["159941"]) * m_us_g * m_us_all * vg_us * prem_cut * corr_cut
            out["513500"] = self.floor["513500"] + (out["513500"] - self.floor["513500"]) * m_us_g * m_us_all * vg_us * prem_cut * corr_cut
            if prem_freed > 0:
                out["159952"] = self.floor["159952"] + (out["159952"] - self.floor["159952"]) * m_cn_g * m_cn_all * vg_cn * corr_cut + prem_freed * m_cn_g * m_cn_all
        if self.premium_tilt and self._premium is not None and dt in self._premium.index:
            _pr = self._premium.loc[dt]
            _p941, _p500 = _pr.get("159941"), _pr.get("513500")
            if pd.notna(_p941) and pd.notna(_p500):
                _diff = float(_p941) - float(_p500)
                _span = max(self.premium_tilt_cap - self.premium_tilt_thr, 1e-9)
                if _diff > self.premium_tilt_thr:
                    _frac = min(self.premium_tilt_max, (_diff - self.premium_tilt_thr) / _span)
                    _sh = (out["159941"] - self.floor["159941"]) * _frac
                    out["159941"] -= _sh; out["513500"] += _sh
                elif _diff < -self.premium_tilt_thr:
                    _frac = min(self.premium_tilt_max, (-_diff - self.premium_tilt_thr) / _span)
                    _sh = (out["513500"] - self.floor["513500"]) * _frac
                    out["513500"] -= _sh; out["159941"] += _sh
        for s in self.exclude:
            out[s] = 0.0
        if self.am_gate:
            i = self.sig._idx(dt)
            if i >= self.am_win:
                seg = self.R.iloc[i - self.am_win + 1: i + 1]
                for s in ("159952", "159941", "513500"):
                    g = float((1 + seg[s].fillna(0.0)).prod()) - 1.0
                    if g < 0:
                        out[s] = self.floor[s] + (out[s] - self.floor[s]) * self.am_cut
        if self.hh_stop:
            i = self.sig._idx(dt)
            if i >= self.hh_win:
                for s in ("159952", "159941", "513500"):
                    seg2 = self.R.iloc[i - self.hh_win + 1: i + 1]
                    lvl = (1 + seg2[s].fillna(0.0)).cumprod()
                    dd = float(lvl.iloc[-1] / lvl.max() - 1.0)
                    if dd < -self.hh_thr:
                        out[s] = self.floor[s] + (out[s] - self.floor[s]) * self.hh_cut
        if self.rsrs_gate:
            i = self.sig._idx(dt)
            targets = ("159232", "515100", "159952", "159941", "513500") if self.rsrs_gate_all else ("159952", "159941", "513500")
            for s in targets:
                if float(self.sig.rsrs[s].iloc[i]) < self.rsrs_thr:
                    out[s] = self.floor[s] + (out[s] - self.floor[s]) * self.rsrs_cut
        if self.ind_gate:
            i = self.sig._idx(dt)
            targets = ("159232", "515100", "159952", "159941", "513500") if self.ind_gate_all else ("159952", "159941", "513500")
            for s in targets:
                if float(self.sig.ind[s][self.ind_name].iloc[i]) < self.ind_thr:
                    out[s] = self.floor[s] + (out[s] - self.floor[s]) * self.ind_cut
        return out

    def regular_target(self, dt, ctx):
        sc = self._eff_score(dt)
        if self.score_confirm > 0:
            sc = self._confirm_score(dt, sc)
        self.state_log.append((dt, sc))
        out = self._base_target(dt, sc)
        pf = ctx.get("pf_rets", pd.Series(dtype=float))
        scale = 1.0
        if len(pf) > 30:
            self._pf_stats(pf)
            if self.vol_gate:
                i = self.sig._idx(dt)
                if i >= self.vol_gate_win:
                    seg = self.R["159952"].iloc[i - self.vol_gate_win + 1: i + 1].fillna(0.0)
                    sd = float(seg.std())
                    if np.isfinite(sd) and sd > 0:
                        v_ann = sd * np.sqrt(252)
                        cut = 1.0
                        for b, c in zip(self.vol_gate_bands, self.vol_gate_cuts):
                            if v_ann >= b:
                                cut = min(cut, c)
                        if cut < 1.0:
                            scale = min(scale, cut)
        if scale < 1.0:
            for s in SLOTS:
                out[s] = self.floor[s] + (out[s] - self.floor[s]) * scale
        if self.cool_off_weeks > 0:
            i = self.sig._idx(dt)
            dti = self.R.index[i] if i < len(self.R) else dt
            for s in SLOTS:
                cf = self._cool_from.get(s)
                if cf is not None:
                    if (dti - cf).days >= self.cool_off_weeks * 7:
                        self._cool_from.pop(s, None)
                    else:
                        out[s] = self.floor[s]
            for s in SLOTS:
                prev = self._prev_target.get(s, self.floor[s])
                if prev > self.floor[s] + 1e-6 and out[s] <= self.floor[s] + 1e-6:
                    self._cool_from.setdefault(s, dti)
            self._prev_target = dict(out)
        if self.regime_gate and self._regime_pct is not None:
            i = self.sig._idx(dt)
            if i >= 0 and not np.isnan(self._regime_pct.iloc[i]):
                p = float(self._regime_pct.iloc[i])
                cut = self.regime_cuts[0]
                for b, c in zip(self.regime_bands, self.regime_cuts[1:]):
                    if p >= b:
                        cut = c
                if cut < 1.0:
                    for s in SLOTS:
                        out[s] = self.floor[s] + (out[s] - self.floor[s]) * cut
                    scale = min(scale, cut)
        if self.reentry_step > 0:
            cur = float(ctx.get("equity", 1.0))
            tgt_eq = sum(out[s] for s in SLOTS) / 100.0
            if tgt_eq > cur + self.reentry_step:
                total_flex = sum(out[s] - self.floor[s] for s in SLOTS)
                scale2 = ((cur + self.reentry_step) * 100.0 - self.floor_eq) / max(total_flex, 1e-9)
                if scale2 < 1.0:
                    for s in SLOTS:
                        out[s] = self.floor[s] + (out[s] - self.floor[s]) * max(scale2, 0.0)
                self.risk_log.append((dt, sc, scale))
        return self._finalize(out)

    def target_with_mkt_cut(self, dt, ctx, mkt, cut):
        """分市场刹车: 仅削减指定市场 floor 以上的弹性, 另一市场与防御仓不动"""
        sc = self._eff_score(dt)
        out = self._base_target(dt, sc)
        codes = ["159952"] if mkt == "CN" else ["159941", "513500"]
        for s in codes:
            out[s] = self.floor[s] + (out[s] - self.floor[s]) * cut
        return self._finalize(out)

    def target_with_eq_cap(self, dt, ctx, cap_pct):
        sc = self._eff_score(dt)
        out = self._base_target(dt, sc)
        total_flex = sum(out[s] - self.floor[s] for s in SLOTS)
        scale = max(0.0, (cap_pct - self.floor_eq) / max(total_flex, 1e-9))
        if scale < 1.0:
            for s in SLOTS:
                out[s] = self.floor[s] + (out[s] - self.floor[s]) * scale
        return self._finalize(out)

    def _finalize(self, out):
        total = sum(out.values())
        cash = 100.0 - total
        cash = max(cash, self.min_cash * 100)
        if 100 - cash > self.max_eq * 100:
            cash = (1 - self.max_eq) * 100
        eq = 100.0 - cash
        if total <= 1e-9:  # 底仓=0且弹性=0(极端深熊全锁)
            for s in SLOTS:
                out[s] = 0.0
        else:
            for s in SLOTS:
                out[s] = out[s] * (eq / total) / 100.0
        out["cash"] = cash / 100.0
        return out
