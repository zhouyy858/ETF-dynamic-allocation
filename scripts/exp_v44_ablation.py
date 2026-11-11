# -*- coding: utf-8 -*-
"""v44 预注册消融（官方引擎）。

只调用 scripts/engine.py 的 run_backtest + evaluate，信号与成交沿用 DynamicStrategy / pnl_panel。
不下载、不刷新行情。窗口终点写死 2026-11-10。

A. 现有规则逐项及 v40 三处改动的小组合。
B. 预注册的新机制（未在禁用清单里，且不是已证伪的速度刹车 / recovery_ramp）。
   recovery_ramp 的函数在当前代码里没有调用点，本轮不重接。
   speed_brake 维持关闭，不重开旧参数路径。

三关：proxy / real / OOS 的 Calmar 都不低于 v40，且 proxy |MDD| < 10%。
"""
import copy, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate, SLOTS
from strategy import DynamicStrategy
from swaps import pnl_panel

END = "2026-11-10"
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out")
os.makedirs(OUT, exist_ok=True)

PERIODS = {
    "2015股灾": ("2015-06-15", "2016-02-29"),
    "2018熊市": ("2018-01-02", "2019-01-03"),
    "2019-2021牛市": ("2019-01-04", "2021-02-18"),
    "2021-2022熊市": ("2021-02-19", "2022-10-31"),
    "2024Q4-2026双牛": ("2024-09-02", END),
}

# 审计引用的 v40 官方数字（PR #15，同一窗口）
EXPECT = {
    "proxy": (13.3536, -7.8264, 1.4196, 1.70624),
    "real": (27.5054, -4.1913, 2.4141, 6.56249),
    "oos": (12.0679, -7.4225, 1.3905, 1.62586),
}


class RuleSet(DynamicStrategy):
    """与 DynamicStrategy 同路径。_exp 为空时结构闸门 / 深熊锁 / 快刹车逐行等价。"""

    def __init__(self, R_full, cfg=None, a_mkt_override=None):
        self._x = dict((cfg or {}).get("_exp") or {})
        super().__init__(R_full, cfg=cfg, a_mkt_override=a_mkt_override)

    def _market_mult(self, dt, market):
        dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, market, self.gate_win)
        fb_thr, fb_cut, deep_thr, recov = self.market_dd[market]
        m_g, m_all = 1.0, 1.0
        deep_scope = self._x.get("deep_scope", "all")
        unlock = self._x.get("unlock", "either")
        if deep_scope != "off":
            if self._lock[market]:
                dd_ok = dd > -recov
                trend_ok = bool(ok120 and ok20_60)
                if unlock == "dd":
                    release = dd_ok
                elif unlock == "trend":
                    release = trend_ok
                else:
                    release = dd_ok or trend_ok
                if release:
                    self._lock[market] = False
                else:
                    return (0.0, 1.0) if deep_scope == "growth" else (0.0, 0.0)
            if dd < -deep_thr:
                self._lock[market] = True
                return (0.0, 1.0) if deep_scope == "growth" else (0.0, 0.0)
        structure = self._x.get("structure", "binary")
        if structure == "off":
            pass
        elif structure == "soft":
            band = float(self._x.get("soft_band", 0.04))
            i = self.sig._idx(dt)
            x = self.sig.a_mkt if market == "CN" else self.sig.u_m
            nm = "a_mkt" if market == "CN" else "u_m"
            px = float(x.iloc[i])
            sg = float(self.sig.sma[nm][self.gate_win].iloc[i])
            if not (sg > 0) or px != px:
                m_g = 0.0
            else:
                m_g = min(1.0, max(0.0, (px / sg - (1.0 - band)) / band))
        else:
            gst = self._gate_state(dt, market) if self.gate_band > 0 else ok120
            if not gst:
                m_g = 0.0
        if self._x.get("fast", True):
            if dd < -fb_thr and not (ok20 and ok20_60):
                m_g = min(m_g, fb_cut)
                if self._x.get("fast_all"):
                    m_all = min(m_all, fb_cut)
        return m_g, m_all


class Tilt(RuleSet):
    """两市都在闸门之上时，按 px/SMA 差距把固定比例的成长预算从弱的一侧挪到强的一侧。
    总成长预算不变。只改 bull 拆分，bear 拆分不动。信号用 _idx，与结构闸门同一根 K 线。"""

    def _base_target(self, dt, sc):
        shift = float(self._x.get("tilt", 0.0))
        thr = float(self._x.get("tilt_thr", 0.05))
        saved = dict(self.growth_split_bull)
        if shift > 0:
            i = self.sig._idx(dt)
            cn_sma = float(self.sig.sma["a_mkt"][self.gate_win].iloc[i])
            us_sma = float(self.sig.sma["u_m"][self.gate_win].iloc[i])
            cn = float(self.sig.a_mkt.iloc[i]) / cn_sma if cn_sma > 0 else 1.0
            us = float(self.sig.u_m.iloc[i]) / us_sma if us_sma > 0 else 1.0
            gap = us - cn
            bull = dict(saved)
            if gap > thr:
                move = min(bull.get("159952", 0.0), shift)
                us_sum = bull.get("159941", 0.0) + bull.get("513500", 0.0)
                if move > 0 and us_sum > 0:
                    bull["159952"] = bull.get("159952", 0.0) - move
                    bull["159941"] = bull.get("159941", 0.0) + move * bull.get("159941", 0.0) / us_sum
                    bull["513500"] = bull.get("513500", 0.0) + move * bull.get("513500", 0.0) / us_sum
                    self.growth_split_bull = bull
            elif gap < -thr:
                us_sum = bull.get("159941", 0.0) + bull.get("513500", 0.0)
                move = min(us_sum, shift)
                if move > 0 and us_sum > 0:
                    bull["159941"] = bull.get("159941", 0.0) - move * bull.get("159941", 0.0) / us_sum
                    bull["513500"] = bull.get("513500", 0.0) - move * bull.get("513500", 0.0) / us_sum
                    bull["159952"] = bull.get("159952", 0.0) + move
                    self.growth_split_bull = bull
        try:
            return super()._base_target(dt, sc)
        finally:
            self.growth_split_bull = saved


class AsymDelta(RuleSet):
    """减仓腿不设缺口门槛；加仓腿仍要单槽 +3.5pp 才执行。引擎 min_delta 由用例设为 0。"""

    def regular_target(self, dt, ctx):
        out = super().regular_target(dt, ctx)
        w = ctx.get("weights")
        if w is None:
            return out
        up_min = float(self._x.get("up_min", 0.035)) * 100.0
        changed = False
        for i, s in enumerate(SLOTS):
            cur = float(w[i]) * 100.0
            if out[s] > cur + 1e-9 and (out[s] - cur) < up_min:
                out[s] = cur
                changed = True
        if changed:
            out = self._finalize(out)
            self.decisions[__import__("pandas").Timestamp(dt).normalize()] = {k: float(v) for k, v in out.items()}
        return out


def load_base():
    return json.load(open(os.path.join(REF, "final_cfg_v40.json")))


def clone(cfg, **kw):
    c = copy.deepcopy(cfg)
    exp = dict(c.get("_exp") or {})
    for k, v in kw.items():
        if k.startswith("x_"):
            exp[k[2:]] = v
        else:
            c[k] = v
    if exp:
        c["_exp"] = exp
    return c


def set_cap(cfg, rows):
    c = copy.deepcopy(cfg)
    for k, g in rows.items():
        c["state_map"][str(k)] = [g, c["state_map"][str(k)][1]]
    return c


def set_split(cfg, cn, nq, spx=0.07):
    c = copy.deepcopy(cfg)
    c["growth_split_bull"] = {"159952": cn, "159941": nq, "513500": spx}
    return c


def set_dd(cfg, edits):
    """edits: [(market, index, value), ...]"""
    c = copy.deepcopy(cfg)
    for mkt, idx, val in edits:
        row = list(c["market_dd"][mkt])
        row[idx] = val
        c["market_dd"][mkt] = row
    return c


def build_cases(base):
    cases = [("A00_v40", base, RuleSet)]
    # --- v40 三处改动：上限乘数、bull 拆分、冷却 ---
    cases += [
        ("A01_cap_x1.00", set_cap(base, {7: 65, 8: 85, 9: 81}), RuleSet),
        ("A02_cap_x1.05", set_cap(base, {7: 68.25, 8: 89.25, 9: 85.05}), RuleSet),
        ("A03_cap_x1.15", set_cap(base, {7: 74.75, 8: 97.75, 9: 93.15}), RuleSet),
        ("A04_split_30_63_7", set_split(base, 0.30, 0.63), RuleSet),
        ("A05_split_31_62_7", set_split(base, 0.31, 0.62), RuleSet),
        ("A06_split_33_60_7", set_split(base, 0.33, 0.60), RuleSet),
        ("A07_split_34_59_7", set_split(base, 0.34, 0.59), RuleSet),
    ]
    for w in (0, 1, 2, 4, 5, 6, 8):
        cases.append((f"A08_cool_{w}", clone(base, cool_off_weeks=w), RuleSet))
    # 三处改动的小组合（缺一角）
    cases += [
        ("A09_drop_cap_and_split", set_split(set_cap(base, {7: 65, 8: 85, 9: 81}), 0.30, 0.63), RuleSet),
        ("A10_drop_cap_and_cool", clone(set_cap(base, {7: 65, 8: 85, 9: 81}), cool_off_weeks=0), RuleSet),
        ("A11_drop_split_and_cool", clone(set_split(base, 0.30, 0.63), cool_off_weeks=0), RuleSet),
        ("A12_drop_all_three_v40", clone(set_split(set_cap(base, {7: 65, 8: 85, 9: 81}), 0.30, 0.63), cool_off_weeks=0), RuleSet),
    ]
    # --- 打分 / 动量映射 ---
    cases += [
        ("A13_defense_mom_off", clone(base, defense_momentum=False), RuleSet),
        ("A14_defmom_win60", clone(base, defense_momentum_win=60), RuleSet),
        ("A15_defmom_win100", clone(base, defense_momentum_win=100), RuleSet),
        ("A16_defmom_t2", clone(base, defense_momentum_t=2.0), RuleSet),
        ("A17_defmom_t6", clone(base, defense_momentum_t=6.0), RuleSet),
        ("A18_confirm_0", clone(base, score_confirm_weeks=0), RuleSet),
        ("A19_confirm_1", clone(base, score_confirm_weeks=1), RuleSet),
        ("A20_confirm_3", clone(base, score_confirm_weeks=3), RuleSet),
        ("A21_hyst_up_0.50", clone(base, hyst_up=0.50), RuleSet),
        ("A22_hyst_up_0.80", clone(base, hyst_up=0.80), RuleSet),
        ("A23_hyst_down_0.16", clone(base, hyst_down=0.16), RuleSet),
        ("A24_hyst_down_0.36", clone(base, hyst_down=0.36), RuleSet),
        ("A25_hyst_off", clone(base, hyst_up=1.0, hyst_down=1.0), RuleSet),
        ("A26_state9_eq_state8", set_cap(base, {9: 93.5}), RuleSet),
        ("A27_state8_eq_state9", set_cap(base, {8: 89.1}), RuleSet),
    ]
    # --- 底仓 / min_delta ---
    cases += [
        ("A28_floor_cn0", clone(base, floor_pct={"cn": 0.0, "us": 10.0}), RuleSet),
        ("A29_floor_cn4", clone(base, floor_pct={"cn": 4.0, "us": 10.0}), RuleSet),
        ("A30_floor_us0", clone(base, floor_pct={"cn": 2.0, "us": 0.0}), RuleSet),
        ("A31_floor_us5", clone(base, floor_pct={"cn": 2.0, "us": 5.0}), RuleSet),
        ("A32_floor_us15", clone(base, floor_pct={"cn": 2.0, "us": 15.0}), RuleSet),
        ("A33_floor_zero", clone(base, floor_pct={"cn": 0.0, "us": 0.0}), RuleSet),
        ("A34_mindelta_0", clone(base, min_delta=0.0), RuleSet),
        ("A35_mindelta_0.02", clone(base, min_delta=0.02), RuleSet),
        ("A36_mindelta_0.05", clone(base, min_delta=0.05), RuleSet),
        ("A37_mindelta_0.08", clone(base, min_delta=0.08), RuleSet),
    ]
    # --- 五层里可关的：溢价、相关、结构、快刹车、深熊 ---
    cases += [
        ("A38_premium_gate_off", clone(base, premium_gate=False), RuleSet),
        ("A39_premium_tilt_off", clone(base, premium_tilt=False), RuleSet),
        ("A40_premium_both_off", clone(base, premium_gate=False, premium_tilt=False), RuleSet),
        ("A41_premium_only_8pct_cut0.45", clone(base, premium_thr=[0.08], premium_cut=[0.45]), RuleSet),
        ("A42_premium_only_8pct_cut0.22", clone(base, premium_thr=[0.08], premium_cut=[0.22]), RuleSet),
        ("A43_premium_only_8pct_cut0.08", clone(base, premium_thr=[0.08], premium_cut=[0.08]), RuleSet),
        ("A44_corr_off", clone(base, corr_risk=False), RuleSet),
        ("A45_corr_win30", clone(base, corr_risk_win=30), RuleSet),
        ("A46_corr_win60", clone(base, corr_risk_win=60), RuleSet),
        ("A47_corr_cut_mild", clone(base, corr_risk_cut=[0.9, 0.75]), RuleSet),
        ("A48_corr_cut_hard", clone(base, corr_risk_cut=[0.7, 0.5]), RuleSet),
        ("A49_structure_off", clone(base, x_structure="off"), RuleSet),
        ("A50_fast_off", set_dd(base, [("CN", 0, 9.0), ("US", 0, 9.0)]), RuleSet),
        ("A51_cn_fb_0.05", set_dd(base, [("CN", 0, 0.05)]), RuleSet),
        ("A52_cn_fb_0.10", set_dd(base, [("CN", 0, 0.10)]), RuleSet),
        ("A53_us_fb_0.10", set_dd(base, [("US", 0, 0.10)]), RuleSet),
        ("A54_us_fb_0.16", set_dd(base, [("US", 0, 0.16)]), RuleSet),
        ("A55_deep_off", set_dd(base, [("CN", 2, 9.0), ("US", 2, 9.0)]), RuleSet),
        ("A56_cn_deep_0.15", set_dd(base, [("CN", 2, 0.15)]), RuleSet),
        ("A57_cn_deep_0.21", set_dd(base, [("CN", 2, 0.21)]), RuleSet),
        ("A58_cn_deep_0.24", set_dd(base, [("CN", 2, 0.24)]), RuleSet),
        ("A59_us_deep_0.20", set_dd(base, [("US", 2, 0.20)]), RuleSet),
        ("A60_us_deep_0.28", set_dd(base, [("US", 2, 0.28)]), RuleSet),
        ("A61_cn_recov_0.10", set_dd(base, [("CN", 3, 0.10)]), RuleSet),
        ("A62_cn_recov_0.18", set_dd(base, [("CN", 3, 0.18)]), RuleSet),
        ("A63_us_recov_0.10", set_dd(base, [("US", 3, 0.10)]), RuleSet),
        ("A64_us_recov_0.20", set_dd(base, [("US", 3, 0.20)]), RuleSet),
        ("A65_unlock_dd_only", clone(base, x_unlock="dd"), RuleSet),
        ("A66_unlock_trend_only", clone(base, x_unlock="trend"), RuleSet),
    ]
    # --- B. 新机制，跑之前写死 ---
    cases += [
        ("B01_corr_conditional", clone(base, corr_conditional=True), RuleSet),
        ("B02_soft_gate_0.03", clone(base, x_structure="soft", x_soft_band=0.03), RuleSet),
        ("B03_soft_gate_0.04", clone(base, x_structure="soft", x_soft_band=0.04), RuleSet),
        ("B04_soft_gate_0.05", clone(base, x_structure="soft", x_soft_band=0.05), RuleSet),
        ("B05_cool_growth_only", clone(base, cool_off_slots=["159952", "159941", "513500"]), RuleSet),
        ("B06_deep_growth_only", clone(base, x_deep_scope="growth"), RuleSet),
        ("B07_tilt_0.10", clone(base, x_tilt=0.10), Tilt),
        ("B08_tilt_0.20", clone(base, x_tilt=0.20), Tilt),
        ("B09_cool_bull1", clone(base, cool_off_bull_weeks=1), RuleSet),
        ("B10_asym_mindelta", clone(base, min_delta=0.0, x_up_min=0.035), AsymDelta),
        ("B11_fast_cuts_defense", clone(base, x_fast_all=True), RuleSet),
    ]
    return cases


def pack(e, res):
    r = res["rets"]
    periods = {}
    for pn, (ps, pe) in PERIODS.items():
        sub = r[(r.index >= ps) & (r.index <= pe)]
        if len(sub) > 5:
            sw = (1 + sub).cumprod()
            from metrics import max_drawdown, annualized_ret
            periods[pn] = {
                "total": round(float(sw.iloc[-1] - 1) * 100, 2),
                "mdd": round(float(max_drawdown(sw)[0]) * 100, 2),
                "cagr": round(float(annualized_ret(sub)) * 100, 2),
            }
    return {
        "cagr": round(float(e["cagr"]) * 100, 4),
        "mdd": round(float(e["max_dd"]) * 100, 4),
        "sharpe": round(float(e["sharpe"]), 4),
        "calmar": round(float(e["calmar"]), 5),
        "turnover": round(float(e["turnover"]), 2),
        "avg_cash": round(float(e["avg_cash"]) * 100, 2),
        "end": str(r.index.max().date()),
        "periods": periods,
    }


def run_one(name, cfg, cls, panels):
    R_proxy, R_real, Rp, Rr, bond = panels
    out = {"name": name}
    specs = (
        ("proxy", R_proxy, Rp, "2014-06-23"),
        ("real", R_real, Rr, "2025-04-23"),
        ("oos", R_proxy, Rp, "2022-01-04"),
    )
    for key, Rsig, Rpnl, start in specs:
        ds = cls(Rsig, cfg=cfg)
        res = run_backtest(
            Rpnl, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
            start=start, end=END, name=name, min_delta=cfg.get("min_delta", 0.02),
            repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
            cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
            rebal_weekday=cfg.get("rebal_weekday", 2), rebal_freq=cfg.get("rebal_freq", "weekly"),
            strict=True,
        )
        e = evaluate(res)
        out[key] = pack(e, res)
    return out


def near(got, exp, ndp):
    return abs(got - exp) < 0.5 * 10 ** (-ndp)


def main():
    t0 = time.time()
    R_proxy, _ = build_panel("proxy")
    R_real, _ = build_panel("real")
    base = load_base()
    bond = rets_from(read_table(f"{base.get('cash_bond_code', '511260')}_nav.csv"), "cum_nav")
    Rp = pnl_panel(R_proxy, "proxy", base)
    Rr = pnl_panel(R_real, "real", base)
    assert str(R_proxy.index.max().date()) == END
    assert str(bond.index.max().date()) <= END
    panels = (R_proxy, R_real, Rp, Rr, bond)
    cases = build_cases(base)
    rows = []
    for i, (name, cfg, cls) in enumerate(cases):
        row = run_one(name, cfg, cls, panels)
        rows.append(row)
        p, r, o = row["proxy"], row["real"], row["oos"]
        print(
            f"{name:<28} P {p['cagr']:7.4f}/{p['mdd']:8.4f}/Sh{p['sharpe']:.4f}/Cal{p['calmar']:.5f}"
            f" | R {r['cagr']:7.4f}/{r['mdd']:8.4f}/Sh{r['sharpe']:.4f}/Cal{r['calmar']:.5f}"
            f" | O {o['cagr']:7.4f}/{o['mdd']:8.4f}/Sh{o['sharpe']:.4f}/Cal{o['calmar']:.5f}",
            flush=True,
        )
        if name == "A00_v40":
            for k, (cagr, mdd, sh, cal) in EXPECT.items():
                g = row[k]
                ok = (near(g["cagr"], cagr, 4) and near(g["mdd"], mdd, 4)
                      and near(g["sharpe"], sh, 4) and near(g["calmar"], cal, 5))
                if not ok:
                    raise SystemExit(f"baseline mismatch {k}: {g}")
            print("baseline matches PR #15 quoted v40", flush=True)
    blob = {
        "end": END,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "expect_v40": EXPECT,
        "rows": rows,
    }
    path = os.path.join(OUT, "exp_v44_ablation.json")
    json.dump(blob, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[ok] {path} n={len(rows)} sec={blob['seconds']}")


if __name__ == "__main__":
    main()
