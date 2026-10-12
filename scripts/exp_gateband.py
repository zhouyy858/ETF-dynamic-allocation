# -*- coding: utf-8 -*-
"""v33 候选预注册评估(2026-09-15, 数据至2026-09-14): 结构闸门滞回带 gate_band
单机制×3 + gate_win 平台扫描, 严格口径(signal_lag=1/溢价T-2/先计提后成交, 无未来函数)

动机: v30 扰动审计 2/112 恶化均来自 gate_win±20%(80/120)时 proxy MDD -11.36% 破线
      —— 结构闸门是 px>SMA(gate_win) 的硬阈值, 均线附近反复穿越使 gate_win 轴脆弱。
机制: 闸门带滞回带 b —— 跌破 SMA_g×(1-b) 才关闸, 站上 SMA_g×(1+b) 才开闸,
      与打分侧滞回(0.66/0.24)同哲学; 仅作用于结构闸门, 深熊锁解锁条件不变。
预注册验收(全部满足才考虑升级):
  1) 基线: proxy Cal 不低于 v30 基线 0.10, proxy MDD > -10%(守线), real MDD > -4.1%;
  2) 平台: 固定 b 下 gate_win∈{80,90,100,110,120} 全部 proxy MDD > -10.5%;
  3) OOS(2022+): Cal 不低于基线 OOS 减 0.10;
  4) 复跑 112 次扰动: Calmar 恶化>10% 次数 <= 2/112 且不再出现 MDD 破 -10.5%。
用法: python3 exp_gateband.py  (输出 out/exp_gateband.json)
"""
import sys, os, json, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

OUT = os.path.join(os.getcwd(), "out"); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = os.path.join(HERE, "..", "references", "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"


class BandedGateStrategy(DynamicStrategy):
    """gate_band>0 时结构闸门带滞回带; =0 时与基线逐位一致"""
    def __init__(self, R_full, cfg=None, a_mkt_override=None):
        super().__init__(R_full, cfg=cfg, a_mkt_override=a_mkt_override)
        self.gate_band = float(cfg.get("gate_band", 0.0)) if cfg else 0.0
        self._gate_st = {}

    def _gate_state(self, dt, market):
        i = self.sig._idx(dt)
        nm = "a_mkt" if market == "CN" else "u_m"
        x = self.sig.a_mkt if market == "CN" else self.sig.u_m
        if i >= len(x):
            return True
        px = float(x.iloc[i])
        sg = float(self.sig.sma[nm][self.gate_win].iloc[i])
        st = self._gate_st.get(market)
        if st is None:
            st = bool(px > sg)  # 初始状态与原始硬阈值一致(NaN比较为False)
            self._gate_st[market] = st
            return st
        if not st and px > sg * (1.0 + self.gate_band):
            self._gate_st[market] = True
            return True
        if st and px < sg * (1.0 - self.gate_band):
            self._gate_st[market] = False
            return False
        return st

    def _market_mult(self, dt, market):
        dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, market, self.gate_win)
        fb_thr, fb_cut, deep_thr, recov = self.market_dd[market]
        m_g, m_all = 1.0, 1.0
        if self._lock[market]:
            if dd > -recov or (ok120 and ok20_60):
                self._lock[market] = False
            else:
                return 0.0, 0.0
        if dd < -deep_thr:
            self._lock[market] = True
            return 0.0, 0.0
        gst = self._gate_state(dt, market) if self.gate_band > 0 else ok120
        if not gst:
            m_g = 0.0
        if dd < -fb_thr and not (ok20 and ok20_60):
            m_g = min(m_g, fb_cut)
        return m_g, m_all


def run_cfg(cfg, R, start, strat_cls=BandedGateStrategy):
    ds = strat_cls(R, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    return evaluate(res)


def metrics(cfg, R, start):
    e = run_cfg(cfg, R, start)
    return {"cagr": e["cagr"], "mdd": e["max_dd"], "cal": e["calmar"],
            "sharpe": e["sharpe"], "cash": e["avg_cash"], "to": e["turnover"]}


CANDS = {
    "base_v30": {},
    "G1_band1": {"gate_band": 0.01},
    "G2_band2": {"gate_band": 0.02},
    "G3_band3": {"gate_band": 0.03},
}

def main():
    base = json.load(open(CFG_FILE))
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    rows = {}
    for name, ov in CANDS.items():
        cfg = copy.deepcopy(base); cfg.update(ov)
        p = metrics(cfg, R, PROXY_START); r = metrics(cfg, Rr, REAL_START)
        rows[name] = {"override": ov, "proxy": p, "real": r}
        print(f"{name:<12} proxy Cal {p['cal']:5.2f} MDD {p['mdd']*100:6.2f}% | real Cal {r['cal']:5.2f} MDD {r['mdd']*100:6.2f}%")

    # gate_win 平台扫描: 对每个 band 候选, 扫 gate_win 80/90/100/110/120 的 proxy MDD/Cal
    scan = {}
    for name in ("G1_band1", "G2_band2", "G3_band3"):
        for gw in (80, 90, 100, 110, 120):
            cfg = copy.deepcopy(base); cfg.update(CANDS[name]); cfg["gate_win"] = gw
            e = run_cfg(cfg, R, PROXY_START)
            scan[f"{name}@gw{gw}"] = {"cagr": e["cagr"], "mdd": e["max_dd"], "cal": e["calmar"]}
            print(f"  平台 {name}@gw{gw}: Cal {e['calmar']:5.2f} MDD {e['max_dd']*100:6.2f}%")
    rows["gate_win_scan"] = scan

    # OOS 窗口对照(基线 vs 各 band)
    oos = {}
    for name, ov in CANDS.items():
        cfg = copy.deepcopy(base); cfg.update(ov)
        oos[name] = metrics(cfg, R, OOS_START)
    rows["oos"] = oos
    bp, br = rows["base_v30"]["proxy"], rows["base_v30"]["real"]
    print(f"\n基线 v30: proxy Cal {bp['cal']:.2f} MDD {bp['mdd']*100:.2f}% | real Cal {br['cal']:.2f} MDD {br['mdd']*100:.2f}%")
    for name in CANDS:
        if name == "base_v30":
            continue
        r = rows[name]
        dcp, dcr = r["proxy"]["cal"] - bp["cal"], r["real"]["cal"] - br["cal"]
        ok = (dcp > -0.10 and r["proxy"]["mdd"] > -0.10 and r["real"]["mdd"] > -0.041
              and oos[name]["cal"] > oos["base_v30"]["cal"] - 0.10)
        print(f"{name:<12} ΔCal proxy {dcp:+.3f} / real {dcr:+.3f}  OOS Cal {oos[name]['cal']:.2f}  {'★预注册1/3关通过' if ok else '✗ 未过'}")
    json.dump(rows, open(os.path.join(OUT, "exp_gateband.json"), "w"), ensure_ascii=False, indent=1)
    print("\n[ok] out/exp_gateband.json")


if __name__ == "__main__":
    main()
