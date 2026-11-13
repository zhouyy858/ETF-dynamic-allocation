# -*- coding: utf-8 -*-
"""v44 邻域与「关掉也不变」确认。沿用 exp_v44_ablation 的官方引擎路径。

只检查两处单独过了三关的格子（防御动量温度 t、CN 深熊解锁回撤），
以及美股快刹车 / 美股深熊锁是否整段样本都不改变成交。
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_v44_ablation import END, RuleSet, clone, load_base, run_one, set_dd
from data_prep import build_panel, read_table, rets_from
from swaps import pnl_panel
from strategy import DynamicStrategy


def main():
    t0 = time.time()
    R_proxy, _ = build_panel("proxy")
    R_real, _ = build_panel("real")
    base = load_base()
    bond = rets_from(read_table(f"{base.get('cash_bond_code', '511260')}_nav.csv"), "cum_nav")
    panels = (R_proxy, R_real, pnl_panel(R_proxy, "proxy", base), pnl_panel(R_real, "real", base), bond)
    ds = DynamicStrategy(R_proxy, cfg=base)
    for mkt, nm, px in (("CN", "a_mkt", ds.sig.a_mkt), ("US", "u_m", ds.sig.u_m)):
        dd = px / ds.sig.dd_max[nm] - 1.0
        sub = dd[dd.index >= "2014-06-23"]
        print(f"{mkt} 3y dd min {float(sub.min())*100:.2f}% on {sub.idxmin().date()}  "
              f"days<-20% {(sub < -0.20).sum()}  days<-24% {(sub < -0.24).sum()}  "
              f"days<-13% {(sub < -0.13).sum()}  days<-16% {(sub < -0.16).sum()}")
    cases = []
    for t in (3, 5, 7, 8, 10):
        cases.append((f"N_t{t}", clone(base, defense_momentum_t=float(t)), RuleSet))
    for rec in (0.15, 0.16, 0.17, 0.19, 0.20, 0.22, 0.24):
        cases.append((f"N_cn_recov_{rec:.2f}", set_dd(base, [("CN", 3, rec)]), RuleSet))
    cases.append(("N_us_fast_off", set_dd(base, [("US", 0, 9.0)]), RuleSet))
    cases.append(("N_us_deep_off", set_dd(base, [("US", 2, 9.0)]), RuleSet))
    # 两处单独过线的格子叠在一起。只在邻域同号时才有资格，数字先算出来。
    both = clone(set_dd(base, [("CN", 3, 0.18)]), defense_momentum_t=6.0)
    cases.append(("C01_t6_and_recov18", both, RuleSet))
    rows = []
    for name, cfg, cls in cases:
        row = run_one(name, cfg, cls, panels)
        rows.append(row)
        p, r, o = row["proxy"], row["real"], row["oos"]
        print(
            f"{name:<24} P {p['cagr']:7.4f}/{p['mdd']:8.4f}/Cal{p['calmar']:.5f}"
            f" | R {r['cagr']:7.4f}/{r['mdd']:8.4f}/Cal{r['calmar']:.5f}"
            f" | O {o['cagr']:7.4f}/{o['mdd']:8.4f}/Cal{o['calmar']:.5f}",
            flush=True,
        )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "exp_v44_neighbors.json")
    json.dump({"end": END, "seconds": round(time.time() - t0, 1), "rows": rows}, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[ok] {path}")


if __name__ == "__main__":
    main()
