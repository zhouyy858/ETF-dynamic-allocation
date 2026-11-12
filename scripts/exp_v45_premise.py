# -*- coding: utf-8 -*-
"""v45 前提层研究：红利/创业板跷跷板 + 纳指科技周期联动（场内价，纸面）。"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import run
from run_backtest import exec_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(ROOT, "references", "final_cfg_v44.json")))


def C(**kw):
    c = copy.deepcopy(BASE)
    c.update(kw)
    c["version"] = "v45_" + kw.pop("_tag", "x") if "_tag" in kw else "v45"
    return c


def cand(tag, **kw):
    c = copy.deepcopy(BASE)
    c.update(kw)
    c["version"] = tag
    return c


CANDS = {
    "base_v44": cand("base_v44"),
    "seesaw_trim10_f25": cand("seesaw_trim10_f25", cn_seesaw=True, cn_seesaw_mode="contrarian", cn_seesaw_thr=0.10, cn_seesaw_frac=0.25),
    "seesaw_trim10_f50": cand("seesaw_trim10_f50", cn_seesaw=True, cn_seesaw_mode="contrarian", cn_seesaw_thr=0.10, cn_seesaw_frac=0.50),
    "seesaw_trim20_f50": cand("seesaw_trim20_f50", cn_seesaw=True, cn_seesaw_mode="contrarian", cn_seesaw_thr=0.20, cn_seesaw_frac=0.50),
    "seesaw_rebound10_f25": cand("seesaw_rebound10_f25", cn_seesaw=True, cn_seesaw_mode="contrarian", cn_seesaw_thr=0.10, cn_seesaw_frac=0.25, cn_seesaw_rebound=True),
    "seesaw_rebound10_f50": cand("seesaw_rebound10_f50", cn_seesaw=True, cn_seesaw_mode="contrarian", cn_seesaw_thr=0.10, cn_seesaw_frac=0.50, cn_seesaw_rebound=True),
    "seesaw_mom0_f25": cand("seesaw_mom0_f25", cn_seesaw=True, cn_seesaw_mode="momentum", cn_seesaw_thr=0.0, cn_seesaw_frac=0.25),
    "seesaw_mom0_f50": cand("seesaw_mom0_f50", cn_seesaw=True, cn_seesaw_mode="momentum", cn_seesaw_thr=0.0, cn_seesaw_frac=0.50),
    "techrot_pro_f25": cand("techrot_pro_f25", tech_rotate=True, tech_rotate_mode="procyclical", tech_rotate_frac=0.25),
    "techrot_pro_f50": cand("techrot_pro_f50", tech_rotate=True, tech_rotate_mode="procyclical", tech_rotate_frac=0.50),
    "techrot_inv_f25": cand("techrot_inv_f25", tech_rotate=True, tech_rotate_mode="inverse", tech_rotate_frac=0.25),
    "techcap20": cand("techcap20", tech_cap=True, tech_cap_pct=0.20, tech_cap_corr=0.0),
    "techcap25": cand("techcap25", tech_cap=True, tech_cap_pct=0.25, tech_cap_corr=0.0),
    "techcap20_corr15": cand("techcap20_corr15", tech_cap=True, tech_cap_pct=0.20, tech_cap_corr=0.15),
}

STRESS = (
    ("2015", "2015-06-15", "2016-02-29"),
    ("2018", "2018-01-02", "2019-01-03"),
    ("2021-22", "2021-02-19", "2022-10-31"),
)


def main():
    P = exec_panel(V.R, "proxy", BASE, "price")
    Pr = exec_panel(V.Rr, "real", BASE, "price")
    report = {"metrics": {}, "stress": {}}
    for name, c in CANDS.items():
        report["metrics"][name] = {}
        print(f"\n{name}", flush=True)
        for layer, sig, pnl, start, end in (
            ("proxy", V.R, P, V.PROXY_START, None),
            ("train", V.R, P, V.PROXY_START, "2021-12-31"),
            ("oos", V.R, P, V.OOS_START, None),
            ("real", V.Rr, Pr, V.REAL_START, None),
        ):
            _, e = run(c, sig, pnl, start, end)
            report["metrics"][name][layer] = V.pack(e)
            print(f"  {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.3f} / TO {e['turnover']:6.1f}", flush=True)
    print("\n== 压力窗口 ==", flush=True)
    for sn, start, end in STRESS:
        report["stress"][sn] = {}
        for name in ("base_v44", "seesaw_trim10_f25", "seesaw_trim10_f50", "seesaw_rebound10_f50", "seesaw_mom0_f25", "techrot_pro_f25", "techcap20"):
            c = CANDS[name]
            _, e = run(c, V.R, P, start, end)
            report["stress"][sn][name] = V.pack(e)
            print(f"  {sn:8} {name:22} {e['cagr']*100:7.2f}% / {e['max_dd']*100:7.2f}% / {e['calmar']:5.2f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v45_premise.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v45_premise.json", flush=True)


if __name__ == "__main__":
    main()
