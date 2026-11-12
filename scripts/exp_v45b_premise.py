# -*- coding: utf-8 -*-
"""v45b：结构化前提候选（打分去重、纳指确认回补、快速科技联动）。"""
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


def cand(tag, **kw):
    c = copy.deepcopy(BASE); c.update(kw); c["version"] = tag; return c


CANDS = {
    "base_v44": cand("base_v44"),
    "score_tech_confirm": cand("score_tech_confirm", score_mode="tech_confirm"),
    "score_dedup_us": cand("score_dedup_us", score_mode="dedup_us"),
    "score_us_lead": cand("score_us_lead", score_mode="us_lead"),
    "score_cn_lead": cand("score_cn_lead", score_mode="cn_lead"),
    "rebound_us_f25": cand("rebound_us_f25", cn_seesaw=True, cn_seesaw_mode="contrarian", cn_seesaw_thr=0.10, cn_seesaw_frac=0.25, cn_seesaw_rebound=True, cn_seesaw_confirm_us=True),
    "rebound_us_f50": cand("rebound_us_f50", cn_seesaw=True, cn_seesaw_mode="contrarian", cn_seesaw_thr=0.10, cn_seesaw_frac=0.50, cn_seesaw_rebound=True, cn_seesaw_confirm_us=True),
    "techrot20_f25": cand("techrot20_f25", tech_rotate=True, tech_rotate_mode="procyclical", tech_rotate_win=20, tech_rotate_frac=0.25),
    "techrot20_f50": cand("techrot20_f50", tech_rotate=True, tech_rotate_mode="procyclical", tech_rotate_win=20, tech_rotate_frac=0.50),
    "techrot60_f25": cand("techrot60_f25", tech_rotate=True, tech_rotate_mode="procyclical", tech_rotate_win=60, tech_rotate_frac=0.25),
    "techrot60_f50": cand("techrot60_f50", tech_rotate=True, tech_rotate_mode="procyclical", tech_rotate_win=60, tech_rotate_frac=0.50),
    "techrot20_inv_f25": cand("techrot20_inv_f25", tech_rotate=True, tech_rotate_mode="inverse", tech_rotate_win=20, tech_rotate_frac=0.25),
}


def main():
    P = exec_panel(V.R, "proxy", BASE, "price"); Pr = exec_panel(V.Rr, "real", BASE, "price")
    report = {}
    for name, c in CANDS.items():
        report[name] = {}
        print(f"\n{name}", flush=True)
        for layer, sig, pnl, start, end in (
            ("proxy", V.R, P, V.PROXY_START, None),
            ("train", V.R, P, V.PROXY_START, "2021-12-31"),
            ("oos", V.R, P, V.OOS_START, None),
            ("real", V.Rr, Pr, V.REAL_START, None),
        ):
            _, e = run(c, sig, pnl, start, end)
            report[name][layer] = V.pack(e)
            print(f"  {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.3f} / TO {e['turnover']:6.1f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v45b_premise.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok]", flush=True)


if __name__ == "__main__":
    main()
