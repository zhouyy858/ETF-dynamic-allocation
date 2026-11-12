# -*- coding: utf-8 -*-
"""v44 候选的合成共振熊压力（场内价）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import BASE, run
from run_backtest import exec_panel
from stress_test import synthetic_resonance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
CAND = json.load(open(os.path.join(OUT, "candidate_v44_cn4_us11.json")))


def main():
    P_REAL = exec_panel(V.Rr, "real", BASE, "price")
    report = {}
    for tag, us, cn, idx in (
        ("synth_std", -0.30, -0.20, -0.25),
        ("synth_hard", -0.40, -0.30, -0.35),
    ):
        synth, s_idx = synthetic_resonance(P_REAL, ("2024-09-02", "2026-07-31"), us, cn, idx)
        report[tag] = {}
        for name, cfg in (("v43", BASE), ("v44", CAND)):
            _, e = run(cfg, synth, synth, "2024-09-02", "2026-07-31")
            report[tag][name] = V.pack(e)
            print(f"{tag:10} {name:4} {e['cagr']*100:7.2f}% / {e['max_dd']*100:7.2f}% / {e['calmar']:5.2f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp_v44_stress.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_v44_stress.json", flush=True)


if __name__ == "__main__":
    main()
