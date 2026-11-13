# -*- coding: utf-8 -*-
"""v45 周五 14:57 同日信号研究（非严格回测口径）。

历史数据只有日线，无法精确回放 14:57 快照；本脚本把周五收盘价作为 14:57
近似值，仅用于估计同日信号+同日尾盘成交的增量。正式 v45 统计仍使用 T-1
信号、周五收盘成交的严格口径。
"""
import copy
import json
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from engine import evaluate, run_backtest
from run_backtest import exec_panel
from strategy import DynamicStrategy, SignalSet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
BASE = json.load(open(os.path.join(ROOT, "references", "final_cfg_v45.json")))
P = exec_panel(V.R, "proxy", BASE, "price")
PR = exec_panel(V.Rr, "real", BASE, "price")


class SameDaySignalSet(SignalSet):
    """审计专用：允许读到决策日已经形成的最后一根日线。"""

    def _idx(self, dt):
        ts = pd.Timestamp(dt)
        pos = int(self.R.index.searchsorted(ts, side="left")) - 1
        if pos + 1 < len(self.R.index) and self.R.index[pos + 1] <= ts:
            pos += 1
        return max(0, pos)


def _cfg(**kw):
    c = copy.deepcopy(BASE)
    c.update(kw)
    return c


def run_cfg(cfg, sig, pnl, start, end=None, fee=0.0005, periods=None, same_day=False):
    DynamicStrategy.SIG_CLASS = SameDaySignalSet if same_day else SignalSet
    ds = DynamicStrategy(sig, cfg=cfg)
    res = run_backtest(
        pnl,
        target_weights_fn=ds.target_fn(),
        daily_override_fn=ds.daily_fn(),
        start=start,
        end=end,
        name="DYN",
        fee=fee,
        min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=V.BOND,
        cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=not same_day,
        accrual_mode="pre",
        exec_lag=0,
    )
    return res, evaluate(res, periods=periods)


def compare():
    report = {}
    for tag, same_day in (("strict_T1", False), ("sameday_proxy", True)):
        report[tag] = {}
        print("\n==", tag, "==", flush=True)
        for layer, sig, pnl, start, end in (
            ("proxy", V.R, P, V.PROXY_START, None),
            ("train", V.R, P, V.PROXY_START, "2021-12-31"),
            ("oos", V.R, P, V.OOS_START, None),
            ("real", V.Rr, PR, V.REAL_START, None),
        ):
            res, e = run_cfg(BASE, sig, pnl, start, end,
                             periods=V.PERIODS if layer == "proxy" else None,
                             same_day=same_day)
            report[tag][layer] = V.pack(e)
            if layer == "proxy":
                report[tag]["periods"] = {k: dict(v) for k, v in e["periods"].items()}
                report[tag]["bootstrap"] = V.bootstrap(res["rets"])
            print(f"  {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / "
                  f"Cal {e['calmar']:5.3f}", flush=True)
        report[tag]["costs"] = {}
        for bp in (5, 10, 20, 30, 50):
            _, ep = run_cfg(BASE, V.R, P, V.PROXY_START, None,
                            fee=bp / 10000.0, same_day=same_day)
            _, eo = run_cfg(BASE, V.R, P, V.OOS_START, None,
                            fee=bp / 10000.0, same_day=same_day)
            report[tag]["costs"][str(bp)] = {"proxy": V.pack(ep), "oos": V.pack(eo)}
            print(f"  {bp:2}bp proxy Cal {ep['calmar']:.3f} | OOS Cal {eo['calmar']:.3f}", flush=True)
    return report


def _floor_worker(x):
    cn, us = x
    c = _cfg(floor_pct={"cn": float(cn) * 2.0, "us": float(us)},
             version=f"v46_cn{cn:g}_us{us:g}")
    _, tr = run_cfg(c, V.R, P, V.PROXY_START, "2021-12-31", same_day=True)
    return cn, us, V.pack(tr)


def _core_worker(x):
    k, sm, ps = x
    c = _cfg(lock_defense_keep=float(k), score_mode=sm, premium_shift=int(ps),
             version=f"v46_k{k:g}_{sm}_p{ps}")
    _, tr = run_cfg(c, V.R, P, V.PROXY_START, "2021-12-31", same_day=True)
    _, oo = run_cfg(c, V.R, P, V.OOS_START, None, same_day=True)
    return k, sm, ps, V.pack(tr), V.pack(oo)


def retrain():
    report = {}
    with mp.get_context("fork").Pool(min(6, mp.cpu_count())) as pool:
        floor_rows = pool.map(
            _floor_worker,
            [(cn, us) for cn in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6)
             for us in (8, 9, 10, 11, 12, 13, 14, 15, 16)],
        )
    floor_rows.sort(key=lambda z: z[2]["calmar"], reverse=True)
    floor_valid = []
    for cn, us, m in floor_rows[:12]:
        c = _cfg(floor_pct={"cn": float(cn) * 2.0, "us": float(us)})
        _, oos = run_cfg(c, V.R, P, V.OOS_START, None, same_day=True)
        floor_valid.append({"cn_eff": cn, "us": us, "train": m, "oos": V.pack(oos)})
    report["floor"] = {"top_train": [
        {"cn_eff": cn, "us": us, "metrics": m} for cn, us, m in floor_rows
    ], "validated": floor_valid}

    with mp.get_context("fork").Pool(min(6, mp.cpu_count())) as pool:
        core_rows = pool.map(
            _core_worker,
            [(k, sm, ps)
             for k in (.05, .1, .15, .2, .25, .3, .5, 1.0)
             for sm in ("base", "dedup_us", "tech_confirm", "us_lead", "cn_lead")
             for ps in (1, 2, 3, 4, 5)],
        )
    core_rows.sort(key=lambda z: z[3]["calmar"], reverse=True)
    _, v45_train = run_cfg(BASE, V.R, P, V.PROXY_START, "2021-12-31", same_day=True)
    _, v45_oos = run_cfg(BASE, V.R, P, V.OOS_START, None, same_day=True)
    bt, bo = V.pack(v45_train), V.pack(v45_oos)
    core_valid = [{"k": k, "score_mode": sm, "premium_shift": ps,
                   "train": tr, "oos": oo}
                  for k, sm, ps, tr, oo in core_rows[:16]]
    pareto = [{"k": k, "score_mode": sm, "premium_shift": ps,
               "train": tr, "oos": oo}
              for k, sm, ps, tr, oo in core_rows
              if tr["calmar"] >= bt["calmar"] and oo["calmar"] >= bo["calmar"]]
    report["core"] = {"v45_sameday": {"train": bt, "oos": bo},
                      "top_train": [{"k": k, "score_mode": sm, "premium_shift": ps,
                                     "metrics": tr} for k, sm, ps, tr, _ in core_rows],
                      "validated": core_valid, "pareto": pareto}
    return report


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode not in ("compare", "retrain", "all"):
        raise SystemExit("usage: exp_v46_sameday.py [compare|retrain|all]")
    report = {}
    if mode in ("compare", "all"):
        report["compare"] = compare()
    if mode in ("retrain", "all"):
        report["retrain"] = retrain()
    path = os.path.join(OUT, f"exp_v46_sameday_{mode}.json")
    json.dump(report, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[ok] {path}", flush=True)


if __name__ == "__main__":
    main()
