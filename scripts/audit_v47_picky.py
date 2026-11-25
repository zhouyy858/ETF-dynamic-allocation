# -*- coding: utf-8 -*-
"""纸面审计：复现 references/v47_picky_code_audit.md 里的关键反事实。

不改配置，不下单，不写 references/。结果打印到 stdout，并写入 out/audit_v47_picky.json。
完整一组大约两分钟。--quick 只跑基线、收益填 0、关掉日度覆盖，以及不需要回测的检查。
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import DATA_DIR, build_panel, read_table, rets_from
from engine import evaluate, run_backtest
from metrics import annualized_ret, max_drawdown, sharpe
from run_backtest import BACKTEST_START, exec_panel
from strategy import DynamicStrategy, SLOTS, live_rebalance_target

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "references", "final_cfg_v47.json")
OUT_DIR = os.path.join(ROOT, "out")


def pack(e):
    keys = ("cagr", "max_dd", "sharpe", "calmar", "turnover")
    return {k: round(float(e[k]), 6) for k in keys}


def run(R, P, cfg, start, daily=True):
    ds = DynamicStrategy(R, cfg=cfg)
    bond = None
    if cfg.get("cash_bond_pct"):
        bond = rets_from(read_table(f"{cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav")
    log = []
    daily_fn = None
    if daily:
        raw = ds.daily_fn()

        def daily_fn(dt, prev, ctx, _raw=raw, _ds=ds, _log=log):
            out = _raw(dt, prev, ctx)
            if out is not None:
                key = pd.Timestamp(dt).normalize()
                stored = _ds.decisions.get(key)
                log.append({
                    "date": str(key.date()),
                    "override_eq": round(1.0 - float(out["cash"]), 4),
                    "decision_eq": None if stored is None else round(1.0 - float(stored["cash"]), 4),
                    "max_abs_vs_decision": None if stored is None else round(
                        max(abs(float(out[k]) - float(stored[k])) for k in list(SLOTS) + ["cash"]), 4),
                    "cool_names": sorted(_ds._cool_from),
                })
            return out

    res = run_backtest(
        P, target_weights_fn=ds.target_fn(), daily_override_fn=daily_fn,
        start=start, min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"), cash_bond_rets=bond,
        cash_bond_pct=cfg.get("cash_bond_pct", 0.0), rebal_weekday=cfg.get("rebal_weekday", 4),
        rebal_freq=cfg.get("rebal_freq", "weekly"), strict=True)
    return ds, res, log, evaluate(res)


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    cfg = json.load(open(CFG_PATH))
    assert cfg.get("version", "").startswith("v47")
    R, _ = build_panel("proxy")
    P = exec_panel(R, "proxy", cfg, "price")
    report = {"cfg": os.path.relpath(CFG_PATH, ROOT), "quick": quick}

    ds, res, log, ev = run(R, P, cfg, BACKTEST_START, daily=True)
    print(f"baseline     {ev['cagr']*100:7.2f}% {ev['max_dd']*100:7.2f}% cal {ev['calmar']:.3f} to {ev['turnover']:.1f} emergencies {len(log)}")
    report["baseline"] = pack(ev)
    report["sharpe_rf_018"] = round(float(sharpe(res["rets"], 0.018)), 4)
    report["sharpe_rf_022"] = round(float(sharpe(res["rets"], 0.022)), 4)
    disagree = [x for x in log if x["max_abs_vs_decision"] and x["max_abs_vs_decision"] > 1e-4]
    uprisk = [x for x in disagree if x["decision_eq"] is not None and x["override_eq"] > x["decision_eq"] + 1e-4]
    report["emergencies"] = {
        "n": len(log),
        "friday_disagree": len(disagree),
        "friday_uprisk": len(uprisk),
        "uprisk_examples": uprisk[:8],
    }
    print(f"  friday overrides differing from the stored decision: {len(disagree)}, higher equity: {len(uprisk)}")

    _, res0, _, ev0 = run(R, P.fillna(0.0), cfg, BACKTEST_START, daily=True)
    print(f"pnl fillna0  {ev0['cagr']*100:7.2f}% {ev0['max_dd']*100:7.2f}% cal {ev0['calmar']:.3f}")
    report["pnl_fillna0"] = pack(ev0)

    _, res_ne, _, ev_ne = run(R, P, cfg, BACKTEST_START, daily=False)
    print(f"no emergency {ev_ne['cagr']*100:7.2f}% {ev_ne['max_dd']*100:7.2f}% cal {ev_ne['calmar']:.3f} to {ev_ne['turnover']:.1f}")
    report["friday_only"] = pack(ev_ne)

    c = copy.deepcopy(cfg)
    c["recovery_ramp"] = True
    _, res_r, _, ev_r = run(R, P, c, BACKTEST_START, daily=True)
    report["recovery_ramp_true_max_abs_ret"] = float((res["rets"] - res_r["rets"]).abs().max())
    print(f"recovery_ramp true vs false max |dret| {report['recovery_ramp_true_max_abs_ret']:.3e}")

    # Holiday rows whose close already equals the next HS300 session.
    hs = read_table("index_sh000300.csv")
    phantom = []
    for fn in ("index_930955.csv", "index_932365.csv"):
        s = read_table(fn)["close"].sort_index()
        extra = s.index.difference(hs.index)
        extra = extra[extra >= BACKTEST_START]
        common = s.index.intersection(hs.index)
        for dt in extra:
            nxt = common[common > dt]
            if len(nxt) == 0:
                continue
            if abs(float(s.loc[dt]) - float(s.loc[nxt[0]])) < 1e-6:
                phantom.append({"file": fn, "row": str(dt.date()), "equals": str(nxt[0].date()),
                                "close": float(s.loc[dt])})
    report["phantom_next_close"] = phantom
    print(f"phantom next-close rows: {len(phantom)}")

    # 270042 wins the overlap; floors after exclude.
    a = rets_from(read_table("270042_nav.csv"), "cum_nav")
    b = rets_from(read_table("159941_nav.csv"), "cum_nav")
    both = a.index.intersection(b.index)
    report["nav_signal_winner"] = {
        "overlap": int(len(both)),
        "panel_equals_270042": bool(np.allclose(R.loc[both, "159941"].fillna(0), a.reindex(both).fillna(0))),
        "mean_abs_bp": round(float((a.reindex(both) - b.reindex(both)).abs().mean() * 1e4), 2),
    }
    report["floors_pct"] = ds.floor
    print("floors", ds.floor, "270042 wins overlap", report["nav_signal_winner"]["panel_equals_270042"])

    if not quick:
        c = copy.deepcopy(cfg)
        c["signal_lag"] = 0
        _, res_l, _, _ = run(R, P, c, BACKTEST_START, daily=True)
        report["signal_lag0_max_abs_ret"] = float((res["rets"] - res_l["rets"]).abs().max())
        for gw in (80, 120):
            c = copy.deepcopy(cfg)
            c["gate_win"] = gw
            _, res_g, _, _ = run(R, P, c, BACKTEST_START, daily=True)
            report[f"gate_win_{gw}_max_abs_ret"] = float((res["rets"] - res_g["rets"]).abs().max())
        c = copy.deepcopy(cfg)
        c["instrument_swaps"] = {}
        Pns = exec_panel(R, "proxy", c, "price")
        _, res_n, _, ev_n = run(R, Pns, c, BACKTEST_START, daily=True)
        report["no_swap"] = pack(ev_n)
        pre = res["rets"].loc[: "2019-01-17"]
        report["no_swap_pre_max_abs"] = float((pre - res_n["rets"].reindex(pre.index)).abs().max())
        R_us = R.copy()
        R_us[["159941", "513500"]] = R_us[["159941", "513500"]].shift(1)
        _, _, _, ev_u = run(R_us, P, cfg, BACKTEST_START, daily=True)
        report["us_signal_shift1"] = pack(ev_u)
        etf = rets_from(read_table("159941_nav.csv"), "cum_nav")
        R_etf = R.copy()
        R_etf.loc[etf.index, "159941"] = etf
        _, _, _, ev_e = run(R_etf, P, cfg, BACKTEST_START, daily=True)
        report["etf_nav_signal"] = pack(ev_e)
        fridays = [d for d in res["weights"].index if int(d.day_of_week) == 4]
        probe = fridays[-2]
        ds_c, res_c, _, _ = run(R.loc[R.index < probe], P.loc[P.index < probe], cfg, BACKTEST_START, daily=True)
        decision, signal_dt, via = live_rebalance_target(ds_c, res_c["rets"], 4)
        ref = ds.decisions[pd.Timestamp(probe).normalize()]
        report["friday_9am"] = {
            "friday": str(probe.date()),
            "signal": str(signal_dt.date()),
            "helper_vs_decision": max(abs(float(via[k]) - float(ref[k])) for k in list(SLOTS) + ["cash"]),
            "decision_vs_fill": max(abs(float(ref[k]) - float(res["weights"].loc[probe, k])) for k in list(SLOTS) + ["cash"]),
        }
        print("friday 9am", report["friday_9am"])
        train = res["rets"].loc[: "2021-12-31"]
        w = (1 + train).cumprod()
        mdd = max_drawdown(w)[0]
        report["train_slice"] = {"cagr": round(float(annualized_ret(train)), 6), "mdd": round(float(mdd), 6)}

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "audit_v47_picky.json")
    json.dump(report, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print(f"[ok] {path}  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
