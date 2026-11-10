# -*- coding: utf-8 -*-
"""v40 路径的最小未来函数抽检。

沿用截断与扰动：把样本在若干日期截断后重跑，截断日之前的权重应与全样本一致。
再把 159509 的某一天收益改大，这一天之前的权重不应变化。
对照：把信号列 159941 的收益向前挪一天（用到下一交易日），权重必须变化，否则这个检查没有辨别力。

不改默认配置。纸面。
用法: python3 scripts/audit_lookahead_v40.py
"""
import os, sys, json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest
from strategy import DynamicStrategy
from swaps import pnl_panel
from exp_v40_universe import splice, ret_from_close, load_close, PROXY_START

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
CUTS = ["2024-12-31", "2025-06-30", "2026-06-30"]
PERTURB_DAY = "2026-08-03"


def weights(R_sig, R_pnl, cfg, bond, start, end):
    sig = R_sig.loc[:end] if end is not None else R_sig
    pnl = R_pnl.loc[:end] if end is not None else R_pnl
    bond_s = bond.loc[:end] if end is not None else bond
    ds = DynamicStrategy(sig, cfg=cfg)
    res = run_backtest(
        pnl, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="DYN",
        min_delta=cfg.get("min_delta", 0.035), repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond_s, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    return res["weights"]


def max_diff(a, b):
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return None, 0
    d = (a.loc[idx] - b.loc[idx]).abs()
    return int(len(idx)), float(d.max().max())


def main():
    cfg = json.load(open(os.path.join(REF, "final_cfg_v39.json")))
    bond = rets_from(read_table("511260_nav.csv"), "cum_nav")
    R, _ = build_panel("proxy")
    Rp = pnl_panel(R, "proxy", cfg)
    report = {"cuts": [], "perturb": None, "negative_control": None, "ok": True}

    print("[截断] v39 全样本")
    w_full = weights(R, Rp, cfg, bond, PROXY_START, None)
    for cut in CUTS:
        w_cut = weights(R, Rp, cfg, bond, PROXY_START, cut)
        # 只比截断日及之前
        n, diff = max_diff(w_full.loc[:cut], w_cut)
        ok = diff is not None and diff < 1e-8
        report["cuts"].append({"path": "v39", "cut": cut, "n": n, "max_abs": diff, "ok": ok})
        print(f"  v39 {cut} n={n} max_abs={diff} ok={ok}")
        if not ok:
            report["ok"] = False

    own = ret_from_close(load_close("159509", "qfq"), "close")
    if own is None:
        report["perturb"] = {"ok": False, "note": "没有 159509 前复权"}
        report["ok"] = False
    else:
        filled, info = splice(Rp["159941"], own, Rp.index, "proxy")
        if filled is None:
            report["perturb"] = {"ok": False, "note": f"拼接失败 {info}"}
            report["ok"] = False
            print("[159509] 拼接失败", info)
        else:
            pnl = Rp.copy()
            pnl["159941"] = filled
            print("[截断] 159509 替换纳指槽")
            w509 = weights(R, pnl, cfg, bond, PROXY_START, None)
            for cut in CUTS:
                w_cut = weights(R, pnl, cfg, bond, PROXY_START, cut)
                n, diff = max_diff(w509.loc[:cut], w_cut)
                ok = diff is not None and diff < 1e-8
                report["cuts"].append({"path": "159509_replace", "cut": cut, "n": n, "max_abs": diff, "ok": ok})
                print(f"  159509 {cut} n={n} max_abs={diff} ok={ok}")
                if not ok:
                    report["ok"] = False
            # 扰动：把 2026-08-03 的 159509 收益 +0.50，这一天之前的权重应不变
            shocked = own.copy()
            day = pd.Timestamp(PERTURB_DAY)
            if day not in shocked.index:
                day = shocked.index[shocked.index.get_indexer([day], method="pad")][0]
            shocked.loc[day] = float(shocked.loc[day]) + 0.50
            filled2, info2 = splice(Rp["159941"], shocked, Rp.index, "proxy")
            pnl2 = Rp.copy()
            pnl2["159941"] = filled2
            w2 = weights(R, pnl2, cfg, bond, PROXY_START, None)
            before = w509.index[w509.index < day]
            n, diff = max_diff(w509.loc[before], w2.loc[before])
            after = w509.index[w509.index >= day]
            _, diff_after = max_diff(w509.loc[after], w2.loc[after])
            ok = diff is not None and diff < 1e-8
            report["perturb"] = {
                "day": str(pd.Timestamp(day).date()), "n_before": n,
                "max_abs_before": diff, "max_abs_on_or_after": diff_after, "ok": ok,
                "gap": info, "note": "之前不变才算没有把这一天的行情用进更早的决策",
            }
            print(f"[扰动] day={day.date()} before={diff} after={diff_after} ok={ok}")
            if not ok:
                report["ok"] = False

    # 负对照：信号列用下一交易日的 159941 收益。权重应当变。
    sig = R.copy()
    sig["159941"] = sig["159941"].shift(-1)
    sig = sig.dropna(subset=["159941"])
    # 负对照只需要证明检查有辨别力，截到 2024 年底以省时间
    w_leak = weights(sig, Rp, cfg, bond, PROXY_START, "2024-12-31")
    w_base = weights(R, Rp, cfg, bond, PROXY_START, "2024-12-31")
    n, diff = max_diff(w_base, w_leak)
    ok = diff is not None and diff > 1e-4
    report["negative_control"] = {"n": n, "max_abs": diff, "ok": ok,
                                  "note": "把 159941 收益向前挪一天后权重必须变化"}
    print(f"[负对照] max_abs={diff} ok={ok}")
    if not ok:
        report["ok"] = False

    path = os.path.join(OUT, "audit_lookahead_v40.json")
    json.dump(report, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[写] {path} ok={report['ok']}")
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
