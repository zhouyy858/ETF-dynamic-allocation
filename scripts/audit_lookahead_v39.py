# -*- coding: utf-8 -*-
"""v39 未来函数检测（截断测试 + 未来扰动 + 周五 09:00 信息集）.

截断: 数据在 T 处截掉之后, 重算 T 及之前的信号和目标权重, 必须与全样本逐位一致.
扰动: 只改 T 之后的价格 / 净值 / 收益, T 及之前的信号和权重必须不变.
09:00: 决策日当天的 K 线还不在样本里时, 用前一交易日信号 + 已经走完的状态,
      得到的目标必须等于全样本回测在该决策日的 regular_target.

用法:
  python3 scripts/audit_lookahead_v39.py          # 30 个随机日 + 近 10 个周五
  python3 scripts/audit_lookahead_v39.py --quick  # 少抽样, 改代码时用
退出码非 0 表示检测到泄漏或清单与回测不一致.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import strategy as strategy_mod
import data_prep
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy, live_rebalance_target
from swaps import pnl_panel, _gap_fill

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
START = "2014-06-23"
REAL_START = "2025-04-23"
OOS_START = "2022-01-04"
KEYS = ["159232", "515100", "159941", "513500", "159952", "cash"]
ATOL = 1e-8


def load_cfg(ver):
    return json.load(open(os.path.join(REF, f"final_cfg_{ver}.json")))


def pack(e):
    return {k: float(e[k]) for k in ("cagr", "max_dd", "calmar", "sharpe", "turnover")}


def fmt(e):
    return f"{e['cagr']*100:.2f}% / {e['max_dd']*100:.2f}% / {e['calmar']:.3f}"


def execute(R, pnl, cfg, start, bond, end=None):
    ds = DynamicStrategy(R, cfg=cfg)
    emergencies = []
    raw = ds.daily_fn()

    def wrapped(dt, prev, ctx):
        out = raw(dt, prev, ctx)
        if out is not None:
            emergencies.append(pd.Timestamp(dt).normalize())
        return out

    res = run_backtest(
        pnl, target_weights_fn=ds.target_fn(), daily_override_fn=wrapped,
        start=start, end=end, min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    return ds, res, emergencies


def max_abs_weights(a, b):
    return max(abs(float(a[k]) - float(b[k])) for k in KEYS)


def legacy_idx(index, dt, lag):
    """改 _idx 之前的公式. 决策日在索引里时, 新公式必须与它逐日相同."""
    i = int(index.searchsorted(pd.Timestamp(dt), side="right")) - 1
    return max(0, i - max(1, int(lag)))


def frame_diff(full, cut):
    common = cut.index.intersection(full.index)
    if len(common) == 0:
        return 0.0
    delta = (full.loc[common].astype(float) - cut.loc[common].astype(float)).abs()
    return float(np.nanmax(delta.to_numpy()))


def own_returns(code):
    direct = os.path.join(data_prep.DATA_DIR, f"{code}_qfq.csv")
    uni = os.path.join(data_prep.DATA_DIR, "v39_universe", "px", f"{code}_qfq.csv")
    path = direct if os.path.exists(direct) else uni
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()


def swapped_pnl(base, slot, code):
    out = base.copy()
    out[slot] = _gap_fill(out[slot], own_returns(code), out.index)
    return out


def slice_metrics(res, start):
    r = res["rets"].loc[res["rets"].index >= start]
    w = (1 + r).cumprod()
    ev = evaluate({"rets": r, "wealth": w, "weights": res["weights"].loc[r.index],
                   "name": "", "turnover": 0})
    return pack(ev)


def main(n_random=30, n_friday=10):
    cfg = load_cfg("v39")
    cfg37 = load_cfg("v37")
    cfg30 = load_cfg("v30")
    bond39 = rets_from(read_table("511260_nav.csv"), "cum_nav")
    bond30 = rets_from(read_table("511010_nav.csv"), "cum_nav")
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    pnl = pnl_panel(R, "proxy", cfg)
    report = {"failures": []}

    print("===== 1. 索引内 _idx 与旧公式逐日相同（回测路径不因修复改数字） =====", flush=True)
    probe = DynamicStrategy(R, cfg=cfg)
    mism = 0
    for dt in probe.R.index:
        if probe.sig._idx(dt) != legacy_idx(probe.R.index, dt, probe.sig.lag):
            mism += 1
    print(f"  不一致 {mism} / {len(probe.R.index)}")
    if mism:
        report["failures"].append(f"in-index _idx mismatch {mism}")

    print("===== 2. 全样本 v39 / v37 / v30 =====", flush=True)
    runs = {}
    for ver, c, bond in (("v39", cfg, bond39), ("v37", cfg37, bond39), ("v30", cfg30, bond30)):
        for layer, Rs, start in (("proxy", R, START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
            p = pnl_panel(Rs, "real" if layer == "real" else "proxy", c)
            ds, res, em = execute(Rs, p, c, start, bond)
            e = evaluate(res)
            runs[(ver, layer)] = (ds, res, em, e)
            print(f"  {ver:4} {layer:5} {fmt(e)}", flush=True)
            report.setdefault("metrics", {}).setdefault(ver, {})[layer] = pack(e)
    full_ds, full_res, emergencies, _full_e = runs[("v39", "proxy")]
    report["v39_proxy_emergencies"] = {
        "n": len(emergencies),
        "first": str(emergencies[0].date()) if emergencies else None,
        "last": str(emergencies[-1].date()) if emergencies else None,
    }
    dw = full_res["weights"].diff().abs().sum(axis=1)
    report["v39_proxy_emergencies"]["weight_change_days_gt_1pp"] = int((dw > 0.01).sum())
    print(f"  v39 日度紧急覆盖 {len(emergencies)} 次; 权重变动>1pp 的交易日 {int((dw > 0.01).sum())} 次")

    trade_idx = full_res["weights"].index
    rng = np.random.default_rng(20261110)
    pool = trade_idx[(trade_idx >= "2016-01-04") & (trade_idx <= trade_idx[-15])]
    chosen = [pd.Timestamp(pool[i]) for i in sorted(rng.choice(len(pool), size=n_random, replace=False))]
    fridays = [pd.Timestamp(d).normalize() for d in trade_idx if int(d.weekday()) == 4][-n_friday:]
    report["sample"] = {
        "random": [str(d.date()) for d in chosen],
        "fridays": [str(d.date()) for d in fridays],
    }

    print(f"===== 3. 截断测试 {len(chosen)} 随机日 + {len(fridays)} 周五 =====", flush=True)
    trunc_worst = 0.0
    for n, T in enumerate(list(chosen) + list(fridays), 1):
        Rt = R.loc[:T]
        pt = pnl.loc[:T]
        ds_t, res_t, _ = execute(Rt, pt, cfg, START, bond39)
        wdiff = frame_diff(full_res["weights"], res_t["weights"])
        rdiff = frame_diff(full_res["rets"].to_frame("r"), res_t["rets"].to_frame("r"))
        td = 0.0
        missing = []
        for dt, tgt in ds_t.decisions.items():
            if dt > T:
                continue
            ref = full_ds.decisions.get(dt)
            if ref is None:
                missing.append(str(dt.date()))
                continue
            td = max(td, max_abs_weights(tgt, ref))
        worst = max(wdiff, rdiff, td)
        trunc_worst = max(trunc_worst, worst)
        if worst > ATOL or missing:
            report["failures"].append(
                f"trunc {T.date()} w {wdiff:.3e} r {rdiff:.3e} tgt {td:.3e} missing {missing[:3]}")
            print(f"  FAIL {T.date()} {worst:.3e}", flush=True)
        elif n % 10 == 0:
            print(f"  {n} done, worst {trunc_worst:.3e}", flush=True)
    print(f"  截断最大绝对差 {trunc_worst:.3e}")
    report["trunc_max_abs"] = trunc_worst

    print("===== 4. 未来扰动测试（同一批日期） =====", flush=True)
    real_read = strategy_mod.pd.read_csv
    pert_worst = 0.0

    def make_reader(cutoff):
        def perturb_csv(path, *args, **kwargs):
            df = real_read(path, *args, **kwargs)
            if not isinstance(df, pd.DataFrame) or "date" not in df.columns:
                return df
            d = pd.to_datetime(df["date"])
            m = (d > cutoff).to_numpy()
            if not m.any():
                return df
            phase = np.arange(len(df))
            factor = np.where(m, 1.0 + 0.07 * np.sin(phase * 0.63 + 0.4), 1.0)
            df = df.copy()
            if "close" in df.columns:
                df.loc[m, "close"] = df.loc[m, "close"].to_numpy(dtype=float) * factor[m]
            if "unit_nav" in df.columns:
                nav_factor = np.where(m, 1.0 + 0.05 * np.cos(phase * 0.41), 1.0)
                df.loc[m, "unit_nav"] = df.loc[m, "unit_nav"].to_numpy(dtype=float) * nav_factor[m]
            if "cum_nav" in df.columns:
                df.loc[m, "cum_nav"] = df.loc[m, "cum_nav"].to_numpy(dtype=float) * (factor[m] * 1.02)
            return df
        return perturb_csv

    try:
        for n, T in enumerate(list(chosen) + list(fridays), 1):
            strategy_mod.pd.read_csv = make_reader(T)
            shock = 0.03 * np.sin(np.arange(len(R)) * 0.29 + 1.7)
            m = np.asarray(R.index > T)
            R2 = R.copy()
            R2.iloc[np.flatnonzero(m)] = R2.to_numpy()[m] + shock[m][:, None]
            p2 = pnl.copy()
            pm = np.asarray(p2.index > T)
            if int(pm.sum()) != int(m.sum()):
                raise RuntimeError("pnl index diverged from signal panel")
            p2.iloc[np.flatnonzero(pm)] = p2.to_numpy()[pm] + shock[m][:, None]
            b2 = bond39.copy()
            b2.loc[b2.index > T] = b2.loc[b2.index > T] + 0.01
            ds_p, res_p, _ = execute(R2, p2, cfg, START, b2)
            wdiff = frame_diff(full_res["weights"].loc[:T], res_p["weights"].loc[:T])
            td = 0.0
            for dt, tgt in full_ds.decisions.items():
                if dt > T:
                    continue
                got = ds_p.decisions.get(dt)
                if got is None:
                    td = 1.0
                    break
                td = max(td, max_abs_weights(tgt, got))
            worst = max(wdiff, td)
            pert_worst = max(pert_worst, worst)
            if worst > ATOL:
                report["failures"].append(f"perturb {T.date()} w {wdiff:.3e} tgt {td:.3e}")
                print(f"  FAIL {T.date()} {worst:.3e}", flush=True)
            elif n % 10 == 0:
                print(f"  {n} done, worst {pert_worst:.3e}", flush=True)
    finally:
        strategy_mod.pd.read_csv = real_read
    print(f"  扰动最大绝对差 {pert_worst:.3e}")
    report["perturb_max_abs"] = pert_worst

    print(f"===== 5. 周五 09:00 信息集 vs 全样本决策（近 {len(fridays)} 个周五） =====", flush=True)
    live_rows = []
    old_max = 0.0
    for fri in fridays:
        Rcut = R.loc[R.index < fri]
        pcut = pnl.loc[pnl.index < fri]
        ds_c, res_c, _ = execute(Rcut, pcut, cfg, START, bond39)
        ref = full_ds.decisions[fri]
        # helper 内部只调用一次 regular_target; 再调一次会把滞回打两次
        decision, signal_dt, via_helper = live_rebalance_target(ds_c, res_c["rets"], 4)
        err = max_abs_weights(via_helper, ref)
        err_h = err
        cold = DynamicStrategy(Rcut, cfg=dict(cfg, signal_lag=0))
        old = cold.regular_target(Rcut.index[-1], {"pf_rets": pd.Series(dtype=float)})
        old_err = max_abs_weights(old, ref)
        old_max = max(old_max, old_err)
        if err > ATOL or err_h > ATOL or signal_dt != pd.Timestamp(Rcut.index[-1]).normalize():
            report["failures"].append(
                f"9am {fri.date()} replay {err:.3e} helper {err_h:.3e} signal {signal_dt.date()}")
        elif decision != fri:
            report["failures"].append(f"9am helper decision {decision.date()} != {fri.date()}")
        live_rows.append({
            "decision": str(fri.date()),
            "signal": str(pd.Timestamp(Rcut.index[-1]).date()),
            "replay_max_abs": err,
            "helper_max_abs": err_h,
            "old_cold_lastbar_max_abs": old_err,
        })
        print(f"  {fri.date()} 信号 {pd.Timestamp(Rcut.index[-1]).date()} "
              f"修复后 {err:.2e}  旧冷启动清单 {old_err*100:.2f}pp", flush=True)
    report["friday_9am"] = live_rows
    if old_max < 0.01:
        report["failures"].append(f"old live path unexpectedly close to backtest, max {old_max}")
    dec, sig, tgt = live_rebalance_target(full_ds, full_res["rets"], 4)
    last_fri = fridays[-1]
    if dec != last_fri or max_abs_weights(tgt, full_ds.decisions[last_fri]) > ATOL:
        report["failures"].append(f"post-close list {dec.date()} != last friday decision {last_fri.date()}")
    report["post_close_list"] = {"decision": str(dec.date()), "signal": str(sig.date())}

    print("===== 6. 对齐与选择的反事实（不改默认配置） =====", flush=True)

    def variant(ver, c, bond, Rs, p, start, label):
        _ds, res, _em = execute(Rs, p, c, start, bond)
        e = evaluate(res)
        report.setdefault("counterfactual", {}).setdefault(label, {})[ver] = pack(e)
        print(f"  {label:28} {ver:4} {fmt(e)}", flush=True)
        return res

    for ver, c, bond in (("v39", cfg, bond39), ("v37", cfg37, bond39), ("v30", cfg30, bond30)):
        for layer, Rs, start in (("proxy", R, START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
            lagged = Rs.copy()
            lagged[["159941", "513500"]] = lagged[["159941", "513500"]].shift(1)
            base_p = pnl_panel(Rs, "real" if layer == "real" else "proxy", c)
            variant(ver, c, bond, lagged, base_p, start, f"qdii_sig_lag1_{layer}")

    for ver, c, bond in (("v39", cfg, bond39), ("v37", cfg37, bond39), ("v30", cfg30, bond30)):
        for layer, Rs, start in (("proxy", R, START), ("real", Rr, REAL_START)):
            p = pnl_panel(Rs, "real" if layer == "real" else "proxy", c).copy()
            p[["159941", "513500"]] = p[["159941", "513500"]].shift(1)
            variant(ver, c, bond, Rs, p.fillna(0.0), start, f"qdii_pnl_shift1_{layer}")

    variant("v39", cfg, bond39, R, pnl.fillna(0.0), START, "pnl_fillna0_proxy")
    wk = R.index.day_of_week < 5
    variant("v39", cfg, bond39, R.loc[wk], pnl.loc[wk], START, "drop_weekend_proxy")

    saved = dict(data_prep.DIV_YIELD)
    data_prep.DIV_YIELD = {}
    try:
        R_nodiv, _ = build_panel("proxy")
    finally:
        data_prep.DIV_YIELD = saved
    p_nodiv = pnl_panel(R_nodiv, "proxy", cfg)
    variant("v39", cfg, bond39, R_nodiv, p_nodiv, START, "no_proxy_div_proxy")
    variant("v39", cfg, bond39, R_nodiv, p_nodiv, OOS_START, "no_proxy_div_oos")

    r39 = runs[("v39", "proxy")][1]["rets"]
    r37 = runs[("v37", "proxy")][1]["rets"]
    yearly = []
    for year, idx in r39.groupby(r39.index.year).groups.items():
        a = float((1 + r39.loc[idx]).prod() - 1)
        b = float((1 + r37.loc[idx]).prod() - 1)
        yearly.append({"year": int(year), "v39": a, "v37": b, "excess": a - b})
    report["v39_minus_v37_by_year"] = yearly
    print("  v39-v37 分年收益差:", ", ".join(
        f"{row['year']} {row['excess']*100:+.2f}pp" for row in yearly))

    q222 = own_returns("159222")
    own_252 = pd.Timestamp(q222.index[251]).normalize()
    report["own_252"] = {
        "159222": str(own_252.date()),
        "563020": str(pd.Timestamp(own_returns("563020").index[251]).date()),
    }
    print(f"  159222 自有样本满 252 日: {own_252.date()}")

    def rank_at(cutoff, codes):
        Rc = R.loc[:cutoff]
        out = {}
        ds, res, _ = execute(Rc, Rc, cfg37, START, bond39)
        base_e = evaluate(res)
        oos = slice_metrics(res, OOS_START)
        out["v37"] = {"proxy": pack(base_e), "oos_cal": oos["calmar"]}
        for slot, code in codes:
            p = swapped_pnl(Rc, slot, code)
            _ds, res, _ = execute(Rc, p, cfg37, START, bond39)
            e = evaluate(res)
            oe = slice_metrics(res, OOS_START)
            own = own_returns(code)
            own = own.loc[:cutoff]
            out[code] = {
                "proxy": pack(e),
                "dcal": float(e["calmar"] - base_e["calmar"]),
                "oos_cal": oe["calmar"],
                "own_n": int(len(own)),
                "own_start": str(own.index.min().date()) if len(own) else None,
            }
        return out

    slots = [("159232", "159222"), ("159232", "159201"),
             ("515100", "563020"), ("515100", "512890"), ("515100", "159547")]
    print("  前向选券（只看到截止日）...", flush=True)
    report["forward_2025_12_31"] = rank_at("2025-12-31", slots)
    report["forward_159222_252"] = rank_at(own_252, slots)

    def pick(table, names):
        best, best_cal = None, -1e9
        for code in names:
            cal = table[code]["proxy"]["calmar"]
            if cal > best_cal:
                best, best_cal = code, cal
        return best

    tab = report["forward_159222_252"]
    fcf = pick(tab, ["159222", "159201"])
    div = pick(tab, ["563020", "512890", "159547"])
    report["forward_pick_at_252"] = {"fcf": fcf, "div": div, "asof": str(own_252.date())}
    p_pick = swapped_pnl(swapped_pnl(pnl_panel(R, "proxy", cfg37), "159232", fcf), "515100", div)
    _ds, res_pick, _ = execute(R, p_pick, cfg37, START, bond39)
    _ds, res_base, _ = execute(R, pnl_panel(R, "proxy", cfg37), cfg37, START, bond39)
    after = own_252 + pd.Timedelta(days=1)
    report["forward_after_252"] = {
        "picked": slice_metrics(res_pick, after),
        "v37": slice_metrics(res_base, after),
        "v39_published_combo": slice_metrics(full_res, after),
    }
    print("  252 日之后 picked", fmt(report["forward_after_252"]["picked"]),
          " v37", fmt(report["forward_after_252"]["v37"]),
          " v39combo", fmt(report["forward_after_252"]["v39_published_combo"]))

    path = os.path.join(OUT, "audit_lookahead_v39.json")
    json.dump(report, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n[ok] {path}")
    if report["failures"]:
        print("FAILURES:")
        for item in report["failures"]:
            print(" -", item)
        return 1
    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    sys.exit(main(5, 3) if quick else main(30, 10))
