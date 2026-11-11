# -*- coding: utf-8 -*-
"""Independent paper audit of default v40. Does not change the default config.

Uses scripts/engine.py run_backtest + evaluate and scripts/strategy.py only.
Does not fetch market data. Windows match the published ones:
  proxy 2014-06-23 → panel end, real 2025-04-23 → panel end, OOS 2022-01-04 → panel end.
"""
import copy, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import DATA_DIR, build_panel, read_table, rets_from
from engine import evaluate, run_backtest
from strategy import DynamicStrategy, live_rebalance_target
from swaps import pnl_panel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REF = os.path.join(ROOT, "references")
START = "2014-06-23"
REAL_START = "2025-04-23"
OOS_START = "2022-01-04"
SLOTS = ["159232", "515100", "159941", "513500", "159952"]
PERIODS = {
    "2014H2-2015牛": ("2014-06-23", "2015-06-12"),
    "2015股灾": ("2015-06-15", "2016-02-29"),
    "2016-2017修复": ("2016-03-01", "2017-12-29"),
    "2018熊市": ("2018-01-02", "2019-01-03"),
    "2019-2021牛": ("2019-01-04", "2021-02-18"),
    "2021-2022熊": ("2021-02-19", "2022-10-31"),
    "2023-2024震荡": ("2023-01-03", "2024-08-30"),
    "2024Q4-2026双牛": ("2024-09-02", "2026-11-10"),
}


def load_cfg(ver):
    return json.load(open(os.path.join(REF, f"final_cfg_{ver}.json")))


def pack(e):
    return {
        "cagr": float(e["cagr"]),
        "max_dd": float(e["max_dd"]),
        "sharpe": float(e["sharpe"]),
        "calmar": float(e["calmar"]),
        "turnover": float(e["turnover"]),
        "avg_cash": float(e["avg_cash"]),
        "mdd_peak": str(pd.Timestamp(e["mdd_peak"]).date()) if e.get("mdd_peak") is not None else None,
        "mdd_trough": str(pd.Timestamp(e["mdd_trough"]).date()) if e.get("mdd_trough") is not None else None,
        "n": int(e["cagr"] == e["cagr"]),
    }


def fmt(e):
    return (f"{e['cagr']*100:.4f}% / {e['max_dd']*100:.4f}% / "
            f"Sh {e['sharpe']:.4f} / Cal {e['calmar']:.6f}")


def execute(R, pnl, cfg, start, bond, end=None, fee=None):
    ds = DynamicStrategy(R, cfg=cfg)
    kwargs = dict(
        target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    if fee is not None:
        kwargs["fee"] = fee
    res = run_backtest(pnl, **kwargs)
    ev = evaluate(res, periods=PERIODS if start == START and end is None else None)
    return ds, res, ev


def with_cap(cfg, mult):
    """Scale only v39 state 7-9 growth caps. Defense column is left as in cfg."""
    c = copy.deepcopy(cfg)
    base = {7: 65.0, 8: 85.0, 9: 81.0}
    for k, g0 in base.items():
        _g, d = c["state_map"][str(k)]
        c["state_map"][str(k)] = [g0 * mult, d]
    return c


def with_published_cap(cfg):
    c = copy.deepcopy(cfg)
    c["state_map"]["7"] = [71.5, 10]
    c["state_map"]["8"] = [93.5, 6]
    c["state_map"]["9"] = [89.1, 3]
    return c


def with_split(cfg, cyb):
    c = copy.deepcopy(cfg)
    c["growth_split_bull"] = {
        "159952": round(cyb, 10),
        "159941": round(0.93 - cyb, 10),
        "513500": 0.07,
    }
    return c


def with_cool(cfg, weeks):
    c = copy.deepcopy(cfg)
    c["cool_off_weeks"] = int(weeks)
    return c


def slice_metrics(res, start, end=None):
    r = res["rets"]
    r = r.loc[r.index >= start]
    if end is not None:
        r = r.loc[r.index <= end]
    w = (1 + r).cumprod()
    ev = evaluate({"rets": r, "wealth": w, "weights": res["weights"].loc[r.index],
                   "name": "", "turnover": 0})
    # turnover over the slice is not the engine total; leave the full-run turnover out
    out = pack(ev)
    out["turnover"] = None
    out["n_days"] = int(len(r))
    out["start"] = str(r.index.min().date())
    out["end"] = str(r.index.max().date())
    return out


def yearly(rets):
    rows = []
    for year, idx in rets.groupby(rets.index.year).groups.items():
        sub = rets.loc[idx]
        rows.append({"year": int(year), "total": float((1 + sub).prod() - 1), "n": int(len(sub))})
    return rows


def block_bootstrap(rets, block=20, n_boot=2000, seed=20261110, circular=True):
    r = np.asarray(rets, dtype=float)
    nobs = len(r)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(nobs / block))
    cals = np.empty(n_boot)
    cagrs = np.empty(n_boot)
    mdds = np.empty(n_boot)
    max_start = nobs if not circular else nobs
    for i in range(n_boot):
        if circular:
            starts = rng.integers(0, nobs, size=nblocks)
        else:
            starts = rng.integers(0, max(nobs - block, 1), size=nblocks)
        parts = []
        for s in starts:
            if circular:
                ix = (np.arange(block) + int(s)) % nobs
                parts.append(r[ix])
            else:
                parts.append(r[int(s):int(s) + block])
        sample = np.concatenate(parts)[:nobs]
        w = np.cumprod(1.0 + sample)
        peak = np.maximum.accumulate(w)
        dd = w / peak - 1.0
        mdd = float(dd.min())
        cagr = float(w[-1] ** (252.0 / nobs) - 1.0)
        cal = cagr / abs(mdd) if mdd < 0 else np.nan
        cals[i] = cal
        cagrs[i] = cagr
        mdds[i] = mdd
    def q(a):
        return {k: float(np.nanquantile(a, p)) for k, p in
                (("p05", 0.05), ("p50", 0.50), ("p95", 0.95))}
    return {
        "n": n_boot, "block": block, "seed": seed, "circular": circular, "nobs": nobs,
        "p_cal_lt_1": float(np.mean(cals < 1.0)),
        "p_cal_lt_0": float(np.mean(cals < 0.0)),
        "p_cagr_lt_0.043": float(np.mean(cagrs < 0.043)),
        "p_cagr_lt_0": float(np.mean(cagrs < 0.0)),
        "calmar": q(cals), "cagr": q(cagrs), "mdd": q(mdds),
    }


def shock_bond(bond, start, days=20, total=-0.08):
    b = bond.copy()
    idx = b.index[b.index >= pd.Timestamp(start)]
    take = idx[:days]
    if len(take) < days:
        raise RuntimeError(f"bond shock {start} only {len(take)} days")
    r = (1.0 + total) ** (1.0 / days) - 1.0
    b.loc[take] = r
    return b, [str(t.date()) for t in take]


def config_diff(a, b):
    keys = sorted(set(a) | set(b))
    diffs = []
    for k in keys:
        if a.get(k) != b.get(k):
            diffs.append({"key": k, "v39": a.get(k, "<missing>"), "v40": b.get(k, "<missing>")})
    return diffs


def weekend_count(path):
    df = pd.read_csv(path, parse_dates=["date"])
    d = pd.to_datetime(df["date"])
    wk = d.dt.day_of_week >= 5
    return {
        "n": int(len(df)),
        "weekend": int(wk.sum()),
        "dup_dates": int(d.duplicated().sum()),
        "min": str(d.min().date()) if len(d) else None,
        "max": str(d.max().date()) if len(d) else None,
        "after_2026_11_10": int((d > "2026-11-10").sum()),
        "weekend_dates": [str(x.date()) for x in d[wk].tolist()[:12]],
    }


def main():
    report = {"tip_expected": "0f91fe84e2daf2150ee65f9b979f1e84e81af119",
              "engine": "scripts/engine.py run_backtest strict=True fee default 0.0005",
              "sharpe_rf": "evaluate() default rf=0.018, not cfg repo_rate 0.022"}
    cfg40 = load_cfg("v40")
    cfg39 = load_cfg("v39")
    report["config_diff_v39_v40"] = config_diff(cfg39, cfg40)
    print("config diffs", [d["key"] for d in report["config_diff_v39_v40"]], flush=True)

    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    bond = rets_from(read_table("511260_nav.csv"), "cum_nav")
    report["panel"] = {
        "proxy": [str(R.index.min().date()), str(R.index.max().date()), int(len(R))],
        "real": [str(Rr.index.min().date()), str(Rr.index.max().date()), int(len(Rr))],
        "bond": [str(bond.index.min().date()), str(bond.index.max().date()), int(len(bond))],
        "proxy_weekend_rows": int((R.index.day_of_week >= 5).sum()),
        "real_weekend_rows": int((Rr.index.day_of_week >= 5).sum()),
    }
    print("panel", report["panel"], flush=True)

    runs = {}

    def run_one(label, cfg, layer):
        if layer == "real":
            Rs, start = Rr, REAL_START
        elif layer == "oos":
            Rs, start = R, OOS_START
        else:
            Rs, start = R, START
        pnl = pnl_panel(Rs, "real" if layer == "real" else "proxy", cfg)
        ds, res, ev = execute(Rs, pnl, cfg, start, bond)
        runs[(label, layer)] = (ds, res, ev)
        print(f"  {label:16} {layer:5} {fmt(pack(ev))}  end {res['rets'].index.max().date()}", flush=True)
        return ds, res, ev

    print("===== published windows v40 / v39 =====", flush=True)
    metrics = {}
    for label, cfg in (("v40", cfg40), ("v39", cfg39)):
        metrics[label] = {}
        for layer in ("proxy", "real", "oos"):
            _ds, res, ev = run_one(label, cfg, layer)
            metrics[label][layer] = pack(ev)
            metrics[label][layer]["start"] = str(res["rets"].index.min().date())
            metrics[label][layer]["end"] = str(res["rets"].index.max().date())
            metrics[label][layer]["n_days"] = int(len(res["rets"]))
    report["metrics"] = metrics

    # three-gate on raw Calmar
    gate = {}
    for layer in ("proxy", "real", "oos"):
        a = metrics["v40"][layer]["calmar"]
        b = metrics["v39"][layer]["calmar"]
        gate[layer] = {"v40": a, "v39": b, "delta": a - b, "not_worse": bool(a + 1e-12 >= b)}
    gate["pass"] = all(gate[k]["not_worse"] for k in ("proxy", "real", "oos"))
    report["three_gate_calmar"] = gate
    print("三关", gate, flush=True)

    # periods + yearly from the proxy runs (state carries from 2014, same as published table)
    ds40, res40, ev40 = runs[("v40", "proxy")]
    ds39, res39, ev39 = runs[("v39", "proxy")]
    periods = {}
    for name, (ps, pe) in PERIODS.items():
        row = {"v40": slice_metrics(res40, ps, pe), "v39": slice_metrics(res39, ps, pe)}
        row["v40_total"] = row["v40"]["cagr"]  # placeholder overwritten below
        for lab, res in (("v40", res40), ("v39", res39)):
            sub = res["rets"][(res["rets"].index >= ps) & (res["rets"].index <= pe)]
            sw = (1 + sub).cumprod()
            from metrics import max_drawdown
            mdd = max_drawdown(sw)[0]
            row[lab] = {"total": float(sw.iloc[-1] - 1), "max_dd": float(mdd), "n": int(len(sub))}
        periods[name] = row
    report["periods_proxy"] = periods

    y40 = {r["year"]: r["total"] for r in yearly(res40["rets"])}
    y39 = {r["year"]: r["total"] for r in yearly(res39["rets"])}
    years = []
    for y in sorted(set(y40) | set(y39)):
        a, b = y40.get(y), y39.get(y)
        ld = None if a is None or b is None else float(np.log1p(a) - np.log1p(b))
        years.append({"year": y, "v40": a, "v39": b, "excess_total": None if a is None else a - b,
                      "log_excess": ld})
    pos = sum(r["log_excess"] for r in years if r["log_excess"] and r["log_excess"] > 0)
    for r in years:
        r["share_of_positive_log_excess"] = (r["log_excess"] / pos) if r["log_excess"] and r["log_excess"] > 0 and pos else 0.0
    report["yearly_proxy"] = years
    print("yearly excess", ", ".join(f"{r['year']} {r['excess_total']*100:+.2f}pp" for r in years), flush=True)

    # ablation: each of the three changes, all three windows
    print("===== ablation =====", flush=True)
    ablation_cfgs = {
        "v39": cfg39,
        "cap_only": with_published_cap(cfg39),
        "split_only": with_split(cfg39, 0.32),
        "cool_only": with_cool(cfg39, 3),
        "cap_split": with_split(with_published_cap(cfg39), 0.32),
        "cap_cool": with_cool(with_published_cap(cfg39), 3),
        "split_cool": with_cool(with_split(cfg39, 0.32), 3),
        "v40": cfg40,
    }
    abl = {}
    abl_rets = {}
    for name, cfg in ablation_cfgs.items():
        abl[name] = {}
        for layer, Rs, start in (("proxy", R, START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
            if name in ("v39", "v40") and layer in metrics[name]:
                abl[name][layer] = {k: metrics[name][layer][k] for k in ("cagr", "max_dd", "sharpe", "calmar")}
                if layer == "proxy":
                    abl_rets[name] = runs[(name, "proxy")][1]["rets"]
                continue
            pnl = pnl_panel(Rs, "real" if layer == "real" else "proxy", cfg)
            _ds, res, ev = execute(Rs, pnl, cfg, start, bond)
            abl[name][layer] = {k: float(ev[k]) for k in ("cagr", "max_dd", "sharpe", "calmar")}
            if layer == "proxy":
                abl_rets[name] = res["rets"]
            print(f"  {name:12} {layer:5} {fmt(pack(ev))}", flush=True)
    report["ablation"] = abl
    mech_year = {}
    for name, rets in abl_rets.items():
        mech_year[name] = {r["year"]: r["total"] for r in yearly(rets)}
    report["ablation_yearly_total"] = mech_year

    # neighbors on top of the other two v40 choices
    print("===== neighbors =====", flush=True)
    neighbors = []

    def add_neighbor(kind, label, cfg):
        row = {"kind": kind, "label": label}
        for layer, Rs, start in (("proxy", R, START), ("real", Rr, REAL_START), ("oos", R, OOS_START)):
            pnl = pnl_panel(Rs, "real" if layer == "real" else "proxy", cfg)
            _ds, _res, ev = execute(Rs, pnl, cfg, start, bond)
            row[layer] = {k: float(ev[k]) for k in ("cagr", "max_dd", "sharpe", "calmar")}
        neighbors.append(row)
        p = row["proxy"]
        print(f"  {kind:8} {label:10} proxy {p['cagr']*100:.2f}/{p['max_dd']*100:.2f}/Cal {p['calmar']:.3f} "
              f"real Cal {row['real']['calmar']:.3f} oos Cal {row['oos']['calmar']:.3f}", flush=True)

    for mult in (1.00, 1.05, 1.10, 1.15, 1.20):
        cfg = with_cool(with_split(with_cap(cfg39, mult), 0.32), 3)
        if abs(mult - 1.10) < 1e-12:
            cfg = cfg40  # published 71.5/93.5/89.1, not binary float
        add_neighbor("cap", f"x{mult:.2f}", cfg)
    for cyb in (0.30, 0.31, 0.32, 0.33, 0.34):
        cfg = with_cool(with_split(with_published_cap(cfg39), cyb), 3)
        if abs(cyb - 0.32) < 1e-12:
            cfg = cfg40
        add_neighbor("split", f"cyb{cyb:.0%}", cfg)
    for weeks in (0, 2, 3, 4):
        cfg = with_cool(with_split(with_published_cap(cfg39), 0.32), weeks)
        if weeks == 3:
            cfg = cfg40
        add_neighbor("cool", f"{weeks}w", cfg)
    report["neighbors"] = neighbors

    # fees. Targets do not depend on fee (pf_rets fee drag is unused while speed_brake/vol_gate are off);
    # still re-run the official engine so the return path is the engine's.
    print("===== fees =====", flush=True)
    fees = {}
    pnl40 = pnl_panel(R, "proxy", cfg40)
    for bps in (5, 10, 15, 20):
        _ds, _res, ev = execute(R, pnl40, cfg40, START, bond, fee=bps / 10000.0)
        fees[str(bps)] = pack(ev)
        print(f"  {bps}bp {fmt(pack(ev))}", flush=True)
    report["fees_proxy_v40"] = fees

    # bond shock. No committed script; this is a literal reconstruction:
    # replace 20 bond sessions from the start date with a flat compound path totaling -8%.
    print("===== bond shock =====", flush=True)
    shocks = {}
    for start in ("2015-06-15", "2022-02-07"):
        b2, days = shock_bond(bond, start)
        _ds, _res, ev = execute(R, pnl40, cfg40, START, b2)
        shocks[start] = {"days": days, **pack(ev)}
        print(f"  shock {start} MDD {ev['max_dd']*100:.2f}% Cal {ev['calmar']:.3f}", flush=True)
    both = bond.copy()
    for start in ("2015-06-15", "2022-02-07"):
        both, _days = shock_bond(both, start)
    _ds, _res, ev = execute(R, pnl40, cfg40, START, both)
    shocks["both"] = pack(ev)
    report["bond_shock"] = shocks
    print(f"  both MDD {ev['max_dd']*100:.2f}%", flush=True)

    print("===== bootstrap =====", flush=True)
    boots = {}
    for label, res in (("v40", res40), ("v39", res39)):
        boots[label] = {
            "circular_seed_20261110": block_bootstrap(res["rets"], circular=True, seed=20261110),
            "circular_seed_0": block_bootstrap(res["rets"], circular=True, seed=0),
            "noncircular_seed_20261110": block_bootstrap(res["rets"], circular=False, seed=20261110),
        }
        b = boots[label]["circular_seed_20261110"]
        print(f"  {label} P(Cal<1)={b['p_cal_lt_1']*100:.2f}% P(CAGR<4.3%)={b['p_cagr_lt_0.043']*100:.2f}%", flush=True)
    report["bootstrap"] = boots
    report["bootstrap_note"] = ("No bootstrap script is in the repo. This is a 20-day circular block "
                                 "bootstrap of the official-engine daily returns, 2000 draws. It does not "
                                 "re-simulate the strategy on resampled prices.")

    # end-date sensitivity (the README residue paragraph quotes 12.13/12.15/12.17, which is the v39 scale)
    print("===== end-date sensitivity =====", flush=True)
    ends = {}
    for ver, cfg in (("v40", cfg40), ("v39", cfg39)):
        ends[ver] = {}
        pnl = pnl_panel(R, "proxy", cfg)
        for end in ("2026-11-06", "2026-11-09", "2026-11-10"):
            _ds, _res, ev = execute(R, pnl, cfg, START, bond, end=end)
            ends[ver][end] = pack(ev)
            print(f"  {ver} end {end} {ev['cagr']*100:.4f}%", flush=True)
    report["end_date_proxy"] = ends

    # ffill / weekend counterfactuals on v40 proxy
    print("===== data counterfactuals =====", flush=True)
    _ds, _res, ev = execute(R, pnl40.fillna(0.0), cfg40, START, bond)
    report["pnl_fillna0_proxy"] = pack(ev)
    wk = R.index.day_of_week < 5
    _ds, _res, ev = execute(R.loc[wk], pnl40.loc[wk], cfg40, START, bond)
    report["drop_weekend_proxy"] = pack(ev)
    print("  fillna0", fmt(report["pnl_fillna0_proxy"]), "drop_weekend", fmt(report["drop_weekend_proxy"]), flush=True)

    # NaN gaps inside the proxy backtest window before engine ffill
    window = pnl40.loc[pnl40.index >= START]
    nan_report = {}
    for col in window.columns:
        s = window[col]
        nan_report[col] = {"nan": int(s.isna().sum()), "nan_dates_head": [str(d.date()) for d in s.index[s.isna()][:8]]}
    report["pnl_nans_in_proxy_window"] = nan_report

    # cool-off path inspection on the v40 proxy run already computed
    print("===== cool_off path =====", flush=True)
    cool = inspect_cool(R, pnl40, cfg40, bond, res40, ds40)
    report["cool_off"] = cool
    print("  episodes", cool["n_episodes"], "forced_week_hist", cool["forced_decision_hist"],
          "idx_violations", cool["n_idx_violations"], "daily_bypass", cool["n_daily_bypass"], flush=True)

    # Friday 2026-11-13 observe-only list from the same proxy run
    print("===== friday list =====", flush=True)
    report["friday_list"] = friday_list(ds40, res40, cfg40)
    print(" ", json.dumps(report["friday_list"]["summary"], ensure_ascii=False), flush=True)

    # v39 switch print uses the signal panel, not the swapped pnl panel. Quantify the last-day gap.
    ds_sig = DynamicStrategy(R, cfg=cfg39)
    res_sig = run_backtest(R, target_weights_fn=ds_sig.target_fn(), daily_override_fn=ds_sig.daily_fn(),
                           start=START, min_delta=cfg39.get("min_delta", 0.02),
                           repo=cfg39.get("repo_rate", 0.022), tranche_weights=cfg39.get("tranche_weights"),
                           cash_bond_rets=bond, cash_bond_pct=cfg39.get("cash_bond_pct", 0.0),
                           rebal_weekday=4, rebal_freq="weekly", strict=True)
    w_sig = res_sig["weights"].iloc[-1]
    w_pnl = res39["weights"].iloc[-1]
    report["v39_signal_vs_pnl_last_weight"] = {
        s: {"signal_panel": float(w_sig[s]), "pnl_panel": float(w_pnl[s]),
            "abs_diff": abs(float(w_sig[s] - w_pnl[s]))}
        for s in SLOTS + ["cash"]
    }

    # file integrity
    print("===== files =====", flush=True)
    report["files"] = scan_files()
    path = os.path.join(REF, "audit_v40_independent.json")
    json.dump(report, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print("[ok]", path, flush=True)
    return report


def inspect_cool(R, pnl, cfg, bond, full_res, full_ds):
    """Re-run once with probes. Signal index must be strictly before an in-sample decision.
    Daily override must not lift a slot that cool_off currently pins to the floor.
    """
    ds = DynamicStrategy(R, cfg=cfg)
    idx_bad = []
    episodes = []
    active = {}  # slot -> {start_decision, start_signal, forced}
    orig = ds.regular_target

    def wrapped_target(dt, ctx):
        ts = pd.Timestamp(dt).normalize()
        i = ds.sig._idx(dt)
        in_index = ts in ds.R.index or (len(ds.R.index) and ds.R.index.asof(ts) == ts)
        # asof equality is the in-index test even with a non-unique concern; index is unique
        in_index = bool((ds.R.index == ts).any())
        sig_dt = pd.Timestamp(ds.R.index[i]).normalize() if i < len(ds.R) else None
        if in_index and sig_dt is not None and sig_dt >= ts:
            idx_bad.append({"decision": str(ts.date()), "signal": str(sig_dt.date())})
        before = dict(ds._cool_from)
        out = orig(dt, ctx)
        after = dict(ds._cool_from)
        dti = sig_dt
        for s in SLOTS:
            if s not in before and s in after:
                active[s] = {"slot": s, "trigger_decision": str(ts.date()),
                             "trigger_signal": str(dti.date()) if dti is not None else None,
                             "forced_decisions_after": 0, "released_decision": None,
                             "calendar_days_at_release": None}
            elif s in before and s in after:
                if s in active:
                    active[s]["forced_decisions_after"] += 1
            elif s in before and s not in after:
                ep = active.pop(s, None)
                if ep:
                    if dti is not None and ep["trigger_signal"]:
                        ep["calendar_days_at_release"] = int((dti - pd.Timestamp(ep["trigger_signal"])).days)
                    ep["released_decision"] = str(ts.date())
                    episodes.append(ep)
        return out

    ds.regular_target = wrapped_target
    bypass = []
    raw_daily = ds.daily_fn()

    def wrapped_daily(dt, prev, ctx):
        out = raw_daily(dt, prev, ctx)
        if out is None or not ds._cool_from:
            return out
        for s, cf in ds._cool_from.items():
            floor_f = ds.floor[s] / 100.0
            got = float(out[s])
            if got > floor_f + 1e-4:
                bypass.append({"date": str(pd.Timestamp(dt).date()), "slot": s,
                               "target": got, "floor": floor_f,
                               "excess": got - floor_f,
                               "cool_from": str(pd.Timestamp(cf).date())})
        return out

    res = run_backtest(
        pnl, target_weights_fn=ds.target_fn(), daily_override_fn=wrapped_daily,
        start=START, min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"), cash_bond_rets=bond,
        cash_bond_pct=cfg.get("cash_bond_pct", 0.0), rebal_weekday=4,
        rebal_freq="weekly", strict=True)
    # weight path must match the unprobed official run (probe is observational)
    common = res["weights"].index.intersection(full_res["weights"].index)
    wdiff = float((res["weights"].loc[common] - full_res["weights"].loc[common]).abs().to_numpy().max())
    hist = {}
    for ep in episodes:
        hist[ep["forced_decisions_after"]] = hist.get(ep["forced_decisions_after"], 0) + 1
    # targeted truncation on up to 8 Fridays that fall inside an episode
    checks = []
    sample_dates = []
    for ep in episodes:
        if ep["released_decision"]:
            sample_dates.append(pd.Timestamp(ep["released_decision"]))
        sample_dates.append(pd.Timestamp(ep["trigger_decision"]))
    # unique, inside the series, not too close to the end
    seen = []
    last = full_res["weights"].index[-1] - pd.Timedelta(days=10)
    for d in sample_dates:
        if d < pd.Timestamp("2016-01-01") or d > last:
            continue
        if d not in full_ds.decisions:
            continue
        if any(abs((d - s).days) < 21 for s in seen):
            continue
        seen.append(d)
        if len(seen) >= 8:
            break
    for T in seen:
        Rt = R.loc[:T]
        pt = pnl.loc[:T]
        ds_t, res_t, _ev = execute(Rt, pt, cfg, START, bond)
        td = 0.0
        for dt, tgt in ds_t.decisions.items():
            if dt > T:
                continue
            ref = full_ds.decisions.get(dt)
            if ref is None:
                td = 1.0
                break
            td = max(td, max(abs(float(tgt[k]) - float(ref[k])) for k in list(SLOTS) + ["cash"]))
        common = full_res["weights"].index.intersection(res_t["weights"].index)
        wmax = float((full_res["weights"].loc[common] - res_t["weights"].loc[common]).abs().to_numpy().max())
        # 09:00: drop the decision day itself
        Rcut = R.loc[R.index < T]
        pcut = pnl.loc[pnl.index < T]
        ds_c, res_c, _ev = execute(Rcut, pcut, cfg, START, bond)
        _dec, sig, via = live_rebalance_target(ds_c, res_c["rets"], 4)
        ref = full_ds.decisions[T]
        err = max(abs(float(via[k]) - float(ref[k])) for k in list(SLOTS) + ["cash"])
        checks.append({"T": str(T.date()), "decision_max_abs": td, "weight_max_abs": wmax,
                       "friday_9am_max_abs": err, "signal": str(pd.Timestamp(sig).date())})
        print(f"    cool sample {T.date()} trunc {max(td, wmax):.3e} 9am {err:.3e}", flush=True)
    by_slot = {}
    for ep in episodes:
        by_slot[ep["slot"]] = by_slot.get(ep["slot"], 0) + 1
    return {
        "n_episodes": len(episodes),
        "by_slot": by_slot,
        "forced_decision_hist": {str(k): v for k, v in sorted(hist.items())},
        "n_idx_violations": len(idx_bad),
        "idx_violations_head": idx_bad[:5],
        "n_daily_bypass": len(bypass),
        "daily_bypass_head": bypass[:8],
        "daily_bypass_max_excess": max((b["excess"] for b in bypass), default=0.0),
        "probe_weight_max_abs_vs_official": wdiff,
        "open_at_end": {s: {"from": str(pd.Timestamp(v).date())} for s, v in ds._cool_from.items()},
        "targeted_lookahead": checks,
        "targeted_lookahead_max_abs": max((max(c["decision_max_abs"], c["weight_max_abs"], c["friday_9am_max_abs"])
                                           for c in checks), default=0.0),
        "calendar_days_at_release": [ep["calendar_days_at_release"] for ep in episodes if ep["calendar_days_at_release"] is not None],
    }


def friday_list(ds, res, cfg):
    decision, signal_dt, tgt = live_rebalance_target(ds, res["rets"], 4)
    w = res["weights"].iloc[-1]
    from swaps import traded_code
    rows = []
    max_abs = 0.0
    for s in SLOTS + ["cash"]:
        held = float(w[s])
        target = float(tgt[s])
        delta = target - held
        if s != "cash":
            max_abs = max(max_abs, abs(delta))
        rows.append({
            "slot": s,
            "traded": "511260" if s == "cash" else traded_code(s, cfg),
            "held": held,
            "target": target,
            "delta": delta,
        })
    sc = ds.state_log[-1] if ds.state_log else None
    mkts = {}
    for mkt in ("CN", "US"):
        dd, ok20, ok2060, rec, ok120 = ds.sig.mkt_info(decision, mkt, ds.gate_win)
        mkts[mkt] = {"dd": float(dd), "above_sma20": bool(ok20), "dual_ma": bool(ok2060),
                     "above_gate": bool(ok120), "recovery": float(rec)}
    prem = ds._premium_row(decision)
    j = ds.sig._idx(decision) - (int(ds.premium_shift) - 1)
    prem_dt = str(pd.Timestamp(ds.R.index[j]).date()) if 0 <= j < len(ds.R.index) else None
    premiums = {}
    if prem is not None:
        for code in ("159941", "513500"):
            premiums[code] = None if pd.isna(prem.get(code)) else float(prem.get(code))
    min_delta = float(cfg.get("min_delta", 0.035))
    return {
        "summary": {
            "data_end": str(res["weights"].index[-1].date()),
            "decision": str(pd.Timestamp(decision).date()),
            "signal": str(pd.Timestamp(signal_dt).date()),
            "score": None if sc is None else int(sc[1]),
            "score_date": None if sc is None else str(pd.Timestamp(sc[0]).date()),
            "lock_CN": bool(ds._lock["CN"]),
            "lock_US": bool(ds._lock["US"]),
            "min_delta": min_delta,
            "max_abs_slot_delta": max_abs,
            "would_trade": bool(max_abs + 1e-12 >= min_delta),
            "premium_date": prem_dt,
            "premiums": premiums,
        },
        "rows": rows,
        "markets": mkts,
    }


def scan_files():
    rows = []
    after = []
    for path in glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True):
        try:
            hdr = pd.read_csv(path, nrows=0)
        except Exception as ex:
            rows.append({"file": os.path.relpath(path, DATA_DIR), "error": str(ex)})
            continue
        if "date" not in hdr.columns:
            continue
        d = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"], errors="coerce")
        n_after = int((d > "2026-11-10").sum())
        rel = os.path.relpath(path, DATA_DIR)
        if n_after:
            after.append(rel)
        # only keep detail for files that feed the default or that have weekends / future dates
        wk = int((d.dt.day_of_week >= 5).sum())
        interesting = n_after or wk or rel in {
            "index_932365.csv", "index_930955.csv", "270042_nav.csv", "159941_nav.csv",
            "513500_nav.csv", "index_sz399006.csv", "159952_nav.csv", "511260_nav.csv",
            "511010_nav.csv", "159222_qfq.csv", "563020_qfq.csv", "050025_nav.csv",
            "qdii_price_159941.csv", "qdii_price_513500.csv",
        } or rel.endswith("159222_qfq.csv") or rel.endswith("563020_qfq.csv")
        if interesting:
            rows.append({
                "file": rel, "n": int(d.notna().sum()), "weekend": wk,
                "min": None if d.dropna().empty else str(d.min().date()),
                "max": None if d.dropna().empty else str(d.max().date()),
                "after_2026_11_10": n_after,
            })
    return {"detailed": rows, "any_after_2026_11_10": after,
            "n_csv": len(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))}


if __name__ == "__main__":
    main()
