# -*- coding: utf-8 -*-
"""低相关池 × 动量共振 旁路消融, 对照 v30。不改默认配置。

预注册中位参数(先写死再跑, 不按结果回改):
  低相关: W_corr=120 (60/120/252 的中位), ρ*=0.30 (2014+ 截面均值 |corr| 的中位附近, 未拟合),
          λ=4 (超额相关 +0.10 → 弹性 ×exp(-0.4)≈0.67)
  动量共振: W_m=60 (20/60/120 的中位), α=1 (1 与 1.5 里取较小), clamp=[0.05, 0.80]
          (使 v30 纳指 bull 63% 落在区间内), 成长三件套, 至少 2 只 score>0 才偏离 growth_split
单因子初值网格只报告、不择优升版。关 1 不过则不做以邻域胜者为中心的再搜索。

用法: python3 scripts/exp_lc_mom.py
产物: out/exp_lc_mom.json , references/exp_lc_mom.json
"""
import sys, os, json, copy, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate, FEE
from strategy import DynamicStrategy, low_corr_softpool, momentum_resonance_weights, _project_simplex_box

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
CFG_V30 = os.path.join(ROOT, "references", "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
PERIODS = {
    "2015股灾": ("2015-06-15", "2016-02-29"),
    "2018熊市": ("2018-01-02", "2019-01-03"),
    "2021-2022熊市": ("2021-02-19", "2022-10-31"),
    "2019-2021牛市": ("2019-01-04", "2021-02-18"),
    "2024Q4-2026双牛": ("2024-09-02", "2026-10-20"),
}

MED_LC = {"low_corr_pool": True, "low_corr_win": 120, "low_corr_rho": 0.30, "low_corr_lam": 4.0}
MED_MOM = {
    "mom_resonance": True, "mom_res_win": 60, "mom_res_alpha": 1.0,
    "mom_res_clamp": [0.05, 0.80], "mom_res_min_pos": 2, "mom_res_defense": False,
}


def _self_check():
    proj = _project_simplex_box({"a": 0.9, "b": 0.1, "c": 0.0}, 0.05, 0.80)
    assert proj is not None and abs(sum(proj.values()) - 1) < 1e-9
    assert abs(proj["a"] - 0.80) < 1e-9 and proj["a"] >= proj["b"] >= proj["c"] >= 0.05
    w = {"159952": 10.0, "159941": 20.0, "513500": 8.0, "159232": 5.0, "515100": 5.0}
    fl = {"159952": 0.0, "159941": 5.0, "513500": 5.0, "159232": 1.0, "515100": 1.0}
    rho = {"159952": 0.20, "159941": 0.55, "513500": 0.50, "159232": 0.25, "515100": 0.22}
    nw = low_corr_softpool(w, fl, rho, 4.0, 0.30, list(w))
    e0 = sum(w[c] - fl[c] for c in w)
    e1 = sum(nw[c] - fl[c] for c in w)
    assert abs(e0 - e1) < 1e-8
    assert all(nw[c] + 1e-9 >= fl[c] for c in w)
    assert nw["159941"] < w["159941"] and nw["159952"] > w["159952"]
    props, fired = momentum_resonance_weights(
        {"159952": 0.2, "159941": -0.1, "513500": -0.05},
        {"159952": 0.30, "159941": 0.63, "513500": 0.07}, 1.0, (0.05, 0.80), 2)
    assert fired is False and abs(props["159941"] - 0.63) < 1e-9
    props, fired = momentum_resonance_weights(
        {"159952": 0.10, "159941": 0.30, "513500": 0.05},
        {"159952": 0.30, "159941": 0.63, "513500": 0.07}, 1.0, (0.05, 0.80), 2)
    assert fired is True and abs(sum(props.values()) - 1) < 1e-8
    assert props["159941"] > props["159952"] > props["513500"]
    print("[ok] 公式自检: 软池守住底仓与弹性总和; 单正分回退固定拆分; 共振按 score 排序")


def _pack(e, n_days):
    years = n_days / 252.0 if n_days else np.nan
    periods = {}
    for pn, pe in (e.get("periods") or {}).items():
        periods[pn] = {"cagr": pe["cagr"], "mdd": pe["max_dd"], "total": pe["total"]}
    return {
        "cagr": e["cagr"], "mdd": e["max_dd"], "cal": e["calmar"], "sharpe": e["sharpe"],
        "turnover": e["turnover"], "cash": e["avg_cash"],
        "fee_ann": (e["turnover"] * FEE / years) if years and years == years else None,
        "periods": periods,
    }


def run_cfg(cfg, R, start, bond, periods=None):
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(
        R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.035),
        repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    e = evaluate(res, periods=periods)
    return _pack(e, len(res["rets"])), res["weights"], res["wealth"], ds


def activity(ds):
    def rate(n, d):
        return (n / d) if d else 0.0
    return {
        "lc_calls": ds._lc_calls, "lc_changed_rate": rate(ds._lc_changed, ds._lc_calls),
        "mom_calls": ds._mom_calls, "mom_fire_rate": rate(ds._mom_fires, ds._mom_calls),
        "mom_changed_rate": rate(ds._mom_changed, ds._mom_calls),
    }


def fmt_row(name, m, base=None):
    dcal = ""
    if base is not None:
        dcal = f"  ΔCal {m['cal'] - base['cal']:+.3f}"
    fee = m["fee_ann"] * 100 if m.get("fee_ann") is not None else float("nan")
    return (f"{name:<16} CAGR {m['cagr']*100:6.2f}%  MDD {m['mdd']*100:7.2f}%  "
            f"Cal {m['cal']:5.2f}{dcal}  TO {m['turnover']:6.1f}  费拖累 {fee:5.2f}pp/年")


def gate1(m, base):
    dp = m["proxy"]["cal"] - base["proxy"]["cal"]
    dr = m["real"]["cal"] - base["real"]["cal"]
    mdd_ok = m["proxy"]["mdd"] > -0.10 + 1e-12
    return bool(dp > 1e-12 and dr > 1e-12 and mdd_ok), dp, dr


def gate3(m, base):
    return bool(m["oos"]["cal"] + 1e-12 >= base["oos"]["cal"])


def main():
    _self_check()
    base_cfg = json.load(open(CFG_V30))
    assert "low_corr_pool" not in base_cfg and "mom_resonance" not in base_cfg, "v30 不得写入旁路开关"
    v30_sha = hashlib.sha256(open(CFG_V30, "rb").read()).hexdigest()
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    bond = rets_from(read_table(f"{base_cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav")
    print(f"数据截止 proxy {R.index[-1].date()}  real {Rr.index[-1].date()}")
    # 共振分数只依赖收益与窗口, 不得随 α / λ / 防御开关变化
    s1 = DynamicStrategy(R, cfg=dict(base_cfg, **MED_MOM))
    ov = dict(MED_MOM, mom_res_alpha=1.5, defense_momentum=False, low_corr_pool=True, low_corr_lam=2.0)
    s2 = DynamicStrategy(R, cfg=dict(base_cfg, **ov))
    assert np.allclose(s1._mom_score.fillna(-999.0), s2._mom_score.fillna(-999.0))
    finite_rate = float(s1._mom_score.loc["2016-01-01":].notna().mean().mean())
    assert finite_rate > 0.9, finite_rate
    print(f"[ok] 共振分数与 α/防御开关无关, 2016 后有限占比 {finite_rate:.1%}")

    cfg_false = copy.deepcopy(base_cfg)
    cfg_false.update(low_corr_pool=False, mom_resonance=False)
    _, _, w_base, _ = run_cfg(base_cfg, R, PROXY_START, bond)
    _, _, w_false, _ = run_cfg(cfg_false, R, PROXY_START, bond)
    max_nav = float((w_base - w_false).abs().max())
    assert max_nav == 0.0, f"开关全关与 v30 净值不一致: {max_nav}"
    print(f"[ok] 开关全关与 v30 净值逐位相同 (max |ΔNAV|={max_nav})")

    arms = {
        "A_v30": {},
        "B_low_corr": dict(MED_LC),
        "C_mom": dict(MED_MOM),
        "D_both": {**MED_LC, **MED_MOM},
    }
    overlap = {
        "A_corr_off": {"corr_risk": False},
        "B_corr_off": {**MED_LC, "corr_risk": False},
        "A_def_off": {"defense_momentum": False},
        "C_def_off": {**MED_MOM, "defense_momentum": False},
        "D_corr_off": {**MED_LC, **MED_MOM, "corr_risk": False},
        "D_def_off": {**MED_LC, **MED_MOM, "defense_momentum": False},
    }
    neighbors = {}
    for w in (60, 252):
        neighbors[f"B_W{w}"] = {**MED_LC, "low_corr_win": w}
    for lam in (2.0, 8.0):
        neighbors[f"B_lam{lam:g}"] = {**MED_LC, "low_corr_lam": lam}
    for rho in (0.25, 0.40):
        neighbors[f"B_rho{rho:.2f}"] = {**MED_LC, "low_corr_rho": rho}
    for w in (20, 120):
        neighbors[f"C_W{w}"] = {**MED_MOM, "mom_res_win": w}
    neighbors["C_a1.5"] = {**MED_MOM, "mom_res_alpha": 1.5}
    for w in (20, 120):
        neighbors[f"C_W{w}_a1.5"] = {**MED_MOM, "mom_res_win": w, "mom_res_alpha": 1.5}
    for w in (60, 252):
        neighbors[f"D_Wcorr{w}"] = {**MED_LC, **MED_MOM, "low_corr_win": w}
    for lam in (2.0, 8.0):
        neighbors[f"D_lam{lam:g}"] = {**MED_LC, **MED_MOM, "low_corr_lam": lam}
    for rho in (0.25, 0.40):
        neighbors[f"D_rho{rho:.2f}"] = {**MED_LC, **MED_MOM, "low_corr_rho": rho}
    for w in (20, 120):
        neighbors[f"D_Wm{w}"] = {**MED_LC, **MED_MOM, "mom_res_win": w}
    neighbors["D_a1.5"] = {**MED_LC, **MED_MOM, "mom_res_alpha": 1.5}

    def eval_arm(name, ov):
        cfg = copy.deepcopy(base_cfg)
        cfg.update(ov)
        cfg["version"] = f"exp_lc_mom_{name}"
        p, wp, _, ds = run_cfg(cfg, R, PROXY_START, bond, periods=PERIODS)
        r, _, _, _ = run_cfg(cfg, Rr, REAL_START, bond)
        o, _, _, _ = run_cfg(cfg, R, OOS_START, bond)
        rec = {"override": ov, "proxy": p, "real": r, "oos": o, "activity_proxy": activity(ds),
               "proxy_weights": wp}
        print(fmt_row(name + "/proxy", p))
        print(fmt_row(name + "/real", r))
        print(fmt_row(name + "/oos", o))
        act = rec["activity_proxy"]
        if act["lc_calls"] or act["mom_calls"]:
            print(f"    活跃 低相关改权重 {act['lc_changed_rate']*100:.1f}% 调用  "
                  f"共振触发 {act['mom_fire_rate']*100:.1f}%  共振改权重 {act['mom_changed_rate']*100:.1f}%")
        return rec

    print("\n===== 主对照 A–D (中位参数) =====")
    primary = {}
    for name, ov in arms.items():
        primary[name] = eval_arm(name, ov)

    base = primary["A_v30"]
    assert abs(base["proxy"]["cagr"] - 0.1121) < 0.0008, base["proxy"]["cagr"]
    assert abs(base["proxy"]["mdd"] + 0.0991) < 0.0008, base["proxy"]["mdd"]
    assert abs(base["proxy"]["cal"] - 1.13) < 0.02, base["proxy"]["cal"]
    assert abs(base["real"]["cagr"] - 0.2527) < 0.002, base["real"]["cagr"]
    assert abs(base["real"]["cal"] - 6.74) < 0.05, base["real"]["cal"]
    assert abs(base["oos"]["cal"] - 1.29) < 0.05, base["oos"]["cal"]
    print("[ok] A 复现 README 10-20 口径 (proxy 11.21/-9.91/1.13, real 25.27/Cal 6.74, OOS Cal ~1.29)")

    print("\n===== 与 corr_risk / defense_momentum 的增量对照 =====")
    ovlap = {}
    for name, ov in overlap.items():
        ovlap[name] = eval_arm(name, ov)

    print("\n===== 预注册单因子邻域 (非搜索, 含三窗口) =====")
    neigh = {}
    for name, ov in neighbors.items():
        neigh[name] = eval_arm(name, ov)

    def slim(rec):
        return {
            "override": rec["override"],
            "proxy": {k: rec["proxy"][k] for k in ("cagr", "mdd", "cal", "sharpe", "turnover", "cash", "fee_ann")},
            "real": {k: rec["real"][k] for k in ("cagr", "mdd", "cal", "sharpe", "turnover", "cash", "fee_ann")},
            "oos": {k: rec["oos"][k] for k in ("cagr", "mdd", "cal", "sharpe", "turnover", "cash", "fee_ann")},
            "activity_proxy": rec["activity_proxy"],
            "periods_mdd": {k: v["mdd"] for k, v in rec["proxy"].get("periods", {}).items()},
        }

    def l1_vs_a(rec):
        a = primary["A_v30"]["proxy_weights"]
        b = rec["proxy_weights"]
        cols = ["159232", "515100", "159941", "513500", "159952", "cash"]
        return float((a[cols] - b[cols]).abs().sum(axis=1).mean())

    print("\n===== 相对 A 的对照与三关 =====")
    verdicts = {}
    for name, rec in list(primary.items()) + list(neigh.items()):
        if name == "A_v30":
            continue
        ok1, dp, dr = gate1(rec, base)
        ok3 = gate3(rec, base)
        verdicts[name] = {
            "gate1_both_windows": ok1,
            "dcal_proxy": dp, "dcal_real": dr,
            "gate3_oos": ok3,
            "dcal_oos": rec["oos"]["cal"] - base["oos"]["cal"],
            "proxy_mdd": rec["proxy"]["mdd"],
            "weight_l1_vs_A": l1_vs_a(rec) if name in primary else None,
        }
        flag = "关1过" if ok1 else "关1未过"
        flag3 = "关3过" if ok3 else "关3未过"
        print(f"{name:<16} {flag} ΔCal proxy {dp:+.3f} real {dr:+.3f} | {flag3} ΔOOS {verdicts[name]['dcal_oos']:+.3f} "
              f"| MDD {rec['proxy']['mdd']*100:.2f}%")

    def plateau(median_name, prefix):
        med = primary[median_name]
        names = [n for n in neigh if n.startswith(prefix)]
        cols = [neigh[n]["proxy"]["cal"] for n in names] + [med["proxy"]["cal"]]
        med_cal = med["proxy"]["cal"]
        worse = sum(1 for c in cols if c < med_cal * 0.9 - 1e-12)
        grid_med = float(np.median(cols))
        spike = (max(cols) > grid_med * 1.10 + 1e-12) and (med_cal >= max(cols) - 1e-12)
        g1_names = [n for n in names if verdicts[n]["gate1_both_windows"]]
        g1_med, _, _ = gate1(med, base)
        return {
            "n_grid_incl_median": len(cols),
            "proxy_cal_min": min(cols), "proxy_cal_median": grid_med, "proxy_cal_max": max(cols),
            "worse_than_90pct_of_median": worse,
            "worse_ratio": worse / len(cols),
            "v30_perturb_ratio_ref": 2 / 104,
            "isolated_peak": bool(spike),
            "median_gate1": bool(g1_med),
            "neighbors_gate1": g1_names,
        }

    plat = {"B": plateau("B_low_corr", "B_"), "C": plateau("C_mom", "C_"), "D": plateau("D_both", "D_")}
    any_g1 = any(verdicts[n]["gate1_both_windows"] for n in ("B_low_corr", "C_mom", "D_both"))
    for k, p in plat.items():
        entered = p["median_gate1"]
        ratio_ok = p["worse_ratio"] <= 2 / 104 + 1e-12
        p["gate2_entered"] = bool(entered)
        # 只在中位臂过关 1 后评审平台; 孤峰或恶化占比高于 v30 的 2/104 则关 2 失败
        p["gate2_pass"] = bool(entered and (not p["isolated_peak"]) and ratio_ok)
        print(f"关2 {k}: {'进入' if entered else '未进入(中位关1未过)'}  "
              f"网格 Cal {p['proxy_cal_min']:.2f}/{p['proxy_cal_median']:.2f}/{p['proxy_cal_max']:.2f}  "
              f"恶化>10% {p['worse_than_90pct_of_median']}/{p['n_grid_incl_median']}  "
              f"孤峰={p['isolated_peak']}  邻域关1={p['neighbors_gate1'] or '无'}")

    # 重叠: 关掉既有模块后, 新模块相对其自身基线的 ΔCal 是否仍在
    def dcal(arm, ref, window):
        return arm[window]["cal"] - ref[window]["cal"]
    overlap_note = {
        "B_vs_A_proxy": dcal(primary["B_low_corr"], base, "proxy"),
        "B_vs_A_given_corr_off_proxy": dcal(ovlap["B_corr_off"], ovlap["A_corr_off"], "proxy"),
        "C_vs_A_proxy": dcal(primary["C_mom"], base, "proxy"),
        "C_vs_A_given_def_off_proxy": dcal(ovlap["C_def_off"], ovlap["A_def_off"], "proxy"),
        "D_vs_A_proxy": dcal(primary["D_both"], base, "proxy"),
        "D_vs_A_given_corr_off_proxy": dcal(ovlap["D_corr_off"], ovlap["A_corr_off"], "proxy"),
        "D_vs_A_given_def_off_proxy": dcal(ovlap["D_def_off"], ovlap["A_def_off"], "proxy"),
        "B_weight_l1": l1_vs_a(primary["B_low_corr"]),
        "C_weight_l1": l1_vs_a(primary["C_mom"]),
        "D_weight_l1": l1_vs_a(primary["D_both"]),
    }

    promote = []
    for name, key, plat_key in (("B_low_corr", "B", "B"), ("C_mom", "C", "C"), ("D_both", "D", "D")):
        v = verdicts[name]
        if v["gate1_both_windows"] and plat[plat_key]["gate2_pass"] and v["gate3_oos"]:
            promote.append(name)
    can_shake = bool(promote)
    if can_shake:
        conclusion = "中位臂三关全过, 仍不得自动改默认; 仅列为待人工确认的升版候选。"
    elif any_g1:
        conclusion = "有臂过关1, 但关2或关3未过, 不能撼动 v30。"
    else:
        conclusion = "A–D 中位臂均未过关1(双窗口 Calmar 同向改善且 MDD<10%), 不能撼动 v30。默认保持 v30。"

    # 落盘实验 cfg (v30 本体不动)
    for fn, ov in (
        ("final_cfg_exp_lc_mom_b.json", MED_LC),
        ("final_cfg_exp_lc_mom_c.json", MED_MOM),
        ("final_cfg_exp_lc_mom_d.json", {**MED_LC, **MED_MOM}),
    ):
        cfg = copy.deepcopy(base_cfg)
        cfg.update(ov)
        cfg["version"] = fn.replace("final_cfg_", "").replace(".json", "")
        cfg["_note"] = ("旁路实验, 不是默认。默认仍为 final_cfg_v30.json。" + conclusion)
        with open(os.path.join(ROOT, "references", fn), "w") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
            f.write("\n")

    payload = {
        "disclaimer": "研究回测, 不构成投资建议。默认配置仍为 v30。",
        "data_end_proxy": str(R.index[-1].date()),
        "data_end_real": str(Rr.index[-1].date()),
        "v30_sha256": v30_sha,
        "median_params": {"low_corr": MED_LC, "mom": MED_MOM},
        "identity_nav_max_abs_diff": max_nav,
        "baseline_reproduced": True,
        "primary": {k: slim(v) for k, v in primary.items()},
        "overlap": {k: slim(v) for k, v in ovlap.items()},
        "neighbors": {k: slim(v) for k, v in neigh.items()},
        "verdicts": verdicts,
        "plateau": plat,
        "overlap_deltas": overlap_note,
        "promote_candidates": promote,
        "can_shake_v30": can_shake,
        "conclusion": conclusion,
        "command": "python3 scripts/exp_lc_mom.py",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    for path in (os.path.join(OUT, "exp_lc_mom.json"), os.path.join(ROOT, "references", "exp_lc_mom.json")):
        with open(path, "w") as f:
            f.write(text)
            f.write("\n")
    print("\n" + conclusion)
    print("[ok] out/exp_lc_mom.json  references/exp_lc_mom.json")
    print(f"升版候选: {promote or '无'}  撼动默认: {can_shake}")


if __name__ == "__main__":
    main()
