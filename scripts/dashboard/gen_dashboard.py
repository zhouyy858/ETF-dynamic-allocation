# -*- coding: utf-8 -*-
"""仪表盘数据生成器: 复用 skill 回测管线, 输出 dashboard.json
用法: ETF_DATA_DIR=~/ETF策略日报/data python3 gen_dashboard.py [输出json路径]
数据内容: 日报全部关键信息 + 净值/回撤/资产对比曲线, 供网页展示"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.environ.get("ETF_SKILL_DIR", os.path.expanduser("~/.codex/skills/etf-dynamic-allocation"))
SCRIPTS = os.path.join(SKILL, "scripts")
sys.path.insert(0, SCRIPTS)

from data_prep import build_panel, read_table, rets_from, DATA_DIR
from engine import run_backtest
from strategy import DynamicStrategy

CFG_FILE = (os.path.join(SKILL, "references", "final_cfg_v30.json")
            if os.path.exists(os.path.join(SKILL, "references", "final_cfg_v30.json"))
            else os.path.join(SKILL, "scripts", "references", "final_cfg_v30.json"))
NAMES = {"159232": "自由现金流", "515100": "红利低波100", "159941": "纳指100",
         "513500": "标普500", "159952": "创业板"}
SLOTS = ["159232", "515100", "159941", "513500", "159952"]
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "dashboard.json")
    cfg = json.load(open(CFG_FILE))
    R, W = build_panel("proxy")
    ds = DynamicStrategy(R, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav") if cfg.get("cash_bond_pct") else None
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start="2014-06-23", min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
                       tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 2), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    wdf, rets = res["weights"], res["rets"]
    wealth = (1 + rets).cumprod()
    last = wdf.index[-1]
    w_act = wdf.iloc[-1]
    # 净值口径最后日期(ETF 净值 T 日晚公布, 与日报"数据截至"一致)
    nav_last = None
    for _f in ["159232_nav.csv", "515100_nav.csv", "159941_nav.csv", "513500_nav.csv",
               "159952_nav.csv", "270042_nav.csv", "050025_nav.csv"]:
        _fp = os.path.join(DATA_DIR, _f)
        if os.path.exists(_fp):
            _d = str(pd.read_csv(_fp, usecols=[0]).iloc[-1, 0])
            if nav_last is None or _d > nav_last:
                nav_last = _d

    # 实盘目标: 最新数据日=T-1收盘, 用 signal_lag=0 实例计算下一调仓日目标(同日报口径)
    ds_live = DynamicStrategy(R, cfg=dict(cfg, signal_lag=0))
    tgt = ds_live.regular_target(last, {"pf_rets": rets})
    sc = ds.state_log[-1][1] if ds.state_log else 0

    today = pd.Timestamp.now().normalize()
    rebal_wd = int(cfg.get("rebal_weekday", 2))
    is_rebal_day = today.weekday() == rebal_wd
    gap = max(abs(w_act[s] - tgt[s]) for s in SLOTS + ["cash"])
    if is_rebal_day:
        action = ("今日是周" + WEEKDAY_CN[rebal_wd][1] +
                  "调仓日：按 v30 规则用前一日(T-1)收盘信号决策，目标缺口 1 笔当日收盘成交；"
                  "QDII 溢价用 T-2 口径（绝对溢价>5/8/12% 自动降仓×0.45/×0.22/×0.08；相对溢价差>2pp 转投低溢价品种），下单前查 IOPV")
    elif gap > 0.02:
        action = ("非调仓日，仓位与目标基本一致（最大偏差 " + f"{gap*100:.1f}%" +
                  "），等待下一个周" + WEEKDAY_CN[rebal_wd][1] + "检查")
    else:
        action = "非调仓日，仓位与目标基本一致，仅观察"

    # 市场状态
    mkt = {}
    for m in ["CN", "US"]:
        dd, ok20, ok20_60, rec, ok120 = ds.sig.mkt_info(last, m, ds.gate_win)
        mkt[m] = {"dd": round(float(dd) * 100, 2), "sma20": int(ok20), "trend": int(ok20_60),
                  "sma120": int(ok120), "rec": round(float(rec) * 100)}

    # QDII 溢价
    premium = []
    for code, nm in [("159941", "纳指100"), ("513500", "标普500")]:
        try:
            px = pd.read_csv(os.path.join(DATA_DIR, f"qdii_price_{code}.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
            nav = pd.read_csv(os.path.join(DATA_DIR, f"{code}_nav.csv"), parse_dates=["date"]).set_index("date")
            nav = nav[~nav.index.duplicated(keep="last")].sort_index()["unit_nav"]
            df = pd.concat([px.rename("px"), nav.rename("nav")], axis=1).dropna()
            if code == "159941" and "2022-07-04" in df.index:
                df = df.drop("2022-07-04")
            prem = float(df["px"].iloc[-1] / df["nav"].iloc[-1] - 1)
            flag = "high" if prem > 0.05 else ("warn" if prem > 0.03 else "ok")
            premium.append({"code": code, "name": nm, "pct": round(prem * 100, 2), "flag": flag})
        except Exception:
            pass

    # 组合表现
    r5 = float((1 + rets.iloc[-5:]).prod() - 1)
    r20 = float((1 + rets.iloc[-20:]).prod() - 1)
    y0 = rets.index.max().year
    ytd_mask = rets.index >= f"{y0}-01-01"
    ytd = float((1 + rets[ytd_mask]).prod() - 1) if ytd_mask.any() else np.nan
    cur_dd = float(wealth.iloc[-1] / wealth.cummax().iloc[-1] - 1)

    # 曲线: 净值 + 回撤 + 资产对比(归一化起点=100)
    dates = [d.strftime("%Y-%m-%d") for d in rets.index]
    dd_series = (wealth / wealth.cummax() - 1) * 100
    curves = {"dates": dates, "wealth": [round(float(v) * 100, 3) for v in wealth.tolist()],
              "dd": [round(float(v), 3) for v in dd_series.tolist()]}
    if len(W):
        ww = W.loc[rets.index]
        for s in SLOTS:
            if s not in ww.columns:
                continue
            col = ww[s].dropna()
            if len(col) == 0:
                continue
            base = float(col.iloc[0])
            if base <= 0:
                continue
            curves[s] = [None if pd.isna(v) else round(float(v) * 100 / base, 3) for v in ww[s]]

    data = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_date": nav_last or str(last.date()),
        "score": {"num": int(sc), "max": 9},
        "lock": {"CN": int(ds._lock["CN"]), "US": int(ds._lock["US"])},
        "mkt": mkt,
        "positions": [{"code": s, "name": NAMES[s], "actual": round(float(w_act[s]) * 100, 1),
                       "target": round(float(tgt[s]) * 100, 1)} for s in SLOTS] +
                     [{"code": "cash", "name": "现金(逆回购)", "actual": round(float(w_act["cash"]) * 100, 1),
                       "target": round(float(tgt["cash"]) * 100, 1)}],
        "premium": premium,
        "action": action,
        "is_rebal_day": int(is_rebal_day),
        "perf": {"last": round(float(rets.iloc[-1]) * 100, 2), "r5": round(r5 * 100, 2), "r20": round(r20 * 100, 2),
                 "ytd": round(ytd * 100, 2) if ytd == ytd else None, "dd": round(cur_dd * 100, 2)},
        "curves": curves,
        "config": {"version": "v30", "signal_lag": 1, "rebal_weekday": rebal_wd},
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[ok] dashboard.json 已生成: {out_path} (数据截至 {data['data_date']}, {len(dates)} 个交易日)")


if __name__ == "__main__":
    main()
