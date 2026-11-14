# -*- coding: utf-8 -*-
"""纸面：v47 去掉全部国内权益，只留纳指 159941、标普 513500 和现金 511260。

不改默认，不下单，不拉新数据。官方引擎是 scripts/engine.py 的 run_backtest，
损益面板与 scripts/run_backtest.py 的 exec_panel 相同。

预注册（写完再跑，跑完不增删格子）：

A0 干净消融。国内槽全部 exclude。159952 成长份额按引擎既有归一化，
把牛市 57:7、熊市 40:15 放大到整个成长预算。有效国内底仓 4% 进现金，
海外底仓保持 11%。state_map 的防御列没有可成交标的，也进现金，不并入美股。

在 A0 上每次只动一个事先说得通的旋钮，另加一个组合格，不做网格搜索：

- F15：把那 4% 有效国内底仓改加到海外底仓（11% → 15%），引擎仍对 159941/513500 平分。
- B80 / B70：牛市成长拆分改为 80/20、70/30。熊市仍用放大后的 40:15。
- Ctop90：只把状态 7/8/9 的成长上限乘 0.9 后四舍五入（73/93/89 → 66/84/80）。
  低分状态已经因为防御列进现金而降了风险，不再砍。
- P：F15 与 B80 合在一起。底仓回到原先 15% 总底仓，同时把缩放后过重的纳指降到 80%。
- Dtog：防御列整列加进成长列，防御列置 0。这是「红利预算进美股」而不是「进现金」。

风控保持 v47：美股快刹车、美股深熊锁、QDII 溢价门控/倾斜、波动率闸门、
绝对动量、单资产急跌、3 周冷却、80/100/120 结构闸门多数投票。
波动率闸门仍读取 159952 的 20 日波动，只作为信号，权重必须为 0。
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import evaluate, run_backtest
from run_backtest import BACKTEST_START, REAL_START, exec_panel
from strategy import DynamicStrategy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "references")
OUT = os.path.join(ROOT, "out")
OOS_START = "2022-01-04"
TRAIN_END = "2021-12-31"
CN_SLOTS = ("159232", "515100", "159952")
US_SLOTS = ("159941", "513500")
WINDOWS = (
    ("proxy", "proxy", BACKTEST_START, None),
    ("train", "proxy", BACKTEST_START, TRAIN_END),
    ("oos", "proxy", OOS_START, None),
    ("real", "real", REAL_START, None),
)


def _base_us():
    cfg = json.load(open(os.path.join(REF, "final_cfg_v47.json"), encoding="utf-8"))
    cfg["exclude"] = ["159232", "515100", "159952"]
    cfg["floor_pct"] = {"cn": 0.0, "us": 11.0}
    # 0.57:0.07 与 0.40:0.15 交给引擎按非 exclude 权重归一化，等价于把创业板份额
    # 按原有纳指/标普比例摊到整个成长预算。159952 显式写 0，不能靠缺省键。
    cfg["growth_split_bull"] = {"159952": 0.0, "159941": 0.57, "513500": 0.07}
    cfg["growth_split_bear"] = {"159952": 0.0, "159941": 0.40, "513500": 0.15}
    cfg["instrument_swaps"] = {}
    cfg["cash_bond_pct"] = 1.0
    cfg["cash_bond_code"] = "511260"
    return cfg


def build_configs():
    a0 = _base_us()
    a0["version"] = "v47_us_cash_A0"
    a0["_note"] = (
        "纸面消融，不改默认。排除 159232/515100/159952，不换 512890/563020。"
        "牛市 57:7、熊市 40:15 按引擎归一化吃下原 159952 成长份额；"
        "有效 CN 底仓 4% 与整列防御预算进现金；US 底仓保持 11%。"
    )

    f15 = _base_us()
    f15["version"] = "v47_us_cash_F15"
    f15["floor_pct"] = {"cn": 0.0, "us": 15.0}
    f15["_note"] = "A0，但把有效 CN 底仓 4% 加到 US 底仓（11%→15%，两只美股 ETF 平分）。"

    b80 = _base_us()
    b80["version"] = "v47_us_cash_B80"
    b80["growth_split_bull"] = {"159952": 0.0, "159941": 0.80, "513500": 0.20}
    b80["_note"] = "A0，牛市成长改为纳指 80 / 标普 20。熊市仍是放大后的 40:15。"

    b70 = _base_us()
    b70["version"] = "v47_us_cash_B70"
    b70["growth_split_bull"] = {"159952": 0.0, "159941": 0.70, "513500": 0.30}
    b70["_note"] = "A0，牛市成长改为纳指 70 / 标普 30。熊市仍是放大后的 40:15。"

    ctop = _base_us()
    ctop["version"] = "v47_us_cash_Ctop90"
    ctop["state_map"] = dict(ctop["state_map"])
    # 0.9 倍，四舍五入：65.7→66，83.7→84，80.1→80。防御列不动。
    ctop["state_map"]["7"] = [66, 10]
    ctop["state_map"]["8"] = [84, 6]
    ctop["state_map"]["9"] = [80, 3]
    ctop["_note"] = "A0，仅状态 7/8/9 成长上限 ×0.9 取整为 66/84/80。低分状态不另砍。"

    pack = _base_us()
    pack["version"] = "v47_us_cash_P"
    pack["floor_pct"] = {"cn": 0.0, "us": 15.0}
    pack["growth_split_bull"] = {"159952": 0.0, "159941": 0.80, "513500": 0.20}
    pack["_note"] = "预注册组合：US 底仓 15%（吃下原 4% CN 底仓）且牛市 80/20。其余同 A0。"

    dtog = _base_us()
    dtog["version"] = "v47_us_cash_Dtog"
    dtog["state_map"] = {
        k: [int(v[0]) + int(v[1]), 0] for k, v in dtog["state_map"].items()
    }
    dtog["_note"] = (
        "A0，但 state_map 防御列整列加进成长列后置 0。"
        "这是把原红利弹性预算改配到美股，而不是留在现金。"
    )

    v47 = json.load(open(os.path.join(REF, "final_cfg_v47.json"), encoding="utf-8"))
    return {
        "v47": v47,
        "A0": a0,
        "F15": f15,
        "B80": b80,
        "B70": b70,
        "Ctop90": ctop,
        "P": pack,
        "Dtog": dtog,
    }


def write_configs(configs):
    names = {
        "A0": "candidate_cfg_v47_us_cash_a0.json",
        "F15": "candidate_cfg_v47_us_cash_f15.json",
        "B80": "candidate_cfg_v47_us_cash_b80.json",
        "B70": "candidate_cfg_v47_us_cash_b70.json",
        "Ctop90": "candidate_cfg_v47_us_cash_ctop90.json",
        "P": "candidate_cfg_v47_us_cash_p.json",
        "Dtog": "candidate_cfg_v47_us_cash_dtog.json",
    }
    for key, fn in names.items():
        path = os.path.join(REF, fn)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(configs[key], handle, ensure_ascii=False, indent=1)
            handle.write("\n")


def _metrics(result):
    ev = evaluate(result)
    weights = result["weights"]
    cn_max = {code: float(weights[code].abs().max()) for code in CN_SLOTS}
    means = {code: float(weights[code].mean()) for code in list(US_SLOTS) + ["cash"]}
    r = result["rets"]
    return {
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "n": int(len(r)),
        "cagr": float(ev["cagr"]),
        "max_dd": float(ev["max_dd"]),
        "sharpe": float(ev["sharpe"]),
        "calmar": float(ev["calmar"]),
        "turnover": float(ev["turnover"]),
        "avg_cash": float(ev["avg_cash"]),
        "mdd_peak": str(pd.Timestamp(ev["mdd_peak"]).date()) if ev["mdd_peak"] is not None else None,
        "mdd_trough": str(pd.Timestamp(ev["mdd_trough"]).date()) if ev["mdd_trough"] is not None else None,
        "cn_abs_max": cn_max,
        "mean_weight": means,
    }


def run_one(cfg, signals, panel, start, end):
    strategy = DynamicStrategy(signals, cfg=cfg)
    bond = None
    if cfg.get("cash_bond_pct"):
        bond = rets_from(read_table(f"{cfg.get('cash_bond_code', '511010')}_nav.csv"), "cum_nav")
    result = run_backtest(
        panel,
        target_weights_fn=strategy.target_fn(),
        daily_override_fn=strategy.daily_fn(),
        start=start,
        end=end,
        name=cfg.get("version", "DYN"),
        min_delta=cfg.get("min_delta", 0.02),
        repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond,
        cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 2),
        rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True,
    )
    splits = {
        "bull": {k: float(v) for k, v in strategy.growth_split_bull.items()},
        "bear": {k: float(v) for k, v in strategy.growth_split_bear.items()},
        "floor": {k: float(v) for k, v in strategy.floor.items()},
        "defense_split": {k: float(v) for k, v in strategy.defense_split_base.items()},
    }
    return _metrics(result), splits


def fmt(row):
    return (f"{row['cagr']*100:7.2f}% / {row['max_dd']*100:7.2f}% / "
            f"Sh {row['sharpe']:5.2f} / Cal {row['calmar']:6.3f}  "
            f"{row['start']}→{row['end']} n={row['n']}")


def main():
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    configs = build_configs()
    write_configs(configs)
    print("loading panels", flush=True)
    r_proxy, _ = build_panel("proxy")
    r_real, _ = build_panel("real")
    signals = {"proxy": r_proxy, "real": r_real}
    report = {
        "windows": {
            "proxy": [BACKTEST_START, None],
            "train": [BACKTEST_START, TRAIN_END],
            "oos": [OOS_START, None],
            "real": [REAL_START, None],
        },
        "fee": 0.0005,
        "sharpe_rf": 0.018,
        "candidates": {},
    }
    for name, cfg in configs.items():
        if only and name not in only:
            continue
        report["candidates"][name] = {"version": cfg.get("version"), "calibers": {}}
        for caliber in ("price", "nav"):
            panels = {
                "proxy": exec_panel(r_proxy, "proxy", cfg, caliber),
                "real": exec_panel(r_real, "real", cfg, caliber),
            }
            block = {}
            splits = None
            print(f"\n== {name} {caliber} ==", flush=True)
            for window, layer, start, end in WINDOWS:
                metrics, splits = run_one(cfg, signals[layer], panels[layer], start, end)
                block[window] = metrics
                cn = metrics["cn_abs_max"]
                leaked = max(cn.values()) > 1e-10
                if name != "v47" and leaked:
                    raise SystemExit(f"{name} {caliber} {window} CN weight leaked: {cn}")
                flag = "CN" if leaked else "noCN"
                print(f"  {window:5} {fmt(metrics)}  {flag} {cn}", flush=True)
            block["_engine_split"] = splits
            report["candidates"][name]["calibers"][caliber] = block
        path = os.path.join(OUT, "exp_v47_us_cash.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
