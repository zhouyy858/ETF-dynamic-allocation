# -*- coding: utf-8 -*-
"""回测引擎: 静态基准 + 动态策略
统一约束: 固定调仓日(默认周三, 配置rebal_weekday, v22起定稿为周五); 分笔由tranche_weights配置
(如[1.0]=决策当日1笔全部成交; 或[1/3]*3=决策当周及随后2个调仓日各1/3三周调整完);
紧急风控任意交易日触发并取代未完成计划; 单边费万5; 闲置现金享逆回购收益(可叠加国债ETF层)
收益口径(无未来函数): 决策信号使用截至决策日前一交易日的收盘数据(signal_lag=1); 成交按决策日
收盘价进行, 成交份额自下一交易日起计收益(先计提当日收益、后执行成交); exec_lag=1时成交顺延至次日收盘"""
import numpy as np, pandas as pd
from metrics import TRADING_DAYS, annualized_ret, annualized_vol, max_drawdown, sharpe

FEE = 0.0005
REPO = 0.018
TRANCHES = 3
SLOTS = ["159232", "515100", "159941", "513500", "159952"]

def run_backtest(R, target_weights_fn=None, daily_override_fn=None, fixed_weights=None,
                 start=None, end=None, tranches=TRANCHES, fee=FEE, repo=REPO,
                 min_delta=0.002, name="", tranche_weights=None,
                 cash_bond_rets=None, cash_bond_pct=0.0, exec_lag=0, accrual_mode="pre",
                 rebal_weekday=2, rebal_freq="weekly", strict=False,
                 cash_gold_rets=None, gold_override_fn=None,
                 cash_sleeve_fn=None, cash_bond_alt_rets=None,
                 weight_quantum=None, lot_prices=None, lot_nav0=1_000_000.0,
                 lot_size=100, lot_mode="nearest"):
    if strict:
        assert accrual_mode == "pre", "严格模式必须 accrual_mode=pre (成交仓位不得计提当日收益)"
        assert exec_lag == 0, "严格模式必须 exec_lag=0 (决策日收盘成交, 前一日信号)";
    """accrual_mode: "pre"=先按交易前权重计提当日收益再收盘成交(严格无未来, 新买入份额自下一
    交易日起计收益); "post"=复现旧版行为(成交后按当日收益计提, 仅供审计对照, 勿用于实盘口径)
    exec_lag: 成交时点 0=决策当日收盘成交(严格口径: 先计提当日收益再成交); 1=决策次日收盘成交
    rebal_weekday: 常规调仓日 0=周一..4=周五 (默认2=周三)
    rebal_freq: "weekly"(每周) / "biweekly"(隔周) / "monthly"(每月首个调仓日)
    tranche_weights: 分笔比例列表(如[1.0]立即执行、[0.5,0.5]、[1/3]*3), 默认None=等分tranches笔
    cash_bond_pct: 闲置现金中配置债券ETF的比例(0~1); cash_bond_rets: 债券ETF日收益序列(如511010),
    缺失日按逆回购repo计息 -> 现金层 = (1-pct)*逆回购 + pct*债券ETF
    cash_sleeve_fn: 可选。None 时现金层与上式逐位一致(生产默认路径)。若给出, 签名 fn(dt)->{"pct","alt_frac"},
    在当日决策之后更新, 自下一交易日起计提(与黄金危机袖套同一时点, 无未来函数)。
    cash_bond_alt_rets: 可选第二债券序列; 仅当 alt_frac>0 时按 (1-alt)*主债券 + alt*第二债券 混合。
    weight_quantum: 可选。None=目标权重保持 float(生产路径)。>0 时在 min_delta 判断前把各资产目标
    舍入到该组合比例粒度(如 0.0001=1bp)。不改变调仓频率与滞回。
    lot_prices: 可选成交价表(列名=SLOTS)。None=不按手数取整。给出时用 lot_nav0 初始资金、
    当日财富×价格把目标股数舍入到 lot_size(默认100股)后再换算回权重; 缺价的资产保持连续权重。
    lot_mode: nearest / floor / ceil。以上钩子均为 None 时, 成交分支与改动前相同。"""
    if tranche_weights is None:
        tw = np.array([1.0 / tranches] * tranches)
    else:
        tw = np.array(tranche_weights, float); tw = tw / tw.sum()
    """顺序模拟回测(无未来函数)。
    fixed_weights: 静态权重, 每周三调回目标(分3周三笔)。
    target_weights_fn(dt, R_up_to_prev, ctx): 动态目标权重(dict, 含cash)或None; 策略侧保证
    信号只使用截至前一日收盘数据, R_up_to_prev 即引擎传入的截至前一日切片。
    分笔规则: 每次决策的目标缺口按tranche_weights分笔, 第1笔决策当日收盘成交(exec_lag=0),
    其余在随后(tranches-1)个周三各成交1笔 -> 每周三调仓、三周调整完; 若上一计划仍有未执行分笔,
    常规决策顺延(计划不重叠), 紧急风控可取代未完成计划并立即重启新的3周三笔计划。
    """
    R = R.copy()
    if start: R = R[R.index >= start]
    if end: R = R[R.index <= end]
    R = R.ffill().fillna(0.0)
    dates = R.index
    n = len(dates)
    k = len(SLOTS)
    repo_d = (1 + repo) ** (1 / TRADING_DAYS) - 1
    bond = None
    if cash_bond_rets is not None and cash_bond_pct > 0:
        bond = cash_bond_rets.reindex(dates).ffill().fillna(repo_d)
    gold = None
    if cash_gold_rets is not None:
        gold = cash_gold_rets.reindex(dates).ffill().fillna(repo_d)
    gold_pct_cur = 0.0
    bond_alt = None
    if cash_bond_alt_rets is not None:
        bond_alt = cash_bond_alt_rets.reindex(dates).ffill().fillna(repo_d)
    # 仅 cash_sleeve_fn 非空时使用; 初值=静态比例, 当日决策后更新、次日计提
    pct_cur = float(cash_bond_pct)
    alt_frac_cur = 0.0
    px_df = None
    nav_prev = 1.0
    if lot_prices is not None:
        # 只向前填充最多 5 个交易日; 不 bfill, 避免用未来价格
        px_df = lot_prices.reindex(dates).ffill(limit=5)

    def cash_ret_at(i):
        # 严格口径: 用上一收盘已生效的黄金配置计提当日收益, 当日决策收盘后生效、次日计收益
        # 语义: gold_pct_cur = 现金层中黄金占比(0~1), 危机时替代债券+逆回购基准收益
        gp = gold_pct_cur
        # 生产路径: 不传 sleeve / 第二债券时, 公式与改动前逐字相同
        if cash_sleeve_fn is None and bond_alt is None:
            if gold is not None:
                base = (1.0 - cash_bond_pct) * repo_d + cash_bond_pct * (
                    float(bond.iloc[i]) if bond is not None else 0.0)
                return gp * float(gold.iloc[i]) + (1.0 - gp) * base
            if bond is not None:
                return (1.0 - cash_bond_pct) * repo_d + cash_bond_pct * float(bond.iloc[i])
            return repo_d
        bret = float(bond.iloc[i]) if bond is not None else repo_d
        if bond_alt is not None and alt_frac_cur > 0.0:
            bret = (1.0 - alt_frac_cur) * bret + alt_frac_cur * float(bond_alt.iloc[i])
        base = (1.0 - pct_cur) * repo_d + pct_cur * bret
        if gold is not None:
            return gp * float(gold.iloc[i]) + (1.0 - gp) * base
        return base

    def fixed_target():
        t = np.array([fixed_weights[s] for s in SLOTS], dtype=float)
        return t / t.sum()

    w = np.zeros(k)
    pf_rets_ctx = []
    weights_history = np.zeros((n, k + 1))
    rets = np.zeros(n)
    turnover_day = np.zeros(n)
    pending = {}

    def freq_ok(d):
        if rebal_freq == "biweekly":
            return d.isocalendar().week % 2 == 0
        if rebal_freq == "monthly":
            return d.day <= 7
        return True

    def next_rebal(i):
        j = i + 1
        while j < n and not (dates[j].weekday() == rebal_weekday and freq_ok(dates[j])):
            j += 1
        return j if j < n else None

    def make_ctx():
        ctx = ({"pf_rets": pd.Series(pf_rets_ctx, index=dates[:len(pf_rets_ctx)])}
               if pf_rets_ctx else {"pf_rets": pd.Series(dtype=float)})
        ctx["equity"] = float(w.sum())
        ctx["weights"] = w.copy()
        return ctx

    for i in range(n):
        dt = dates[i]
        fee_today = 0.0
        if accrual_mode == "pre":
            # 严格口径: 先按"上一收盘持仓"计提当日收益, 再收盘执行成交 ->
            # 新买入/卖出份额自下一交易日起计收益, 不享受成交当日涨跌(消除1日超前收益)
            cash = 1.0 - w.sum()
            r = R.iloc[i].values
            cash_ret = cash_ret_at(i)
            g = w * (1.0 + r)
            c = cash * (1.0 + cash_ret)
            factor = float(g.sum() + c)
            pf_ret = factor - 1.0
            w = g / factor
            cash = c / factor
            for delta in pending.pop(i, []):
                w = w + delta
                fee_today += float(np.abs(delta).sum())
        else:
            # 旧版口径(仅审计对照): 先执行到期分笔再决策, 当日收益按成交后权重计提
            for delta in pending.pop(i, []):
                w = w + delta
                fee_today += float(np.abs(delta).sum())
        scheduled = sum(len(v) for v in pending.values())
        # 决策: 首日建仓/常规调仓日/日度紧急刹车 (信号用截至前一日数据, 无未来函数)
        is_rebal = (i == 0) or (dt.weekday() == rebal_weekday and freq_ok(dt))
        target = None
        if is_rebal:
            if fixed_weights is not None:
                target = fixed_target()
            elif target_weights_fn is not None:
                tgt = target_weights_fn(dt, R.iloc[:i], make_ctx())
                if tgt is not None:
                    target = np.array([tgt[s] for s in SLOTS], dtype=float)
        is_emergency = False
        if daily_override_fn is not None:
            tgt2 = daily_override_fn(dt, R.iloc[:i], make_ctx())
            if tgt2 is not None:
                target = np.array([tgt2[s] for s in SLOTS], dtype=float)
                is_emergency = True
        if target is not None:
            # 常规决策: 上一计划未执行完毕则顺延, 保证同一时点只有一份计划、三周内调整完
            if not is_emergency and scheduled > 0:
                target = None
        if target is not None:
            total = target.sum()
            if total > 1.0 + 1e-9:
                target = target / total
            if weight_quantum is not None and weight_quantum > 0:
                q = float(weight_quantum)
                target = np.round(np.clip(target, 0.0, None) / q) * q
                s = float(target.sum())
                if s > 1.0 + 1e-12:
                    target = target / s
            if px_df is not None and accrual_mode == "pre":
                nav_now = nav_prev * (1.0 + pf_ret)
                port = float(lot_nav0) * nav_now
                if port > 0:
                    snapped = target.copy()
                    for si, sname in enumerate(SLOTS):
                        if sname not in px_df.columns:
                            continue
                        try:
                            px = float(px_df[sname].iloc[i])
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(px) or px <= 0:
                            continue
                        raw = float(target[si]) * port / px
                        if raw <= 0:
                            lots = 0.0
                        elif lot_mode == "floor":
                            lots = np.floor(raw / lot_size + 1e-12)
                        elif lot_mode == "ceil":
                            lots = np.ceil(raw / lot_size - 1e-12)
                        else:
                            lots = np.round(raw / lot_size)  # numpy 银行家舍入
                        snapped[si] = lots * lot_size * px / port
                    if float(snapped.sum()) > 1.0 + 1e-9:
                        snapped = snapped / snapped.sum()
                    target = snapped
            delta = target - w
            if i == 0 or np.abs(delta).max() >= min_delta:
                if is_emergency:
                    pending = {}  # 紧急风控取代未完成计划
                # 分笔: 第1笔当日执行(exec_lag=0)或次日执行(exec_lag=1), 其余在随后各决策周三分笔
                d0 = delta * tw[0]
                if exec_lag == 0:
                    w = w + d0
                    fee_today += float(np.abs(d0).sum())
                    j = i
                else:
                    ex = i + 1
                    if ex < n:
                        pending.setdefault(ex, []).append(d0)
                    j = ex if ex < n else i
                for t in range(1, len(tw)):
                    j = next_rebal(j)
                    if j is None:
                        break
                    pending.setdefault(j, []).append(delta * tw[t])
        if accrual_mode == "pre":
            # 严格口径: 当日收益已在成交前计提完毕, 此处仅扣除当日费用
            pf_ret -= fee_today * fee
        else:
            # 旧版口径: 按成交后权重计提当日收益并扣除费用
            cash = 1.0 - w.sum()
            r = R.iloc[i].values
            cash_ret = cash_ret_at(i)
            g = w * (1.0 + r)
            c = cash * (1.0 + cash_ret)
            pf_ret = float(g.sum() + c - 1.0 - fee_today * fee)
            factor = 1.0 + pf_ret
            w = g / factor
            cash = c / factor
        turnover_day[i] = fee_today
        if gold_override_fn is not None:
            try:
                gp_new = float(gold_override_fn(dt) or 0.0)
            except Exception:
                gp_new = 0.0
            gold_pct_cur = min(max(gp_new, 0.0), 1.0)
        if cash_sleeve_fn is not None:
            spec = cash_sleeve_fn(dt) or {}
            pct_cur = min(max(float(spec.get("pct", cash_bond_pct)), 0.0), 1.0)
            alt_frac_cur = min(max(float(spec.get("alt_frac", 0.0)), 0.0), 1.0)
        # 记录日终持仓(成交后): 资产权重与现金必须同口径一致, 供展示/avg_cash使用
        weights_history[i] = np.concatenate([w, [1.0 - w.sum()]])
        rets[i] = pf_ret
        pf_rets_ctx.append(pf_ret)
        if px_df is not None:
            nav_prev *= (1.0 + pf_ret)

    rs = pd.Series(rets, index=dates)
    W = (1 + rs).cumprod()
    wdf = pd.DataFrame(weights_history, index=dates, columns=SLOTS + ["cash"])
    return {"rets": rs, "wealth": W, "weights": wdf, "name": name,
            "turnover": float(turnover_day.sum()), "schedule_n": int(sum(len(v) for v in pending.values()))}

def evaluate(result, rf=REPO, periods=None):
    r, w = result["rets"], result["wealth"]
    mdd, pk, tr = max_drawdown(w)
    out = {"name": result.get("name", ""), "cagr": annualized_ret(r), "vol": annualized_vol(r),
           "max_dd": mdd, "mdd_peak": pk, "mdd_trough": tr,
           "sharpe": sharpe(r, rf),
           "calmar": (annualized_ret(r) / abs(mdd) if mdd == mdd and mdd < 0 else np.nan),
           "total_ret": w.iloc[-1] - 1, "final_wealth": w.iloc[-1],
           "turnover": result["turnover"], "avg_cash": float(result["weights"]["cash"].mean()),
           "avg_equity": float((1 - result["weights"]["cash"]).mean())}
    if periods:
        out["periods"] = {}
        for pn, (ps, pe) in periods.items():
            sub = r[(r.index >= ps) & (r.index <= pe)]
            if len(sub) > 5:
                sw = (1 + sub).cumprod()
                out["periods"][pn] = {"cagr": annualized_ret(sub), "vol": annualized_vol(sub),
                                      "max_dd": max_drawdown(sw)[0], "total": sw.iloc[-1] - 1}
    return out

def fmt_eval(e):
    return (f"{e['name']:<18} CAGR={e['cagr']*100:6.2f}%  Vol={e['vol']*100:5.2f}%  "
            f"MDD={e['max_dd']*100:6.2f}%  Sharpe={e['sharpe']:5.2f}  Calmar={e['calmar']:5.2f}  "
            f"Cash={e['avg_cash']*100:4.1f}%  换手={e['turnover']:.2f}")
