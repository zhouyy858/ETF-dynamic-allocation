# -*- coding: utf-8 -*-
"""v40 纸面实验：159509，以及相对 v39 的全市场换券 / 机制网格。

规则在 PRE_RULES 里写死。先 --fingerprint-only，再跑。结果不得回改规则。
不改 references/final_cfg_v39.json，不改 README 持仓段，不改 SKILL.md，不改引擎默认路径。
纸面，不实盘。过关才写 references/final_cfg_v40.json，并标明草案、未激活。

用法:
  python3 scripts/exp_v40_universe.py --fingerprint-only
  python3 scripts/exp_v40_universe.py
"""
import sys, os, json, hashlib, copy, math
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from, DATA_DIR
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from swaps import pnl_panel, _gap_fill, _real_fill

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
UNI = os.path.join(DATA_DIR, "v40_universe")
META = os.path.join(UNI, "meta.json")
CFG39 = os.path.join(REF, "final_cfg_v39.json")
CFG37 = os.path.join(REF, "final_cfg_v37.json")
CFG30 = os.path.join(REF, "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
# 本环境在写规则之前、用未改过的 v39 配置测得。对不上就中止，不解释候选。
ANCHOR = {
    "proxy": {"cagr": 0.12218366412644954, "mdd": -0.09563306611941624, "cal": 1.2776298939728628},
    "real": {"cagr": 0.25571062264795885, "mdd": -0.037226252269268345, "cal": 6.869093907125792},
    "oos": {"cal": 1.5323794903341015},
}
THR = 0.10
THR_STRICT = 0.15  # 技能文档的周度复检线。过关用 0.10；建议升默认还要过 0.15。
BLENDS = (0.30, 0.50, 0.70)  # 并存臂只晋升 0.50。两侧是平台，不能事后改选。
MECH = [
    ("mom_win", "defense_momentum_win", [60, 80, 100, 120], 80),
    ("min_delta", "min_delta", [0.02, 0.035, 0.05, 0.08], 0.035),
    ("gate_win", "gate_win", [60, 80, 100, 120, 140], 100),
    ("prem_cut", "premium_cut", [
        [0.60, 0.40, 0.20],
        [0.45, 0.22, 0.08],
        [0.35, 0.18, 0.06],
        [0.25, 0.12, 0.04],
    ], [0.45, 0.22, 0.08]),
]
ABLATE = [
    ("premium_gate_off", "premium_gate", False),
    ("corr_risk_off", "corr_risk", False),
    ("defense_momentum_off", "defense_momentum", False),
    ("premium_tilt_off", "premium_tilt", False),
    ("speed_brake_on", "speed_brake", True),
]

PRE_RULES = """
## 1. 预注册（本节在跑数之前写入，不随结果改写）

对照是同一次跑出来的 v39（`references/final_cfg_v39.json`：现金 100% 511260，159232→159222，515100→563020）。v37、v30 只做对照，不作为 Δ 的分母。不改 v39 文件，不改 README 当前持仓段，不改 SKILL.md，不改 `scripts/engine.py`，不把默认路径指到新文件。纸面，不实盘。`allow_live` 不出现。

引擎、费用（单边万 5）、信号（`signal_lag=1`、溢价 T-2、先计提后成交、`strict=True`）、周五一笔，与 v39 相同。打分、深熊锁、底仓、溢价门控、溢价倾斜的输入仍是原来的五列面板。换券只替换损益，不把新券的价格送进信号。

窗口：

- proxy 全历史从 2014-06-23 起
- real 从 2025-04-23 起
- OOS 是 proxy 上的独立起点 2022-01-04（新策略实例）
- 自有样本：从该代码第一根自身收益起到同一末日，基线用同一起点重跑

锚（写规则之前、在这台机器上用未改配置测得的 v39）：proxy CAGR 0.12218366412644954、MDD -0.09563306611941624、Calmar 1.2776298939728628；real 0.25571062264795885 / -0.037226252269268345 / 6.869093907125792；OOS Calmar 1.5323794903341015。对不上就中止，容差 1e-6。公布口径是 12.22% / -9.56% / 1.28，25.57% / -3.72% / 6.87，OOS 1.532。

### 1.1 宇宙（不看策略收益）

东财 `fundcode_search.js` 里名称含 ETF、不含联接、不含杠杆/反向/做空/两倍/货币的场内基金，全部拉 2025-06-01 以来的前复权日 K，算近端成交额（最多 120 个成交额>0 的交易日，均值，单位元）。这是全市场普查，不是等用户点名。

主题桶按名称先匹配者优先，写在 `scripts/exp_v40_collect.py` 的 `bucket_of`：纳指科技、纳指生物、纳指100、标普生物、标普500、恒生科技、中概、现金流、红利低波/红利质量/中证红利（不含港股、恒生、标普、A500、300、800、50）、科创50（不含增强）、创业板宽基（不含 50/200/综/行业主题）、沪深300、中证500、中证1000、A 股科技 ETF、国债/地方债。

资格：近端成交额 ≥ 2000 万，近端 K 线 ≥ 40。159509 强制纳入。现金剔除 v37/v39 已经比过的代码，期限规则与 v39 相同（不要 1-3 年、0-4 年、30 年、政金、国开）。当前成交标的 159222、563020、159941、513500、159952、511260 不作为新候选。

每个（桶，指数名）按 `log10(成交额) - 50*总费率 + （QDII 才有）(-20*溢价中位数)` 取前 3。缺费率按 1% 计。QDII 缺溢价中位数按 5% 计，用来惩罚，不拿去填假溢价。总费率 = 管理费 + 托管费。

另外取不在主题桶里、名称不含债/黄金/白银/豆粕/货币、成交额过线的前 20 名做通配。槽位按与五列代理日收益的相关系数最高者分配（重叠至少 60 日）。这是日收益相关，不是策略 Calmar。相关低于 0.30 的仍跑，但不能晋升。

短名单以 `assets/data/v40_universe/meta.json` 为准。指纹覆盖这份文件。脚本只接受 `kept` 的记录。跑完不准按收益把落选的补进来。

### 1.2 159509

用户点名，不是扫描挑出来的。身份以东财基金详情为准，写在结果里（名称、指数、成立日）。三条臂：

- (a) 替换纳指槽：`159941` 的损益换成 159509。主口径是前复权，与 `scripts/swaps.py` 一致。
- (b) 与 159941 并存：纳指槽收益 = (1-α)×原槽 + α×159509。α 只测 0.30、0.50、0.70。唯一可以晋升的是 0.50。0.30 和 0.70 是平台，不许事后改选最高的那个。
- (c) 全换对照：同一张表里的其他科技/成长 QDII（纳指100、纳指生物、标普生物、恒生科技、中概）也是整槽替换，不是只换一半。

上市前用原槽收益拼接。内部缺口最长不超过 5 个交易日、占比不超过 2%，否则那些日子记 0 的规则与 `swaps._gap_fill` / `_real_fill` 相同；超过则该臂不能晋升，数字若仍打印，只作诊断。real 不把上市前的原券算进「全窗口」。盖不住 2025-04-23 的，real 从双方都有自身收益的第一天起重跑，基线同一起点，这种 real 不能过 (a)。

QDII 另跑累计净值口径。前复权与净值的 proxy ΔCal 或 real ΔCal 反号，则口径冲突，不能晋升。净值缺失也不能晋升。溢价只陈列，不叠进收益（前复权已经是场内价）。信号和溢价门控仍用 159941 / 513500，不改读 159509 的溢价。

### 1.3 其他换券与现金

一次只动一个槽，底是 v39 已经换过 159222 与 563020 的损益面板。A 股与通配用前复权。QDII 主口径前复权，净值是口径核对。现金不拼到 511260 上。盖不住 2014-06-23 的，proxy 只在缩短窗口上做诊断，不能过 (a)。

### 1.4 机制网格（样本内选参，事先声明）

在 v39 上每次只改一个旋钮。网格端点可以测，不能当选中点。选中点只能是内点里 proxy Calmar 最高、且 proxy MDD 严格好于 -10% 的那个。这是样本内选参：real 和 OOS 不参与挑选。

- 防御动量窗：60、80、100、120。v39 是 80。内点是 80 和 100。
- min_delta：0.02、0.035、0.05、0.08。v39 是 0.035。内点是 0.035 和 0.05。
- gate_win：60、80、100、120、140。v39 是 100。内点是 80、100、120。
- premium_cut：0.60/0.40/0.20，0.45/0.22/0.08（v39），0.35/0.18/0.06，0.25/0.12/0.04。内点是 v39 和 0.35/0.18/0.06。

平台：选中点两侧相邻格的 proxy Calmar 与选中点之差的绝对值不超过 0.05，且这两格的 proxy MDD 也严格好于 -10%。选中点若就是 v39 的现值，这一轴不晋升。

下列开关没有两侧平台，只归档，不晋升：关掉溢价门控、关掉相关风控、关掉防御动量、关掉溢价倾斜、打开速度刹车。

### 1.5 三关

ΔCal 相对同一次 v39。

(a) proxy 与 real 的 ΔCalmar 都大于 0，至少一个达到 +0.10；proxy MDD 严格好于 -10%（`mdd > -0.10`）。real 必须是从 2025-04-23 起的全窗口。proxy 必须从 2014-06-23 可评。现金因此要自身覆盖这段。技能文档的周度线是 +0.15，另记一列，不替代 +0.10。

(b) 平台。换券：同一槽、同一指数名里，至少两只的 proxy ΔCal 与 real ΔCal 都大于 0，而且两只都是全窗口。只有一只为正，算幸运代码，不晋升。同一指数的两只 ETF 收益高度相关，这一关只排除单代码事故，不是两次独立样本。通配还要与所分配槽的日收益相关 ≥ 0.30。并存臂不靠同指数第二只，靠 α 平台：0.50 过 (a)(c) 和识别，且 0.30 与 0.70 的 proxy ΔCal 与 0.50 同号、绝对差不超过 0.05，三档 proxy MDD 都严格好于 -10%。机制臂的平台见 1.4。

(c) OOS Calmar 不低于同一次 v39。容差只吃浮点（1e-9），不吃真正的变差。

识别：自有样本不少于 252 个交易日，且这段 proxy ΔCal 与全样本 proxy ΔCal 同号。不满足则不晋升。为省时间，识别和净值核对只对已经通过 (a) 的臂、以及全部 159509 臂计算。没过 (a) 的不靠识别翻案。

### 1.6 组合

只有单臂过了自己的 (a)(b)(c) 和识别（QDII 还要口径同号），才进组合。同一槽多只都过，留 proxy Calmar 更高的一只。并存和纳指槽替换冲突时同样只留更高的一只，不叠加。机制轴最多留一条，不把多条旋钮叠在一起。组合自己要再过 (a)(c) 和识别。识别的自有起点取各腿自有起点里最晚的那天。

过了才写 `references/final_cfg_v40.json`。文件标明草案、未激活。默认仍是 v39。并存若过关，配置里会写 `nasdaq_blend`，但引擎没有这条规则，这种草案不能执行，也不建议升默认。单臂过、组合不过，就不写文件。没有任何臂过，也不写。

建议升默认还要同时满足：组合（或唯一过关臂）的 ΔCal 至少有一窗达到 +0.15，引擎现有代码能执行，并且不是只靠缺口不合格的诊断数字。否则即使数字过了 +0.10，也只归档，不建议替换 v39。

事先不做的事：不改 v39 的阈值来迁就结果，不重开日频，不按收益补票，不改 README / SKILL / 引擎。
"""


def fingerprint(meta_bytes):
    h = hashlib.sha256()
    h.update(PRE_RULES.encode())
    h.update(meta_bytes)
    h.update(json.dumps({"blends": BLENDS, "mech": MECH, "ablate": ABLATE}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def pct(x):
    return "nan" if x is None or x != x else f"{x * 100:.2f}%"


def calf(x):
    return "nan" if x is None or x != x else f"{x:.2f}"


def dlt(x):
    return "nan" if x is None or x != x else f"{x:+.3f}"


def cell(m):
    if not m:
        return "—"
    return f"{pct(m['cagr'])} / {pct(m['mdd'])} / {calf(m['cal'])}"


def load_close(code, kind):
    path = os.path.join(UNI, "px" if kind != "nav" else "nav",
                        f"{code}.csv" if kind == "nav" else f"{code}_{kind}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def ret_from_close(df, col):
    if df is None or col not in df.columns:
        return None
    s = df[col].dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 5:
        return None
    return s.pct_change().dropna()


def gap_stats(own, index, first_floor):
    own = own.dropna().sort_index()
    first = own.index.min()
    cut = first if first_floor is None else max(pd.Timestamp(first), pd.Timestamp(first_floor))
    after = index[index >= cut]
    got = own.reindex(after)
    miss = got.isna()
    longest = run = 0
    for v in miss.tolist():
        run = run + 1 if v else 0
        longest = max(longest, run)
    frac = float(miss.sum()) / max(len(after), 1)
    return {
        "first": str(pd.Timestamp(first).date()),
        "n_own": int((index >= first).sum()),
        "longest": int(longest),
        "frac": frac,
        "ok": bool(longest <= 5 and frac <= 0.02),
    }


def splice(base, own, index, layer):
    """与 swaps.py 相同。失败时不返回失真序列。"""
    info = gap_stats(own, index, REAL_START if layer == "real" else None)
    if not info["ok"]:
        return None, info
    try:
        if layer == "real":
            out = _real_fill(base, own, index)
        else:
            out = _gap_fill(base, own, index)
    except RuntimeError as e:
        info["ok"] = False
        info["err"] = str(e)
        return None, info
    return out, info


def blend(base, own, index, alpha, layer):
    """上市后 α 用新券、1-α 用原槽。缺日沿用原槽，不记 0。缺口过大则失败。"""
    info = gap_stats(own, index, REAL_START if layer == "real" else None)
    if not info["ok"]:
        return None, info
    own = own.dropna().sort_index()
    first = own.index.min()
    cut = first if layer != "real" else max(first, pd.Timestamp(REAL_START))
    out = base.reindex(index).astype(float).copy()
    after = index[index >= cut]
    got = own.reindex(after)
    mixed = (1.0 - alpha) * out.loc[after] + alpha * got
    mixed = mixed.where(got.notna(), out.loc[after])
    out.loc[after] = mixed
    info["alpha"] = alpha
    return out, info


def metrics(res):
    e = evaluate(res)
    return {"cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
            "to": float(e["turnover"]), "end": str(res["rets"].index.max().date()),
            "n": int(len(res["rets"]))}


def run_dyn(R_sig, R_pnl, cfg, start, bond):
    ds = DynamicStrategy(R_sig, cfg=cfg)
    res = run_backtest(
        R_pnl, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=None, name="DYN",
        min_delta=cfg.get("min_delta", 0.035), repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True)
    return metrics(res)


def near(m, pin, keys, tol=1e-6):
    bad = []
    for k in keys:
        if abs(m[k] - pin[k]) > tol:
            bad.append(f"{k}:{m[k]:.12g}!={pin[k]:.12g}")
    return bad


def gate_a(base, row):
    if not row.get("proxy") or not row.get("full_proxy") or not row.get("full_real") or not row.get("real"):
        return {"ok": False, "same": False, "one": False, "mdd": False, "dp": None, "dr": None, "strict": False}
    dp = row["proxy"]["cal"] - base["proxy"]["cal"]
    dr = row["real"]["cal"] - base["real"]["cal"]
    same = dp > 0 and dr > 0
    one = same and max(dp, dr) >= THR
    mdd = row["proxy"]["mdd"] > -0.10
    strict = same and max(dp, dr) >= THR_STRICT and mdd
    return {"ok": bool(same and one and mdd), "same": same, "one": one, "mdd": mdd,
            "dp": dp, "dr": dr, "strict": bool(strict)}


def main():
    meta_bytes = open(META, "rb").read()
    meta = json.loads(meta_bytes.decode())
    digest = fingerprint(meta_bytes)
    print(f"[预注册指纹] {digest}")
    short = meta["shortlist"]
    if not any(s["code"] == "159509" for s in short):
        print("159509 不在短名单里，中止")
        sys.exit(2)
    if "--fingerprint-only" in sys.argv:
        print(fingerprint_banner(meta, digest))
        return

    cfg39 = json.load(open(CFG39))
    cfg37 = json.load(open(CFG37))
    cfg30 = json.load(open(CFG30))
    bond260 = rets_from(read_table("511260_nav.csv"), "cum_nav")
    bond010 = rets_from(read_table("511010_nav.csv"), "cum_nav")
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    Rp = pnl_panel(R, "proxy", cfg39)
    Rr_p = pnl_panel(Rr, "real", cfg39)
    print(f"[数据] proxy {R.index.min().date()}~{R.index.max().date()} "
          f"real {Rr.index.min().date()}~{Rr.index.max().date()} 短名单 {len(short)}")

    # 与 swaps.py 对齐：159222 / 563020 的拼接必须和 pnl_panel 一致。
    chk = Rp.copy()
    for slot, code in (("159232", "159222"), ("515100", "563020")):
        own = ret_from_close(load_close(code, "qfq") if os.path.exists(
            os.path.join(DATA_DIR, f"{code}_qfq.csv")) else None, "close")
        # 生产文件在 assets/data，不在 v40_universe。
        raw = pd.read_csv(os.path.join(DATA_DIR, f"{code}_qfq.csv"), parse_dates=["date"]).set_index("date")["close"]
        raw = raw[~raw.index.duplicated(keep="last")].sort_index().pct_change().dropna()
        filled, info = splice(R[slot], raw, R.index, "proxy")
        if filled is None or (filled - Rp[slot]).abs().max() > 1e-12:
            print(f"[swaps 对齐失败] {code} info={info}")
            sys.exit(3)
    print("[swaps 对齐] 159222 与 563020 的 proxy 拼接和 pnl_panel 一致")

    base = {}
    base["proxy"] = run_dyn(R, Rp, cfg39, PROXY_START, bond260)
    base["real"] = run_dyn(Rr, Rr_p, cfg39, REAL_START, bond260)
    base["oos"] = run_dyn(R, Rp, cfg39, OOS_START, bond260)
    bad = []
    bad += near(base["proxy"], ANCHOR["proxy"], ["cagr", "mdd", "cal"])
    bad += near(base["real"], ANCHOR["real"], ["cagr", "mdd", "cal"])
    bad += near({"cal": base["oos"]["cal"]}, ANCHOR["oos"], ["cal"])
    if bad:
        print("[锚失败]", bad)
        sys.exit(4)
    print(f"[锚] v39 proxy {cell(base['proxy'])} real {cell(base['real'])} OOS {base['oos']['cal']:.4f}")

    rows = []

    def add_control(name, cfg, bond, Rsig, Rpnl, Rsig_r, Rpnl_r):
        row = {"id": name, "kind": "control", "proxy": run_dyn(Rsig, Rpnl, cfg, PROXY_START, bond),
               "real": run_dyn(Rsig_r, Rpnl_r, cfg, REAL_START, bond),
               "oos": run_dyn(Rsig, Rpnl, cfg, OOS_START, bond),
               "full_proxy": True, "full_real": True}
        g = gate_a(base, row)
        row["dp"], row["dr"] = g["dp"], g["dr"]
        row["gate_a"] = False  # 对照不晋升
        rows.append(row)
        print(f"[对照] {name} {cell(row['proxy'])} Δ {dlt(row['dp'])} | real {cell(row['real'])} Δ {dlt(row['dr'])}")

    add_control("v37", cfg37, bond260, R, R, Rr, Rr)
    add_control("v30", cfg30, bond010, R, R, Rr, Rr)
    rows.append({"id": "v39", "kind": "control", "proxy": base["proxy"], "real": base["real"],
                 "oos": base["oos"], "full_proxy": True, "full_real": True, "dp": 0.0, "dr": 0.0,
                 "gate_a": False})

    own_cache = {}

    def own_of(code, kind):
        key = (code, kind)
        if key not in own_cache:
            if kind == "qfq":
                own_cache[key] = ret_from_close(load_close(code, "qfq"), "close")
            else:
                own_cache[key] = ret_from_close(load_close(code, "nav"), "cum_nav")
        return own_cache[key]

    def eval_arm(arm_id, kind, pnl_p, pnl_r, info, full_proxy, full_real, extra, cfg=None, bond=None):
        cfg = cfg or cfg39
        bond = bond260 if bond is None else bond
        row = {"id": arm_id, "kind": kind, "full_proxy": full_proxy, "full_real": full_real,
               "info": info, **extra}
        if pnl_p is None:
            row["proxy"] = row["real"] = row["oos"] = None
            row["gate_a"] = False
            row["dp"] = row["dr"] = None
            rows.append(row)
            print(f"[臂] {arm_id} 缺口不合格 {info}")
            return row
        start_p = PROXY_START if full_proxy else info["first"]
        start_r = REAL_START if full_real else info["first"]
        # 缩短窗口的基线单独算，Δ 仍相对全窗口 v39 只在 full 时有意义。
        # 拼接后的 proxy 从 2014 就有定义，OOS 仍从 2022-01-04 起，不因上市日晚于 2022 而缩短。
        row["proxy"] = run_dyn(R, pnl_p, cfg, start_p, bond)
        row["real"] = run_dyn(Rr, pnl_r, cfg, start_r, bond)
        if full_proxy:
            row["oos_full"] = True
            oos_start = OOS_START
        elif pd.Timestamp(info.get("first", PROXY_START)) > pd.Timestamp(OOS_START):
            row["oos_full"] = False
            oos_start = info["first"]
        else:
            row["oos_full"] = True
            oos_start = OOS_START
        row["oos"] = run_dyn(R, pnl_p, cfg, oos_start, bond)
        if not full_proxy or not full_real:
            # 诊断 Δ 用同一缩短起点的 v39，不拿去过 (a)。
            row["proxy_base_short"] = run_dyn(R, Rp, cfg39, start_p, bond260)
            row["real_base_short"] = run_dyn(Rr, Rr_p, cfg39, start_r, bond260)
            row["dp"] = row["proxy"]["cal"] - row["proxy_base_short"]["cal"]
            row["dr"] = row["real"]["cal"] - row["real_base_short"]["cal"]
            row["gate_a"] = False
            row["oos_base_short"] = run_dyn(R, Rp, cfg39, oos_start, bond260)
        else:
            g = gate_a(base, row)
            row["dp"], row["dr"] = g["dp"], g["dr"]
            row["gate_a"] = g["ok"]
            row["gate_strict"] = g["strict"]
            row["mdd_ok"] = g["mdd"]
        rows.append(row)
        print(f"[臂] {arm_id} proxy {cell(row['proxy'])} Δ {dlt(row['dp'])} | "
              f"real {cell(row['real'])} Δ {dlt(row['dr'])} full={full_proxy}/{full_real}")
        return row

    def cover_real(info):
        return pd.Timestamp(info["first"]) <= pd.Timestamp(REAL_START)

    # ---- 159509 替换与并存 ----
    s509 = next(s for s in short if s["code"] == "159509")
    q509 = own_of("159509", "qfq")
    n509 = own_of("159509", "nav")
    for kind, own in (("qfq", q509), ("nav", n509)):
        if own is None:
            rows.append({"id": f"159509_replace_{kind}", "kind": "etf159509", "proxy": None,
                         "note": "无序列", "gate_a": False, "full_proxy": False, "full_real": False})
            print(f"[臂] 159509 {kind} 无序列")
            continue
        pp, ip = splice(Rp["159941"], own, Rp.index, "proxy")
        rr, ir = splice(Rr_p["159941"], own, Rr_p.index, "real")
        full_p = pp is not None
        full_r = rr is not None and cover_real(ir)
        if rr is not None and not cover_real(ir):
            # 诊断用缩短起点；splice 的 real 在 first 之前仍是原槽，所以改从 first 起跑
            full_r = False
        pnl_p = Rp.copy()
        pnl_r = Rr_p.copy()
        if pp is not None:
            pnl_p["159941"] = pp
        if rr is not None:
            pnl_r["159941"] = rr
        eval_arm(f"159509_replace_{kind}", "etf159509",
                 pnl_p if pp is not None else None,
                 pnl_r if rr is not None else None,
                 ip, full_p, full_r,
                 {"code": "159509", "slot": "159941", "index_name": s509.get("index_name"),
                  "bucket": "nasdaq_tech", "px_kind": kind, "name": s509.get("name")})

    if q509 is not None:
        for alpha in BLENDS:
            pp, ip = blend(Rp["159941"], q509, Rp.index, alpha, "proxy")
            rr, ir = blend(Rr_p["159941"], q509, Rr_p.index, alpha, "real")
            pnl_p = Rp.copy(); pnl_r = Rr_p.copy()
            if pp is not None:
                pnl_p["159941"] = pp
            if rr is not None:
                pnl_r["159941"] = rr
            eval_arm(f"159509_blend_{alpha:.2f}", "blend",
                     pnl_p if pp is not None else None,
                     pnl_r if rr is not None else None,
                     ip, pp is not None, rr is not None and cover_real(ir),
                     {"code": "159509", "slot": "159941", "alpha": alpha,
                      "index_name": s509.get("index_name"), "px_kind": "qfq",
                      "bucket": "nasdaq_tech"})

    # ---- 短名单换券（159509 的替换臂上面已跑，这里跳过重复的强制项若桶是 nasdaq_tech 且代码是 159509）----
    for s in short:
        if s["code"] == "159509" and s.get("bucket") == "nasdaq_tech":
            continue
        slot = s.get("slot")
        bucket = s.get("bucket")
        own = own_of(s["code"], "qfq")
        if bucket == "cash":
            if own is None:
                rows.append({"id": f"cash:{s['code']}", "kind": "cash", "proxy": None,
                             "gate_a": False, "full_proxy": False, "full_real": False,
                             "code": s["code"], "note": "无行情"})
                continue
            first = str(own.index.min().date())
            full_p = pd.Timestamp(first) <= pd.Timestamp(PROXY_START)
            full_r = pd.Timestamp(first) <= pd.Timestamp(REAL_START)
            info = {"first": first, "ok": True, "n_own": int(len(own)), "longest": 0, "frac": 0.0}
            # 不拼接。缩短窗口只诊断。
            start_p = PROXY_START if full_p else first
            start_r = REAL_START if full_r else first
            # 用 v39 权益面板，只换债券序列。引擎会在起点之前没有数据时用逆回购填，所以起点必须卡在自身第一天。
            row = {"id": f"cash:{s['code']}", "kind": "cash", "code": s["code"],
                   "slot": None, "bucket": "cash", "index_name": s.get("index_name"),
                   "name": s.get("name"), "full_proxy": full_p, "full_real": full_r, "info": info,
                   "max_corr": None}
            row["proxy"] = run_dyn(R, Rp, cfg39, start_p, own)
            row["real"] = run_dyn(Rr, Rr_p, cfg39, start_r, own)
            oos_start = OOS_START if pd.Timestamp(first) <= pd.Timestamp(OOS_START) else first
            row["oos_full"] = oos_start == OOS_START
            row["oos"] = run_dyn(R, Rp, cfg39, oos_start, own)
            if full_p and full_r:
                g = gate_a(base, row)
                row["dp"], row["dr"] = g["dp"], g["dr"]
                row["gate_a"] = g["ok"]
                row["gate_strict"] = g["strict"]
            else:
                row["proxy_base_short"] = run_dyn(R, Rp, cfg39, start_p, bond260)
                row["real_base_short"] = run_dyn(Rr, Rr_p, cfg39, start_r, bond260)
                row["dp"] = row["proxy"]["cal"] - row["proxy_base_short"]["cal"]
                row["dr"] = row["real"]["cal"] - row["real_base_short"]["cal"]
                row["gate_a"] = False
            rows.append(row)
            print(f"[臂] cash:{s['code']} proxy {cell(row['proxy'])} Δ {dlt(row['dp'])} full={full_p}")
            continue
        if slot not in Rp.columns or own is None:
            rows.append({"id": f"{bucket}:{s['code']}", "kind": "swap", "proxy": None,
                         "gate_a": False, "full_proxy": False, "full_real": False,
                         "code": s["code"], "note": "无槽或无行情"})
            print(f"[臂] {bucket}:{s['code']} 跳过 slot={slot}")
            continue
        pp, ip = splice(Rp[slot], own, Rp.index, "proxy")
        rr, ir = splice(Rr_p[slot], own, Rr_p.index, "real")
        pnl_p = Rp.copy(); pnl_r = Rr_p.copy()
        if pp is not None:
            pnl_p[slot] = pp
        if rr is not None:
            pnl_r[slot] = rr
        full_r = rr is not None and cover_real(ir)
        eval_arm(f"{bucket}:{s['code']}", "swap",
                 pnl_p if pp is not None else None,
                 pnl_r if rr is not None else None,
                 ip, pp is not None, full_r,
                 {"code": s["code"], "slot": slot, "bucket": bucket,
                  "index_name": s.get("index_name") or "", "name": s.get("name"),
                  "px_kind": "qfq", "max_corr": s.get("max_corr"),
                  "amt_120": s.get("amt_120"), "fee": s.get("fee"),
                  "estab": s.get("estab")})

    # ---- 机制 ----
    for axis, key, grid, center in MECH:
        for i, val in enumerate(grid):
            cfg = copy.deepcopy(cfg39)
            cfg[key] = val
            tag = f"mech:{axis}={val if not isinstance(val, list) else '/'.join(str(x) for x in val)}"
            row = {"id": tag, "kind": "mech", "axis": axis, "value": val, "idx": i,
                   "is_center": val == center, "is_end": i == 0 or i == len(grid) - 1,
                   "full_proxy": True, "full_real": True, "oos_full": True}
            row["proxy"] = run_dyn(R, Rp, cfg, PROXY_START, bond260)
            row["real"] = run_dyn(Rr, Rr_p, cfg, REAL_START, bond260)
            row["oos"] = run_dyn(R, Rp, cfg, OOS_START, bond260)
            g = gate_a(base, row)
            row["dp"], row["dr"] = g["dp"], g["dr"]
            row["gate_a"] = g["ok"]
            row["gate_strict"] = g["strict"]
            row["mdd_ok"] = g["mdd"]
            rows.append(row)
            print(f"[机制] {tag} proxy {cell(row['proxy'])} Δ {dlt(row['dp'])} real Δ {dlt(row['dr'])}")

    for name, key, val in ABLATE:
        cfg = copy.deepcopy(cfg39)
        cfg[key] = val
        row = {"id": f"ablate:{name}", "kind": "ablate", "full_proxy": True, "full_real": True,
               "oos_full": True}
        row["proxy"] = run_dyn(R, Rp, cfg, PROXY_START, bond260)
        row["real"] = run_dyn(Rr, Rr_p, cfg, REAL_START, bond260)
        row["oos"] = run_dyn(R, Rp, cfg, OOS_START, bond260)
        g = gate_a(base, row)
        row["dp"], row["dr"] = g["dp"], g["dr"]
        row["gate_a"] = False  # 无平台，不晋升
        row["would_a"] = g["ok"]
        rows.append(row)
        print(f"[消融] {name} proxy Δ {dlt(row['dp'])} real Δ {dlt(row['dr'])}")

    # ---- 识别与净值核对：只对过 (a) 的，以及全部 159509 臂 ----
    def needs_extra(row):
        if row.get("kind") == "etf159509" or row.get("kind") == "blend":
            return True
        return bool(row.get("gate_a"))

    for row in rows:
        if not needs_extra(row) or not row.get("proxy"):
            continue
        # 机制没有上市日。预注册：自有样本就是全样本，同号自动成立。
        if row.get("kind") == "mech":
            row["ident_dp"] = row["dp"]
            row["ident_ok"] = True
            row["ident_n"] = row["proxy"]["n"]
            continue
        first = (row.get("info") or {}).get("first")
        if not first:
            row["ident_ok"] = False
            continue
        if row.get("kind") == "cash":
            # 现金的自有样本就是缩短或全长的那次，不再另跑。
            n = row["proxy"]["n"]
            row["ident_n"] = n
            row["ident_dp"] = row["dp"]
            row["ident_ok"] = bool(n >= 252 and row["dp"] is not None and row.get("full_proxy")
                                   and (row["dp"] > 0) == (row["dp"] > 0))
            # 现金全窗口才谈得上与全样本同号；缩短的已经不能过 (a)。
            continue
        n_own = (row.get("info") or {}).get("n_own") or 0
        row["ident_n"] = n_own
        if n_own < 252 or not row.get("full_proxy"):
            row["ident_ok"] = False
            row["ident_dp"] = None
            continue
        row["_need_ident_rerun"] = True

    # 第二遍：换券臂的识别重跑。pnl 没有存在 row 里，按规则重建一次（确定性，不用结果选券）。
    def rebuild_pnl(row, layer_R, layer_base, layer):
        code = row.get("code")
        slot = row.get("slot")
        kind = row.get("px_kind") or "qfq"
        own = own_of(code, "nav" if kind == "nav" else "qfq")
        if own is None or slot not in layer_base.columns:
            return None
        if row.get("kind") == "blend":
            out, _ = blend(layer_base[slot], own, layer_base.index, row["alpha"], layer)
        else:
            out, _ = splice(layer_base[slot], own, layer_base.index, layer)
        if out is None:
            return None
        pnl = layer_base.copy()
        pnl[slot] = out
        return pnl

    for row in rows:
        if not row.get("_need_ident_rerun"):
            continue
        first = row["info"]["first"]
        pnl = rebuild_pnl(row, R, Rp, "proxy")
        if pnl is None:
            row["ident_ok"] = False
            continue
        m = run_dyn(R, pnl, cfg39, first, bond260)
        b = run_dyn(R, Rp, cfg39, first, bond260)
        row["ident_dp"] = m["cal"] - b["cal"]
        row["ident_proxy"] = m
        same_sign = (row["ident_dp"] > 0 and (row.get("dp") or 0) > 0) or (
            row["ident_dp"] < 0 and (row.get("dp") or 0) < 0)
        row["ident_ok"] = bool(same_sign and row["ident_n"] >= 252)
        print(f"[识别] {row['id']} ownΔ {dlt(row['ident_dp'])} ok={row['ident_ok']}")

    # QDII 净值核对：159509 已有 nav 臂。其他过 (a) 的 QDII 补一条 nav。
    qdii_buckets = {"nasdaq", "nasdaq_tech", "nasdaq_bio", "hs_tech", "china_net", "spx", "spx_bio"}
    for row in list(rows):
        if row.get("kind") != "swap" or not row.get("gate_a"):
            continue
        if row.get("bucket") not in qdii_buckets and row.get("bucket") != "wildcard":
            continue
        own = own_of(row["code"], "nav")
        if own is None:
            if row.get("bucket") in qdii_buckets:
                row["nav_agree"] = False
                row["nav_note"] = "无净值"
            continue
        slot = row["slot"]
        pp, ip = splice(Rp[slot], own, Rp.index, "proxy")
        rr, ir = splice(Rr_p[slot], own, Rr_p.index, "real")
        if pp is None or rr is None:
            row["nav_agree"] = False
            row["nav_note"] = "净值缺口"
            continue
        pnl_p = Rp.copy(); pnl_p[slot] = pp
        pnl_r = Rr_p.copy(); pnl_r[slot] = rr
        mp = run_dyn(R, pnl_p, cfg39, PROXY_START, bond260)
        mr = run_dyn(Rr, pnl_r, cfg39, REAL_START, bond260)
        dp = mp["cal"] - base["proxy"]["cal"]
        dr = mr["cal"] - base["real"]["cal"]
        row["nav_dp"], row["nav_dr"] = dp, dr
        row["nav_agree"] = (dp > 0) == (row["dp"] > 0) and (dr > 0) == (row["dr"] > 0)
        print(f"[净值] {row['id']} navΔ {dlt(dp)}/{dlt(dr)} agree={row['nav_agree']}")

    # 159509 qfq vs nav 同号
    rep_q = next((r for r in rows if r["id"] == "159509_replace_qfq"), None)
    rep_n = next((r for r in rows if r["id"] == "159509_replace_nav"), None)
    if rep_q and rep_n and rep_q.get("dp") is not None and rep_n.get("dp") is not None:
        rep_q["nav_agree"] = (rep_q["dp"] > 0) == (rep_n["dp"] > 0) and (
            rep_q.get("dr") is None or rep_n.get("dr") is None or (rep_q["dr"] > 0) == (rep_n["dr"] > 0))
        rep_q["nav_dp"] = rep_n.get("dp")
        rep_q["nav_dr"] = rep_n.get("dr")

    # ---- (b) 同指数 ----
    families = {}
    for row in rows:
        if row.get("kind") not in ("swap", "etf159509", "cash"):
            continue
        if row.get("px_kind") not in (None, "qfq"):
            continue
        if not row.get("full_proxy") or not row.get("full_real"):
            continue
        key = (row.get("slot"), row.get("index_name") or "")
        families.setdefault(key, []).append(row)
    for key, group in families.items():
        pos = [r for r in group if (r.get("dp") or 0) > 0 and (r.get("dr") or 0) > 0]
        ok = len(pos) >= 2
        for r in group:
            r["family_n"] = len(group)
            r["family_pos"] = len(pos)
            r["gate_b"] = ok

    # 通配相关
    for row in rows:
        if row.get("bucket") == "wildcard":
            corr = row.get("max_corr")
            row["corr_ok"] = corr is not None and corr >= 0.30
            if not row["corr_ok"]:
                row["gate_b"] = False
        elif row.get("kind") in ("swap", "etf159509", "cash") and "gate_b" not in row:
            row["gate_b"] = False

    # 并存平台
    blends = {round(r.get("alpha"), 2): r for r in rows if r.get("kind") == "blend" and r.get("alpha") is not None}
    b50 = blends.get(0.50)
    if b50 and b50.get("dp") is not None:
        sides_ok = True
        for a in (0.30, 0.70):
            s = blends.get(a)
            if not s or s.get("dp") is None or s["proxy"]["mdd"] <= -0.10:
                sides_ok = False
                break
            if (s["dp"] > 0) != (b50["dp"] > 0) or abs(s["dp"] - b50["dp"]) > 0.05:
                sides_ok = False
        b50["plateau"] = sides_ok and b50["proxy"]["mdd"] > -0.10
    # 机制平台
    for axis, key, grid, center in MECH:
        group = [r for r in rows if r.get("kind") == "mech" and r.get("axis") == axis]
        group.sort(key=lambda r: r["idx"])
        interiors = [r for r in group if not r["is_end"] and r.get("mdd_ok")]
        if not interiors:
            continue
        winner = max(interiors, key=lambda r: r["proxy"]["cal"])
        winner["mech_selected"] = True
        if winner["is_center"]:
            winner["plateau"] = False
            winner["mech_promote"] = False
            continue
        i = winner["idx"]
        left = group[i - 1]
        right = group[i + 1]
        flat = (abs(left["proxy"]["cal"] - winner["proxy"]["cal"]) <= 0.05
                and abs(right["proxy"]["cal"] - winner["proxy"]["cal"]) <= 0.05
                and left["proxy"]["mdd"] > -0.10 and right["proxy"]["mdd"] > -0.10)
        winner["plateau"] = flat
        oos_ok = winner.get("oos") and winner["oos"]["cal"] + 1e-9 >= base["oos"]["cal"]
        ident = winner.get("ident_ok", True)
        winner["mech_promote"] = bool(winner.get("gate_a") and flat and oos_ok and ident)

    def oos_ok(row):
        if not row.get("oos") or not row.get("oos_full", True):
            return False
        return row["oos"]["cal"] + 1e-9 >= base["oos"]["cal"]

    def passed(row):
        if row.get("kind") == "blend":
            if row.get("alpha") != 0.50:
                return False
            nav_ok = True if rep_q is None else bool(rep_q.get("nav_agree"))
            return bool(row.get("gate_a") and row.get("plateau") and oos_ok(row)
                        and row.get("ident_ok") and nav_ok)
        if row.get("kind") == "mech":
            return bool(row.get("mech_promote"))
        if row.get("kind") not in ("swap", "etf159509", "cash"):
            return False
        if row.get("px_kind") not in (None, "qfq"):
            return False
        nav_ok = True
        if row.get("bucket") in qdii_buckets or "nav_agree" in row:
            nav_ok = bool(row.get("nav_agree"))
        return bool(row.get("gate_a") and row.get("gate_b") and oos_ok(row)
                    and row.get("ident_ok") and nav_ok)

    for r in rows:
        r["promote"] = bool(passed(r))
    winners = [r for r in rows if r["promote"]]
    print(f"[过关单臂] {[r['id'] for r in winners]}")

    # 组合
    combo = None
    combo_note = "没有单臂过关，不组组合。"
    if winners:
        by_slot = {}
        mech_w = []
        for r in winners:
            if r["kind"] == "mech":
                mech_w.append(r)
            else:
                by_slot.setdefault(r.get("slot"), []).append(r)
        chosen = []
        for slot, group in by_slot.items():
            group.sort(key=lambda r: r["proxy"]["cal"], reverse=True)
            chosen.append(group[0])
        mech_pick = None
        if mech_w:
            mech_w.sort(key=lambda r: r["proxy"]["cal"], reverse=True)
            mech_pick = mech_w[0]
        # 重建组合 pnl
        pnl_p = Rp.copy(); pnl_r = Rr_p.copy()
        cfg = copy.deepcopy(cfg39)
        blend_alpha = None
        bond_c = bond260
        latest = []
        for r in chosen:
            if r["kind"] == "blend":
                blend_alpha = r["alpha"]
                own = own_of("159509", "qfq")
                pp, _ = blend(Rp["159941"], own, Rp.index, blend_alpha, "proxy")
                rr, _ = blend(Rr_p["159941"], own, Rr_p.index, blend_alpha, "real")
                pnl_p["159941"] = pp
                pnl_r["159941"] = rr
            elif r["kind"] == "cash":
                bond_c = own_of(r["code"], "qfq")
                cfg["cash_bond_code"] = r["code"]
            elif r["kind"] in ("swap", "etf159509"):
                own = own_of(r["code"], "qfq")
                pp, _ = splice(Rp[r["slot"]], own, Rp.index, "proxy")
                rr, _ = splice(Rr_p[r["slot"]], own, Rr_p.index, "real")
                pnl_p[r["slot"]] = pp
                pnl_r[r["slot"]] = rr
                swaps = dict(cfg.get("instrument_swaps") or {})
                swaps[r["slot"]] = r["code"]
                cfg["instrument_swaps"] = swaps
            latest.append(pd.Timestamp((r.get("info") or {}).get("first", PROXY_START)))
        if mech_pick:
            for axis, key, grid, center in MECH:
                if mech_pick.get("axis") == axis:
                    cfg[key] = mech_pick["value"]
        combo_proxy = run_dyn(R, pnl_p, cfg, PROXY_START, bond_c)
        combo_real = run_dyn(Rr, pnl_r, cfg, REAL_START, bond_c)
        combo_oos = run_dyn(R, pnl_p, cfg, OOS_START, bond_c)
        crow = {"proxy": combo_proxy, "real": combo_real, "oos": combo_oos,
                "full_proxy": True, "full_real": True, "oos_full": True}
        g = gate_a(base, crow)
        crow["dp"], crow["dr"] = g["dp"], g["dr"]
        crow["gate_a"] = g["ok"]
        crow["gate_strict"] = g["strict"]
        # 识别
        start_id = max(latest) if latest else pd.Timestamp(PROXY_START)
        if mech_pick and not chosen:
            start_id = pd.Timestamp(PROXY_START)
        mid = run_dyn(R, pnl_p, cfg, str(start_id.date()), bond_c)
        mb = run_dyn(R, Rp, cfg39, str(start_id.date()), bond260)
        ident_dp = mid["cal"] - mb["cal"]
        # 交易日数
        n_id = int((R.index >= start_id).sum())
        ident_ok = n_id >= 252 and ((ident_dp > 0 and crow["dp"] > 0) or (ident_dp < 0 and crow["dp"] < 0))
        oos_pass = combo_oos["cal"] + 1e-9 >= base["oos"]["cal"]
        crow["ident_ok"] = ident_ok
        crow["ident_dp"] = ident_dp
        crow["ident_n"] = n_id
        crow["passed"] = bool(g["ok"] and oos_pass and ident_ok)
        crow["chosen"] = [r["id"] for r in chosen]
        crow["mech"] = None if not mech_pick else mech_pick["id"]
        crow["cfg"] = cfg
        crow["blend_alpha"] = blend_alpha
        combo = crow
        combo_note = "组合过关" if crow["passed"] else "单臂过了，组合没有再过 (a)(c) 和识别"
        print(f"[组合] {cell(combo_proxy)} Δ {dlt(crow['dp'])} real Δ {dlt(crow['dr'])} "
              f"OOS {combo_oos['cal']:.3f} passed={crow['passed']}")

    # 写草案
    wrote = None
    recommend = False
    if combo and combo["passed"]:
        cfg = combo["cfg"]
        engine_ok = combo["blend_alpha"] is None and combo["cfg"].get("cash_bond_code", "511260") == "511260"
        strict_ok = bool(combo.get("gate_strict"))
        recommend = bool(engine_ok and strict_ok)
        cfg["version"] = "v40-draft"
        cfg["_note"] = (
            "v40 草案、未激活，默认仍是 final_cfg_v39.json。不要合并，不要 allow_live。"
            f" 来自预注册指纹 {digest}。单臂 {combo['chosen']} 机制 {combo['mech']}。"
            + ("引擎没有 nasdaq_blend，这份草案不能执行。" if combo["blend_alpha"] is not None else
               "instrument_swaps 由 scripts/swaps.py 读取。")
            + (" 达到 +0.15，仍不改默认。" if strict_ok else " 未到技能文档 +0.15，不建议升默认。")
        )
        if combo["blend_alpha"] is not None:
            cfg["nasdaq_blend"] = {"159941": 1.0 - combo["blend_alpha"], "159509": combo["blend_alpha"]}
            cfg["engine_implements_blend"] = False
        path = os.path.join(REF, "final_cfg_v40.json")
        json.dump(cfg, open(path, "w"), ensure_ascii=False, indent=1)
        wrote = path
        # 让 swaps.py 能读到前复权文件，只复制组合里新出现的代码。不改 v39。
        for r in winners:
            code = r.get("code")
            if not code:
                continue
            src_p = os.path.join(UNI, "px", f"{code}_qfq.csv")
            dst_p = os.path.join(DATA_DIR, f"{code}_qfq.csv")
            if os.path.exists(src_p) and not os.path.exists(dst_p):
                open(dst_p, "w").write(open(src_p).read())

    text = render(meta, digest, base, rows, combo, combo_note, wrote, recommend, s509)
    rep_path = os.path.join(REF, "exp_v40_universe_prereg_report.md")
    open(rep_path, "w").write(text)
    # 结果 json 去掉 cfg 里的大段，保留数字
    dump = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in ("cfg",)}
        dump.append(d)
    out = {"fingerprint": digest, "anchor": ANCHOR, "base": base,
           "rows": dump, "combo_note": combo_note,
           "combo": None if not combo else {k: v for k, v in combo.items() if k != "cfg"},
           "wrote": wrote, "recommend_default": recommend}
    json.dump(out, open(os.path.join(REF, "exp_v40_universe_results.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    print(f"[写] {rep_path}")
    print(f"[结论] 草案={wrote} 建议升默认={recommend} {combo_note}")


def fingerprint_banner(meta, digest):
    lines = [f"指纹 {digest}", f"短名单 {len(meta['shortlist'])} 只"]
    for s in meta["shortlist"]:
        lines.append(f"  {s.get('bucket')} {s['code']} {s.get('name')} {s.get('index_name')}")
    return "\n".join(lines)


def render(meta, digest, base, rows, combo, combo_note, wrote, recommend, s509):
    sm = meta["summary"]
    L = []
    A = L.append
    A("# v40 纸面实验：159509 与相对 v39 的全市场扫描")
    A("")
    A(f"> 预注册指纹 `{digest}`。普查了 {sm['n_census']} 只场内 ETF"
      f"（名称筛到 {sm['n_etf_name']}），近端成交额 ≥ 2000 万的有 {sm['n_above_floor']} 只。"
      f"主题名称命中 {sm['n_themed_name']}，过线 {sm['n_themed_pass']}。"
      f"短名单 {sm['n_shortlist']}。普查失败 {sm['n_census_err']}。"
      f"第 1 节是跑数之前写死的规则。第 2 节才是结果。规则没有按结果回改。")
    A("")
    A("默认仍是 v39。本文件是纸面记录，不下单，不改 `final_cfg_v39.json`。")
    A("")
    A(PRE_RULES.strip())
    A("")
    A("### 1.7 短名单（由上面的公式从 meta.json 算出，跑数之前已经固定）")
    A("")
    A("| 桶 | 代码 | 名称 | 槽 | 指数 | 成立 | 行情 | 近端成交额(亿) | 费率 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(meta["shortlist"], key=lambda x: (x.get("bucket") or "", x.get("code"))):
        amt = s.get("amt_120")
        fee = s.get("fee")
        A(f"| {s.get('bucket')} | {s['code']} | {s.get('name')} | {s.get('slot')} | {s.get('index_name')} | "
          f"{s.get('estab')} | {s.get('px_start')}~{s.get('px_end')} | "
          f"{'—' if not amt else f'{amt/1e8:.2f}'} | {'—' if fee is None else f'{fee*100:.2f}%'} |")
    A("")
    A("## 2. 结果")
    A("")
    A(f"159509 东财简称「{s509.get('name')}」，指数「{s509.get('index_name')}」"
      f"（{s509.get('index_code')}），成立 {s509.get('estab')}。"
      f"全称以采集时的基金详情为准：景顺长城纳斯达克科技市值加权 ETF（QDII）。"
      f"行情 {s509.get('px_start')} ~ {s509.get('px_end')}，{s509.get('px_n')} 根。"
      f"近端成交额 {'—' if not s509.get('amt_120') else f'{s509['amt_120']/1e8:.2f} 亿'}。"
      f"这不是纳指100。")
    prem = s509.get("prem") or {}
    if prem:
        A(f"溢价中位数 {prem.get('median'):.2%}，末日 {prem.get('last'):.2%}，"
          f"样本 {prem.get('start')} ~ {prem.get('end')}。溢价没有叠进收益。")
    A("")
    A(f"**v39** proxy {cell(base['proxy'])}；real {cell(base['real'])}；OOS Cal {base['oos']['cal']:.4f}。")
    A("Δ 相对这一条。缩短窗口的 Δ 是相对同一起点重跑的 v39，不能过 (a)，表里标成诊断。")
    A("")
    A("| 臂 | proxy | ΔCal | real | ΔCal | OOS Cal | (a) | (b)/平台 | 识别 | 过关 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") == "control" and r["id"] not in ("v37", "v30", "v39"):
            continue
        btxt = "—"
        if r.get("kind") == "swap":
            btxt = f"{'是' if r.get('gate_b') else '否'} ({r.get('family_pos','?')}/{r.get('family_n','?')})"
        elif r.get("kind") == "blend" and r.get("alpha") == 0.50:
            btxt = "是" if r.get("plateau") else "否"
        elif r.get("kind") == "mech" and r.get("mech_selected"):
            btxt = "是" if r.get("plateau") else "否"
        elif r.get("kind") == "ablate":
            btxt = "无关可平台"
        ident = "—"
        if "ident_ok" in r:
            ident = "是" if r["ident_ok"] else "否"
        if r["id"] in ("v39", "v37", "v30") or r.get("kind") == "ablate":
            ok = "—"
        else:
            ok = "是" if r.get("promote") else "否"
        oos = "—" if not r.get("oos") else f"{r['oos']['cal']:.3f}"
        if r.get("oos_full") is False:
            oos = oos + " 缩短"
        A(f"| {r['id']} | {cell(r.get('proxy'))} | {dlt(r.get('dp'))} | {cell(r.get('real'))} | {dlt(r.get('dr'))} | "
          f"{oos} | {'是' if r.get('gate_a') else '否'} | {btxt} | {ident} | {ok} |")
    A("")
    A("### 2.1 没进短名单、但公式排在同指数后面的")
    A("")
    runners = meta.get("runners") or []
    if not runners:
        A("没有（每组不足 4 只，或采集文件没写落选者）。")
    else:
        for s in runners:
            A(f"- {s.get('bucket')} {s['code']} {s.get('name')} {s.get('index_name')}")
    A("")
    A("## 3. 结论")
    A("")
    A(conclusion(rows, combo, combo_note, wrote, recommend, base))
    A("")
    A("## 4. 泄漏与样本内")
    A("")
    A("换券不把新价格送进打分、深熊锁、均线或溢价门控。决策日的目标权重只用原五列截至前一日的数据，以及已经发生的组合收益。这不是用未来行情做信号。")
    A("")
    A("上市前拼接原槽。全历史 proxy 的 ΔCal 因此几乎只来自上市之后的那段，和 v39 把 159222、563020 拼进去是同一类偏差。识别关要求自有样本至少 252 个交易日且与全样本同号，就是为了不让拼接段假过关。即便识别通过，外推仍要打折：这段行情只有这一次。")
    A("")
    A("短名单按成交额、费率、溢价中位数和指数名固定，通配槽位按日收益相关分配。没有用策略 Calmar 决定谁能进表。表内「谁最高」是跑完才知道的，只能陈列，不能倒回去改短名单。")
    A("")
    A("机制网格的选中点用的是全历史 proxy Calmar。这是样本内选参。real 与 OOS 没有参与挑选，只用于确认。消融开关没有平台，即使数字好看也不晋升。")
    A("")
    A(f"指纹 `{digest}`。规则文本、meta.json、网格一起哈希。结果文件是 `references/exp_v40_universe_results.json`。")
    A("")
    return "\n".join(L) + "\n"


def conclusion(rows, combo, combo_note, wrote, recommend, base):
    def find(i):
        return next((r for r in rows if r["id"] == i), None)
    q = find("159509_replace_qfq")
    b = find("159509_blend_0.50")
    lines = []
    if q and q.get("proxy"):
        lines.append(
            f"159509 替换 159941：proxy {cell(q['proxy'])}（ΔCal {dlt(q.get('dp'))}），"
            f"real {cell(q.get('real'))}（ΔCal {dlt(q.get('dr'))}），"
            f"OOS {('—' if not q.get('oos') else f'{q['oos']['cal']:.3f}')}。"
            f"(a) {'过' if q.get('gate_a') else '不过'}。"
            f"同指数家族 {q.get('family_pos')}/{q.get('family_n')}，(b) {'过' if q.get('gate_b') else '不过'}。"
            f"净值核对 {'同号' if q.get('nav_agree') else '未同号或缺失'}。"
        )
    elif q:
        lines.append(f"159509 替换臂没有可评序列：{q.get('info') or q.get('note')}。")
    if b and b.get("proxy"):
        lines.append(
            f"159509 与 159941 以 50/50 并存：proxy ΔCal {dlt(b.get('dp'))}，real ΔCal {dlt(b.get('dr'))}，"
            f"平台 {'平坦' if b.get('plateau') else '不平坦'}，"
            f"(a) {'过' if b.get('gate_a') else '不过'}。"
        )
    # 扫描里 proxy Δ 最高的全窗口臂，只陈列
    cands = [r for r in rows if r.get("kind") in ("swap", "blend", "mech") and r.get("full_proxy") and r.get("full_real") and r.get("dp") is not None]
    if cands:
        best = max(cands, key=lambda r: r["dp"])
        lines.append(
            f"全窗口里 proxy ΔCal 最高的是 {best['id']}（Δ {dlt(best['dp'])} / real Δ {dlt(best.get('dr'))}）。"
            f"这是跑完才看到的最大值，不是预注册的选择。"
            f"它 {'过了单臂全部门槛' if best.get('promote') else '没有过三关'}。"
        )
    fails = [r for r in rows if r.get("kind") in ("swap", "blend", "mech", "cash", "etf159509") and not r.get("gate_a")]
    lines.append(f"没过 (a) 的臂有 {len(fails)} 条，原因是不同向、没到 +0.10、MDD 不优于 -10%，或窗口被缩短/缺口不合格。明细在第 2 节。")
    lines.append(combo_note)
    if wrote:
        lines.append(f"已写草案 `{wrote}`。未接入默认路径。")
    else:
        lines.append("没有写 `final_cfg_v40.json`。")
    if recommend:
        lines.append("数字上达到了预注册的晋升线，也到了 +0.15，引擎能执行。即便如此，默认 JSON 不改，不合并。增益若主要在新券上市之后，外推要打折。")
    else:
        lines.append("不建议把任何一条臂升成默认。v39 维持。v37、v30 仍是对照。")
    return "\n\n".join(lines)


if __name__ == "__main__":
    main()
