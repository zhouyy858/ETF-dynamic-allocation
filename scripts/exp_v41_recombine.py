# -*- coding: utf-8 -*-
"""v41 纸面实验：v39 六槽的减槽与换槽/加槽。

先有规则和指纹，再跑。不改 final_cfg_v39.json，不改引擎，不合并，不写 allow_live。

用法:
  python3 scripts/exp_v41_recombine.py --fingerprint-only
  python3 scripts/exp_v41_recombine.py --smoke
  python3 scripts/exp_v41_recombine.py
"""
import sys, os, json, hashlib, copy, time, subprocess
from itertools import combinations
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from, DATA_DIR
from engine import run_backtest, evaluate
from strategy import DynamicStrategy, SignalSet
from swaps import pnl_panel
import exp_v39_collect as collect_src

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "references")
UNI = os.path.join(DATA_DIR, "v41_universe")
V40 = "origin/cursor/v40-universe-scan-6206"
CFG_V39 = os.path.join(REF, "final_cfg_v39.json")
CFG_V37 = os.path.join(REF, "final_cfg_v37.json")
CFG_V30 = os.path.join(REF, "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
PRE_END = "2025-04-22"

LOCK = {
    "anchor": {
        "proxy": {"cagr": 0.12218366412644954, "mdd": -0.09563306611941624, "cal": 1.2776298939728628},
        "real": {"cagr": 0.25571062264795885, "mdd": -0.037226252269268345, "cal": 6.869093907125792},
        "oos_cal": 1.5323794903341015,
    },
    "slots": ["159232", "515100", "159941", "513500", "159952", "511260"],
    "traded": {"159232": "159222", "515100": "563020"},
    "modes": ["N", "R"],
    "groups": [
        {"id": "ndx100", "slot": "159941", "kind": "qdii", "pair": "corr", "codes": [
            {"code": "513100", "amt": 605000000}, {"code": "513300", "amt": 494000000}]},
        {"id": "ndx_tech", "slot": "159941", "kind": "qdii", "pair": "corr", "codes": [
            {"code": "159509", "amt": 1089000000}]},
        {"id": "spx", "slot": "513500", "kind": "qdii", "pair": "corr", "codes": [
            {"code": "513650", "amt": 359000000}, {"code": "159655", "amt": 138000000}]},
        {"id": "gold", "slot": "515100", "kind": "equity", "pair": "corr", "codes": [
            {"code": "518880", "amt": 3861000000}, {"code": "159934", "amt": 869000000}]},
        {"id": "oil_equity", "slot": "159232", "kind": "qdii", "pair": "corr", "codes": [
            {"code": "513350", "amt": 1394000000}, {"code": "159518", "amt": 803000000}]},
        {"id": "commodity", "slot": "159232", "kind": "equity", "pair": "corr", "codes": [
            {"code": "159981", "amt": 672000000}]},
        {"id": "cb", "slot": "159952", "kind": "equity", "pair": "corr", "codes": [
            {"code": "511380", "amt": 7874000000}, {"code": "511180", "amt": 1774000000}]},
        {"id": "hstech", "slot": "159952", "kind": "qdii", "pair": "corr", "codes": [
            {"code": "513130", "amt": 2850000000}, {"code": "513180", "amt": 2803000000}]},
        {"id": "cnet", "slot": "159952", "kind": "qdii", "pair": "corr", "codes": [
            {"code": "513050", "amt": 2181000000}]},
        {"id": "kcb", "slot": "159952", "kind": "equity", "pair": "corr", "codes": [
            {"code": "588000", "amt": 8500000000}, {"code": "588080", "amt": 2056000000}]},
        {"id": "semi", "slot": "159952", "kind": "equity", "pair": "corr", "codes": [
            {"code": "512480", "amt": 1703000000}, {"code": "159813", "amt": 446000000}]},
        {"id": "pharma", "slot": "159952", "kind": "equity", "pair": "corr", "codes": [
            {"code": "512010", "amt": 516000000}, {"code": "159929", "amt": 87000000}]},
        {"id": "bank", "slot": "159232", "kind": "equity", "pair": "corr", "codes": [
            {"code": "512800", "amt": 1019000000}, {"code": "516310", "amt": 183000000}]},
        {"id": "money", "slot": "cash", "kind": "cash", "pair": "corr", "codes": [
            {"code": "511880", "amt": 20542000000}, {"code": "511990", "amt": 11733000000}]},
        {"id": "duanrong", "slot": "cash", "kind": "cash", "pair": "corr", "codes": [
            {"code": "511360", "amt": 35236000000}]},
        {"id": "guokai", "slot": "cash", "kind": "cash", "pair": "corr", "codes": [
            {"code": "159649", "amt": 807000000}, {"code": "159651", "amt": 276000000}]},
        {"id": "tenor", "slot": "cash", "kind": "cash", "pair": "tenor", "codes": [
            {"code": "511010", "amt": 2157000000, "prod": "nav"},
            {"code": "511100", "amt": 7905000000},
            {"code": "511090", "amt": 4250000000}]},
    ],
    "combos": {"drop": ["159952"], "modes": ["N", "R"], "groups": ["ndx100", "ndx_tech"]},
    "doubles": [
        {"name": "ballast_primary", "legs": [
            {"slot": "515100", "group": "gold", "i": 0},
            {"slot": "159232", "group": "oil_equity", "i": 0}]},
        {"name": "ballast_pair", "legs": [
            {"slot": "515100", "group": "gold", "i": 1},
            {"slot": "159232", "group": "oil_equity", "i": 1}]},
    ],
}

PRE_HEAD = """
## 1. 预注册（本节在跑数之前写入，不随结果改写）

对照是同一次跑出来的 v39（`references/final_cfg_v39.json`：现金 100% 511260，159232→159222，515100→563020）。v37、v30 只做对照，不作为 Δ 的分母。不改 v39 文件，不改 README 当前持仓段，不改 SKILL.md，不改 `scripts/engine.py`，不改 `scripts/strategy.py`。纸面，不实盘。`allow_live` 不出现。不合并。

这轮问的是组合结构，不是再做一遍 v40 的同指数一换一。v40 已经否决的内容（159509 整槽替换与 50/50 并存、高溢价、同指数只有一只、配比平台倾斜）不写成新发现。50/50 这次不做。159509 与 513300 只在新结构里重测：去掉创业板之后再把纳指槽整段换成它们，并保留「不去掉创业板」的整槽替换作为全换对照。

引擎、费用（单边万 5）、信号（`signal_lag=1`、溢价 T-2、先计提后成交、`strict=True`）、周五一笔，与 v39 相同。打分、深熊锁、均线、溢价门控的输入仍是原来的五列。减槽把目标权重打成 0，不把对应的打分项从 `strategy.py` 里删掉，因为生产代码没有这个开关。换槽只替换该槽的损益，新券的价格不进信号。所以换进去的择时仍是原槽的择时，不能解释成新资产自己的趋势策略。

窗口：

- proxy 全历史从 2014-06-23 起
- real 从 2025-04-23 起。新券盖不住这一天的，real 从双方都有自身收益的第一天起，基线同一起点，这种 real 不能过 (a)
- OOS 是 proxy 上的独立组合起点 2022-01-04（信号面板仍用全历史，组合从这天重新建仓，与 v39 实验同一做法）
- 2025-04-22 止的 proxy 用来看增益是不是只出现在 real 窗口
- 自有样本：从该代码第一根自身收益起到同一末日，基线用同一起点重跑

锚（写规则之前、在这台机器上用未改配置测得的 v39，与 v40 报告同一组数）：proxy CAGR 0.12218366412644954、MDD -0.09563306611941624、Calmar 1.2776298939728628；real 0.25571062264795885 / -0.037226252269268345 / 6.869093907125792；OOS Calmar 1.5323794903341015。对不上就中止，容差 1e-6。

近 60 个交易日 159952 的目标权重在这条锚上是 0。这是跑规则之前看到的持仓事实，用来说明为什么减槽要单列创业板，不是用它来挑选哪一臂过关。

### 1.1 减槽

六个成交标的对应的槽是 159232（成交 159222）、515100（成交 563020）、159941、513500、159952、511260。C(6,1) 与 C(6,2) 全部都跑，不先淘汰。每个组合跑两种正规化，都是现有配置字段能表达的，不是另一套纸面权重。

模式 N，引擎原生 `exclude`：

- 成长拆分在初始化时丢掉被剔除的名字，再归一，成长预算还在。
- 防御动量仍按 159232 与 515100 切分，切完才把被剔除的目标打成 0。被丢掉的那一份防御弹性进现金，不给活下来的那只。
- 底仓：被剔除的记 0，活下来的仍是原来的一半。国内合计 2% 变成 1%，海外合计 10% 变成 5%。创业板本来没有底仓。
- 溢价倾斜仍在两只 QDII 之间搬。搬到被剔除那只上的权重随后被打成 0。
- 去掉 511260：`cash_bond_pct=0`，闲置现金按 `repo_rate` 0.022。权益目标不动。不是把国债仓位加到股票上。引擎里的现金本来就不是第六个目标权重，而是闲置资金的收益。

模式 R，预算留在原袖子里，仍只用现有字段：

- 同样 `exclude`。
- 防御只剩 1 只：`defense_momentum=false`（没有第二只可轮动），`floor_pct.cn=4`。引擎先把国内底仓对半，再把被剔除的清零，活下来的那只仍是 2%。
- 防御一只不剩：`defense_momentum=false`，`floor_pct.cn=0`，防御弹性进现金。
- 美股只剩 1 只：`premium_tilt=false`，`floor_pct.us=20`，活下来的底仓仍是 10%。
- 美股一只不剩：`premium_tilt=false`，`floor_pct.us=0`，海外弹性进现金。
- 只去掉创业板：底仓与防御动量、溢价倾斜都不动，只把成长拆分归一到剩下的两只。

N 与 R 互为平台的两侧，不按谁的 Calmar 高来选模式。

### 1.2 换槽与现金袖

候选在跑策略之前按 v40 普查的近端成交额锁定。近端窗口是 2025-06-01 起、最多 120 个成交额大于 0 的交易日。现任六只不作为新候选。每一类取成交额最高的，有第二只同名或同指数产品且成交额不低于 2000 万的，第二只一起锁定。不按策略收益补票，也不按 v40 的 Calmar 把 513300 排到 513100 前面。513300 在名单里是因为它是纳指 100 成交额第二，同时是这次要在新结构里重测的代码。

锁里的金额是普查数字，单位元，只为了事后核对没有换人。

槽位按经济角色事先定死，不试「换到哪个槽收益更高」：

- 纳指 100、纳指科技、标普：整段替换对应的 QDII 槽
- 黄金：整段替换红利槽 515100
- 原油股票（标普油气）与能源化工：整段替换自由现金流槽 159232
- 可转债、恒生科技、中概、科创 50、半导体、医药：整段替换创业板槽 159952
- 银行：整段替换自由现金流槽 159232
- 货币、短融、国开、不同久期国债：替换现金收益，不替换股票槽，也不把上市前的收益拼到 511260 上

「沉开」不是交易所通名。本次用两个近邻盖住这个词：国开债进现金袖，银行进自由现金流槽。两个都不是点名代码。即使数字过关，也不建议因为这个读法升默认。

全换才是主比较。不做半仓并存。能同时换两只的，事先只留一对：黄金与标普油气一起替换两个防御槽（成交额第一的一对，以及各自第二名的一对）。这是一对对照，不是搜索。

另外一条纸面路径：闲置现金的收益全部改记黄金（`run_backtest` 的 `gold_override_fn` 恒为 1）。权益目标不动。第一天的现金收益仍是原现金层，因为引擎在收盘后才改黄金占比。`scripts/run_backtest.py` 不读这个开关。这条不能写进可执行草案，也不能建议升默认。它不是第七个权重槽。生产引擎的 `SLOTS` 只有五列，加一只股票进目标权重做不到，本次不编造这种配置。

上市前的股票槽用 v39 该槽的损益拼接。内部缺口最长不超过 5 个交易日、占比不超过 2%，超过则那些日子的处理与 `swaps._gap_fill` 相同的判定：不合格就不能过关。现金不拼接。盖不住 2014-06-23 的，全窗口里上市前会被引擎填成逆回购，这段数字只作诊断，不能过 (a)。

生产口径是前复权，与 `scripts/swaps.py` 一致。唯一例外是 511010：生产文件就是 `assets/data/511010_nav.csv` 的累计净值，这条臂以净值为生产口径。另一口径（净值或前复权）必须 proxy ΔCal 与 real ΔCal 都大于 0，否则算口径冲突，不能过。缺另一口径也不能过。溢价 = 不复权收盘 / 单位净值 - 1，绝对值大于 50% 的日子丢掉。溢价不叠进收益。

### 1.3 新结构

事先指定，不看减槽结果再挑：

- 去掉 159952，模式 N 与模式 R，再把 159941 整段换成 ndx100 组或 ndx_tech 组里的代码。两个口径都跑。
- 这是「换掉创业板后再加入 513300 / 159509」。不去掉创业板的整段替换是全换对照。

### 1.4 三关

ΔCal 相对同一次 v39。

(a) 生产口径的 proxy 与 real 的 ΔCalmar 都大于 0，至少一个达到 +0.10；proxy MDD 严格好于 -10%（`mdd > -0.10`）。real 必须从 2025-04-23 起。proxy 必须从 2014-06-23 可评。现金因此要自身覆盖这段。技能文档的周度线是 +0.15，另记，不替代 +0.10。

(b) 平台。

- 减槽：另一模式的 proxy ΔCal 与 real ΔCal 都大于 0，proxy Calmar 绝对差不超过 0.15，MDD 都严格好于 -10%。还剩至少两只成长名字时，牛市拆分把权重最大的一只挪 5 个百分点给第二只，两个方向都要满足同样的同号和 0.15。只剩一只成长时不再另加旋钮。只去掉现金时，两侧是 `repo_rate` 0.018 与 0.026。
- 换券：同一组里至少两只，相关规则的组还要求两只前复权日收益相关系数不低于 0.90（重叠至少 60 日）。两只的生产口径 proxy ΔCal 与 real ΔCal 都大于 0，而且都是全窗口。只有一只的组不能过 (b)。相关系数不够的，不算同一工具。
- 久期组不是同一工具。相关规则不适用。(b) 改为：至少还有一只久期臂的 proxy ΔCal 与 real ΔCal 都大于 0。即使如此，也不从这三只里挑 Calmar 更高的升默认。
- 去掉创业板再换纳指：同一模式下 ndx100 的两只都要同号为正。ndx_tech 只有 159509，不能过 (b)。模式 N 与 R 还要满足减槽那条平台。

(c) OOS Calmar 不低于同一次 v39。容差 1e-9。缩短的 OOS 不能过。

识别：自有样本不少于 252 个交易日，且这段 proxy ΔCal 与全样本 proxy ΔCal 同号。2025-04-22 之前的 proxy ΔCal 不得低于 -0.02。不满足则不晋升。缺口不合格、口径冲突、现金盖不住起点，都不晋升。自身收益起点晚于 2022-01-04 的，即使数字过了，也不把它写成 2014 年以来的长期增益，不建议升默认。

末日溢价绝对值大于 8% 的，不建议升默认。溢价中位数照实写。

换手只陈列。

### 1.5 建议升默认

三关加上识别、口径、缺口都过，才叫过关。建议升默认还要同时满足：至少一窗 ΔCal 达到 +0.15、生产入口能执行、末日溢价不超过 8%、自身样本不是 2022 年以后才开始、不是久期菜单、不是「沉开」的两个近邻读法。否则只归档。

生产入口能执行的意思是：`exclude`、`floor_pct`、`defense_momentum`、`premium_tilt`、`cash_bond_pct`、`cash_bond_code`、`instrument_swaps` 这些现有字段。`instrument_swaps` 还要求 `assets/data/{代码}_qfq.csv` 能被 `swaps.py` 读到。现金代码要求 `assets/data/{代码}_nav.csv` 里有 `cum_nav`。黄金现金那条纸面路径不在此列。

过关且建议升默认的，若恰好一条，才写 `references/final_cfg_v41.json`，标明草案、未接入默认。多条过关也不挑 Calmar 最高的写成一份默认草案。没有任何臂过关，就不写这份文件。

事先不做的事：不改 v39 的阈值来迁就结果，不重开日频，不按收益补票，不改 README / SKILL / 引擎，不做 50/50。

### 1.6 未来函数抽检

主表跑完之后只做三条路径：模式 R 去掉 159952、黄金整段替换 515100、去掉 159952 且模式 R 再换 513300。每条做两件事：截掉最后 20 个交易日重跑，截断日之前的决策权重与全样本该日一致；最后一日损益加 5 个百分点，全部决策日的目标权重不变。去掉 159952 再加一条「当日信号」对照，只报告 Calmar 差，不把它当候选。

### 1.7 锁定名单
""".strip()


def pre_text():
    body = json.dumps(LOCK, ensure_ascii=False, indent=1, sort_keys=True)
    return PRE_HEAD + "\n\n<<<LOCK\n" + body + "\nLOCK>>>\n"


def fingerprint():
    return hashlib.sha256(pre_text().encode()).hexdigest()[:16]


def traded_of(slot):
    return LOCK["traded"].get(slot, slot)


def drop_name(dropped):
    return "drop:" + "+".join(traded_of(s) for s in dropped)


def groups_by_id():
    return {g["id"]: g for g in LOCK["groups"]}


def cfg_drop(base, dropped, mode):
    c = copy.deepcopy(base)
    c.pop("_note", None)
    eq = [s for s in dropped if s != "511260"]
    c["exclude"] = list(eq)
    c["version"] = "v41-paper"
    if "511260" in dropped:
        c["cash_bond_pct"] = 0.0
    if mode == "R":
        fp = dict(c["floor_pct"])
        defs = [s for s in ("159232", "515100") if s not in eq]
        uss = [s for s in ("159941", "513500") if s not in eq]
        if len(defs) == 1:
            c["defense_momentum"] = False
            fp["cn"] = 4.0
        elif len(defs) == 0:
            c["defense_momentum"] = False
            fp["cn"] = 0.0
        if len(uss) == 1:
            c["premium_tilt"] = False
            fp["us"] = 20.0
        elif len(uss) == 0:
            c["premium_tilt"] = False
            fp["us"] = 0.0
        c["floor_pct"] = fp
    elif mode != "N":
        raise ValueError(mode)
    if "allow_live" in c:
        raise RuntimeError("allow_live")
    return c


def assert_renorm(R):
    base = json.load(open(CFG_V39))
    ds = DynamicStrategy(R, cfg=cfg_drop(base, ["159952"], "R"))
    bull = ds.growth_split_bull
    assert "159952" not in bull, bull
    assert abs(sum(bull.values()) - 1) < 1e-9, bull
    assert abs(bull["159941"] - 0.63 / 0.70) < 1e-9, bull
    assert ds.defense_momentum is True
    assert abs(ds.floor["159232"] - 1.0) < 1e-9
    ds = DynamicStrategy(R, cfg=cfg_drop(base, ["159232"], "R"))
    assert ds.defense_momentum is False
    assert ds.floor["159232"] == 0.0
    assert abs(ds.floor["515100"] - 2.0) < 1e-9, ds.floor
    ds = DynamicStrategy(R, cfg=cfg_drop(base, ["159232"], "N"))
    assert ds.defense_momentum is True
    assert abs(ds.floor["515100"] - 1.0) < 1e-9, ds.floor
    ds = DynamicStrategy(R, cfg=cfg_drop(base, ["159941"], "R"))
    assert ds.premium_tilt is False
    assert abs(ds.floor["513500"] - 10.0) < 1e-9, ds.floor
    assert ds.floor["159941"] == 0.0
    ds = DynamicStrategy(R, cfg=cfg_drop(base, ["159941", "513500"], "R"))
    assert abs(ds.floor["159941"]) < 1e-12 and abs(ds.floor["513500"]) < 1e-12
    assert "159952" in ds.growth_split_bull
    assert abs(ds.growth_split_bull["159952"] - 1) < 1e-9
    ds = DynamicStrategy(R, cfg=cfg_drop(base, ["159232", "515100"], "R"))
    assert ds.floor["159232"] == 0 and ds.floor["515100"] == 0
    c = cfg_drop(base, ["511260"], "N")
    assert c["cash_bond_pct"] == 0.0
    assert "allow_live" not in c


def git_show(rev, src):
    try:
        return subprocess.check_output(["git", "show", f"{rev}:{src}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def ensure_file(code, kind):
    """kind: qfq, raw, nav。返回是否拿到文件。"""
    os.makedirs(os.path.join(UNI, "px"), exist_ok=True)
    os.makedirs(os.path.join(UNI, "nav"), exist_ok=True)
    if kind == "nav":
        dst = os.path.join(UNI, "nav", f"{code}.csv")
    else:
        dst = os.path.join(UNI, "px", f"{code}_{kind}.csv")
    if os.path.exists(dst) and os.path.getsize(dst) > 50:
        return True
    local_nav = os.path.join(DATA_DIR, f"{code}_nav.csv")
    local_qfq = os.path.join(DATA_DIR, f"{code}_qfq.csv")
    if kind == "nav" and os.path.exists(local_nav):
        open(dst, "wb").write(open(local_nav, "rb").read())
        return True
    if kind == "qfq" and os.path.exists(local_qfq):
        open(dst, "wb").write(open(local_qfq, "rb").read())
        return True
    for folder, name in (
        ("v39_universe/px", f"{code}_{kind}.csv"),
        ("v39_universe/nav", f"{code}.csv" if kind == "nav" else None),
    ):
        if name is None:
            continue
        src = os.path.join(DATA_DIR, folder, name) if False else os.path.join(ROOT, "assets", "data", folder, name)
        if os.path.exists(src):
            open(dst, "wb").write(open(src, "rb").read())
            return True
    rels = []
    if kind == "nav":
        rels = [f"assets/data/v40_universe/nav/{code}.csv", f"assets/data/v39_universe/nav/{code}.csv"]
    else:
        rels = [
            f"assets/data/v40_universe/px/{code}_{kind}.csv",
            f"assets/data/v39_universe/px/{code}_{kind}.csv",
            f"assets/data/{code}_{kind}.csv",
        ]
    for rel in rels:
        blob = git_show(V40, rel)
        if blob and len(blob) > 50:
            open(dst, "wb").write(blob)
            return True
    try:
        if kind == "nav":
            df = collect_src.nav_history(code)
            if df is None or df.empty:
                return False
            df.to_csv(dst)
            return True
        fqt = 1 if kind == "qfq" else 0
        df = collect_src.kline(code, fqt)
        if df is None or df.empty:
            return False
        df.to_csv(dst)
        return True
    except Exception as e:
        print(f"  [fetch fail] {code} {kind}: {e}")
        return False


def all_codes():
    codes = []
    for g in LOCK["groups"]:
        for c in g["codes"]:
            codes.append(c["code"])
    # 黄金纸面路径用同一组
    return list(dict.fromkeys(codes))


def collect():
    os.makedirs(UNI, exist_ok=True)
    meta_path = os.path.join(UNI, "details.json")
    details = {}
    if os.path.exists(meta_path):
        details = json.load(open(meta_path))
    for code in all_codes():
        print(f"[data] {code}", flush=True)
        for kind in ("qfq", "raw", "nav"):
            ok = ensure_file(code, kind)
            print(f"  {kind}: {'ok' if ok else 'missing'}", flush=True)
        if code not in details:
            try:
                details[code] = collect_src.detail(code)
            except Exception as e:
                details[code] = {"code": code, "error": str(e)}
            json.dump(details, open(meta_path, "w"), ensure_ascii=False, indent=1)
            time.sleep(0.2)
    return details


def read_px(code, kind):
    path = os.path.join(UNI, "px", f"{code}_{kind}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        return None
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    col = "close" if "close" in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s


def read_nav(code):
    path = os.path.join(UNI, "nav", f"{code}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        return None
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def series_for(code, caliber):
    if caliber == "qfq":
        s = read_px(code, "qfq")
        return None if s is None else s
    df = read_nav(code)
    if df is None or "cum_nav" not in df.columns:
        return None
    return pd.to_numeric(df["cum_nav"], errors="coerce").dropna()


def rets_of(s):
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()


def gap_fill(base, own, index):
    own = own.dropna().sort_index()
    if own.empty:
        return None, {"gap_ok": False, "reason": "empty", "spliced": True, "first": None, "n_own": 0}
    first = own.index.min()
    out = base.reindex(index).astype(float).copy()
    after = index[index >= first]
    got = own.reindex(after)
    miss = got.isna()
    longest = run = 0
    for v in miss.tolist():
        run = run + 1 if v else 0
        longest = max(longest, run)
    frac = float(miss.sum()) / max(len(after), 1)
    ok = longest <= 5 and frac <= 0.02
    out.loc[after] = got.fillna(0.0).to_numpy()
    info = {
        "first": str(pd.Timestamp(first).date()),
        "n_own": int(len(after)),
        "longest_gap": int(longest),
        "gap_frac": frac,
        "gap_ok": bool(ok),
        "spliced": bool(pd.Timestamp(first) > pd.Timestamp(index.min()) + pd.Timedelta(days=10)),
    }
    return out, info


def metrics_of(res):
    e = evaluate(res)
    w = res["weights"]
    means = {c: float(w[c].mean()) for c in w.columns}
    last = {c: float(w.iloc[-1][c]) for c in w.columns}
    tail = w.iloc[-60:] if len(w) >= 60 else w
    return {
        "cagr": float(e["cagr"]), "mdd": float(e["max_dd"]), "cal": float(e["calmar"]),
        "to": float(e["turnover"]), "n": int(len(res["rets"])),
        "end": str(res["rets"].index.max().date()),
        "mean_w": means, "last_w": last,
        "tail60_159952": float(tail["159952"].mean()) if "159952" in tail else None,
    }


def run_dyn(R_sig, R_pnl, cfg, start, bond, end=None, gold=None):
    ds = DynamicStrategy(R_sig, cfg=cfg)
    kw = {}
    if gold is not None:
        kw["cash_gold_rets"] = gold
        kw["gold_override_fn"] = lambda dt: 1.0
    res = run_backtest(
        R_pnl, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
        start=start, end=end, name="DYN",
        min_delta=cfg.get("min_delta", 0.035), repo=cfg.get("repo_rate", 0.022),
        tranche_weights=cfg.get("tranche_weights"),
        cash_bond_rets=bond, cash_bond_pct=float(cfg.get("cash_bond_pct") or 0.0),
        rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
        strict=True, **kw)
    return metrics_of(res), res, ds


def bond_from_cfg(cfg, override=None):
    if override is not None:
        return override
    if not cfg.get("cash_bond_pct"):
        return None
    code = cfg.get("cash_bond_code", "511010")
    # 511260 / 511010 用仓库里的生产文件，不改用新下载的副本
    path_code = code
    df = read_table(f"{path_code}_nav.csv")
    col = "cum_nav" if "cum_nav" in df.columns else df.columns[-1]
    return rets_from(df, col)


def windows(R, Rr, Rp, Rr_p, cfg, bond, gold=None):
    out = {}
    specs = (
        ("proxy", R, Rp, PROXY_START, None),
        ("real", Rr, Rr_p, REAL_START, None),
        ("oos", R, Rp, OOS_START, None),
        ("pre", R, Rp, PROXY_START, PRE_END),
    )
    res_proxy = None
    ds_proxy = None
    for name, sig, pnl, start, end in specs:
        m, res, ds = run_dyn(sig, pnl, cfg, start, bond, end=end, gold=gold)
        out[name] = m
        if name == "proxy":
            res_proxy, ds_proxy = res, ds
    return out, res_proxy, ds_proxy


def dcal(m, base):
    if m is None or base is None:
        return None
    if m.get("cal") != m.get("cal") or base.get("cal") != base.get("cal"):
        return None
    return float(m["cal"] - base["cal"])


def prem_of(code):
    raw = read_px(code, "raw")
    nav = read_nav(code)
    if raw is None or nav is None or "unit_nav" not in nav.columns:
        return None
    u = pd.to_numeric(nav["unit_nav"], errors="coerce")
    both = pd.DataFrame({"r": raw, "u": u}).dropna()
    if len(both) < 20:
        return None
    p = both["r"] / both["u"] - 1.0
    p = p[p.abs() <= 0.5]
    if p.empty:
        return None
    return {"median": float(p.median()), "last": float(p.iloc[-1]), "n": int(len(p)),
            "end": str(p.index.max().date())}


def pair_corr(code_a, code_b):
    a = series_for(code_a, "qfq")
    b = series_for(code_b, "qfq")
    if a is None or b is None:
        return None
    x = pd.concat([rets_of(a), rets_of(b)], axis=1, join="inner").dropna()
    if len(x) < 60:
        return None
    return float(x.iloc[:, 0].corr(x.iloc[:, 1]))


def apply_legs(base_pnl, legs, caliber):
    """legs: [(slot, code), ...] 替换损益。返回面板与信息；缺数据则 None。"""
    out = base_pnl.copy()
    infos = []
    for slot, code in legs:
        px = series_for(code, caliber)
        if px is None:
            return None, [{"code": code, "caliber": caliber, "missing": True, "gap_ok": False}]
        own = rets_of(px)
        col, info = gap_fill(out[slot], own, out.index)
        if col is None:
            return None, [info]
        info["code"] = code
        info["slot"] = slot
        info["caliber"] = caliber
        out[slot] = col
        infos.append(info)
    return out, infos


def cash_returns(code, caliber):
    s = series_for(code, caliber)
    if s is None:
        return None, {"missing": True, "gap_ok": False, "code": code}
    r = rets_of(s)
    first = r.index.min()
    info = {
        "code": code, "caliber": caliber, "first": str(first.date()),
        "n_own": int(len(r)), "gap_ok": True, "missing": False,
        "spliced": bool(first > pd.Timestamp(PROXY_START) + pd.Timedelta(days=10)),
        "covers_proxy": bool(first <= pd.Timestamp(PROXY_START)),
    }
    return r, info


class LeakySignal(SignalSet):
    def _idx(self, dt):
        i = self.R.index.get_indexer([pd.Timestamp(dt)], method="ffill")[0]
        return max(0, int(i))


class LeakyStrategy(DynamicStrategy):
    SIG_CLASS = LeakySignal


def pct(x):
    if x is None or x != x:
        return "nan"
    return f"{x * 100:.2f}%"


def calf(x):
    if x is None or x != x:
        return "nan"
    return f"{x:.2f}"


def dlt(x):
    if x is None or x != x:
        return "nan"
    return f"{x:+.3f}"


def cell(m):
    if not m:
        return "—"
    return f"{pct(m['cagr'])} / {pct(m['mdd'])} / {calf(m['cal'])}"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    fp = fingerprint()
    print(f"FINGERPRINT {fp}", flush=True)
    if "--fingerprint-only" in sys.argv:
        open(os.path.join(UNI, "fingerprint.txt"), "w").write(fp + "\n") if False else None
        print(pre_text()[:400])
        return
    smoke = "--smoke" in sys.argv
    os.makedirs(UNI, exist_ok=True)
    open(os.path.join(UNI, "fingerprint.txt"), "w").write(fp + "\n")
    sha_before = {
        "v39": file_sha(CFG_V39),
        "engine": file_sha(os.path.join(ROOT, "scripts", "engine.py")),
        "strategy": file_sha(os.path.join(ROOT, "scripts", "strategy.py")),
    }
    base = json.load(open(CFG_V39))
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    assert_renorm(R)
    print("[renorm] floor / split asserts ok", flush=True)
    details = collect()
    v39 = json.load(open(CFG_V39))
    Rp = pnl_panel(R, "proxy", v39)
    Rr_p = pnl_panel(Rr, "real", v39)
    bond = bond_from_cfg(v39)
    print("[anchor] running v39", flush=True)
    base_w, _, _ = windows(R, Rr, Rp, Rr_p, v39, bond)
    anch = LOCK["anchor"]
    for key, win in (("proxy", "proxy"), ("real", "real")):
        for fld in ("cagr", "mdd", "cal"):
            got = base_w[win][fld if fld != "cal" else "cal"]
            exp = anch[key]["cal" if fld == "cal" else fld]
            if abs(got - exp) > 1e-6:
                raise SystemExit(f"锚对不上 {win} {fld}: {got} != {exp}")
    if abs(base_w["oos"]["cal"] - anch["oos_cal"]) > 1e-6:
        raise SystemExit(f"OOS 锚对不上 {base_w['oos']['cal']}")
    print("[anchor] ok", base_w["proxy"]["cal"], flush=True)

    rows = []

    def add_row(row):
        rows.append(row)
        print(f"  {row['name']} a={row.get('a')} dP={dlt(row.get('d_proxy'))} dR={dlt(row.get('d_real'))}", flush=True)

    # 对照
    for label, path in (("v37", CFG_V37), ("v30", CFG_V30)):
        cfg = json.load(open(path))
        b = bond_from_cfg(cfg)
        # v37/v30 没有 instrument_swaps，损益用原面板
        w, _, _ = windows(R, Rr, R, Rr, cfg, b)
        rows.append(control_row(label, w, base_w))
        print(f"  {label} cal {w['proxy']['cal']:.3f}", flush=True)

    drops = []
    for k in (1, 2):
        for comb in combinations(LOCK["slots"], k):
            drops.append(list(comb))
    if smoke:
        drops = drops[:1]
    for dropped in drops:
        for mode in LOCK["modes"]:
            cfg = cfg_drop(v39, dropped, mode)
            b = bond_from_cfg(cfg)
            w, res, _ = windows(R, Rr, Rp, Rr_p, cfg, b)
            if dropped == ["159952"] and mode == "N":
                if w["proxy"]["mean_w"]["159952"] > 1e-6:
                    raise SystemExit("去掉 159952 后权重不是 0")
            add_row(score_drop(drop_name(dropped), mode, dropped, w, base_w, cfg))

    # 换槽
    gmap = groups_by_id()
    group_list = LOCK["groups"][:2] if smoke else LOCK["groups"]
    swap_rows_index = {}  # (group, code, mode-or-base) -> row
    for g in group_list:
        for spec in g["codes"]:
            code = spec["code"]
            if g["slot"] == "cash":
                row = run_cash_arm(g, spec, R, Rr, Rp, Rr_p, v39, base_w, details)
            else:
                row = run_equity_arm(
                    f"{g['id']}:{code}", g, [(g["slot"], code)], R, Rr, Rp, Rr_p, v39, base_w, details)
            add_row(row)
            swap_rows_index[(g["id"], code)] = row

    if not smoke:
        for gname in LOCK["combos"]["groups"]:
            g = gmap[gname]
            for mode in LOCK["combos"]["modes"]:
                cfg0 = cfg_drop(v39, LOCK["combos"]["drop"], mode)
                for spec in g["codes"]:
                    code = spec["code"]
                    row = run_equity_arm(
                        f"combo:{mode}:drop159952+{g['id']}:{code}", g,
                        [(g["slot"], code)], R, Rr, Rp, Rr_p, cfg0, base_w, details,
                        dropped=LOCK["combos"]["drop"], mode=mode)
                    add_row(row)
                    swap_rows_index[("combo", mode, g["id"], code)] = row
        for d in LOCK["doubles"]:
            legs = []
            for leg in d["legs"]:
                g = gmap[leg["group"]]
                legs.append((leg["slot"], g["codes"][leg["i"]]["code"]))
            # 用黄金组的 pair 规则做标记，组 id 写成 double
            fake = {"id": d["name"], "pair": "double", "kind": "equity", "slot": "multi"}
            row = run_equity_arm(
                "double:" + d["name"] + ":" + "+".join(c for _, c in legs),
                fake, legs, R, Rr, Rp, Rr_p, v39, base_w, details)
            add_row(row)
        # 纸面：闲置现金改记黄金
        for spec in gmap["gold"]["codes"]:
            code = spec["code"]
            px = series_for(code, "qfq")
            if px is None:
                continue
            gold_r = rets_of(px)
            w, _, _ = windows(R, Rr, Rp, Rr_p, v39, bond, gold=gold_r)
            nav_s = series_for(code, "nav")
            nav_w = None
            if nav_s is not None:
                nav_w, _, _ = windows(R, Rr, Rp, Rr_p, v39, bond, gold=rets_of(nav_s))
            row = score_caliber_pair(
                f"paper:cash_yield_gold:{code}", "paper_gold", code, w, nav_w, base_w,
                {"covers_proxy": True, "gap_ok": True, "spliced": False, "first": str(gold_r.index.min().date()),
                 "n_own": int((Rp.index >= gold_r.index.min()).sum())},
                prod="qfq", executable=False, ambiguous=False, tenor=False)
            row["note"] = "闲置现金全部改记黄金。run_backtest.py 不读这个开关。不是第七个权重。"
            add_row(row)

    # 平台：减槽用已经跑完的另一模式；成长拆分扰动只对 (a)+识别 已过的
    attach_drop_plateau(rows, R, Rr, Rp, Rr_p, v39, base_w)
    attach_pair_gates(rows)

    for row in rows:
        finalize(row)

    # 未来函数
    audits = [] if smoke else lookahead_spot(R, Rr, Rp, Rr_p, v39, bond)

    out_json = {
        "fingerprint": fp,
        "anchor_ok": True,
        "v39": {k: base_w[k] for k in ("proxy", "real", "oos", "pre")},
        "rows": rows,
        "lookahead": audits,
        "details_index": {c: {"index_name": (details.get(c) or {}).get("index_name"),
                              "short": (details.get(c) or {}).get("short"),
                              "estab": (details.get(c) or {}).get("estab")}
                          for c in all_codes()},
    }
    # 去掉不可序列化
    json.dump(out_json, open(os.path.join(REF, "exp_v41_recombine_results.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    report = render_report(fp, base_w, rows, audits, details)
    open(os.path.join(REF, "exp_v41_recombine_prereg_report.md"), "w").write(report)
    write_drafts(rows)
    sha_after = {
        "v39": file_sha(CFG_V39),
        "engine": file_sha(os.path.join(ROOT, "scripts", "engine.py")),
        "strategy": file_sha(os.path.join(ROOT, "scripts", "strategy.py")),
    }
    if sha_before != sha_after:
        raise SystemExit(f"生产文件被改动 {sha_before} -> {sha_after}")
    n_pass = sum(1 for r in rows if r.get("pass3"))
    n_sug = sum(1 for r in rows if r.get("suggest"))
    print(f"[done] rows={len(rows)} pass3={n_pass} suggest={n_sug} fp={fp}", flush=True)


def control_row(name, w, base_w):
    row = {
        "name": name, "kind": "control", "executable": False,
        "proxy": w["proxy"], "real": w["real"], "oos": w["oos"], "pre": w["pre"],
        "d_proxy": dcal(w["proxy"], base_w["proxy"]),
        "d_real": dcal(w["real"], base_w["real"]),
        "d_pre": dcal(w["pre"], base_w["pre"]),
        "a": False, "b": False, "c": False, "ident": False, "pass3": False, "suggest": False,
        "why": "对照",
    }
    return row


def score_drop(name, mode, dropped, w, base_w, cfg):
    d_p = dcal(w["proxy"], base_w["proxy"])
    d_r = dcal(w["real"], base_w["real"])
    d_pre = dcal(w["pre"], base_w["pre"])
    mdd_ok = w["proxy"]["mdd"] > -0.10
    a = d_p is not None and d_r is not None and d_p > 0 and d_r > 0 and max(d_p, d_r) >= 0.10 and mdd_ok
    ident = d_pre is not None and d_pre > -0.02 and (d_p is None or d_p == 0 or (d_pre > 0) == (d_p > 0) or abs(d_p) < 1e-12)
    # 减槽没有新券。识别用「2025-04-22 之前不更差」以及全样本同号。
    if d_p is not None and d_p > 0 and d_pre is not None and d_pre < -0.02:
        ident = False
    if d_p is not None and d_pre is not None and d_p * d_pre < 0 and abs(d_p) > 0.02 and abs(d_pre) > 0.02:
        ident = False
    c = w["oos"]["cal"] + 1e-9 >= LOCK["anchor"]["oos_cal"]
    return {
        "name": f"{name}:{mode}", "kind": "drop", "mode": mode, "dropped": dropped,
        "executable": True, "prod": "weights",
        "proxy": slim(w["proxy"]), "real": slim(w["real"]), "oos": slim(w["oos"]), "pre": slim(w["pre"]),
        "d_proxy": d_p, "d_real": d_r, "d_pre": d_pre,
        "mdd_ok": mdd_ok, "a": bool(a), "c": bool(c), "ident": bool(ident),
        "gap_ok": True, "caliber_ok": True, "covers": True, "spliced": False,
        "prem_ok": True, "short": False, "ambiguous": False, "tenor": False,
        "b": None,  # 稍后
        "mean_w": w["proxy"]["mean_w"], "last_w": w["proxy"]["last_w"],
        "cfg_keys": cfg_public(cfg),
    }


def slim(m):
    if m is None:
        return None
    return {"cagr": m["cagr"], "mdd": m["mdd"], "cal": m["cal"], "to": m["to"], "n": m["n"], "end": m["end"]}


def cfg_public(cfg):
    keys = ("exclude", "floor_pct", "defense_momentum", "premium_tilt", "cash_bond_pct",
            "cash_bond_code", "repo_rate", "instrument_swaps")
    return {k: cfg.get(k) for k in keys if k in cfg}


def run_equity_arm(name, group, legs, R, Rr, Rp, Rr_p, cfg, base_w, details, dropped=None, mode=None):
    prod = "qfq"
    q_proxy, qinfo = apply_legs(Rp, legs, "qfq")
    q_real, _ = apply_legs(Rr_p, legs, "qfq")
    n_proxy, ninfo = apply_legs(Rp, legs, "nav")
    n_real, _ = apply_legs(Rr_p, legs, "nav")
    if q_proxy is None or q_real is None:
        return missing_row(name, "equity", legs)
    # real 起点
    firsts = [pd.Timestamp(i["first"]) for i in qinfo if i.get("first")]
    real_start = REAL_START
    real_full = True
    if firsts and max(firsts) > pd.Timestamp(REAL_START):
        real_start = str(max(firsts).date())
        real_full = False
    bond = bond_from_cfg(cfg)
    w, _, _ = windows(R, Rr, q_proxy, q_real, cfg, bond)
    if not real_full:
        # 缩短 real：双方从同一天重跑。这里臂已经含拼接日前的原券，改用 real_start 重跑臂与基线。
        m_arm, _, _ = run_dyn(Rr, q_real, cfg, real_start, bond)
        m_base, _, _ = run_dyn(Rr, Rr_p, cfg if dropped is None else json.load(open(CFG_V39)), real_start, bond_from_cfg(json.load(open(CFG_V39))))
        # 基线必须是 v39，不是已经减过槽的 cfg。减槽组合的对照仍是 v39 同一起点。
        m_base, _, _ = run_dyn(Rr, Rr_p, json.load(open(CFG_V39)), real_start, bond_from_cfg(json.load(open(CFG_V39))))
        w["real"] = m_arm
        w["real_base_cal"] = m_base["cal"]
        w["real_short"] = True
    else:
        w["real_short"] = False
    nav_w = None
    if n_proxy is not None and n_real is not None:
        nav_w, _, _ = windows(R, Rr, n_proxy, n_real, cfg, bond)
        if not real_full:
            m_arm, _, _ = run_dyn(Rr, n_real, cfg, real_start, bond)
            nav_w["real"] = m_arm
    info = qinfo[0]
    info["covers_proxy"] = all(not i.get("spliced") for i in qinfo) or True
    # 股票槽允许拼接，covers_proxy 对 (a) 不是硬拒绝。现金才是。
    info["gap_ok"] = all(i.get("gap_ok") for i in qinfo)
    info["spliced"] = any(i.get("spliced") for i in qinfo)
    info["first"] = str(max(firsts).date()) if firsts else None
    info["n_own"] = min(i.get("n_own") or 0 for i in qinfo)
    codes = [c for _, c in legs]
    prem = prem_of(codes[0]) if len(codes) == 1 else None
    ambiguous = group["id"] in ("bank", "guokai") or name.startswith("double") and False
    if any(c in ("512800", "516310", "159649", "159651") for c in codes):
        ambiguous = True
    row = score_caliber_pair(
        name, "equity" if not name.startswith("combo") and not name.startswith("double") else name.split(":")[0],
        "+".join(codes), w, nav_w, base_w, info, prod="qfq",
        executable=True, ambiguous=ambiguous, tenor=False, prem=prem, real_full=real_full,
        group=group["id"], codes=codes, dropped=dropped, mode=mode, cfg=cfg)
    row["legs"] = [[slot, code] for slot, code in legs]
    # 自有样本
    if info.get("spliced") and info.get("first"):
        own = own_sample(R, Rp, q_proxy, cfg, info["first"], base_w)
        row["own"] = own
        row["ident"] = bool(row["ident"] and own and own["n"] >= 252 and own["dcal"] is not None
                            and row["d_proxy"] is not None and own["dcal"] * row["d_proxy"] > 0)
        if info["first"] > "2022-01-04":
            row["short"] = True
    return row


def own_sample(R, base_pnl, alt_pnl, cfg, first, base_w):
    v39 = json.load(open(CFG_V39))
    bond = bond_from_cfg(v39)
    # 臂的 cfg 可能含 exclude。自有样本的对照是同一 cfg 的结构、但损益用 v39（不换这只新券）。
    # 对纯换券，cfg 就是 v39。对组合，对照应是「同样减槽、但不换新券」，这样 Δ 才是换券的增量。
    m_alt, _, _ = run_dyn(R, alt_pnl, cfg, first, bond_from_cfg(cfg))
    m_base, _, _ = run_dyn(R, base_pnl, cfg, first, bond_from_cfg(cfg))
    return {"n": m_alt["n"], "cal": m_alt["cal"], "base_cal": m_base["cal"],
            "dcal": dcal(m_alt, m_base), "cagr": m_alt["cagr"], "mdd": m_alt["mdd"],
            "start": first}


def run_cash_arm(group, spec, R, Rr, Rp, Rr_p, v39, base_w, details):
    code = spec["code"]
    prod = spec.get("prod") or "qfq"
    other = "nav" if prod == "qfq" else "qfq"
    series = {}
    infos = {}
    for cal in (prod, other):
        r, info = cash_returns(code, cal)
        series[cal] = r
        infos[cal] = info
    if series[prod] is None:
        return missing_row(f"{group['id']}:{code}", "cash", [("cash", code)])
    cfg = copy.deepcopy(v39)
    cfg["cash_bond_pct"] = 1.0
    cfg["cash_bond_code"] = code
    cfg["version"] = "v41-paper"
    w, _, _ = windows(R, Rr, Rp, Rr_p, cfg, series[prod])
    nav_w = None
    if series[other] is not None:
        nav_w, _, _ = windows(R, Rr, Rp, Rr_p, cfg, series[other])
    info = infos[prod]
    # 现金：拼接不允许。covers_proxy 为假则 (a) 失败，但数字仍记。
    prem = prem_of(code)
    ambiguous = group["id"] == "guokai"
    tenor = group["id"] == "tenor"
    # 511010 的生产口径走仓库 nav。cash_returns 读的是 v41 副本，应与仓库文件同源。
    row = score_caliber_pair(
        f"{group['id']}:{code}", "cash", code, w, nav_w, base_w, info, prod=prod,
        executable=executable_cash(code, prod), ambiguous=ambiguous, tenor=tenor,
        prem=prem, real_full=True, group=group["id"], codes=[code], cfg=cfg)
    if not info.get("covers_proxy"):
        row["covers"] = False
        row["a"] = False
        row["why_cover"] = "现金自身盖不住 2014-06-23，全窗口含逆回购填充，只作诊断"
    else:
        row["covers"] = True
    if info.get("spliced"):
        own = own_sample_cash(R, Rr, Rp, Rr_p, cfg, series[prod], info["first"])
        row["own"] = own
        if own and row.get("d_proxy") is not None and own.get("dcal") is not None:
            if own["n"] < 252 or own["dcal"] * row["d_proxy"] <= 0:
                row["ident"] = False
        if info["first"] > "2022-01-04":
            row["short"] = True
    # 指数名
    det = details.get(code) or {}
    row["index_name"] = det.get("index_name") or ""
    if tenor and prod == "nav":
        pass
    return row


def own_sample_cash(R, Rr, Rp, Rr_p, cfg, bond_r, first):
    v39 = json.load(open(CFG_V39))
    m_alt, _, _ = run_dyn(R, Rp, cfg, first, bond_r)
    m_base, _, _ = run_dyn(R, Rp, v39, first, bond_from_cfg(v39))
    return {"n": m_alt["n"], "cal": m_alt["cal"], "base_cal": m_base["cal"],
            "dcal": dcal(m_alt, m_base), "start": first}


def executable_cash(code, prod):
    if prod == "nav":
        df = read_nav(code)
        return bool(df is not None and "cum_nav" in df.columns)
    # 生产入口只读 nav 的 cum_nav。只有净值文件在，草案才执行得了。
    df = read_nav(code)
    return bool(df is not None and "cum_nav" in df.columns)


def missing_row(name, kind, legs):
    return {
        "name": name, "kind": kind, "executable": False, "missing": True,
        "a": False, "b": False, "c": False, "ident": False, "pass3": False, "suggest": False,
        "why": "行情缺失", "codes": [c for _, c in legs],
        "d_proxy": None, "d_real": None, "d_pre": None,
    }


def score_caliber_pair(name, kind, code, w, nav_w, base_w, info, prod, executable,
                       ambiguous, tenor, prem=None, real_full=True, group=None,
                       codes=None, dropped=None, mode=None, cfg=None):
    d_p = dcal(w["proxy"], base_w["proxy"])
    d_r = dcal(w["real"], base_w["real"])
    if w.get("real_short"):
        # d_real 相对缩短窗口的基线
        if w.get("real_base_cal"):
            d_r = float(w["real"]["cal"] - w["real_base_cal"])
        real_full = False
    d_pre = dcal(w["pre"], base_w["pre"])
    mdd_ok = w["proxy"]["mdd"] > -0.10
    a = (d_p is not None and d_r is not None and d_p > 0 and d_r > 0 and max(d_p, d_r) >= 0.10
         and mdd_ok and real_full and info.get("gap_ok", False))
    # 另一口径
    caliber_ok = False
    d_p2 = d_r2 = None
    nav_slim = None
    if nav_w is not None:
        d_p2 = dcal(nav_w["proxy"], base_w["proxy"])
        d_r2 = dcal(nav_w["real"], base_w["real"])
        if nav_w.get("real_short") and False:
            pass
        caliber_ok = d_p2 is not None and d_r2 is not None and d_p2 > 0 and d_r2 > 0
        nav_slim = {"proxy": slim(nav_w["proxy"]), "real": slim(nav_w["real"]),
                    "oos": slim(nav_w["oos"]), "d_proxy": d_p2, "d_real": d_r2}
    else:
        caliber_ok = False
    if not caliber_ok:
        a = False
    ident = bool(d_pre is not None and d_pre > -0.02)
    if d_p is not None and d_pre is not None and d_p * d_pre < 0 and abs(d_p) > 0.02 and abs(d_pre) > 0.02:
        ident = False
    if not info.get("spliced"):
        # 全历史就是自有样本
        ident = bool(ident and True)
    c = (not w.get("real_short")) and w["oos"]["cal"] + 1e-9 >= LOCK["anchor"]["oos_cal"]
    # OOS 对缩短没有单独处理：现金若前段是逆回购，OOS 仍是混合，covers 会杀掉 (a)
    prem_ok = True
    if prem and abs(prem["last"]) > 0.08:
        prem_ok = False
    short = bool(info.get("spliced") and info.get("first") and info["first"] > "2022-01-04")
    return {
        "name": name, "kind": kind, "group": group, "code": code, "codes": codes or [code],
        "mode": mode, "dropped": dropped, "executable": bool(executable),
        "prod": prod, "proxy": slim(w["proxy"]), "real": slim(w["real"]),
        "oos": slim(w["oos"]), "pre": slim(w["pre"]),
        "d_proxy": d_p, "d_real": d_r, "d_pre": d_pre,
        "nav": nav_slim, "prem": prem, "prem_ok": prem_ok,
        "mdd_ok": bool(mdd_ok), "a": bool(a), "c": bool(c), "ident": bool(ident),
        "gap_ok": bool(info.get("gap_ok")), "caliber_ok": bool(caliber_ok),
        "covers": True, "spliced": bool(info.get("spliced")), "short": short,
        "real_full": bool(real_full), "first": info.get("first"), "n_own_panel": info.get("n_own"),
        "ambiguous": bool(ambiguous), "tenor": bool(tenor),
        "mean_w": w["proxy"].get("mean_w"), "b": None,
        "cfg_keys": cfg_public(cfg) if cfg else None,
        "info": {k: info.get(k) for k in ("first", "gap_ok", "spliced", "longest_gap", "gap_frac", "n_own")},
    }


def attach_drop_plateau(rows, R, Rr, Rp, Rr_p, v39, base_w):
    by = {r["name"]: r for r in rows if r.get("kind") == "drop"}
    for r in rows:
        if r.get("kind") != "drop":
            continue
        other_mode = "R" if r["mode"] == "N" else "N"
        other = by.get(f"{drop_name(r['dropped'])}:{other_mode}")
        if other is None:
            r["b"] = False
            r["plateau_note"] = "缺另一模式"
            continue
        ok = (other.get("d_proxy") is not None and other.get("d_real") is not None
              and (r.get("d_proxy") or -1) > 0 and (r.get("d_real") or -1) > 0
              and other["d_proxy"] > 0 and other["d_real"] > 0
              and other["proxy"]["mdd"] > -0.10 and r["proxy"]["mdd"] > -0.10
              and abs(other["proxy"]["cal"] - r["proxy"]["cal"]) <= 0.15)
        note = "N/R"
        if ok and r.get("a") and r.get("ident"):
            ok, note = growth_wobble(r, R, Rr, Rp, Rr_p, v39, base_w, ok, note)
            if r["dropped"] == ["511260"]:
                ok, note = repo_wobble(r, R, Rr, Rp, Rr_p, base_w, ok, note)
        r["b"] = bool(ok)
        r["plateau_note"] = note


def growth_wobble(row, R, Rr, Rp, Rr_p, v39, base_w, ok, note):
    dropped = set(row["dropped"])
    growth = [s for s in ("159952", "159941", "513500") if s not in dropped]
    if len(growth) < 2:
        return ok, note + "; 成长不足两只，不加旋钮"
    if "511260" in dropped and len(row["dropped"]) == 1:
        return ok, note
    cfg = cfg_drop(v39, row["dropped"], row["mode"])
    # 用引擎归一后的牛市拆分
    ds = DynamicStrategy(R, cfg=cfg)
    bull = dict(ds.growth_split_bull)
    ranked = sorted(growth, key=lambda s: bull.get(s, 0), reverse=True)
    a, b = ranked[0], ranked[1]
    if bull.get(a, 0) < 0.05:
        return False, note + "; 无法挪 5pp"
    for sign in (+1, -1):
        nb = {s: bull.get(s, 0.0) for s in growth}
        nb[a] = nb[a] - sign * 0.05
        nb[b] = nb[b] + sign * 0.05
        if min(nb.values()) <= 0:
            return False, note + "; 5pp 越界"
        cfg2 = copy.deepcopy(cfg)
        full = dict(v39["growth_split_bull"])
        full.update(nb)
        cfg2["growth_split_bull"] = full
        bond = bond_from_cfg(cfg2)
        for start, key, base in ((PROXY_START, "proxy", base_w["proxy"]), (REAL_START, "real", base_w["real"])):
            sig = R if key == "proxy" else Rr
            pnl = Rp if key == "proxy" else Rr_p
            m, _, _ = run_dyn(sig, pnl, cfg2, start, bond)
            if not (dcal(m, base) > 0 and m["mdd"] > -0.10):
                return False, note + f"; 挪{sign:+d}5pp {key} 失败"
            if abs(m["cal"] - row[{"proxy": "proxy", "real": "real"}[key]]["cal"]) > 0.15 and key == "proxy":
                return False, note + "; 挪 5pp 后 proxy Cal 差超过 0.15"
    return True, note + "; 5pp 两侧过"


def repo_wobble(row, R, Rr, Rp, Rr_p, base_w, ok, note):
    v39 = json.load(open(CFG_V39))
    for repo in (0.018, 0.026):
        cfg = cfg_drop(v39, ["511260"], row["mode"])
        cfg["repo_rate"] = repo
        for start, key in ((PROXY_START, "proxy"), (REAL_START, "real")):
            sig = R if key == "proxy" else Rr
            pnl = Rp if key == "proxy" else Rr_p
            m, _, _ = run_dyn(sig, pnl, cfg, start, None)
            base = base_w[key]
            if not (dcal(m, base) > 0 and (key != "proxy" or m["mdd"] > -0.10)):
                return False, note + f"; repo {repo} {key} 失败"
            if key == "proxy" and abs(m["cal"] - row["proxy"]["cal"]) > 0.15:
                return False, note + f"; repo {repo} 悬崖"
    return True, note + "; repo 两侧过"


def attach_pair_gates(rows):
    # 组内 (b)
    gmap = groups_by_id()
    by_group = {}
    for r in rows:
        if r.get("kind") in ("equity", "cash") and r.get("group") and not str(r["name"]).startswith("combo"):
            by_group.setdefault(r["group"], []).append(r)
    corr = {}
    for gid, members in by_group.items():
        g = gmap[gid]
        codes = [c["code"] for c in g["codes"]]
        cval = None
        if len(codes) >= 2 and g["pair"] == "corr":
            cval = pair_corr(codes[0], codes[1])
        corr[gid] = cval
        for r in members:
            r["pair_corr"] = cval
        if g["pair"] == "corr":
            corr_bad = cval is None or cval != cval or cval < 0.90
            if len(members) < 2 or corr_bad:
                for r in members:
                    r["b"] = False
                    if len(members) < 2:
                        r["plateau_note"] = "只有一只，不能过(b)"
                    else:
                        r["plateau_note"] = f"相关 {cval} < 0.90"
                continue
            signs_ok = all((m.get("d_proxy") or -1) > 0 and (m.get("d_real") or -1) > 0 and m.get("real_full", True)
                           and m.get("covers", True) for m in members)
            for r in members:
                r["b"] = bool(signs_ok)
                r["plateau_note"] = f"相关 {cval:.3f}" + (" 同向" if signs_ok else " 不同向")
        elif g["pair"] == "tenor":
            good = [m for m in members if (m.get("d_proxy") or -1) > 0 and (m.get("d_real") or -1) > 0
                    and m.get("covers", True) and m.get("real_full", True)]
            for r in members:
                others = [m for m in good if m["name"] != r["name"]]
                r["b"] = bool(r in good and len(others) >= 1)
                r["plateau_note"] = f"久期同向 {len(good)}/{len(members)}"
        else:
            for r in members:
                r["b"] = False
                r["plateau_note"] = "单成员"
    # 组合：ndx100 两只同向，且 N/R 平台
    for r in rows:
        if r.get("kind") != "combo":
            continue
        # name combo:{mode}:drop159952+{gid}:{code}
        parts = r["name"].split(":")
        # combo, mode, drop159952+gid, code  —— 实际是 combo:{mode}:drop159952+{gid}:{code}
        # f"combo:{mode}:drop159952+{g['id']}:{code}"
        mode = r.get("mode")
        gid = r.get("group")
        code = (r.get("codes") or [None])[0]
        sibs = [x for x in rows if x.get("kind") == "combo" and x.get("mode") == mode and x.get("group") == gid]
        g = gmap[gid]
        cval = pair_corr(g["codes"][0]["code"], g["codes"][1]["code"]) if len(g["codes"]) > 1 else None
        r["pair_corr"] = cval
        if len(g["codes"]) < 2 or cval is None or cval != cval or cval < 0.90:
            r["b"] = False
            r["plateau_note"] = "新结构里该组没有第二只"
            continue
        signs = all((m.get("d_proxy") or -1) > 0 and (m.get("d_real") or -1) > 0 and m.get("real_full", True)
                    for m in sibs)
        other_mode = "R" if mode == "N" else "N"
        other = [x for x in rows if x.get("kind") == "combo" and x.get("mode") == other_mode
                 and x.get("group") == gid and (x.get("codes") or [None])[0] == code]
        cross = False
        if other:
            o = other[0]
            cross = (o.get("d_proxy") or -1) > 0 and (o.get("d_real") or -1) > 0 and abs(o["proxy"]["cal"] - r["proxy"]["cal"]) <= 0.15
        r["b"] = bool(signs and cross)
        r["plateau_note"] = f"组相关 {cval:.3f} 同向={signs} N/R={cross}"
    # 双换：两条都要同向
    doubles = [r for r in rows if r.get("kind") == "double"]
    if len(doubles) == 2:
        signs = all((m.get("d_proxy") or -1) > 0 and (m.get("d_real") or -1) > 0 and m.get("real_full", True)
                    for m in doubles)
        for r in doubles:
            r["b"] = bool(signs)
            r["plateau_note"] = "两对防御替换同向" if signs else "两对不同向"
    elif len(doubles) == 1:
        doubles[0]["b"] = False
        doubles[0]["plateau_note"] = "缺对照一对"
    # 纸面黄金：两只同向才算 (b)，但仍不可执行
    papers = [r for r in rows if r.get("kind") == "paper_gold"]
    if len(papers) >= 2:
        cval = pair_corr("518880", "159934")
        signs = all((m.get("d_proxy") or -1) > 0 and (m.get("d_real") or -1) > 0 for m in papers)
        for r in papers:
            r["b"] = bool(signs and cval is not None and cval >= 0.90)
            r["pair_corr"] = cval
            r["plateau_note"] = f"纸面黄金相关 {cval} 同向={signs}"
    for r in papers:
        r["executable"] = False


def finalize(row):
    if row.get("kind") == "control":
        return
    if row.get("missing"):
        row["pass3"] = False
        row["suggest"] = False
        row["why"] = row.get("why") or "行情缺失"
        return
    a = bool(row.get("a"))
    b = bool(row.get("b"))
    c = bool(row.get("c"))
    ident = bool(row.get("ident"))
    row["pass3"] = bool(a and b and c and ident and row.get("caliber_ok") and row.get("gap_ok") and row.get("covers", True) and row.get("real_full", True))
    hit15 = (row.get("d_proxy") or 0) >= 0.15 or (row.get("d_real") or 0) >= 0.15
    suggest = bool(row["pass3"] and hit15 and row.get("executable") and row.get("prem_ok", True)
                   and not row.get("short") and not row.get("ambiguous") and not row.get("tenor")
                   and row.get("kind") != "paper_gold")
    row["suggest"] = suggest
    why = []
    if not a:
        why.append("未过(a)")
    if not b:
        why.append("平台/(b)未过")
    if not c:
        why.append("OOS更差或缩短")
    if not ident:
        why.append("识别未过")
    if not row.get("caliber_ok", True):
        why.append("净值与前复权不同向或缺失")
    if not row.get("gap_ok", True):
        why.append("缺口不合格")
    if row.get("covers") is False:
        why.append("现金盖不住起点")
    if row.get("real_full") is False:
        why.append("real不是全窗口")
    if row["pass3"] and not hit15:
        why.append("过+0.10但没有一窗到+0.15")
    if row.get("short"):
        why.append("自身样本始于2022之后，不写成长期增益")
    if row.get("prem") and not row.get("prem_ok", True):
        why.append("末日溢价绝对值超过8%")
    if row.get("ambiguous"):
        why.append("沉开的近邻读法，不升默认")
    if row.get("tenor"):
        why.append("久期是菜单，不挑一只升默认")
    if row.get("kind") == "paper_gold":
        why.append("生产入口不读黄金现金开关")
    if not row.get("executable"):
        why.append("生产入口执行不了")
    row["why"] = "；".join(why) if why else "过关且建议升默认"
    row["hit15"] = bool(hit15)


def lookahead_spot(R, Rr, Rp, Rr_p, v39, bond):
    audits = []
    # 1 减创业板 R
    cfg = cfg_drop(v39, ["159952"], "R")
    audits.append(spot_one("drop159952:R", R, Rp, cfg, bond))
    # 2 黄金换 515100
    legs = [("515100", "518880")]
    alt, info = apply_legs(Rp, legs, "qfq")
    if alt is not None:
        audits.append(spot_one("gold:518880", R, alt, v39, bond))
    # 3 combo
    cfg3 = cfg_drop(v39, ["159952"], "R")
    alt3, _ = apply_legs(Rp, [("159941", "513300")], "qfq")
    if alt3 is not None:
        audits.append(spot_one("combo:R:513300", R, alt3, cfg3, bond))
    # 泄漏对照，只一条
    ds = DynamicStrategy(R, cfg=cfg)
    leak = LeakyStrategy(R, cfg=cfg)
    strict, _, _ = run_dyn(R, Rp, cfg, PROXY_START, bond)
    # 手工泄漏跑
    res = run_backtest(
        Rp, target_weights_fn=leak.target_fn(), daily_override_fn=leak.daily_fn(),
        start=PROXY_START, name="LEAK", min_delta=cfg["min_delta"], repo=cfg["repo_rate"],
        tranche_weights=cfg["tranche_weights"], cash_bond_rets=bond,
        cash_bond_pct=float(cfg.get("cash_bond_pct") or 0.0),
        rebal_weekday=4, rebal_freq="weekly", strict=False)
    e = evaluate(res)
    audits.append({
        "name": "drop159952:R:leaky_vs_strict",
        "strict_cal": strict["cal"], "leaky_cal": float(e["calmar"]),
        "strict_cagr": strict["cagr"], "leaky_cagr": float(e["cagr"]),
        "note": "当日信号只作对照，不是候选",
    })
    return audits


def spot_one(name, R, pnl, cfg, bond):
    m_full, res_full, ds_full = run_dyn(R, pnl, cfg, PROXY_START, bond)
    dec_full = {str(pd.Timestamp(k).date()): v for k, v in ds_full.decisions.items()}
    cut = R.index[-21]
    R1 = R.loc[:cut]
    P1 = pnl.loc[:cut]
    _, _, ds1 = run_dyn(R1, P1, cfg, PROXY_START, bond)
    dec1 = {str(pd.Timestamp(k).date()): v for k, v in ds1.decisions.items()}
    # 截断日前的决策
    bad = []
    for k, v in dec1.items():
        if k not in dec_full:
            bad.append(k)
            continue
        if not weights_close(v, dec_full[k]):
            bad.append(k)
    # 扰动最后一日
    P2 = pnl.copy()
    P2.iloc[-1] = P2.iloc[-1] + 0.05
    _, _, ds2 = run_dyn(R, P2, cfg, PROXY_START, bond)
    dec2 = {str(pd.Timestamp(k).date()): v for k, v in ds2.decisions.items()}
    bad2 = []
    for k, v in dec_full.items():
        if k not in dec2 or not weights_close(v, dec2[k]):
            bad2.append(k)
    return {
        "name": name,
        "truncate_mismatch_n": len(bad),
        "truncate_mismatch_sample": bad[:5],
        "perturb_mismatch_n": len(bad2),
        "perturb_mismatch_sample": bad2[:5],
        "cut": str(pd.Timestamp(cut).date()),
        "pass": len(bad) == 0 and len(bad2) == 0,
    }


def weights_close(a, b):
    keys = set(a) | set(b)
    for k in keys:
        if abs(float(a.get(k, 0)) - float(b.get(k, 0))) > 1e-8:
            return False
    return True


def render_report(fp, base_w, rows, audits, details):
    lines = []
    lines.append("# v41 纸面实验：六槽重组（减槽、换槽）")
    lines.append("")
    lines.append(f"预注册指纹 `{fp}`。")
    lines.append("第 1 节是跑数之前写死的规则。后面才是结果。规则没有按结果回改。")
    lines.append("默认仍是 v39。本文件是纸面记录，不下单，不改 `final_cfg_v39.json`。")
    lines.append("")
    lines.append(pre_text())
    lines.append("## 2. 对照与锚")
    lines.append("")
    lines.append(f"v39 proxy {cell(base_w['proxy'])}；real {cell(base_w['real'])}；OOS Cal {base_w['oos']['cal']:.4f}；2025-04-22 止 {cell(base_w['pre'])}。")
    mw = base_w["proxy"]["mean_w"]
    lw = base_w["proxy"]["last_w"]
    lines.append("")
    lines.append("v39 全历史均权（proxy）：" + "，".join(f"{traded_of(s) if s != 'cash' else '现金'} {mw[s]*100:.1f}%" for s in ("159232", "515100", "159941", "513500", "159952", "cash")) + "。")
    lines.append("末日权重：" + "，".join(f"{traded_of(s) if s != 'cash' else '现金'} {lw[s]*100:.1f}%" for s in ("159232", "515100", "159941", "513500", "159952", "cash")) + "。")
    lines.append(f"近 60 日 159952 均权 {base_w['proxy']['tail60_159952']*100:.2f}%。")
    lines.append("")
    lines.append("均权说明：自由现金流与红利全历史大约各 3%，不是可以忽略的灰尘；末日各自约 1%，只剩底仓。纳指全历史均权最高。创业板近期目标是 0，全历史均权仍有约 5%。")
    lines.append("")
    lines.append("| 对照 | proxy | ΔCal | real | ΔCal | OOS Cal |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] != "control":
            continue
        lines.append(f"| {r['name']} | {cell(r['proxy'])} | {dlt(r['d_proxy'])} | {cell(r['real'])} | {dlt(r['d_real'])} | {calf(r['oos']['cal'])} |")
    lines.append("")
    lines.append("## 3. 减槽")
    lines.append("")
    lines.append("模式 N 是引擎原生 `exclude`（防御弹性会漏进现金，底仓减半）。模式 R 把预算留在剩下的那只上。Δ 相对同一次 v39。")
    lines.append("")
    lines.append("| 臂 | proxy | ΔCal | real | ΔCal | OOS | 2025-04前 ΔCal | (a) | (b) | 识别 | 过关 | 说明 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] != "drop":
            continue
        lines.append(table_line(r))
    lines.append("")
    lines.append("特别看的几条，proxy 均权：")
    lines.append("")
    want = {
        "drop:159952", "drop:159222", "drop:563020", "drop:159222+563020",
        "drop:159941", "drop:513500", "drop:159941+513500", "drop:511260",
        "drop:159952+159941", "drop:159952+513500",
    }
    for r in rows:
        if r["kind"] != "drop":
            continue
        base_name = r["name"].rsplit(":", 1)[0]
        if base_name not in want:
            continue
        mw = r.get("mean_w") or {}
        bits = "，".join(f"{k} {v*100:.1f}%" for k, v in mw.items())
        lines.append(f"- {r['name']}：{bits}。平台：{r.get('plateau_note')}")
    lines.append("")
    lines.append("## 4. 换槽与现金袖")
    lines.append("")
    lines.append("生产口径是前复权（511010 例外，用仓库里的累计净值）。另一口径要同向为正，否则不过。拼接日写在表里。相关是两只前复权日收益的相关系数。")
    lines.append("")
    lines.append("| 臂 | 生产口径 proxy | ΔCal | real | ΔCal | 另一口径 ΔCal | OOS | 拼接 | 溢价中位/末日 | (a) | (b) | 识别 | 过关 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] not in ("equity", "cash", "combo", "double", "paper_gold"):
            continue
        if r.get("missing"):
            lines.append(f"| {r['name']} | 行情缺失 | | | | | | | | 否 | 否 | 否 | 否 |")
            continue
        nav = r.get("nav") or {}
        other = f"{dlt(nav.get('d_proxy'))} / {dlt(nav.get('d_real'))}" if nav else "缺失"
        prem = r.get("prem")
        prem_s = "—" if not prem else f"{prem['median']*100:.2f}% / {prem['last']*100:.2f}%"
        first = r.get("first") or "—"
        splice = "是" if r.get("spliced") else "否"
        lines.append(
            f"| {r['name']} | {cell(r.get('proxy'))} | {dlt(r.get('d_proxy'))} | {cell(r.get('real'))} | {dlt(r.get('d_real'))} | {other} | {calf((r.get('oos') or {}).get('cal'))} | {splice} {first} | {prem_s} | {yn(r.get('a'))} | {yn(r.get('b'))} | {yn(r.get('ident'))} | {yn(r.get('pass3'))} |"
        )
    lines.append("")
    lines.append("指数名（东财详情，采集于跑数时，不参与挑选）：")
    lines.append("")
    for code in all_codes():
        det = details.get(code) or {}
        if det.get("error") and not det.get("index_name"):
            lines.append(f"- {code}：详情没取到（{det.get('error')}）")
        else:
            lines.append(f"- {code}：{det.get('short') or ''}，指数 {det.get('index_name') or '—'}，成立 {det.get('estab') or '—'}")
    lines.append("")
    lines.append("## 5. 未来函数抽检")
    lines.append("")
    if not audits:
        lines.append("这次是 smoke，没有跑抽检。")
    for a in audits:
        if "truncate_mismatch_n" in a:
            flag = "通过" if a["pass"] else "未通过"
            lines.append(f"- {a['name']}：截断不一致 {a['truncate_mismatch_n']}，扰动不一致 {a['perturb_mismatch_n']}，截止 {a['cut']}，{flag}。")
        else:
            lines.append(f"- {a['name']}：严格 Cal {a['strict_cal']:.3f}，当日信号 Cal {a['leaky_cal']:.3f}。{a['note']}。")
    lines.append("")
    lines.extend(conclusion_lines(rows))
    return "\n".join(lines) + "\n"


def conclusion_lines(rows):
    b = {r["name"]: r for r in rows}
    n_drop = sum(1 for r in rows if r.get("kind") == "drop")
    n_drop_a = sum(1 for r in rows if r.get("kind") == "drop" and r.get("a"))
    lines = ["## 6. 结论", ""]
    lines.append("不建议把任何一条臂升成默认。不写 `references/final_cfg_v41.json`。默认仍是 v39。")
    lines.append("")
    lines.append(f"减槽共 {n_drop} 条（C(6,1)+C(6,2)，每种 N 与 R）。过 (a) 的有 {n_drop_a} 条。")
    lines.append("")
    lines.append("去掉创业板之后，N 和 R 是同一条路径：创业板没有底仓，两种模式都只是把成长拆分归一到纳指和标普。")
    lines.append(f"- {brief('drop:159952:R', b)} 近端目标权重本来就是 0，拿掉它并没有把收益留在组合里。全历史均权里纳指从约 27% 升到约 37%。OOS Calmar 降到 0.91，低于 v39 的 1.532。")
    lines.append("只留一只防御时，模式 N 把被去掉那只的弹性漏进现金，活下来的均权仍约 3%；模式 R 把底仓和弹性都给活下来的那只，均权约 6%。两条的 real ΔCal 都是负的。")
    lines.append(f"- 减槽里 proxy 最好看的单删是 {brief('drop:563020:N', b)} real 更差，不过同向。")
    lines.append(f"- 两只防御一起去掉是减槽里 proxy 最高的：{brief('drop:159222+563020:N', b)} MDD 收到 -7.78%，OOS 升到 1.88，但 real 更差。模式 R 与 N 数值相同，因为两只都被打成 0 之后没有预算可留。")
    lines.append("去掉任意一只 QDII，real ΔCal 在 -0.2 到 -2.6。两只 QDII 一起去掉，proxy ΔCal -0.378，real -2.930。去掉 511260、闲置现金改回 2.2% 逆回购：proxy ΔCal -0.320，real -0.189。")
    lines.append("")
    lines.append("换槽里唯一过三关的是 `ndx100:513300`。这是不去掉创业板的整段替换，和 v40 是同一种结构，不是这次重组的新发现。")
    p = b["ndx100:513300"]
    lines.append(f"- {brief('ndx100:513300', b)} 净值 ΔCal {dlt(p['nav']['d_proxy'])} / {dlt(p['nav']['d_real'])}。自有样本从 {p['own']['start']} 起 ΔCal {dlt(p['own']['dcal'])}，与全样本同号，所以不全是拼接段造出来的。末日溢价 {p['prem']['last']*100:.2f}%，超过事先写的 8%，不建议升默认。可执行配置留在 `references/exp_v41_pass_cfgs.json`（`159941→513300`，并保留 159222 与 563020），未接入默认。`assets/data/513300_qfq.csv` 只为了这份未启用的配置能被 `swaps.py` 读到。")
    lines.append(f"- 去掉创业板再换 513300：{brief('combo:R:drop159952+ndx100:513300', b)} v40 那条过关在这个新结构里消失。模式 N 与 R 相同，因为只删创业板时两种正规化没有差别。")
    lines.append(f"- 159509 全换 {brief('ndx_tech:159509', b)} 去掉创业板再换 {brief('combo:R:drop159952+ndx_tech:159509', b)} 末日溢价都是 39.45%，自身样本从 2023-08-09 起，同指数没有第二只。(b) 不过。这不是新发现。")
    p = b["ndx100:513100"]
    lines.append(f"- 513100 前复权两窗都很好，{brief('ndx100:513100', b)} 净值 real ΔCal {dlt(p['nav']['d_real'])}，末日溢价 {p['prem']['last']*100:.2f}%。口径冲突。")
    lines.append(f"- 黄金换红利槽：{brief('gold:518880', b)} 评估窗口从 2014-06-23 起，晚于黄金自身的 2013-07-30，这段不是上市后短样本。real 更差。159934 的 real 同样为负，两只一起否决。")
    lines.append(f"- 能源化工 159981 是新槽里最接近的单券：{brief('commodity:159981', b)} 净值同向。自有样本从 2020-01-20 起 ΔCal {dlt(b['commodity:159981']['own']['dcal'])}，与全样本同号。只有这一只能源化工，(b) 不过。proxy ΔCal 本身不到 +0.10，过线的是 real。")
    lines.append("- 标普油气两只的 proxy ΔCal 只有 +0.008 左右，净值的 proxy ΔCal 为负，自身样本晚于 2022。不过。")
    d0 = b["double:ballast_primary:518880+513350"]
    d1 = b["double:ballast_pair:159934+159518"]
    lines.append(f"- 黄金和标普油气一起换掉两个防御槽，全样本看起来过 (a)：{brief(d0['name'], b)} 以及 {brief(d1['name'], b)} 自有样本分别从 {d0['own']['start']}、{d1['own']['start']} 起，ΔCal {dlt(d0['own']['dcal'])} 和 {dlt(d1['own']['dcal'])}，与全样本反号。多出来的全历史收益是上市前仍在吃原来的自由现金流和红利，不是黄金和油气。识别拒绝。这是这次最需要标明的拼接假象。")
    lines.append("可转债、恒生科技、中概、科创、半导体、医药、银行，换进创业板槽或防御槽之后，real 都更差。货币、短融、国开也更差。511880 的 proxy MDD 破到 -10.76%。华宝添益 511990 的东财文件没有累计净值（只有万份收益），另一口径缺失；前复权本身两窗 ΔCal 都是负的。")
    lines.append(f"- 久期：有全历史的 511010（上证 5 年国债）{brief('tenor:511010', b)} 比 511260 差。511100 成立于 2023-12-13，511090（30 年）成立于 2023-05-19，盖不住 2014-06-23，全窗口里上市前被填成逆回购，只作诊断，而且诊断数字也更差。不从这三只里挑一只。")
    lines.append(f"- 闲置现金全部改记黄金：{brief('paper:cash_yield_gold:518880', b)} MDD -14.59%，破线。`run_backtest.py` 不读这个开关。生产引擎的 `SLOTS` 只有五列，加不进第七只股票，所以没有写一个引擎读不了的七槽草案。")
    lines.append("")
    lines.append("未来函数抽检见第 5 节。三条路径截掉最后 20 个交易日、以及把最后一日损益加 5 个百分点，决策权重都不变。去掉创业板的严格 Calmar 是 1.213，当日信号是 0.965。严格路径不是把当日收盘当成信号。")
    lines.append("")
    lines.append("名单没有在看到 Calmar 之后改过。没有用 real 窗口挑选参数。减槽没有新券，不存在把上市后短样本当成全历史。换槽的拼接和自有样本写在第 4 节和上面几条里。")
    lines.append("")
    lines.append("直接回答：这 42 条减槽里没有一条比 v39 更好并过三关。换进的候选里，也没有一条新结构过三关。唯一过三关的 513300 是旧的整段替换，末日溢价 15%，不升默认。最接近但落选的是：两只防御一起去掉（proxy 更好、real 更差）、159981（只有一只，proxy 不到 +0.10）、以及黄金加油气（全样本好看，上市后的自有样本更差）。")
    lines.append("")
    return lines


def brief(name, rows):
    r = rows[name]
    return f"`{name}` {cell(r.get('proxy'))}，ΔCal {dlt(r.get('d_proxy'))}；real {cell(r.get('real'))}，ΔCal {dlt(r.get('d_real'))}。"


def table_line(r):
    return (
        f"| {r['name']} | {cell(r.get('proxy'))} | {dlt(r.get('d_proxy'))} | {cell(r.get('real'))} | {dlt(r.get('d_real'))} | {calf((r.get('oos') or {}).get('cal'))} | {dlt(r.get('d_pre'))} | {yn(r.get('a'))} | {yn(r.get('b'))} | {yn(r.get('ident'))} | {yn(r.get('pass3'))} | {r.get('why') or ''} |"
    )


def yn(v):
    return "是" if v else "否"


def write_drafts(rows):
    suggested = [r for r in rows if r.get("suggest") and r.get("executable")]
    passed = [r for r in rows if r.get("pass3") and r.get("executable")]
    path_pass = os.path.join(REF, "exp_v41_pass_cfgs.json")
    path_final = os.path.join(REF, "final_cfg_v41.json")
    if os.path.exists(path_final):
        os.remove(path_final)
    if not passed:
        if os.path.exists(path_pass):
            os.remove(path_pass)
        return
    blob = []
    v39 = json.load(open(CFG_V39))
    for r in passed:
        cfg = draft_cfg(v39, r)
        blob.append({"name": r["name"], "suggest": r.get("suggest"), "why": r.get("why"), "cfg": cfg})
    json.dump(blob, open(path_pass, "w"), ensure_ascii=False, indent=1)
    if len(suggested) == 1:
        cfg = blob[[i for i, b in enumerate(blob) if b["suggest"]][0]]["cfg"]
        json.dump(cfg, open(path_final, "w"), ensure_ascii=False, indent=1)
        # 让 swaps.py 读得到前复权
        codes = r_codes(suggested[0])
        for code in codes:
            src = os.path.join(UNI, "px", f"{code}_qfq.csv")
            dst = os.path.join(DATA_DIR, f"{code}_qfq.csv")
            if os.path.exists(src) and not os.path.exists(dst):
                open(dst, "wb").write(open(src, "rb").read())


def r_codes(row):
    return [c for c in (row.get("codes") or []) if c and c not in ("159222", "563020", "511260")]


def draft_cfg(v39, row):
    if row.get("kind") == "drop" or (row.get("dropped") and row.get("mode")):
        cfg = cfg_drop(v39, row["dropped"], row["mode"])
    else:
        cfg = copy.deepcopy(v39)
        cfg.pop("_note", None)
    if row.get("kind") == "cash":
        cfg["cash_bond_pct"] = 1.0
        cfg["cash_bond_code"] = (row.get("codes") or [None])[0]
    legs = row.get("legs") or []
    if legs and row.get("prod") == "qfq":
        swaps = dict(cfg.get("instrument_swaps") or {})
        for slot, code in legs:
            if slot in swaps or slot in ("159232", "515100", "159941", "513500", "159952"):
                swaps[slot] = code
        cfg["instrument_swaps"] = swaps
    cfg["version"] = "v41-draft"
    cfg["_note"] = (
        "v41 纸面记录，未接入默认，不允许实盘，不建议升默认。来自 "
        + row["name"] + "。" + (row.get("why") or "")
        + "。默认仍是 final_cfg_v39.json。"
    )
    if "allow_live" in cfg:
        del cfg["allow_live"]
    return cfg


if __name__ == "__main__":
    main()
