# Independent audit of default v40 (paper)

Tip `0f91fe84e2daf2150ee65f9b979f1e84e81af119` (2026-11-11 12:19 +0800). Data stopped at the 2026-11-10 close. No live trading. Default `references/final_cfg_v40.json` was not edited. This v40 is the promoted default (bull cap ×1.10, bull split 32/61/7, `cool_off_weeks=3`). It is a different strategy from the unpromoted 159509 experiment in PR #11.

Engine: `scripts/run_backtest.py` and `scripts/engine.py:run_backtest(..., strict=True)`, fee 5bp one way, `signal_lag=1`, `premium_shift=2`, Friday, one tranche. Sharpe uses `evaluate()`'s default rf of 1.8%. The config `repo_rate` of 2.2% is the residual cash rate. Windows: proxy 2014-06-23 → 2026-11-10 (3022 days), real 2025-04-23 → 2026-11-10 (376 days), OOS 2022-01-04 → 2026-11-10 (1179 days). v39 was re-run on this same panel.

## Verdict

| Check | Result |
|---|---|
| Published v40 proxy / real / OOS | Match at the printed precision |
| 三关 (v40 Calmar not worse than v39 on proxy, real, and OOS) | **Fail.** Real Calmar 6.562 vs v39 6.819 |
| Lookahead, including cool-off dates | **Pass.** Max absolute difference 0 |
| Must-fix code or data bug | **None** that breaks the published path or lookahead. One named rule gap below |
| 15% CAGR and \|MDD\| < 10% on the long sample | CAGR **misses** (13.35% vs 15%). \|MDD\| **meets** (7.83%) |

## 1. Official-engine figures

Exact floats from `run_backtest.py` / `engine.evaluate` (also `scripts/audit_lookahead_v40.py` section 2 for v40).

| Window | CAGR | MDD | Sharpe | Calmar | Turnover | Avg cash |
|---|---:|---:|---:|---:|---:|---:|
| v40 proxy | 0.13353630430388663 (13.35%) | -0.0782635650999407 (-7.83%) | 1.419619310782515 (1.42) | 1.7062384537857938 (1.71 / 1.706) | 119.2825 | 53.75% |
| v40 real | 0.2750539076375813 (27.51%) | -0.041913039387507345 (-4.19%) | 2.414073550286434 (2.41) | 6.562490138082524 (6.56) | 14.5711 | 39.84% |
| v40 OOS | 0.12067897550967954 (12.07%) | -0.07422487687790669 (-7.42%) | 1.3904558815637758 (1.39) | 1.6258561898079764 (1.63) | 40.1137 | 60.67% |
| v39 proxy, same data | 0.12166706001731376 (12.17%) | -0.09563306611941624 (-9.56%) | 1.3131569920556665 (1.31) | 1.272227953722503 (1.27) | 123.3585 | 53.70% |
| v39 real, same data | 0.25384145171321015 (25.38%) | -0.037226252269268345 (-3.72%) | 2.355601098149273 (2.36) | 6.818882810901857 (6.82) | 13.97 | 40.59% |
| v39 OOS, same data | 0.1128252499799749 (11.28%) | -0.07446729341874136 (-7.45%) | 1.328440528784969 (1.33) | 1.5150980356643375 (1.515) | 38.69 | 60.81% |

Gap versus the README / `references/v40_release_report.md` printed figures: **0** at two decimals (Calmar three decimals 1.706 and 1.515 also match). MDD dates: v40 proxy peak 2015-06-03, trough 2015-07-03. v39 proxy trough is 2015-08-25. v40 real peak 2025-10-29, trough 2025-11-21. v39 real peak 2025-10-09, trough 2025-10-14.

Config diff versus v39 is only `_note`, `version`, `state_map` states 7–9 (65/85/81 → 71.5/93.5/89.1), `growth_split_bull` (0.30/0.63/0.07 → 0.32/0.61/0.07), and `cool_off_weeks` 3. Same tickers, same 511260 cash sleeve, same five risk layers.

Stage totals (proxy), which match the release report to the printed hundredth:

| Stage | v39 total / MDD | v40 total / MDD |
|---|---:|---:|
| 2014H2–2015 bull | +30.39% / -5.58% | +33.62% / -5.81% |
| 2015 crash | -6.60% / -7.65% | -4.56% / -5.75% |
| 2016–2017 | +10.38% / -3.71% | +10.27% / -3.98% |
| 2018 bear | +9.87% / -3.49% | +10.67% / -3.42% |
| 2019–2021 bull | +46.80% / -6.59% | +52.35% / -5.09% |
| 2021–2022 bear | +3.89% / -6.57% | +4.29% / -6.85% |
| 2023–2024 | +23.86% / -2.46% | +23.86% / -2.60% |
| 2024Q4–2026 | +42.92% / -4.48% | +47.67% / -4.23% |

2021–22 total return is higher under v40; the stage drawdown is deeper (-6.85% vs -6.57%).

## 2. 三关

Rule used here: v40 Calmar is not worse than v39 on proxy, real, and OOS, same 11-10 panel.

| Window | v40 Calmar | v39 Calmar | Delta | Gate |
|---|---:|---:|---:|---|
| proxy | 1.706238 | 1.272228 | +0.434 | pass |
| real | 6.562490 | 6.818883 | **-0.256** | **fail** |
| OOS | 1.625856 | 1.515098 | +0.111 | pass |

Real CAGR is higher (27.51% vs 25.38%) and real MDD is wider (-4.19% vs -3.72%). The release note already says this. The real-window failure is confirmed, not a rounding artifact. Every one-knob and two-knob subset of the three changes also has a worse real Calmar than v39. The package does not dominate v39.

## 3. Lookahead and cool-off

`scripts/audit_lookahead_v40.py`: **ALL ASSERTIONS PASSED**. In-index `_idx` mismatches 0/5076. Truncation max abs **0**. Future-price perturbation max abs **0**. Ten Fridays, 09:00 information set versus the stored decision, max abs **0**. Post-close list is the 2026-11-13 preview (signal 2026-11-10); the script correctly does not compare that preview to the 2026-11-06 decision.

Code (`strategy.py` `regular_target`, cool-off block): the clock reads `dti = R.index[_idx(dt)]`. For a decision that is already in the index, `_idx` is the previous bar (`searchsorted(side="left")-1`), so the same-day return is not an input. For a Friday 09:00 call, the Friday bar is absent and `_idx` lands on that same previous bar. The two paths use one signal date. Probe on the full v40 proxy run: **0** decisions whose signal timestamp was on or after the decision date. Eight cool-off episode dates, re-truncated and re-checked at 09:00: max abs **0**.

The cool-off clock is 21 calendar days between signal dates (`>= cool_off_weeks * 7`), not three decision-weeks. Of 74 finished episodes, 69 released at 21 days and 5 at 28. Forced intermediate decisions: 60 episodes had 2, 12 had 1, 2 had 0. That is holiday spacing, not a future read.

Daily override does not apply the floor. `daily_fn` returns `_base_target` when target equity is at least 4pp below current equity, and `_base_target` never consults `_cool_from`. That fired **10** times. Largest lift: 2025-03-03, 159941 target 17.08% against a 5% floor (cool started at signal date 2025-02-27), plus 513500 +1.39pp the same day. A counterfactual that forces cooled slots back to the floor inside `daily_fn` leaves MDD unchanged on all three windows and moves CAGR to 13.380% / 27.673% / 12.130% and Calmar to 1.710 / 6.603 / 1.634. Real Calmar stays below v39. This is a prose-versus-code gap. It is not a lookahead bug and it does not flip the audit verdict. The default was left as the code that produced the published numbers.

## 4. Friday 2026-11-13 observe-only list

Official `run_backtest.py` print and a direct `live_rebalance_target` agree.

| Item | Engine | README |
|---|---|---|
| Data end / decision / signal | 2026-11-10 / 2026-11-13 / 2026-11-10 | same |
| Score | 1/9 | 1/9 |
| CN lock / US lock | yes / no | yes / no |
| CN 3y DD, SMA20, dual MA, SMA100, recovery | -19.78%, no, no, no, 47.4% | -19.8%, no, no, no, 47% |
| US same | -4.14%, no, no, no, 90.7% | -4.1%, no, no, no, 91% |
| Premium date (T-2 bar) | 2026-11-09 | 2026-11-09 |
| 159941 / 513500 premium | 12.591% / 8.563% | 12.59% / 8.56% |
| Held 159222, 563020, 159941, 513500, 159952, cash | 1.000%, 1.024%, 4.956%, 6.120%, 0, 86.899% | 1.00, 1.02, 4.96, 6.12, 0, 86.90 |
| Target | 1 / 1 / 5 / 5 / 0 / 88 | same |
| Max slot \|delta\| | 1.120pp (513500) | 1.12pp |
| `min_delta` 3.5% | no trade | 仅观察 |

Slots 159232 and 515100 are still in cool-off from signal date 2026-10-22, as are 159941 and 513500. CN is also deep-bear locked, so the natural target is already the floor. v39 and v40 targets match, and the upgrade does not create an order.

## 5. Where the gain comes from, and the parameter neighborhood

Positive log-excess versus v39, share of the positive part: **2015 43.1%**, 2020 15.4%, 2025 14.2%, 2021 9.7%, 2026 7.9%, 2018 4.9%, 2019 3.0%. 2015 + 2019–21 + 2025–26 is **93%** of the positive log-excess. 2014, 2016, 2022, and 2023 are flat to negative. The release note's concentration claim is right, and it is a reason to discount the 13.35% figure. 2015 alone is the single largest piece, and it is the path that also produces the MDD improvement (trough pulled forward from 2015-08-25 to 2015-07-03).

One knob at a time, proxy / real / OOS Calmar:

| Variant | Proxy Cal | Real Cal | OOS Cal | 2015 total |
|---|---:|---:|---:|---:|
| v39 | 1.272 | 6.819 | 1.515 | +22.31% |
| cap ×1.10 only | 1.313 | 6.405 | 1.543 | +24.10% |
| split 32% only | 1.272 | 6.706 | 1.537 | +23.86% |
| cool 3w only | 1.687 | 5.229 | 1.561 | +26.99% |
| cap + split, no cool | 1.308 | 6.251 | 1.575 | +25.67% |
| cap + cool, split 30% | 1.736 | 4.776 | 1.594 | +28.87% |
| v40 (all three) | 1.706 | 6.562 | 1.626 | +30.37% |

Proxy Calmar is mostly the cool-off. The 32% split by itself does not move proxy Calmar (1.272148 vs 1.272228). Putting the 32% split on top of cap+cool **lowers** proxy Calmar from 1.736 to 1.706 and **raises** real Calmar from 4.776 to 6.562. Real MDD of -4.19% already appears in cap+split with cool-off off; cool-off does not widen that particular real drawdown further. The split is doing its work on the 376-day real window, which is the window where the three-gate fails.

Neighbors, other two knobs held at the v40 choice. Proxy Calmar / real Calmar / OOS Calmar:

| Knob | Neighbor | Proxy / real / OOS Calmar |
|---|---|---|
| Cap | ×1.00, 1.05, **1.10**, 1.15, 1.20 | 1.660/5.033/1.582, 1.642/4.834/1.605, **1.706/6.562/1.626**, 1.706/6.409/1.632, 1.644/6.050/1.633 |
| ChiNext bull | 30, 31, **32**, 33, 34% | 1.736/4.776/1.594, 1.717/6.533/1.582, **1.706/6.562/1.626**, 1.683/6.446/1.651, 1.670/6.351/1.690 |
| Cool-off | 0, 2, **3**, 4 weeks | 1.308/6.251/1.575, 1.501/6.314/1.602, **1.706/6.562/1.626**, 1.670/6.489/1.613 |

×1.10 is a real-Calmar peak; ×1.05 is a cliff (real Calmar 4.83, MDD -5.23%). 33% matches the README (CAGR 13.373%, MDD -7.947%, Calmar 1.683 versus 13.354% / -7.826% / 1.706). Proxy Calmar prefers 30%, OOS Calmar prefers 34%, real Calmar prefers 32%. Cool-off of 3 weeks is the best of {0, 2, 3, 4} on all three Calmars, and both neighbors are worse. That is a one-step peak, not a flat plateau. Combined with the 2015 concentration, the 1.71 Calmar is a fitted point on this history.

`scripts/audit_robustness.py references/final_cfg_v40.json v40`: baseline Calmar 1.7062. Calmar worse by more than 10% in **4/104**, all `gate_win` ×0.8/0.9/1.1/1.2 (Calmar 1.262, 1.339, 1.398, 1.328; MDD -9.58% to -9.17%). The other axes stay inside the 10% band. This matches the release note.

## 6. Data through 2026-11-10

174 CSVs under `assets/data`. **Zero** rows dated after 2026-11-10. Proxy panel ends 2026-11-10 and contains 8 weekend rows. Real panel contains 7. Weekend counts the README names: `050025_nav.csv` 10, `159941_nav.csv` 7, `513500_nav.csv` 7, `270042_nav.csv` 8. Also present, and used by the proxy/real build: `159952_nav.csv` 6, `515100_nav.csv` 3.

QDII nav files `159941_nav.csv`, `513500_nav.csv`, and `270042_nav.csv` end on **2026-11-09**. The 2026-11-10 proxy return for those slots is an engine `ffill`. `513500` has 63 NaNs inside the proxy window (US holidays), which is the "63 gaps" sentence. Filling those NaNs with 0 instead of ffill changes v40 proxy from 13.3536% / Calmar 1.70624 to 13.3537% / 1.70635. Dropping weekend rows changes it to **13.294% / -7.826% / Calmar 1.699**.

The results-section sentence that quotes 11-06 / 11-09 / 11-10 as 12.13% / 12.15% / 12.17%, and real NAV 25.38%, is the **v39** scale. Recomputed v39 proxy CAGR at those ends: 12.126% / 12.148% / 12.167%. Recomputed **v40** proxy CAGR at those ends: **13.313% / 13.336% / 13.354%**. README was not rewritten.

`final_cfg_v39.json` `_note` still says v39 is the current default and quotes proxy 12.22% / Calmar 1.28 and real 25.57% / 6.87. The 11-10 engine produces 12.17% / 1.27 and 25.38% / 6.82, which is what the README v39 paragraph says. The note was not edited.

## 7. Bootstrap, 20bp, bond shock

No bootstrap script and no bond-shock script are in the repo. The fee path is the official engine `fee` argument.

20-day block bootstrap of the official daily returns, 2000 draws:

| Sample | P(Calmar < 1) | P(CAGR < 4.3%) |
|---|---:|---:|
| v40 circular, seed 20261110 | 18.3% | 0 |
| v40 circular, seed 0 | 19.3% | 0 |
| v40 non-circular, seed 20261110 | 19.3% | 0 |
| v39 circular, seed 20261110 | 28.7% | 0 |
| v39 non-circular, seed 20261110 | 29.9% | 0 |

Published 20.1% and 29.8% sit next to this range. v39's 29.8% matches the non-circular draw (29.9%). v40's 20.1% was not reproduced to the digit; the independent draws are 18.3–19.3%. Median bootstrapped v40 Calmar is about 1.37–1.39, against the point estimate 1.71. The point estimate is not a tight lower bound.

Fee on v40 proxy, official engine:

| Fee | CAGR | MDD | Calmar |
|---|---:|---:|---:|
| 5bp | 13.3536% | -7.8264% | 1.70624 |
| 10bp | 12.7916% | -7.8912% | 1.62099 |
| 15bp | 12.2323% | -7.9561% | 1.53748 |
| 20bp | 11.6756% | -8.3520% | 1.39794 |

20bp prints as 11.68% / -8.35% / Calmar 1.40. Match.

Bond sleeve: there is no committed shock script. Replacing 20 bond sessions from 2022-02-07 with a flat path that compounds to -8% takes v40 proxy MDD from -7.83% to **-12.93%**. A single-session -8% bond return on the first bond day on or after 2022-02-07 takes it to **-13.49%**. The published "约 -13.4%" is that neighborhood, not a digit match. The 2015-06-15 20-session path takes MDD to -12.17%. Either shock breaks the 10% drawdown bar. Equity risk layers do not see the bond sleeve.

## 8. Bugs

No must-fix defect in the published backtest path.

- Lookahead, Friday alignment, and the cool-off clock's use of the lagged bar all pass.
- The daily-override cool-off gap in section 3 is real and is left unpatched. MDD is unchanged and 三关 stays failed on real Calmar either way.
- README residue levels in section 6 are v39 figures sitting in the v40 results section. Reported here, not edited.
- `run_backtest.py`'s "相对 v39" lines print v39's last weight from the signal panel. For 563020 that weight is 1.035%, which prints as 1.04%, while the v40 holdings block (pnl panel) is 1.024% and prints as 1.02%. The README holdings table follows the pnl block. The weight gap itself is 0.01pp.

## 9. The 15% / 10% bar

Long-sample proxy CAGR **13.35%** is **1.65 percentage points short of 15%**. OOS CAGR **12.07%** is also short. Absolute MDD **7.83%** is inside 10%. The real window's 27.51% is 376 trading days and is the window whose Calmar got worse. Bootstrap P(Calmar < 1) near 19%, the one-step peaks in section 5, and the 2015 share of the gain are the reasons not to treat 13.35% / 1.71 as an expected result.

Raw floats and the neighbor grid: `references/audit_v40_independent.json`. Regenerated by `python3 scripts/audit_v40_independent.py` with no data download.
