# -*- coding: utf-8 -*-
"""v45 候选完整验证：分段、费用、压力、bootstrap、共振熊（场内价）。"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import run
from run_backtest import exec_panel
from stress_test import synthetic_resonance
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(ROOT,'out')
BASE=json.load(open(os.path.join(ROOT,'references','final_cfg_v44.json')))
CAND=json.load(open(os.path.join(ROOT,'references','final_cfg_v45.json')))
STRESS=(('2015股灾','2015-06-15','2016-02-29'),('2018熊市','2018-01-02','2019-01-03'),('2021-22熊市','2021-02-19','2022-10-31'),('2019-21牛','2019-01-04','2021-02-18'),('2023-24震荡','2023-01-03','2024-08-30'))
def main():
    P=exec_panel(V.R,'proxy',BASE,'price'); Pr=exec_panel(V.Rr,'real',BASE,'price')
    rep={'metrics':{},'costs':{},'stress':{},'bootstrap':{},'synth':{}}
    for name,c in (('base_v44',BASE),('v45',CAND)):
        rep['metrics'][name]={}
        print('\n'+name,flush=True)
        for layer,sig,pnl,start,end in (('proxy',V.R,P,V.PROXY_START,None),('train',V.R,P,V.PROXY_START,'2021-12-31'),('oos',V.R,P,V.OOS_START,None),('real',V.Rr,Pr,V.REAL_START,None)):
            res,e=run(c,sig,pnl,start,end); rep['metrics'][name][layer]=V.pack(e)
            if layer=='proxy': rep['bootstrap'][name]=V.bootstrap(res['rets'])
            print(f"  {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.3f}",flush=True)
        rep['costs'][name]={}
        for bp in (5,10,20,30):
            _,e=run(c,V.R,P,V.PROXY_START,None,fee=bp/10000.0); rep['costs'][name][f'{bp}bp']=V.pack(e)
            print(f"  {bp}bp {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.3f}",flush=True)
    print('\n== stress ==',flush=True)
    for sn,start,end in STRESS:
        rep['stress'][sn]={}
        for name,c in (('base_v44',BASE),('v45',CAND)):
            _,e=run(c,V.R,P,start,end); rep['stress'][sn][name]=V.pack(e)
            print(f"  {sn:10} {name:8} {e['cagr']*100:7.2f}% / {e['max_dd']*100:7.2f}% / {e['calmar']:5.2f}",flush=True)
    print('\n== synth ==',flush=True)
    for tag,us,cn,idx in (('std',-0.30,-0.20,-0.25),('hard',-0.40,-0.30,-0.35)):
        synth,_=synthetic_resonance(Pr,('2024-09-02','2026-07-31'),us,cn,idx)
        rep['synth'][tag]={}
        for name,c in (('base_v44',BASE),('v45',CAND)):
            _,e=run(c,synth,synth,'2024-09-02','2026-07-31'); rep['synth'][tag][name]=V.pack(e)
            print(f"  {tag:4} {name:8} {e['cagr']*100:7.2f}% / {e['max_dd']*100:7.2f}% / {e['calmar']:5.2f}",flush=True)
    json.dump(rep,open(os.path.join(OUT,'exp_v45g_validate.json'),'w'),ensure_ascii=False,indent=1); print('[ok]',flush=True)
if __name__=='__main__': main()
