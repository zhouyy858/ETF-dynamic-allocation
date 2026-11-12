# -*- coding: utf-8 -*-
"""v45c：深熊锁保留红利防御腿（跷跷板对冲）候选。"""
import copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import run
from run_backtest import exec_panel

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(ROOT,'out')
BASE=json.load(open(os.path.join(ROOT,'references','final_cfg_v44.json')))
def C(tag,**kw):
    c=copy.deepcopy(BASE); c.update(kw); c['version']=tag; return c
CANDS={
 'base_v44':C('base_v44'),
 'keep25':C('keep25',lock_defense_keep=0.25),
 'keep50':C('keep50',lock_defense_keep=0.50),
 'keep75':C('keep75',lock_defense_keep=0.75),
 'keep100':C('keep100',lock_defense_keep=1.0),
 'keep50_rel':C('keep50_rel',lock_defense_keep=0.50,lock_defense_rel=True),
 'keep100_rel':C('keep100_rel',lock_defense_keep=1.0,lock_defense_rel=True),
 'keep50_rel_trim':C('keep50_rel_trim',lock_defense_keep=0.50,lock_defense_rel=True,cn_seesaw=True,cn_seesaw_mode='contrarian',cn_seesaw_thr=0.20,cn_seesaw_frac=0.5),
}
STRESS=(('2015','2015-06-15','2016-02-29'),('2018','2018-01-02','2019-01-03'),('2021-22','2021-02-19','2022-10-31'))
def main():
    P=exec_panel(V.R,'proxy',BASE,'price'); Pr=exec_panel(V.Rr,'real',BASE,'price'); rep={'metrics':{},'stress':{}}
    for name,c in CANDS.items():
        rep['metrics'][name]={}; print('\n'+name,flush=True)
        for layer,sig,pnl,start,end in (('proxy',V.R,P,V.PROXY_START,None),('train',V.R,P,V.PROXY_START,'2021-12-31'),('oos',V.R,P,V.OOS_START,None),('real',V.Rr,Pr,V.REAL_START,None)):
            _,e=run(c,sig,pnl,start,end); rep['metrics'][name][layer]=V.pack(e)
            print(f"  {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.3f} / TO {e['turnover']:6.1f}",flush=True)
    print('\n== stress ==',flush=True)
    for sn,start,end in STRESS:
        rep['stress'][sn]={}
        for name,c in CANDS.items():
            _,e=run(c,V.R,P,start,end); rep['stress'][sn][name]=V.pack(e)
            print(f"  {sn:8} {name:16} {e['cagr']*100:7.2f}% / {e['max_dd']*100:7.2f}% / {e['calmar']:5.2f}",flush=True)
    json.dump(rep,open(os.path.join(OUT,'exp_v45c_defkeep.json'),'w'),ensure_ascii=False,indent=1); print('[ok]',flush=True)
if __name__=='__main__': main()
