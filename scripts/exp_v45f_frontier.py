# -*- coding: utf-8 -*-
"""v45f：红利腿保留比例前沿（配合 cn_lead）。"""
import copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v41_validate as V
from exp_v43_floors import run
from run_backtest import exec_panel
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(ROOT,'out')
BASE=json.load(open(os.path.join(ROOT,'references','final_cfg_v44.json')))
def C(tag,**kw):
    c=copy.deepcopy(BASE); c.update(kw); c['version']=tag; return c
CANDS={'base_v44':C('base_v44')}
for k in (5,10,15,20,25,30):
    CANDS[f'k{k}_cnlead']=C(f'k{k}_cnlead',lock_defense_keep=k/100,score_mode='cn_lead')
def main():
    P=exec_panel(V.R,'proxy',BASE,'price'); Pr=exec_panel(V.Rr,'real',BASE,'price'); rep={}
    for name,c in CANDS.items():
        rep[name]={}; print('\n'+name,flush=True)
        for layer,sig,pnl,start,end in (('proxy',V.R,P,V.PROXY_START,None),('train',V.R,P,V.PROXY_START,'2021-12-31'),('oos',V.R,P,V.OOS_START,None),('real',V.Rr,Pr,V.REAL_START,None)):
            _,e=run(c,sig,pnl,start,end); rep[name][layer]=V.pack(e)
            print(f"  {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.3f}",flush=True)
    json.dump(rep,open(os.path.join(OUT,'exp_v45f_frontier.json'),'w'),ensure_ascii=False,indent=1); print('[ok]',flush=True)
if __name__=='__main__': main()
