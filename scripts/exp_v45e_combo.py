# -*- coding: utf-8 -*-
"""v45e：红利腿保留 × OOS 友好层的组合。"""
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
 'k25_cnlead':C('k25_cnlead',lock_defense_keep=0.25,score_mode='cn_lead'),
 'k50_cnlead':C('k50_cnlead',lock_defense_keep=0.50,score_mode='cn_lead'),
 'k25_mom':C('k25_mom',lock_defense_keep=0.25,cn_seesaw=True,cn_seesaw_mode='momentum',cn_seesaw_thr=0.0,cn_seesaw_frac=0.25),
 'k25_inv20':C('k25_inv20',lock_defense_keep=0.25,tech_rotate=True,tech_rotate_mode='inverse',tech_rotate_win=20,tech_rotate_frac=0.25),
 'k50_inv20':C('k50_inv20',lock_defense_keep=0.50,tech_rotate=True,tech_rotate_mode='inverse',tech_rotate_win=20,tech_rotate_frac=0.25),
 'k25_cnlead_inv20':C('k25_cnlead_inv20',lock_defense_keep=0.25,score_mode='cn_lead',tech_rotate=True,tech_rotate_mode='inverse',tech_rotate_win=20,tech_rotate_frac=0.25),
 'k35_cnlead':C('k35_cnlead',lock_defense_keep=0.35,score_mode='cn_lead'),
}
def main():
    P=exec_panel(V.R,'proxy',BASE,'price'); Pr=exec_panel(V.Rr,'real',BASE,'price'); rep={}
    for name,c in CANDS.items():
        rep[name]={}; print('\n'+name,flush=True)
        for layer,sig,pnl,start,end in (('proxy',V.R,P,V.PROXY_START,None),('train',V.R,P,V.PROXY_START,'2021-12-31'),('oos',V.R,P,V.OOS_START,None),('real',V.Rr,Pr,V.REAL_START,None)):
            _,e=run(c,sig,pnl,start,end); rep[name][layer]=V.pack(e)
            print(f"  {layer:5} {e['cagr']*100:6.2f}% / {e['max_dd']*100:6.2f}% / {e['calmar']:5.3f} / TO {e['turnover']:6.1f}",flush=True)
    json.dump(rep,open(os.path.join(OUT,'exp_v45e_combo.json'),'w'),ensure_ascii=False,indent=1); print('[ok]',flush=True)
if __name__=='__main__': main()
