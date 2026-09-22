"""Bounded all-21-parameter refinement using original Maxwell and DLL flow band.

Pressure-band score is an exploratory penalty, not an experimental chi-square.
Keep all four groups' original marginal M-R objectives and anchor penalties.
"""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import run_gqhd_continuous_fullspace as c

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'output/gqhd_pressure_refine_20260908/data'
GROUPS = {'1':('1b','GQHD2',('6620','4715')),
          '2':('2c','GQHD1',('6620','4715','3329')),
          '3':('3b','GQHD1',('6620','4715','347')),
          '4':('4b','GQHD1',('6620','4715','3329','347'))}
BAND = json.loads((DATA/'flow_band.json').read_text())
UP,LO = np.array(BAND['upper_x_P']),np.array(BAND['lower_x_P'])
GRID = np.unique(np.r_[np.linspace(2,4.6,53),UP[:,0],LO[:,0]])
PLO = np.exp(np.interp(GRID,LO[:,0],np.log(LO[:,1])))
PHI = np.exp(np.interp(GRID,UP[:,0],np.log(UP[:,1])))

def parameters(group):
    rows=list(csv.reader((DATA/'parameters-current-nine.csv').open()))
    col=rows[0].index(GROUPS[group][0])
    vals={r[0]:float(r[col]) for r in rows[2:]}
    return np.array([vals[k] for k in c.NAMES])

def pressure_score(x):
    model=c.vector_to_params(x,'pressure-refine')
    pressures=[]
    for n in GRID*BAND['reference_n0_fm3']:
        mu=c.fit.search.chemical_potentials(n/2,n/2,model)
        eps,_=c.rmf.epsilon(n/2,n/2,model)
        pressures.append(float(n*.5*sum(mu)-eps))
    prs=np.array(pressures)
    if not np.isfinite(prs).all() or np.any(prs<=0):
        raise ValueError('nonpositive/nonfinite SNM pressure in flow domain')
    logp=np.log(prs); lower=np.log(PLO);upper=np.log(PHI)
    pull=(np.maximum(lower-logp,0)+np.maximum(logp-upper,0))/((upper-lower)/2)
    integral=np.sum(.5*(pull[1:]**2+pull[:-1]**2)*np.diff(GRID))
    score=float(BAND['weight']*integral/(GRID[-1]-GRID[0]))
    return score,{'n_fm3':(GRID*BAND['reference_n0_fm3']).tolist(),
                  'pressure_MeV_fm3':prs.tolist(),'lower':PLO.tolist(),'upper':PHI.tolist(),
                  'outside_max_halfwidth':float(pull.max()),'weight':BAND['weight']}

def evaluate(payload):
    x,group,dense=payload
    _,anchor,obs=GROUPS[group]
    q=c.evaluate((x,'local',anchor,obs),final=dense)
    q['resolution']='dense' if dense else 'screening'
    q['loss_without_flow']=q['loss']
    if q.get('valid_numerics'):
        try:
            score,diag=pressure_score(x)
            q['penalties']['flow_pressure']=score
            q['loss']+=score
            q['flow']=diag
        except Exception as e:
            q['valid_numerics']=False;q['error']=str(e);q['loss']=1e15
    # The historical label does not imply a validated crust-core transition.
    q['matching_status']='original wide-window formal Maxwell; crust provenance unresolved'
    q['eligible']=bool(q.get('valid_numerics') and all(q.get('acceptance',{}).values()))
    q.pop('screening_core_eos',None)
    return q

def small(q):
    return {k:v for k,v in q.items() if k not in ('core_eos','maxwell_eos','mr_curve','stability_grid')}

def comparison(before,after):
    return {'before':small(before),'after':small(after),
            'delta_loss':after['loss']-before['loss'],
            'relative_improvement':(before['loss']-after['loss'])/before['loss'],
            'same_dense_objective':True,'updated':time.strftime('%F %T %Z')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=GROUPS,required=True)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--steps',type=int,default=8)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--baseline-only',action='store_true')
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        raise RuntimeError('Output already initialized; use a fresh directory to preserve results')
    x0=parameters(args.group)
    scale=np.maximum(.05*np.abs(x0),.005*c.WIDTH)
    low=np.maximum(c.LOW,x0-scale);high=np.minimum(c.HIGH,x0+scale)
    # Preserve the seed sign branch throughout this genuinely local run.
    def project(z):return (np.clip(x0+scale*z,low,high)-x0)/scale
    config={'group':'GQHD'+args.group,'source_candidate':GROUPS[args.group][0],
            'constraints':GROUPS[args.group][2],'anchor':GROUPS[args.group][1],
            'names':c.NAMES,'seed_parameters':x0.tolist(),'local_lower':low.tolist(),'local_upper':high.tolist(),
            'method':'16 small random probes + two starts, projected finite-difference steepest descent with line search',
            'steps':args.steps,'workers':args.workers,'pressure_band':BAND,
            'maxwell_crust_search':[1e-5,.12],'maxwell_core_search':[3e-4,.20],
            'snapshot':'2026-09-04','started':time.strftime('%F %T %Z')}
    c.atomic_json(out/'config.json',config)
    with mp.get_context('spawn').Pool(args.workers) as pool:
        before,screen=pool.map(evaluate,[(x0.tolist(),args.group,True),(x0.tolist(),args.group,False)])
        c.atomic_json(out/'before_dense.json',before);c.atomic_json(out/'before_screening.json',screen)
        if not before.get('eligible'):
            c.atomic_json(out/'status.json',{'state':'baseline_failed','error':before.get('error'),'acceptance':before.get('acceptance')})
            raise RuntimeError('Baseline did not pass the original dense acceptance gates')
        best=before
        c.atomic_json(out/'best_dense.json',best);c.atomic_json(out/'comparison.json',comparison(before,best))
        print(json.dumps({'group':args.group,'baseline_old_loss':before['loss_without_flow'],'baseline_new_loss':before['loss'],'pressure_penalty':before['penalties']['flow_pressure']}),flush=True)
        if args.baseline_only:return
        rng=np.random.default_rng(2026090800+int(args.group))
        def batch(zs):
            reports=pool.map(evaluate,[((x0+scale*project(z)).tolist(),args.group,False) for z in zs])
            with (out/'evaluations.jsonl').open('a') as f:
                for q in reports:f.write(json.dumps(small(q))+'\n')
            return reports
        trials=batch([rng.normal(0,.12,len(x0)) for _ in range(16)])
        initial=sorted([screen]+[q for q in trials if q.get('valid_numerics')],key=lambda q:q['loss'])
        states=initial[:2]
        def validate(candidates,tag):
            nonlocal best
            uniq={tuple(q['vector']):q for q in candidates if q.get('valid_numerics')}
            checks=pool.map(evaluate,[(list(x),args.group,True) for x in uniq])
            for i,q in enumerate(checks):
                c.atomic_json(out/'dense_checks'/f'{tag}_{i}.json',q)
                if q.get('eligible') and q['loss']<best['loss']:
                    best=q;c.atomic_json(out/'best_dense.json',q)
                    c.atomic_json(out/'improvements'/f'{tag}_{i}.json',q)
            c.atomic_json(out/'comparison.json',comparison(before,best))
        validate(states,'initial')
        for step in range(1,args.steps+1):
            if (out/'STOP').exists():break
            zs=[(np.array(q['vector'])-x0)/scale for q in states]
            h=globals().get('DERIVATIVE_STEP',.002)
            probes=[]
            for z in zs:
                for j in range(len(x0)):
                    v=np.zeros(len(x0));v[j]=h;probes.extend([project(z+v),project(z-v)])
            vals=batch(probes)
            candidates=[]
            for s,z in enumerate(zs):
                grad=[]
                for j in range(len(x0)):
                    k=2*(s*len(x0)+j);den=probes[k][j]-probes[k+1][j]
                    grad.append((vals[k]['loss']-vals[k+1]['loss'])/max(den,1e-12))
                grad=np.asarray(grad);direction=grad/max(np.max(np.abs(grad)),1e-12)
                candidates.extend([project(z-a*direction) for a in (.3,.1,.03,.01)])
            reps=batch(candidates)
            for s in range(len(states)):
                options=[states[s]]+[q for q in reps[4*s:4*s+4] if q.get('valid_numerics')]
                states[s]=min(options,key=lambda q:q['loss'])
            if step%2==0 or step==args.steps:validate(states,f'step_{step:03d}')
            c.atomic_json(out/'checkpoint.json',{'step':step,'states':[small(q) for q in states]})
            status={'state':'running','step':step,'steps':args.steps,'best_dense_loss':best['loss'],
                    'before_dense_loss':before['loss'],'best_screening_loss':min(q['loss'] for q in states),
                    'updated':time.strftime('%F %T %Z')}
            c.atomic_json(out/'status.json',status);print(json.dumps(status),flush=True)
        c.atomic_json(out/'status.json',{'state':'complete','best_dense_loss':best['loss'],
                    'before_dense_loss':before['loss'],'updated':time.strftime('%F %T %Z')})
        print('COMPLETE',flush=True)

if __name__=='__main__':main()
