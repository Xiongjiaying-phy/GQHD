"""Continue the four flow-refined seeds with the complete 4D GW event term."""
import json
import time
import numpy as np
import run_gqhd_pressure_refine as r
import gqhd_gw170817_joint as gw

MODEL=None
PREFLIGHT=gw.ROOT/'output/gqhd_tidal_refine_20260908/preflight.json'
if not json.loads(PREFLIGHT.read_text()).get('passed'):
    raise RuntimeError('Tidal preflight has not passed')

def parameters(group):
    path=gw.ROOT/f'output/gqhd_pressure_refine_20260908/results/GQHD{group}/best_dense.json'
    q=json.loads(path.read_text())
    return np.array([q['parameters'][name] for name in r.c.NAMES])

def evaluate(payload):
    global MODEL
    x,group,dense=payload
    _,anchor,obs=r.GROUPS[group]
    q=r.c.evaluate((x,'local',anchor,obs),final=dense)
    q['resolution']='dense' if dense else 'screening'
    q['loss_without_flow']=q['loss']
    q['matching_status']='original wide-window formal Maxwell; same piecewise-linear EOS; explicit Love y jump'
    q['eligible']=bool(q.get('valid_numerics') and all(q.get('acceptance',{}).values()))
    if not q['eligible']:
        q['loss']=max(q['loss'],1.e12);q.pop('screening_core_eos',None);return q
    try:
        score,diag=r.pressure_score(x)
        q['flow']=diag;q['penalties']['flow_pressure']=score;q['loss']+=score
        q['loss_without_gw']=q['loss']
        eos=gw.make_eos(q,r.c.BPS_ROWS)
        curve=gw.tidal_curve(eos,dense=dense)
        if MODEL is None:MODEL=gw.GW170817Joint(.30)
        logl=MODEL.log_integral(curve,32,96) # Same joint quadrature in fast/dense.
        q['penalties']['GW170817']=-2*logl;q['loss']-=2*logl
        q['tidal_curve']=curve;q['logL_GW170817']=logl
        q['diagnostics']['Lambda14']=curve['Lambda14']
        if dense:
            hi=MODEL.log_integral(curve,64,192)
            q['gw_quadrature_logL_high']=hi
            q['gw_quadrature_error']=abs(logl-hi)
            if abs(logl-hi)>.02:raise ValueError('joint GW mass quadrature failed convergence')
    except Exception as e:
        q['valid_numerics']=False;q['eligible']=False;q['loss']=1e15;q['error']=f'{type(e).__name__}: {e}'
    q.pop('screening_core_eos',None)
    return q

original_atomic=r.c.atomic_json
def atomic(path,obj):
    if path.name=='config.json':
        obj.update({'stage':'GW170817 joint posterior refinement after flow refinement',
             'snapshot':'four best_dense.json from completed flow refinement 2026-09-08',
             'source_candidate':'GQHD'+str(obj['group'][-1])+' flow-refined best',
             'GW170817':{'source':'https://dcc.ligo.org/LIGO-P1800061/public','branch':'low_spin_PhenomPNRT',
               'dimensions':['Mc_detector','q','Lambda1','Lambda2'],'bandwidth':.30,
               'geocentric_redshift':.0099,'mass_quadrature':[32,96],'dense_check_quadrature':[64,192],
               'PE_prior':'uniform ordered detector masses, independent uniform Lambda_i in [0,5000]',
               'event_mass_prior':'same as PE; Jacobian cancels against prior correction in the event integral',
               'score':'-2 log integral, single common arbitrary additive constant omitted; no clipping',
               'nuisances':'spins, distance, inclination etc marginalized with original released low-spin PE priors; no strain reanalysis'},
             'tidal_solver':'coupled TOV-Love, exact original linear epsilon(P) tables and finite-pressure y jump',
             'derivative_step_normalized':.02,'preflight':str(PREFLIGHT)})
    original_atomic(path,obj)

r.parameters=parameters
r.evaluate=evaluate
r.DERIVATIVE_STEP=.02
r.c.atomic_json=atomic
if __name__=='__main__':r.main()
