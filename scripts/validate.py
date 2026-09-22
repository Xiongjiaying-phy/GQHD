"""Release smoke tests; optional objective checks require the historical BPS."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import reproduce

root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--objective',action='store_true');args=ap.parse_args()
report={'source_parse':True,'saved_results':True,'fresh_nuclear':{},'objective':{},
        'full_optimization_rerun':False,'full_gw_recalculation':False}
for p in root.rglob('*.py'):
    ast.parse(p.read_text(),filename=str(p.relative_to(root)))
reproduce.check()
reproduce.nuclear()
fresh=json.loads((root/'generated/Nuclear.json').read_text())
for name,props in fresh.items():
    saved=json.loads((root/'models'/f'{name}.json').read_text())['nuclear_properties']
    errors={k:abs(float(v)-float(saved[k])) for k,v in props.items()
            if k in saved and isinstance(v,(float,int))}
    # Numerical differentiation can vary slightly between platforms.
    assert all(e<.02 for e in errors.values()),(name,errors)
    report['fresh_nuclear'][name]=errors
if args.objective:
    import run_gqhd_pressure_refine as r
    for i in range(1,5):
        name=f'GQHD{i}';q=json.loads((root/'models'/f'{name}.json').read_text())
        x=np.array([q['parameters'][k] for k in r.c.NAMES])
        flow,_=r.pressure_score(x)
        nuclear=r.c.nuclear_penalty(fresh[name])
        mr=q['mr_curve'];m=np.array([p['mass_Msun'] for p in mr]);rad=np.array([p['radius_km'] for p in mr])
        obs=sum(r.c.fit.observation_term(m,rad,r.c.fit.OBSERVATIONS[k])['chi2'] for k in r.GROUPS[str(i)][2])
        errors={'flow':abs(flow-q['penalties']['flow_pressure']),
                'nuclear':abs(nuclear-q['penalties']['nuclear']),
                'MR':abs(obs-q['observation_chi2'])}
        assert max(errors.values())<1e-5,(name,errors)
        report['objective'][name]=errors
(root/'generated/Validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: validation written to generated/Validation.json')
