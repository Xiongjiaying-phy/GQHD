"""Data-independent checks, four-model plots and fresh saturation calculations."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def check():
    rows = json.loads((ROOT/'audit/Summary.json').read_text())
    for row in rows:
        total = sum(v for k,v in row.items() if k not in ('model','total'))
        assert np.isclose(total,row['total'],rtol=0,atol=1e-8), row['model']
        q=json.loads((ROOT/'models'/f"{row['model']}.json").read_text())
        assert np.isclose(q['loss'],total,rtol=0,atol=1e-8)
        for key in ('stability','symmetry','thermo','mass'):
            assert row[key] == 0, (row['model'],key)
        assert q['valid_numerics']
        assert all(q['acceptance'].values())
        print(row['model'], 'saved checks PASS; loss =',total)

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    styles=json.loads((ROOT/'data/Styles.json').read_text())
    charts=[('MR','MR',1,0,'R (km)',r'M ($M_\odot$)'),
            ('Pressure','nuclear',0,3,r'n (fm$^{-3}$)',r'P (MeV fm$^{-3}$)'),
            ('Symmetry','nuclear',0,2,r'n (fm$^{-3}$)',r'$E_{sym}$ (MeV)'),
            ('Energy','nuclear',0,1,r'n (fm$^{-3}$)','E/A - M (MeV)'),
            ('Sound','sound_speed',0,1,r'n (fm$^{-3}$)',r'$c_s^2$')]
    for title,suffix,x,y,xlab,ylab in charts:
        fig,ax=plt.subplots(figsize=(6,4.5))
        for i in range(1,5):
            name=f'GQHD{i}'; a=np.loadtxt(ROOT/'data/curves'/f'{name}_{suffix}.dat')
            style=dict(styles[name]);ls=style.get('linestyle')
            if isinstance(ls,list):style['linestyle']=(ls[0],tuple(ls[1]))
            ax.plot(a[:,x],a[:,y],label=name,**style)
        ax.set(xlabel=xlab,ylabel=ylab);ax.legend();ax.grid(alpha=.15)
        if title=='MR':
            ax.set_ylim(0,3.35)
            ax.set_xlim(8.5,15.5)
        fig.tight_layout()
        for ext in ('png','pdf','svg'):fig.savefig(out/f'{title}.{ext}',dpi=180)
        plt.close(fig)
    print('Saved model-only plots to',out)

def nuclear():
    path=ROOT/'tmp/rmf_verified_convention_properties.py'
    spec=importlib.util.spec_from_file_location('release_rmf',path)
    rmf=importlib.util.module_from_spec(spec);sys.modules[spec.name]=rmf;spec.loader.exec_module(rmf)
    results={}
    for i in range(1,5):
        name=f'GQHD{i}';q=json.loads((ROOT/'models'/f'{name}.json').read_text())
        p=rmf.P(**q['parameters']);results[name]=rmf.props(p)
        print(name,results[name])
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    (out/'Nuclear.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('check','plot','nuclear'))
    globals()[ap.parse_args().action]()
