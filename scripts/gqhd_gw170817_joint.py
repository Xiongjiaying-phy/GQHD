"""Joint GW170817 mass/tidal posterior recycling and legacy-Maxwell Love.

The observational prior is uniform in ordered detector masses and in each
Lambda in [0,5000] (P1800061, II.D). We adopt that same detector-mass prior
for the EOS event integral. In (Mc_det,q), posterior density contains the
mass-coordinate Jacobian J; dividing by the PE prior J and integrating
with the same mass prior J cancel. Thus integrate the 4D density on the EOS
surface over dMc_det dq, up to one EOS-independent constant. No fixed Mc,
no Lambda14 box, no tilde-Lambda projection, no EOS-dependent renormalizing.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'nuclear_astro_posterior_20260720'))
from src.tov_love import MEVFM3_TO_KM2
from src.maxwell_eos import integrate_star_maxwell

class LinearBranch:
    """Exactly the original piecewise-linear epsilon(P), with its derivative."""
    def __init__(self,rows):
        rows=sorted(rows,key=lambda q:q['pressure_total_mev_fm3'])
        self.p_mev=np.array([q['pressure_total_mev_fm3'] for q in rows])
        self.p=self.p_mev*MEVFM3_TO_KM2
        self.e=np.array([q['epsilon_total_mev_fm3'] for q in rows])*MEVFM3_TO_KM2
        if not (np.all(np.diff(self.p)>0) and np.all(np.diff(self.e)>0)):
            raise ValueError('nonmonotonic legacy EOS branch; no silent cleanup')
        self.slope=np.diff(self.e)/np.diff(self.p)
    def index(self,p):return int(np.clip(np.searchsorted(self.p,p,side='right')-1,0,len(self.p)-2))
    def epsilon(self,p):
        i=self.index(p);return float(self.e[i]+self.slope[i]*(p-self.p[i]))
    def cs2(self,p):return float(1/self.slope[self.index(p)])

class LegacyLoveEOS:
    def __init__(self,low,high,metadata):
        self.crust=LinearBranch(low);self.core=LinearBranch(high)
        self.transition=SimpleNamespace(delta_epsilon_MeV_fm3=metadata['energy_density_jump_MeV_fm3'])
        self.p_transition=metadata['pressure_transition_MeV_fm3']*MEVFM3_TO_KM2
        self.p=np.r_[self.crust.p,self.core.p[1:]]
        self.p_mev=np.r_[self.crust.p_mev,self.core.p_mev[1:]]
    def branch(self,p):return self.core if p>self.p_transition else self.crust

def make_eos(report,bps):
    mt=report['diagnostics']['crust_core_matching']
    if 'maxwell_eos' in report:
        mx=report['maxwell_eos'];return LegacyLoveEOS(mx['low_branch'],mx['high_branch'],mt)
    low=[]
    for r in bps:
        mu=(r['energy_density_mev_fm3']+r['pressure_mev_fm3'])/r['rho_b']
        if mu<mt['mu_transition_MeV'] and r['pressure_mev_fm3']<mt['pressure_transition_MeV_fm3']:
            low.append({'pressure_total_mev_fm3':r['pressure_mev_fm3'],'epsilon_total_mev_fm3':r['energy_density_mev_fm3']})
    low.append({'pressure_total_mev_fm3':mt['pressure_transition_MeV_fm3'],'epsilon_total_mev_fm3':mt['epsilon_crust_MeV_fm3']})
    high=[{'pressure_total_mev_fm3':mt['pressure_transition_MeV_fm3'],'epsilon_total_mev_fm3':mt['epsilon_core_MeV_fm3']}]+report['screening_core_eos']
    return LegacyLoveEOS(low,high,mt)

def tidal_curve(eos,dense=False):
    pc=np.geomspace(1.,.98*float(eos.core.p_mev[-1]),64 if dense else 28)
    stars=[integrate_star_maxwell(eos,float(t),rtol=2e-7 if dense else 2e-5,
               atol=1e-12,max_step_km=.03 if dense else .10) for t in pc]
    mass=np.array([s.mass_Msun for s in stars]);end=int(mass.argmax())+1
    stars=stars[:end];mass=mass[:end]
    if np.any(np.diff(mass)<=0):raise ValueError('non-single stable branch in tidal curve')
    lam=np.array([s.Lambda for s in stars])
    if not (np.isfinite(lam).all() and np.all(lam>0) and mass[0]<1.0 and mass[-1]>1.9):
        raise ValueError('tidal curve invalid or insufficient mass support')
    return {'mass':mass.tolist(),'Lambda':lam.tolist(),'radius':[s.radius_km for s in stars],
            'pc':[s.central_pressure_MeV_fm3 for s in stars],
            'Lambda14':float(np.exp(PchipInterpolator(mass,np.log(lam))(1.4)))}

class GW170817Joint:
    def __init__(self,bandwidth=.30):
        path=ROOT/'output/gqhd_tidal_refine_20260908/data/low_spin_PhenomPNRT_posterior_samples.dat.gz'
        a=np.loadtxt(path,skiprows=1);m1,m2=a[:,2],a[:,3]
        mc=(m1*m2)**.6/(m1+m2)**.2
        self.samples=np.column_stack([mc,m2/m1,a[:,4],a[:,5]])
        self.kde=gaussian_kde(self.samples.T,bw_method=bandwidth)
        self.bandwidth=bandwidth
        pad=6*math.sqrt(self.kde.covariance[0,0])
        self.mc_bounds=(float(mc.min()-pad),float(mc.max()+pad))
        self.z=.0099 # The original paper's geocentric host-redshift conversion.
    def density(self,points):
        points=np.asarray(points,float);out=np.zeros(len(points))
        # Preserve joint covariance. Reflect queries, not the covariance fit.
        for qflip in (False,True):
            for l1flip in (False,True):
                for l2flip in (False,True):
                    z=points.copy()
                    if qflip:z[:,1]=2-z[:,1]
                    if l1flip:z[:,2]*=-1
                    if l2flip:z[:,3]*=-1
                    out+=self.kde(z.T)
        return out
    def log_integral(self,curve,nmc=32,nq=96):
        mass=np.asarray(curve['mass']);lam=np.asarray(curve['Lambda'])
        f=PchipInterpolator(mass,np.log(lam),extrapolate=False)
        xm,wm=np.polynomial.legendre.leggauss(nmc);xq,wq=np.polynomial.legendre.leggauss(nq)
        lo,hi=self.mc_bounds
        mc=(lo+hi)/2+(hi-lo)*xm/2;qq=.55+.45*xq
        mc,q=np.meshgrid(mc,qq,indexing='ij');weights=np.outer(wm*(hi-lo)/2,wq*.45)
        mc,q,weights=mc.ravel(),q.ravel(),weights.ravel()
        m1=mc*(1+q)**.2/q**.6;m2=q*m1
        l1=np.exp(f(m1/(1+self.z)));l2=np.exp(f(m2/(1+self.z)))
        valid=np.isfinite(l1+l2)&(l1>=0)&(l1<=5000)&(l2>=0)&(l2<=5000)&(m2>=.5)&(m1<=7.7)&(mc>=1.184)&(mc<=2.168)
        d=self.density(np.column_stack([mc[valid],q[valid],l1[valid],l2[valid]]))
        keep=d>0
        if not np.any(keep):raise ValueError('zero GW EOS-track likelihood')
        return float(logsumexp(np.log(d[keep])+np.log(weights[valid][keep])))
