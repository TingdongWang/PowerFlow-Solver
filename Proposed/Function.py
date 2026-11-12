"""Float64 PF arithmetic and a bounded single neural rollout."""
import torch
from torch import nn

class PowerFlow(nn.Module):
    def __init__(self,meta):
        super().__init__()
        self.n=int(meta['n'])
        self.slack=int(meta['slack_idx'])
        for name in ('G','B','non','pq','slack_state'):
            self.register_buffer(name,meta[name].clone())
        nz=(meta['G'][meta['non']]!=0)|(meta['B'][meta['non']]!=0)
        degree=int(nz.sum(1).max())
        neighbors=torch.zeros(self.n,degree,dtype=torch.long)
        g=torch.zeros(self.n,degree,dtype=torch.float64)
        b=torch.zeros_like(g)
        for i,full in enumerate(meta['non']):
            js=torch.where(nz[i])[0]
            neighbors[i,:len(js)]=js
            g[i,:len(js)]=meta['G'][full,js]
            b[i,:len(js)]=meta['B'][full,js]
        self.register_buffer('neighbors',neighbors)
        self.register_buffer('g',g)
        self.register_buffer('b',b)

    def forward(self,state):
        slack=self.slack_state.expand(state.shape[0],1,2)
        full=torch.cat((state[:,:self.slack],slack,state[:,self.slack:]),1)
        other=full[:,self.neighbors]
        angle=state[...,0,None]-other[...,0]
        scale=state[...,1,None]*other[...,1]
        c,s=angle.cos(),angle.sin()
        p=(scale*(self.g*c+self.b*s)).sum(-1)
        q=(scale*(self.g*s-self.b*c)).sum(-1)
        return torch.stack((p,q),-1)

    def residual(self,state,spec):
        r=spec-self(state)
        return torch.stack((r[...,0],r[...,1]*self.pq),-1)

def padded_rollout(model,pf,initial,spec,calls=60,tol=1e-4,early_stop=True):
    assert 1<=calls<=60,'At most ten six-correction blocks'
    state=initial.clone()
    residual=pf.residual(state,spec)
    maximum=residual.abs().flatten(1).amax(1)
    active=torch.isfinite(maximum) & (maximum>=tol)
    used=torch.zeros(len(state),device=state.device,dtype=torch.long)
    beta=float(getattr(model,'momentum_beta',0.0))
    momentum_start=int(getattr(model,'momentum_start_call',6))
    assert 0.0<=beta<1.0
    velocity=torch.zeros_like(state) if beta else None
    for k in range(calls):
        if early_stop and not bool(active.any()): break
        if k==0:
            candidate=model.initialize(state,spec)
        else:
            delta=model(state,residual)
            if beta and k>=momentum_start:
                velocity=delta+beta*velocity
                delta=velocity
            candidate=state+delta
        state=torch.where(active[:,None,None],candidate,state)
        used=used+active.long()
        residual=pf.residual(state,spec)
        maximum=residual.abs().flatten(1).amax(1)
        active=active & torch.isfinite(maximum) & (maximum>=tol)
        if early_stop and not bool(active.any()): break
    return state,maximum,used

def summarize(values):
    r=values.detach().double().cpu()
    success=torch.isfinite(r)&(r<1e-4)
    finite=r[torch.isfinite(r)]
    if not len(r):
        return {'n':0,'success':0,'rate':None,'mean':None,'p50':None,'p90':None,'p99':None,'max':None,'nonfinite':0}
    return {'n':len(r),'success':int(success.sum()),'rate':float(success.double().mean()),
            'mean':float(finite.mean()) if len(finite) else None,
            'p50':float(finite.quantile(.5)) if len(finite) else None,
            'p90':float(finite.quantile(.9)) if len(finite) else None,
            'p99':float(finite.quantile(.99)) if len(finite) else None,
            'max':float(finite.max()) if len(finite) else None,
            'nonfinite':int((~torch.isfinite(r)).sum()),'statistics_scope':'finite samples; nonfinite always fail'}


def rollout(model,pf,initial,spec,calls=60,tol=1e-4,early_stop=True):
    """Compact converged samples; identical per-sample update order and budget.

    Calls include one initializer and at most calls-1 linear corrections.
    No online Jacobian, factorization, inverse, or linear-system solution.
    """
    if not early_stop:
        return padded_rollout(model,pf,initial,spec,calls,tol,early_stop=False)
    if not 1<=calls<=60:raise ValueError('calls must be in [1,60]')
    state=initial.clone();r=pf.residual(state,spec);maximum=r.abs().flatten(1).amax(1)
    used=torch.zeros(len(state),device=state.device,dtype=torch.long)
    ids=torch.where(torch.isfinite(maximum)&(maximum>=tol))[0]
    if not len(ids):return state,maximum,used
    x=model.initialize(state[ids],spec[ids]);used[ids]+=1;state[ids]=x
    r=pf.residual(x,spec[ids]);mx=r.abs().flatten(1).amax(1);maximum[ids]=mx
    keep=torch.isfinite(mx)&(mx>=tol);ids=ids[keep];x=x[keep];r=r[keep]
    if not len(ids):return state,maximum,used
    target=spec[ids];velocity=torch.zeros_like(x)
    beta=float(model.momentum_beta);onset=int(model.momentum_start_call)
    for correction in range(1,calls):
        delta=model(x,r)
        if beta and correction>=onset:
            velocity=delta+beta*velocity;delta=velocity
        x=x+delta;used[ids]+=1;state[ids]=x
        r=pf.residual(x,target);mx=r.abs().flatten(1).amax(1);maximum[ids]=mx
        keep=torch.isfinite(mx)&(mx>=tol)
        if not bool(keep.any()):break
        ids=ids[keep];x=x[keep];r=r[keep];target=target[keep];velocity=velocity[keep]
    return state,maximum,used
