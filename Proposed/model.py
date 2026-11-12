"""MLP initializer + bias-free NN correction, at most 60 calls per rollout.
Online execution contains no Jacobian, factorization or equation solve.
"""
import torch
from torch import nn

class RJMNC(nn.Module):
    def __init__(self,meta):
        super().__init__()
        self.n=int(meta['n'])
        self.momentum_beta=float(meta.get('momentum_beta',.7))
        self.momentum_start_call=int(meta.get('momentum_start_call',6))
        for key in ('nm','ns','om','ost','pq'):
            self.register_buffer(key,meta[key].clone().float())
        self.include_state=bool(meta.get('include_state',False))
        width=int(meta.get('initializer_width',1536))
        self.initializer=nn.Sequential(nn.Linear(self.n*(4 if self.include_state else 2),width),nn.SiLU(),nn.LayerNorm(width),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,self.n*2))
        nn.init.zeros_(self.initializer[-1].weight)
        nn.init.zeros_(self.initializer[-1].bias)
        self.corrector=nn.Linear(self.n*2,self.n*2,bias=False)
        nn.init.zeros_(self.corrector.weight)
        self.register_buffer('rscale',self.ns[2:4].repeat(self.n))
        self.register_buffer('dscale',self.ost.repeat(self.n))
        self.use_gain=False

    def condition(self,spec,initial=None):
        p=(spec[...,0].float()-self.nm[0])/self.ns[0]
        q=((spec[...,1].float()-self.nm[1])/self.ns[1])*self.pq
        features=torch.stack((p,q),-1)
        if self.include_state:
            if initial is None: raise ValueError('Raw initial state is required')
            voltage=(initial.float()-self.nm[4:6])/self.ns[4:6]
            features=torch.cat((features,voltage),-1)
        return features.flatten(1)

    def initialize(self,initial,spec):
        predicted=self.initializer(self.condition(spec,initial)).reshape(-1,self.n,2)
        predicted=predicted.double()*self.ns[4:6].double()+self.nm[4:6].double()
        return torch.stack((predicted[...,0],torch.where(self.pq.bool(),predicted[...,1],initial[...,1])),-1)

    def forward(self,state,residual):
        r=residual.float().flatten(1)/self.rscale
        d=(self.corrector(r)*self.dscale).reshape(-1,self.n,2).double()
        return torch.stack((d[...,0].clamp(-.2,.2),d[...,1].clamp(-.1,.1)*self.pq),-1)
