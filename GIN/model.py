from Proposed._baseline.paths import DATA_ROOT
from Proposed._baseline.paths import ROOT
import numpy as np
import torch
from torch import nn
from Proposed._baseline.common import Common

class GIN(nn.Module):
    def __init__(self, meta, defaults=None):
        super().__init__()
        self.n=int(meta['n']); self.slack=int(meta['slack_idx'])
        for k in ('G','B','non','pq','slack_state'):
            self.register_buffer(k,meta[k].clone())
        if defaults is None:
            from Proposed._baseline.defaults import default_state
            defaults=default_state(meta)
        self.register_buffer('default_angle',defaults['default_angle'].clone().double())
        self.register_buffer('default_vm',defaults['default_vm'].clone().double())
        self.H=nn.Parameter(torch.ones(3,2*self.n,dtype=torch.float64))
        self.bias=nn.Parameter(torch.zeros(3,2*self.n,dtype=torch.float64))

    def start(self,initial):
        if getattr(self,'use_dataset_initial',False):
            vm,angle=initial[...,1],initial[...,0]
            return torch.cat((vm*angle.cos(),vm*angle.sin()),-1)
        vm=torch.where(self.pq.bool(),self.default_vm,initial[...,1])
        angle=self.default_angle.expand_as(vm)
        return torch.cat((vm*angle.cos(),vm*angle.sin()),-1)

    def equations(self,z,spec,initial,jacobian=True):
        n=self.n; dtype=z.dtype; b=len(z)
        e=torch.zeros((b,n+1),device=z.device,dtype=dtype)
        f=torch.zeros_like(e)
        e[:,self.non]=z[:,:n]; f[:,self.non]=z[:,n:]
        e[:,self.slack]=self.slack_state[1]*self.slack_state[0].cos()
        f[:,self.slack]=self.slack_state[1]*self.slack_state[0].sin()
        ir=e@self.G.T-f@self.B.T; ii=e@self.B.T+f@self.G.T
        en=e[:,self.non]; fn=f[:,self.non]; irn=ir[:,self.non]; iin=ii[:,self.non]
        p=en*irn+fn*iin; q=fn*irn-en*iin
        a=torch.cat((p-spec[...,0],torch.where(self.pq.bool(),q-spec[...,1],en.square()+fn.square()-initial[...,1].square())),-1)
        if not jacobian:return a
        g=self.G[self.non][:,self.non]; bb=self.B[self.non][:,self.non]
        pe=en[:,:,None]*g+fn[:,:,None]*bb+torch.diag_embed(irn)
        pf=-en[:,:,None]*bb+fn[:,:,None]*g+torch.diag_embed(iin)
        qe=fn[:,:,None]*g-en[:,:,None]*bb-torch.diag_embed(iin)
        qf=-fn[:,:,None]*bb-en[:,:,None]*g+torch.diag_embed(irn)
        pq=self.pq.bool()[None,:,None]
        qe=torch.where(pq,qe,torch.diag_embed(2*en));qf=torch.where(pq,qf,torch.diag_embed(2*fn))
        j=torch.cat((torch.cat((pe,pf),-1),torch.cat((qe,qf),-1)),1)
        return a,j

    def rectangular(self,initial,spec):
        z=self.start(initial)
        for layer in range(3):
            a,j=self.equations(z,spec,initial)
            step=torch.linalg.solve(j,-a.unsqueeze(-1)).squeeze(-1)
            z=z+self.H[layer]*step+self.bias[layer]
        return z

    def forward(self,initial,spec):
        z=self.rectangular(initial,spec);e=z[:,:self.n];f=z[:,self.n:]
        # No PV projection or extra refinement is added to the original GIN.
        return torch.stack((torch.atan2(f,e),torch.sqrt(e.square()+f.square())),-1)

    @torch.inference_mode()
    def sparse_cpu(self,initial,spec):
        """Algebraically identical CPU inference with sparse SuperLU solves.

        Training still uses differentiable torch solves. Sparse Jacobians exploit
        network structure and avoid benchmarking GIN with a dense-only handicap.
        """
        from scipy.sparse import csr_matrix,diags,bmat
        from scipy.sparse.linalg import splu
        if not hasattr(self,'_sparse'):
            non=self.non.numpy();g=csr_matrix(self.G.numpy());b=csr_matrix(self.B.numpy())
            self._sparse=(g,b,g[non][:,non],b[non][:,non],diags(self.pq.numpy()),diags(1-self.pq.numpy()))
        g,b,gn,bn,pq,pv=self._sparse;non=self.non.numpy();n=self.n
        zs=self.start(initial).numpy().copy();ss=spec.numpy();vm=initial[...,1].numpy()
        hh=self.H.numpy();bias=self.bias.numpy();out=[]
        slack=self.slack_state.numpy();sv=slack[1]*np.exp(1j*slack[0])
        for z,power,vset in zip(zs,ss,vm):
            try:
                for layer in range(3):
                    e=np.zeros(n+1);f=np.zeros(n+1);e[non]=z[:n];f[non]=z[n:];e[self.slack]=sv.real;f[self.slack]=sv.imag
                    ir=(g@e-b@f)[non];ii=(b@e+g@f)[non];en=e[non];fn=f[non]
                    pa=en*ir+fn*ii;qa=fn*ir-en*ii
                    a=np.r_[pa-power[:,0],np.where(self.pq.numpy().astype(bool),qa-power[:,1],en**2+fn**2-vset**2)]
                    pe=diags(en)@gn+diags(fn)@bn+diags(ir)
                    pf=-diags(en)@bn+diags(fn)@gn+diags(ii)
                    qe=pq@(diags(fn)@gn-diags(en)@bn-diags(ii))+pv@diags(2*en)
                    qf=pq@(-diags(fn)@bn-diags(en)@gn+diags(ir))+pv@diags(2*fn)
                    j=bmat([[pe,pf],[qe,qf]],format='csc')
                    z=z+hh[layer]*splu(j).solve(-a)+bias[layer]
                out.append(np.stack((np.arctan2(z[n:],z[:n]),np.hypot(z[:n],z[n:])),-1))
            except RuntimeError:out.append(np.full((n,2),np.nan))
        return torch.from_numpy(np.stack(out))

