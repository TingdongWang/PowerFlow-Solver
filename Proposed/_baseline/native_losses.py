import torch
from Proposed._baseline.common import complex_state

class ProxRMS(torch.optim.Optimizer):
    """RMS-scaled proximal gradient, with an explicitly chosen L1 coefficient."""
    def __init__(self, params, lr=1e-4, decay=.99, l1=1e-8):
        super().__init__(params, dict(lr=lr, decay=decay, l1=l1))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                st = self.state[p]
                if not st: st['v'] = torch.zeros_like(p)
                st['v'].mul_(group['decay']).addcmul_(p.grad, p.grad, value=1-group['decay'])
                step = group['lr']/(st['v'].sqrt()+1e-8)
                updated = p-step*p.grad
                p.copy_(updated.sign()*torch.relu(updated.abs()-step*group['l1']))

def pinn_loss(model,d,ids,epoch,meta):
    pred=model(d['initial'][ids],d['spec'][ids]);z=complex_state(pred);truth=complex_state(d['label'][ids])
    supervised=(z-truth).abs().square().mean()
    schedule=meta['pinn_schedule']
    if epoch <= schedule['physics_warmup_epochs']:
        return supervised
    pq=meta['pq'].bool();non=meta['non']
    diag=torch.complex(meta['G'].diagonal()[non][pq],meta['B'].diagonal()[non][pq])
    power=torch.complex(d['spec'][ids,:,0][:,pq],d['spec'][ids,:,1][:,pq])
    psi=(power/truth[:,pq]).conj()-diag*truth[:,pq]
    physical_voltage=((power/z[:,pq]).conj()-psi)/diag
    physical=(physical_voltage-truth[:,pq]).abs().square().mean()
    progress=max(0,epoch-schedule['physics_warmup_epochs'])/max(1,schedule['epochs']-schedule['physics_warmup_epochs'])
    weight=schedule['physics_weight']*progress
    return (1-weight)*supervised+weight*physical