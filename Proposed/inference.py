"""Load a plain MLP solver and solve inputs without loading datasets or fitting."""
import runtime
import torch
from model import RJMNC
from Function import PowerFlow,rollout

def load_solver(checkpoint_path,device='cpu'):
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=RJMNC(saved['meta']);model.load_state_dict(saved['state_dict'],strict=True)
    model.eval().to(device);pf=PowerFlow(saved['meta']).to(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return model,pf

@torch.inference_mode()
def solve(model,pf,initial,spec,max_blocks=10):
    """Return state [B,N-1,2], maximum mismatch [B], neural-call count [B].

    initial: angles in radians, voltage magnitudes in p.u.; spec: P,Q in p.u.
    Rows follow checkpoint metadata non; PV Q values are ignored. PV magnitudes
    and slack voltage remain prescribed. A 2D single sample is also accepted.
    The first neural call is initialization; remaining calls are corrections.
    """
    device=next(model.parameters()).device
    x=torch.as_tensor(initial,dtype=torch.float64,device=device)
    s=torch.as_tensor(spec,dtype=torch.float64,device=device)
    if x.ndim==2:x=x.unsqueeze(0)
    if s.ndim==2:s=s.unsqueeze(0)
    if x.ndim!=3 or x.shape!=s.shape or x.shape[1:]!=(model.n,2):
        raise ValueError('Expected [batch,N-1,2] or [N-1,2] in checkpoint bus order')
    if not len(x) or not torch.isfinite(x).all() or not torch.isfinite(s).all():
        raise ValueError('Inputs must be nonempty and finite')
    if type(max_blocks) is not int or not 1<=max_blocks<=10:
        raise ValueError('max_blocks must be an integer in [1,10]')
    return rollout(model,pf,x,s,calls=6*max_blocks,tol=1e-4)
