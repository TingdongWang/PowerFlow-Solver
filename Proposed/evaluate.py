"""Complete test, per-sample CSV and independent complex128 PF checks."""
import csv
import math
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch

from Function import PowerFlow, rollout, summarize
from runtime import sync, json_write


@torch.inference_mode()
def evaluate(model,pf,data,config,log=None,return_state=False):
    model.eval()
    device = next(model.parameters()).device
    batch = config['evaluation_batch_size']
    offset = 0
    residuals,counts,states = [],[],[]
    sync(device)
    start = time.perf_counter()
    while offset < len(data['initial']):
        try:
            initial = data['initial'][offset:offset+batch].to(device)
            spec = data['spec'][offset:offset+batch].to(device)
            state,r,c = rollout(model,pf,initial,spec,calls=6*config['inference']['max_blocks'],tol=1e-4)
            residuals.append(r.cpu())
            counts.append(c.cpu())
            if return_state:
                states.append(state.cpu())
            offset += len(initial)
        except torch.OutOfMemoryError:
            raise
    sync(device)
    r = torch.cat(residuals) if residuals else torch.empty(0,dtype=torch.float64)
    c = torch.cat(counts) if counts else torch.empty(0,dtype=torch.long)
    stats = summarize(r)
    stats.update(seconds=time.perf_counter()-start,mean_calls=float(c.double().mean()) if len(c) else 0.,
                 max_calls=int(c.max()) if len(c) else 0,effective_batch=batch,
                 timing_scope='wall time including batch transfers/output copies, no file IO; use benchmark for warmed compute timings')
    if return_state:
        return stats,r,c,torch.cat(states) if states else data['initial'][:0].clone()
    return stats,r,c


def voltage_error_metrics(state, label):
    """All non-slack buses and samples; no convergence-based filtering."""
    if state.shape != label.shape or state.ndim != 3 or state.shape[-1] != 2 or not state.numel():
        raise ValueError('Expected matching nonempty [samples, non-slack buses, 2] tensors')
    state = state.detach().double()
    label = label.detach().to(device=state.device, dtype=torch.float64)
    angle = state[..., 0] - label[..., 0]
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))
    voltage = state[..., 1] - label[..., 1]
    nonfinite = ~(torch.isfinite(state).all(-1) & torch.isfinite(label).all(-1))
    # Retain failed outputs in the denominator; never silently drop them.
    angle = angle.masked_fill(nonfinite, float('inf'))
    voltage = voltage.masked_fill(nonfinite, float('inf'))
    return {'angle_mae': float(angle.abs().mean()),
            'angle_rmse': float(angle.square().mean().sqrt()),
            'voltage_mae': float(voltage.abs().mean()),
            'voltage_rmse': float(voltage.square().mean().sqrt()),
            'nonfinite_bus_states': int(nonfinite.sum()),
            'scope': 'all samples and all non-slack buses, including PV'}


def independent_residual(meta,state,spec):
    from scipy.sparse import csr_matrix
    non = meta['non'].numpy()
    full = np.empty((len(state),meta['n']+1),dtype=np.complex128)
    s = state.numpy()
    full[:,non] = s[...,1]*np.exp(1j*s[...,0])
    slack = meta['slack_state'].numpy()
    full[:,meta['slack_idx']] = slack[1]*np.exp(1j*slack[0])
    Y = csr_matrix(meta['G'].numpy()+1j*meta['B'].numpy())
    power = full*np.conj((Y@full.T).T)
    actual = np.stack((power[:,non].real,power[:,non].imag),-1)
    residual = spec.numpy()-actual
    residual[...,1] *= meta['pq'].numpy()
    return torch.from_numpy(residual)


def write_csv(path,data,r,c):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id','x_signed','pq_signed','max_pf_mismatch','success_lt_1e-4','neural_calls','blocks','x_low','x_high','pq_low','pq_high'])
        for i,(residual,calls) in enumerate(zip(r.tolist(),c.tolist())):
            writer.writerow([int(data['sample_id'][i]),float(data['level'][i,4]),float(data['level'][i,5]),
                residual,math.isfinite(residual) and residual<1e-4,calls,math.ceil(calls/6),
                *[float(v) for v in data['level'][i,:4]]])


def full_test(model,meta,test,config,out,log):
    from data import subset
    if int((~test['hard']).sum()) != 4000 or int(test['hard'].sum()) != 100:
        raise ValueError('Expected complete 4000 heatmap + 100 hard samples')
    device = next(model.parameters()).device
    pf = PowerFlow(meta).to(device)
    report = {'system':config['system'],'source_precision':meta['source_precision'],
              'criterion':'per-sample max |P mismatch(non-slack), Q mismatch(PQ)| strictly <1e-4',
              'start':'dataset_initial_from_csv',
              'max_blocks':config['inference']['max_blocks'],'calls_per_block':6,'neural_only':True,
              'network_dtype':'float32, TF32 disabled','physics_dtype':'float64'}
    for name,select in [('heatmap',~test['hard']),('ill',test['hard'])]:
        group = subset(test,select)
        stats,r,c,state = evaluate(model,pf,group,config,log,return_state=True)
        stats['voltage_accuracy'] = voltage_error_metrics(state, group['label'])
        write_csv(out/(name+'.csv'),group,r,c)
        independent = independent_residual(meta,state,group['spec']).abs().flatten(1).amax(1)
        if not torch.allclose(independent,r,atol=1e-8,rtol=1e-10,equal_nan=True):
            raise RuntimeError('Independent complex128 PF implementation disagrees')
        if not torch.equal(state[:,~meta['pq'].bool(),1],group['initial'][:,~meta['pq'].bool(),1]):
            raise RuntimeError('PV boundary voltage changed')
        if not torch.equal(torch.isfinite(r)&(r<1e-4),torch.isfinite(independent)&(independent<1e-4)):
            raise RuntimeError('Independent PF pass/fail differs at threshold')
        stats['independent_complex128'] = summarize(independent)
        finite = torch.isfinite(r) & torch.isfinite(independent)
        stats['independent_max_difference'] = float((r[finite]-independent[finite]).abs().max()) if finite.any() else None
        report[name] = stats
        log('full_test',name=name,**stats)
    if report['heatmap']['n'] != 4000 or report['ill']['n'] != 100:
        raise ValueError('Expected complete 4000 heatmap + 100 hard samples; no full-data acceptance claim allowed')
    # Disable common solver interfaces, then execute a separate small rollout.
    interfaces = ['torch.linalg.'+k for k in ('solve','solve_ex','inv','inv_ex','pinv','lstsq','lu','lu_factor','lu_solve','cholesky')]
    interfaces += ['numpy.linalg.solve','numpy.linalg.inv','scipy.sparse.linalg.spsolve','scipy.sparse.linalg.splu']
    with ExitStack() as stack:
        for interface in interfaces:
            stack.enter_context(patch(interface,side_effect=RuntimeError('Forbidden online solver '+interface)))
        with torch.inference_mode():
            rollout(model,pf,test['initial'][:17].to(device),test['spec'][:17].to(device),calls=6*config['inference']['max_blocks'])
    report['solver_guard'] = {'passed':True,'samples':17,'interfaces':interfaces}
    report['accuracy_target_met'] = report['heatmap']['success']>=3400 and report['heatmap']['max_calls']<=60
    json_write(out/'summary.json',report)
    return report
