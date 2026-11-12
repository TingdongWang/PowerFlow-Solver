"""Evaluate newly trained methods on all 4100 cases, never fit in evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Proposed._baseline.paths import ROOT,DATA_ROOT
import argparse,json,csv,time,hashlib,math
import torch
from Function import PowerFlow,summarize,rollout
from evaluate import independent_residual
from Proposed._baseline.defaults import default_state
METHODS=('Proposed','GIN','UPINN','PINN4PF')

def run(method,system,device='cuda',config=None):
    if method not in METHODS: raise ValueError(f'Unsupported method: {method}')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from Proposed._baseline.data import load_data
    base=ROOT/'Proposed' if method=='Proposed' else ROOT/method
    out=base/'Results'/system
    out.mkdir(parents=True,exist_ok=True)
    if config is None:config=json.loads((out/(system+'.json')).read_text())
    meta,groups,data_report=load_data(config,lambda event,**fields:print(json.dumps({'event':event,**fields}),flush=True))
    data={'signature':data_report['signature']};test=groups['test'];path=DATA_ROOT/system
    from runtime import json_write
    json_write(out/'data_report.json',data_report)
    out.mkdir(parents=True,exist_ok=True)
    defaults=default_state(meta)
    initial=test['initial'].clone()
    assert torch.equal(initial,test['initial'])
    pf=PowerFlow(meta).to(device)
    checkpoint=None
    if method=='Proposed':
        from inference import load_solver
        checkpoint=out/'model.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for key in ('G','B','non','pq','slack_state'):
            assert torch.equal(saved['meta'][key],meta[key]),'Checkpoint physical metadata differs from CSV: '+key
        model,pf=load_solver(checkpoint,device=device)
    else:
        checkpoint=out/'model.pt';c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert c['signature']==data['signature']
        if method=='GIN':
            from GIN.model import GIN
            model=GIN(meta,defaults=defaults)
            
            with torch.no_grad():model.H.copy_(c['state_dict']['H']);model.bias.copy_(c['state_dict']['bias'])
            model=model.double();model.use_dataset_initial=True
        else:
            if method=='UPINN':from UPINN.model import UPINN as Model
            else:from PINN4PF.model import PINN4PF as Model
            model=Model(meta,{'initial':test['initial'][:2],'spec':test['spec'][:2]});model.load_state_dict(c['state_dict'])
        model.eval().to(device)
    report={'method':method,'system':system,'device':device,'data_path':str(path),'data_signature':data['signature'],
        'source_precision':meta['source_precision'],'start':'dataset_initial_from_csv',
        'start_source':str(path/'bus_state.csv'),
        'raw_input_matches_csv':True,
        'initial_state_handling':{'Proposed':'raw X is passed to rollout; the learned MLP initializer remains the first internal neural step','GIN':'raw X is converted to rectangular voltage for the first learned iteration; training and evaluation use the same initial-state protocol','UPINN':'raw X is passed to the network; architecture uses P/Q and PV setpoints, not free initial coordinates','PINN4PF':'raw X is passed to the network; architecture uses P/Q and PV setpoints, not free initial coordinates'}[method],'criterion':'strict max P(non-slack)/Q(PQ) mismatch <1e-4; PV error separately and jointly reported',
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        'precision':'GIN FP64 inference; Proposed/UPINN/PINN4PF FP32 networks with FP64 PF',
        'fresh_run':True,'timing_note':'Evaluation wall time, not a paired speed benchmark; includes transfers and independent validation',
        'neural_only':method!='GIN'}
    with torch.inference_mode():
        for name,mask in [('heatmap',~test['hard']),('ill',test['hard'])]:
            a=initial[mask];s=test['spec'][mask];rrs=[];counts=[];pv_errors=[];differences=[];p_errors=[];q_errors=[]
            if device=='cuda':torch.cuda.synchronize()
            started=time.perf_counter();batch=config['evaluation_batch_size']
            for off in range(0,len(a),batch):
                x0=a[off:off+batch].to(device);sp=s[off:off+batch].to(device)
                if method=='Proposed':x,res,k=rollout(model,pf,x0,sp)
                else:
                    if method=='GIN':
                        # Preserve failed cases as NaN rows rather than aborting or dropping them.
                        try:
                            expected=torch.cat((x0[...,1]*x0[...,0].cos(),x0[...,1]*x0[...,0].sin()),-1)
                            assert torch.equal(model.start(x0),expected),'GIN start differs from dataset X'
                            x=model(x0,sp)
                        except torch.linalg.LinAlgError:
                            rows=[]
                            for j in range(len(x0)):
                                try:rows.append(model(x0[j:j+1],sp[j:j+1]))
                                except torch.linalg.LinAlgError:rows.append(torch.full_like(x0[j:j+1],float('nan')))
                            x=torch.cat(rows)
                    else:x=model(x0,sp)
                    res=pf.residual(x,sp).abs().flatten(1).amax(1);k=torch.full((len(x),),3 if method=='GIN' else 1,dtype=torch.long)
                x=x.cpu();res=res.cpu();component=independent_residual(meta,x,sp.cpu()).abs()
                ep=component[...,0].amax(1);eq=component[...,1].amax(1)
                ind=torch.maximum(ep,eq)
                assert torch.allclose(ind,component.flatten(1).amax(1),rtol=0,atol=0,equal_nan=True)
                p_errors.append(ep);q_errors.append(eq)
                assert torch.equal(torch.isfinite(res)&(res<1e-4),torch.isfinite(ind)&(ind<1e-4))
                pq=meta['pq'].bool();pv=(x[:,~pq,1]-a[off:off+len(x),~pq,1]).abs().amax(1)
                rrs.append(ind);counts.append(k.cpu());pv_errors.append(pv)
                finite=torch.isfinite(res)&torch.isfinite(ind)
                if finite.any():differences.append(float((res[finite]-ind[finite]).abs().max()))
            if device=='cuda':torch.cuda.synchronize()
            residual=torch.cat(rrs);ep=torch.cat(p_errors);eq=torch.cat(q_errors);calls=torch.cat(counts);pv=torch.cat(pv_errors);ok=torch.isfinite(residual)&(residual<1e-4)
            metrics=summarize(residual);metrics.update(mean_calls=float(calls.double().mean()),max_calls=int(calls.max()),
                joint_pf_pv_success=int((ok&torch.isfinite(pv)&(pv<1e-4)).sum()),max_pv_error=float(pv[torch.isfinite(pv)].max()) if torch.isfinite(pv).any() else None,
                nonfinite_pv=int((~torch.isfinite(pv)).sum()),independent_max_difference=max(differences,default=None),evaluation_seconds=time.perf_counter()-started)
            report[name]=metrics;d={k:v[mask] for k,v in test.items()}
            with (out/(name+'.csv')).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['sample_id','x_signed','pq_signed','max_pf_mismatch','success_lt_1e-4','neural_calls','pv_error','joint_success','x_low','x_high','pq_low','pq_high','max_abs_p_mismatch','max_abs_q_mismatch'])
                for j in range(len(a)):
                    w.writerow([int(d['sample_id'][j]),float(d['level'][j,4]),float(d['level'][j,5]),float(residual[j]),bool(ok[j]),int(calls[j]),float(pv[j]),bool(ok[j] and torch.isfinite(pv[j]) and pv[j]<1e-4),*[float(v) for v in d['level'][j,:4]],float(ep[j]),float(eq[j])])
            with (out/'test.log').open('a') as f:f.write(json.dumps({'event':'raw_dataset_initial_evaluation','split':name,**metrics},allow_nan=False)+'\n')
            print(method,system,name,json.dumps(metrics),flush=True)
    report.update(data_source='csv',data_cache_read=False,data_cache_written=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if config.get('save_heatmap',False):
        from plot import save_heatmap
        save_heatmap(out)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--system',required=True);p.add_argument('--device',default='cuda');p.add_argument('--config',type=Path)
    a=p.parse_args();run(a.method,a.system,a.device,json.loads(a.config.read_text()) if a.config else None)
