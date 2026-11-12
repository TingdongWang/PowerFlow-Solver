"""JSON-driven full-data CUDA training for the retained local baselines."""
import argparse,json,time,pathlib,fcntl,copy,math
import numpy as np
import torch
from Proposed._baseline.paths import ROOT
from Proposed._baseline.common import load,write,log,complex_state
from Proposed._baseline.native_losses import ProxRMS,pinn_loss
from Function import PowerFlow

def target(x):return torch.cat((x[...,1]*x[...,0].cos(),x[...,1]*x[...,0].sin()),-1)
def build_model(method,meta,train):
    if method=='GIN':
        from GIN.model import GIN
        return GIN(meta)
    if method=='UPINN':
        from UPINN.model import UPINN
        return UPINN(meta,train)
    if method=='PINN4PF':
        from PINN4PF.model import PINN4PF
        return PINN4PF(meta,train)
    raise ValueError(method)
def objective(method,model,pf,d,ids,epoch,meta):
    if method=='GIN':return (model.rectangular(d['initial'][ids],d['spec'][ids])-target(d['label'][ids])).abs().sum(-1).mean()
    if method=='PINN4PF':
        loss=pinn_loss(model,d,ids,epoch,meta)
        if meta['pinn_schedule'].get('scale_objective'):
            # Normalize the overall objective; preserve the relative physics weight.
            loss=loss/model.voltage_scale.square().mean().clamp_min(1e-8)
        return loss
    x=model(d['initial'][ids],d['spec'][ids]);return .5*pf.residual(x,d['spec'][ids]).square().sum((1,2)).mean()/(pf.n+1)
@torch.inference_mode()
def validation(method,model,pf,d,batch):
    model.eval();total=0.;residuals=[]
    for ids in torch.arange(len(d['initial']),device='cuda').split(batch):
        if method=='GIN':
            x=model(d['initial'][ids],d['spec'][ids]).double()
            residuals.append(pf.residual(x,d['spec'][ids]).abs().flatten(1).amax(1))
            continue
        elif method=='UPINN':loss=pf.residual(model(d['initial'][ids],d['spec'][ids]),d['spec'][ids]).square().sum((1,2)).mean()
        else:loss=(complex_state(model(d['initial'][ids],d['spec'][ids]))-complex_state(d['label'][ids])).abs().square().mean()
        total+=float(loss)*len(ids)
    if method=='GIN':return float(torch.cat(residuals).quantile(.9))
    return total/len(d['initial'])
def train(config,out):
    method=config['method'];epochs=config['training']['epochs'];batch=config['batch_size'];evbatch=config['evaluation_batch_size']
    torch.set_num_threads(config['cpu_threads']);torch.set_num_interop_threads(1);torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    logger=lambda event,**fields:log(out,event,**fields)
    groups,meta,report,signature=load(config['system'],config,logger);write(out/'data_report.json',report)
    model=build_model(method,meta,groups['train']);params=sum(t.numel() for t in model.parameters())
    if method=='GIN':
        model=model.float()
        model.use_dataset_initial=True
        previous=config['training'].get('initial_checkpoint')
        if previous:
            saved=torch.load(previous,map_location='cpu',weights_only=False)
            assert saved['signature']==signature
            model.load_state_dict(saved['state_dict'])
    model=model.cuda();pf=PowerFlow(meta).cuda()
    dtype=torch.float32 if method=='GIN' else torch.float64
    d={name:{k:v.cuda().to(dtype) for k,v in groups[name].items() if k in ('initial','spec','label')} for name in ('train','val')}
    mt={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in meta.items()}
    mt['pinn_schedule']=config['training']
    if method=='GIN':opt=torch.optim.Adam(model.parameters(),lr=config['training']['learning_rate'],betas=(.99,.999))
    elif method=='UPINN':opt=ProxRMS(model.parameters(),lr=config['training']['learning_rate'])
    else:opt=torch.optim.Adam(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    write(out/'protocol.json',{'method':method,'implementation':'paper-aligned local rerun on the shared CSV split','epochs':epochs,'batch_size':batch,'initial_state':'dataset_initial_from_csv' if method=='GIN' else 'native_direct_predictor','gradient_accumulation':False,'training_dtype':str(dtype),'source':'raw CSV only','data_signature':signature,'parameters':params,'checkpoint_selection':'validation PF mismatch p90' if method=='GIN' else 'validation native objective','seed':config['seed']})
    log(out,'training_start',method=method,system=config['system'],parameters=params,batch_size=batch,train_samples=len(d['train']['initial']),epochs=epochs,device='cuda')
    best=float('inf');best_epoch=0;steps=0;seconds=0.
    torch.cuda.reset_peak_memory_stats()
    stop_epoch=config['training'].get('stop_epoch',epochs)
    for epoch in range(1,stop_epoch+1):
        model.train();torch.cuda.synchronize();start=time.perf_counter();total=0.
        if method=='GIN':opt.param_groups[0]['lr']=config['training']['learning_rate']*(epoch/5 if epoch<=5 else ((epochs-epoch)/max(1,epochs-5))**2)
        if method=='UPINN':opt.param_groups[0]['lr']=1e-4+(1e-6-1e-4)*(epoch-1)/max(epochs-1,1)
        for ids in torch.randperm(len(d['train']['initial']),device='cuda').split(batch):
            opt.zero_grad(set_to_none=True);loss=objective(method,model,pf,d['train'],ids,epoch,mt)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; samples are not dropped')
            loss.backward()
            if method!='GIN':
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            elif not all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters()):raise RuntimeError('Nonfinite GIN gradient')
            opt.step();steps+=1;total+=float(loss.detach())*len(ids)
        torch.cuda.synchronize();duration=time.perf_counter()-start;seconds+=duration
        if epoch % config['training'].get('validate_every',1) and epoch != stop_epoch:
            log(out,'epoch',epoch=epoch,epochs=epochs,steps=steps,loss=total/len(d['train']['initial']),seconds=duration)
            continue
        metric=validation(method,model,pf,d['val'],evbatch)
        if not math.isfinite(metric):raise RuntimeError('Nonfinite validation objective')
        if metric<best:
            best=metric;best_epoch=epoch
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'meta':meta,'signature':signature,'method':method,'epoch':epoch,'epochs':epochs,'steps':steps,'validation_metric':metric,'training_seconds':seconds,'config':config},out/'model.pt')
        status={'complete':epoch==epochs,'epoch':epoch,'epochs':epochs,'best_epoch':best_epoch,'best':best,'steps':steps,'training_seconds':seconds,'batch_size':batch,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        write(out/'training_status.json',status);log(out,'epoch',loss=total/len(d['train']['initial']),seconds=duration,validation_metric=metric,**status)
    log(out,'training_complete',best_epoch=best_epoch)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=pathlib.Path,required=True);args=parser.parse_args()
    config=json.loads(args.config.read_text());out=args.config.resolve().parent
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);train(config,out)
