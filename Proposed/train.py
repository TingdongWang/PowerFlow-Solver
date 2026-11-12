"""Identical full-data training procedure for 14, 118, 300 and 1354 buses."""
import copy
import time
import torch

from Function import PowerFlow, summarize
from Loss import regression_statistics
from model import RJMNC
from runtime import model_meta, sync, torch_save, sha256
from evaluate import evaluate


def preflight(model, pf, train, config, log):
    device = next(model.parameters()).device
    original = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    batch = config['batch_size']
    while True:
        optimizer = None
        try:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            initial = train['initial'][:batch].to(device)
            spec = train['spec'][:batch].to(device)
            target = train['label'][:batch].to(device)
            optimizer = torch.optim.AdamW(model.initializer.parameters(),lr=config['training']['learning_rate'])
            prediction = model.initialize(initial,spec)
            loss = ((prediction-target)/model.ns[4:6]).square().mean()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            norm = sum(g.double().square().sum() for g in gradients).sqrt().item()
            if not norm > 0 or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError('Preflight gradients are invalid')
            optimizer.step()  # Allocate Adam state too; all weights are restored below.
            with torch.no_grad():
                if not torch.equal(model(initial,torch.zeros_like(initial)),torch.zeros_like(initial)):
                    raise RuntimeError('Zero residual must produce exactly zero correction')
                residual = pf.residual(target,spec).abs().flatten(1).amax(1)
            sync(device)
            log('preflight_pass',requested_batch=config['batch_size'],effective_batch=batch,
                actual_samples=len(initial),gradient_norm=norm,label_pf=summarize(residual),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                optimizer_probe='temporary Adam step; weights/RNG restored, not counted as training')
            return batch
        finally:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(original,strict=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng,device)
            optimizer = None
            # Release probe activations before retrying a smaller batch.
            for key in ('initial','spec','target','prediction','loss','gradients','residual'):
                if key in locals():
                    if key == 'prediction': prediction = None
                    elif key == 'loss': loss = None
                    elif key == 'gradients': gradients = None
                    elif key == 'initial': initial = None
                    elif key == 'spec': spec = None
                    elif key == 'target': target = None
                    elif key == 'residual': residual = None
            if device.type == 'cuda':
                torch.cuda.empty_cache()


@torch.no_grad()
def fit_operator(model,pf,train,config,batch,log):
    settings = config['training']
    active = torch.stack((torch.ones_like(model.pq),model.pq),-1).flatten().bool()
    device = model.pq.device
    dim = int(active.sum())
    gram = torch.zeros(dim,dim,device=device,dtype=torch.float64)
    cross = torch.zeros_like(gram)
    rscale,dscale = model.rscale.double()[active],model.dscale.double()[active]
    eye = torch.eye(dim,device=device,dtype=torch.float64)
    for epoch in range(settings['operator_passes']):
        start = time.perf_counter()
        for i in range(0,len(train['label']),batch):
            target = train['label'][i:i+batch].to(device)
            noise = torch.randn_like(target)*settings['operator_noise_std']
            noise[...,1] *= model.pq
            residual = (pf(target-noise)-pf(target+noise))*.5
            residual[...,1] *= model.pq
            r = residual.flatten(1)[:,active]/rscale
            y = (-noise).flatten(1)[:,active]/dscale
            gg,xy = regression_statistics(y,r)
            gram += gg
            cross += xy
        forward = torch.linalg.solve(gram+gram.diag().mean()*settings['operator_ridge']*eye,cross).T
        inverse = torch.linalg.solve(forward,eye)  # OFFLINE training, never called in rollout.
        full = torch.zeros(model.n*2,model.n*2,device=device,dtype=torch.float64)
        ix = torch.where(active)[0]
        full[ix[:,None],ix[None,:]] = inverse
        if not torch.isfinite(full).all():
            raise RuntimeError('Nonfinite offline correction weights')
        model.corrector.weight.copy_(full.float())
        sync(device)
        log('operator_epoch',epoch=epoch+1,samples=len(train['label']),
            cumulative_examples=(epoch+1)*len(train['label']),seconds=time.perf_counter()-start,
            optimizer='closed-form forward least squares then offline inverse')


def save_checkpoint(path,model,meta,config,data_report,info):
    torch_save(path,{'format_version':1,'system':config['system'],
        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'meta':copy.deepcopy(meta),'config':copy.deepcopy(config),'data_signature':data_report['signature'],
        'info':copy.deepcopy(info),'use_gain':False})


def score(metrics):
    return metrics['success'],-metrics['mean_calls'],-(metrics['mean'] if metrics['mean'] is not None else float('inf'))


def train_full(meta,groups,config,data_report,out,log):
    meta = model_meta(meta,config)
    device = torch.device(config['device'])
    model = RJMNC(meta).to(device)
    pf = PowerFlow(meta).to(device)
    training,validation = groups['train'],groups['val']
    ns,nm = meta['ns'].float(),meta['nm'].float()
    targets = ((training['label'].float()-nm[4:6])/ns[4:6]).flatten(1)
    with torch.no_grad():
        model.initializer[-1].bias.copy_(targets.mean(0).to(device))
    batch = preflight(model,pf,training,config,log)
    log('training_start',system=config['system'],train=len(targets),validation=len(validation['initial']),
        effective_batch=batch,parameters=sum(p.numel() for p in model.parameters()),
        initializer='from scratch, training-only mean bias',source_precision=meta['source_precision'])
    fit_operator(model,pf,training,config,config.get('operator_batch_size',batch),log)
    baseline,_,_ = evaluate(model,pf,validation,config,log)
    best = score(baseline)
    path = out/'model.pt'
    info = {'selected_epoch':0,'selected_optimizer_steps':0,'validation':baseline,'effective_batch_size':batch}
    save_checkpoint(path,model,meta,config,data_report,info)
    with torch.no_grad():
        conditions=torch.cat([model.condition(training['spec'][i:i+batch].to(device),training['initial'][i:i+batch].to(device)).cpu() for i in range(0,len(targets),batch)])
    mask = torch.stack((torch.ones_like(model.pq),model.pq),-1).flatten()
    settings = config['training']
    optimizer = torch.optim.AdamW(model.initializer.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings['epochs'],eta_min=settings['min_learning_rate'])
    conditions=conditions.to(device);targets=targets.to(device);mask=mask.to(device)
    steps = 0
    for epoch in range(1,settings['epochs']+1):
        sync(device)
        start = time.perf_counter()
        model.train()
        total = 0.
        for ids in torch.randperm(len(conditions)).split(batch):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.initializer(conditions[ids.to(device)])
            loss = ((prediction-targets[ids.to(device)])*mask).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.initializer.parameters(),settings['gradient_clip'])
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite training gradient')
            optimizer.step()
            steps += 1
            total += float(loss.detach())*len(ids)
        scheduler.step()
        sync(device)
        fields = {'epoch':epoch,'samples':len(conditions),'optimizer_steps':steps,
                  'train_loss':total/len(conditions),'seconds':time.perf_counter()-start}
        if epoch==1 or epoch%settings['validate_every']==0 or epoch==settings['epochs']:
            metrics,_,_ = evaluate(model,pf,validation,config,log)
            fields['validation'] = metrics
            if score(metrics) > best:
                best = score(metrics)
                info.update(selected_epoch=epoch,selected_optimizer_steps=steps,validation=metrics)
                save_checkpoint(path,model,meta,config,data_report,info)
                fields['checkpoint_saved'] = True
        log('mlp_epoch',**fields)
    selected = torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(selected['state_dict'],strict=True)
    info = selected['info']
    policy = None
    for beta in config['inference']['momentum_candidates']:
        model.momentum_beta = beta
        metrics,_,_ = evaluate(model,pf,validation,config,log)
        log('validation_momentum',beta=beta,**metrics)
        if policy is None or score(metrics) > policy[0]:
            policy = score(metrics),beta,metrics
    meta['momentum_beta'] = policy[1]
    model.momentum_beta = policy[1]
    info.update(total_epochs=settings['epochs'],total_optimizer_steps=steps,
                train_samples=len(conditions),operator_passes=settings['operator_passes'],validation=policy[2],
                beta=policy[1],policy_selection='validation only')
    save_checkpoint(path,model,meta,config,data_report,info)
    log('training_complete',**info,weight_sha256=sha256(path))
    return model.eval(),meta
