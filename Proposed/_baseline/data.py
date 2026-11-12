"""CSV-only dataset loading. No data cache read/write path exists."""
import csv,hashlib,sys,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Function import PowerFlow
DATA_ROOT=Path(os.environ.get('PF_DATA_ROOT',str(Path(__file__).resolve().parents[3]/'Data'))).expanduser().resolve()
LEVEL=['x_low','x_high','pq_low','pq_high','x_signed','pq_signed']

def enable_csv_guard():
    if getattr(sys,'_rjmnc_csv_guard',False):return
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).absolute()
            if path.is_relative_to(DATA_ROOT):
                if path.suffix.lower()!='.csv':raise PermissionError('CSV-only policy forbids this data path: '+str(path))
                mode=args[1] if len(args)>1 else 'r'
                flags=args[2] if len(args)>2 else 0
                if isinstance(mode,str) and any(x in mode for x in 'wax+'):raise PermissionError('Dataset is read-only')
                if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):raise PermissionError('Dataset is read-only')
    sys.addaudithook(audit);sys._rjmnc_csv_guard=True
enable_csv_guard()

def split_data(data, split_names, config):
    train_indices=np.flatnonzero(np.asarray(split_names)=='train')
    test_indices=np.flatnonzero(np.asarray(split_names)=='test')
    count=int(config['data']['validation_samples'])
    np.random.default_rng(config['data']['split_seed']).shuffle(train_indices)
    boundary=len(train_indices)-count
    if boundary<1:raise ValueError('Insufficient real training samples')
    expected=int(config['data']['expected_train_samples'])
    if boundary!=expected:raise ValueError(f'CSV has {boundary} training after holding out {count}; config expects {expected}. Never duplicate samples.')
    return {'train':subset(data,train_indices[:boundary]),'val':subset(data,train_indices[boundary:]),'test':subset(data,test_indices)}

def subset(data, indices):
    return {name: tensor[indices].contiguous() for name, tensor in data.items()}

def table_from_csv(root):
    frame = pd.read_csv(root / 'meta.csv').sort_values('sample_id').reset_index(drop=True)
    if frame.sample_id.duplicated().any():
        raise ValueError('Duplicate sample_id in meta.csv')
    if not frame['split'].isin(['train', 'test']).all():
        raise ValueError('Only train/test splits are supported')
    # Rejected training labels are excluded, but the complete test is mandatory.
    frame = frame[(frame['split'] == 'test') | (frame.valid_label == 1)].reset_index(drop=True)
    if not (frame.valid_label == 1).all():
        raise ValueError('Test has invalid labels; inspect data generation before full evaluation')
    return frame

def static_from_csv(root, expected_buses):
    buses = pd.read_csv(root / 'bus_static.csv').sort_values('bus0')
    if len(buses) != expected_buses or buses.bus0.tolist() != list(range(expected_buses)):
        raise ValueError(f'bus_static.csv does not describe {expected_buses} contiguous buses')
    slack_rows = buses[buses.is_slack == 1]
    if len(slack_rows) != 1:
        raise ValueError('Exactly one slack bus is required')
    slack = int(slack_rows.iloc[0].bus0)
    non = torch.tensor([i for i in range(expected_buses) if i != slack])
    pq = torch.tensor(buses.is_pq.to_numpy()[non.numpy()], dtype=torch.float64)
    G = torch.zeros(expected_buses, expected_buses, dtype=torch.float64)
    B = torch.zeros_like(G)
    ybus = pd.read_csv(root / 'ybus.csv')
    if ybus.duplicated(['from_bus0', 'to_bus0']).any():
        raise ValueError('Duplicate Ybus entries')
    for row in ybus.itertuples(index=False):
        G[int(row.from_bus0), int(row.to_bus0)] = float(row.g)
        B[int(row.from_bus0), int(row.to_bus0)] = float(row.b)
    return {'n': expected_buses - 1, 'non': non, 'pq': pq, 'G': G, 'B': B,
            'slack_idx': slack, 'slack_state': torch.tensor(
                [slack_rows.iloc[0].va_init_rad, slack_rows.iloc[0].vm_init], dtype=torch.float64)}

def metadata_columns(frame):
    return {'sample_id': torch.tensor(frame.sample_id.to_numpy().copy(), dtype=torch.long),
            'level': torch.tensor(frame[LEVEL].to_numpy().copy(), dtype=torch.float64),
            'hard': torch.tensor(frame.source.str.lower().eq('ill-conditioned').to_numpy().copy())}

def read_raw(root, config, log):
    buses = int(config['system'].split('_')[1])
    meta = static_from_csv(root, buses)
    frame = table_from_csv(root)
    data = metadata_columns(frame)
    for key in ('initial', 'spec', 'label'):
        data[key] = torch.empty(len(frame), buses - 1, 2, dtype=torch.float64)
    arrays = {key: data[key].numpy() for key in ('initial', 'spec', 'label')}
    seen = np.zeros((len(frame), buses), dtype=bool)
    index = pd.Index(frame.sample_id)
    columns = ['sample_id', 'bus0', 'nonslack0', 'vm_start', 'va_start_rad',
               'p_spec_pu', 'q_spec_pu', 'vm_true', 'va_true_rad']
    log('raw_csv_begin',path=str(root),samples=len(frame),buses=buses)
    chunks = 0
    for chunk in pd.read_csv(root / 'bus_state.csv', usecols=columns, chunksize=250000):
        sample = index.get_indexer(chunk.sample_id)
        keep = sample >= 0
        chunk = chunk.loc[keep]
        sample = sample[keep]
        bus = chunk.bus0.to_numpy(dtype=np.int64)
        if (bus < 0).any() or (bus >= buses).any():
            raise ValueError('bus_state.csv contains an out-of-range bus')
        flat_indices = sample * buses + bus
        if len(np.unique(flat_indices)) != len(flat_indices) or seen[sample, bus].any():
            raise ValueError('Duplicate sample/bus row in bus_state.csv')
        seen[sample, bus] = True
        is_slack = bus == meta['slack_idx']
        if is_slack.any():
            actual = chunk.loc[is_slack, ['va_start_rad', 'vm_start']].to_numpy()
            if not np.allclose(actual, meta['slack_state'].numpy(), rtol=0, atol=1e-12):
                raise ValueError('Slack boundary is not constant')
        ns = chunk.loc[~is_slack]
        i = sample[~is_slack]
        j = ns.nonslack0.to_numpy(dtype=np.int64)
        if (j < 0).any() or (j >= buses - 1).any() or not np.array_equal(meta['non'].numpy()[j], bus[~is_slack]):
            raise ValueError('nonslack0 does not match bus0')
        arrays['initial'][i, j] = ns[['va_start_rad', 'vm_start']].to_numpy()
        arrays['spec'][i, j] = ns[['p_spec_pu', 'q_spec_pu']].to_numpy()
        arrays['label'][i, j] = ns[['va_true_rad', 'vm_true']].to_numpy()
        chunks += 1
        if chunks % 8 == 0:
            log('raw_csv_progress',chunks=chunks,seen_bus_rows=int(seen.sum()))
    if not seen.all():
        raise ValueError(f'Missing {int((~seen).sum())} sample/bus rows; cache was not saved')
    meta['source_precision'] = 'original CSV float64 Ybus, voltage, P/Q and labels'
    return meta, split_data(data, frame['split'].to_numpy(), config)

def normalization(meta, train):
    """Recompute the same six-feature statistics on each training split only."""
    pf = PowerFlow(meta)
    sums = torch.zeros(6, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    delta_square = torch.zeros(2, dtype=torch.float64)
    count = 0
    for i in range(0, len(train['initial']), 256):
        initial = train['initial'][i:i+256]
        calculated = pf(initial)
        residual = train['spec'][i:i+256] - calculated
        residual[..., 1] *= meta['pq']
        features = torch.cat((calculated, residual, initial), -1).flatten(0, 1)
        sums += features.sum(0)
        squares += features.square().sum(0)
        delta_square += (train['label'][i:i+256] - initial).square().sum((0, 1))
        count += len(features)
    mean = sums / count
    meta.update(nm=mean, ns=(squares/count-mean.square()).clamp_min(1e-12).sqrt(),
                om=torch.zeros(2,dtype=torch.float64), ost=(delta_square/count).clamp_min(1e-12).sqrt())

def validate(meta, groups, system):
    n = int(system.split('_')[1]) - 1
    if meta['n'] != n or meta['G'].shape != (n+1, n+1):
        raise ValueError('Topology dimensions do not match system')
    for key in ('G', 'B', 'slack_state'):
        if not torch.isfinite(meta[key]).all():
            raise ValueError(f'Nonfinite topology {key}')
    ids = []
    for name, group in groups.items():
        size = len(group['sample_id'])
        if size == 0:
            raise ValueError(f'Empty {name} split')
        for key in ('initial', 'spec', 'label'):
            value = group[key]
            if value.shape != (size, n, 2) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid {name}.{key} shape or nonfinite values')
        if name != 'test' and group['hard'].any():
            raise ValueError('Hard test samples entered training/validation')
        ids.append(group['sample_id'])
    ids = torch.cat(ids)
    if len(ids.unique()) != len(ids):
        raise ValueError('Duplicate or overlapping split sample IDs')

def signature(meta, groups):
    digest = hashlib.sha256()
    for key in ('G', 'B', 'non', 'pq', 'slack_state'):
        digest.update(meta[key].contiguous().numpy().tobytes())
    for name in ('train', 'val', 'test'):
        for key in ('sample_id', 'initial', 'spec', 'label', 'level', 'hard'):
            digest.update(groups[name][key].contiguous().numpy().tobytes())
    return digest.hexdigest()

def load_data(config,log):
    root=DATA_ROOT/config['system']
    if root.parent.resolve()!=DATA_ROOT:raise ValueError('Unauthorized data directory')
    files={name:root/name for name in ('meta.csv','bus_static.csv','ybus.csv','bus_state.csv')}
    before={name:(f.stat().st_size,f.stat().st_mtime_ns) for name,f in files.items()}
    meta,groups=read_raw(root,config,log)
    validate(meta,groups,config['system'])
    if len(groups['val']['initial'])!=config['data']['validation_samples'] or len(groups['test']['initial'])!=4100:
        raise ValueError('Unexpected validation/test sample counts')
    normalization(meta,groups['train'])
    sig=signature(meta,groups)
    if any((f.stat().st_size,f.stat().st_mtime_ns)!=before[name] for name,f in files.items()):raise RuntimeError('CSV changed while being read')
    report={'system':config['system'],'path':str(root),'source_files':{k:str(v) for k,v in files.items()},
        'signature':sig,'source_precision':meta['source_precision'],'train':len(groups['train']['initial']),
        'validation':len(groups['val']['initial']),'heatmap':int((~groups['test']['hard']).sum()),
        'hard':int(groups['test']['hard'].sum()),'pq_buses':int(meta['pq'].sum()),
        'slack_idx':int(meta['slack_idx']),'data_cache_read':False,'data_cache_written':False,
        'split_seed':config['data']['split_seed'],'csv_size_mtime':before}
    log('csv_dataset_ready',**report)
    return meta,groups,report
