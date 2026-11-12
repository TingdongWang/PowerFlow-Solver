"""Case-default starts from original static data or verified PYPOWER case118."""
import csv
import numpy as np
import torch
from Proposed._baseline.paths import DATA_ROOT

def default_state(meta):
    buses=int(meta['n'])+1
    path=DATA_ROOT/('IEEE_'+str(buses))/'bus_static.csv'
    if path.exists():
        with path.open() as f:rows=sorted(csv.DictReader(f),key=lambda x:int(x['bus0']))
        assert len(rows)==buses
        angle=torch.tensor([float(x['va_init_rad']) for x in rows],dtype=torch.float64)
        mag=torch.tensor([float(x['vm_init']) for x in rows],dtype=torch.float64)
        origin=str(path)
    else:raise FileNotFoundError('Required original static CSV missing: '+str(path))
    angle=angle+meta['slack_state'][0]-angle[int(meta['slack_idx'])]
    return {'default_angle':angle[meta['non']], 'default_vm':mag[meta['non']], 'source':origin}
