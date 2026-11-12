import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Proposed._baseline.paths import DATA_ROOT, ROOT
import json,time
import numpy as np
import torch
from torch import nn
from Function import PowerFlow,summarize

def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def log(out, event, **fields):
    text = json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), event=event, **fields), allow_nan=False)
    print(text, flush=True)
    with (out/'train.log').open('a') as f: f.write(text+'\n')


def complex_state(x):
    return torch.polar(x[..., 1], x[..., 0])


class Common(nn.Module):
    def __init__(self, meta):
        super().__init__()
        self.n = int(meta['n'])
        self.slack = int(meta['slack_idx'])
        self.register_buffer('pq', meta['pq'].bool())
        self.register_buffer('slack_state', meta['slack_state'].double())
        self.register_buffer('non', meta['non'].long())

    def features(self, initial, spec):
        return torch.cat((spec[..., 0], spec[..., 1]*self.pq,
                          initial[..., 1]*(~self.pq)), -1).float()

    def preserve(self, angle, mag, initial):
        return torch.stack((angle.double(), torch.where(self.pq, mag.double(), initial[..., 1])), -1)


def load(system='IEEE_300',config=None,logger=None):
    from Proposed._baseline.data import load_data
    if config is None:raise ValueError('An explicit per-system JSON configuration is required')
    meta,groups,report=load_data(config,logger or (lambda *a,**k:None))
    return groups,meta,report,report['signature']
