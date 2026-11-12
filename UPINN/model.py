from Proposed._baseline.paths import ROOT
import numpy as np
import torch
from torch import nn
from Proposed._baseline.common import Common

class UPINN(Common):
    """Two GCNs, mean pooling, global FC path; mismatch-only training."""
    def __init__(self, meta, train):
        super().__init__(meta)
        y = (meta['G'].square()+meta['B'].square()).sqrt().float()
        # Real-valued admittance-magnitude GCN; retain diagonal self information.
        degree = y.sum(1).clamp_min(1e-12)
        a = y/torch.sqrt(degree[:, None]*degree[None, :])
        self.register_buffer('adj', a.to_sparse())
        from Proposed._baseline.defaults import default_state
        defaults=default_state(meta)
        self.register_buffer('reference_angle', defaults['default_angle'].double())
        self.register_buffer('reference_vm', defaults['default_vm'].double())
        self.g1, self.g2 = nn.Linear(7, 32), nn.Linear(32, 32)
        layers = []
        d = 3*self.n+32+2
        for _ in range(4):
            layers += [nn.Linear(d, 256), nn.SiLU()]
            d = 256
        self.fc = nn.Sequential(*layers)
        self.output = nn.Linear(d, 2*self.n)
        self.skip = nn.Linear(3*self.n, 2*self.n, bias=False)
        nn.init.normal_(self.output.weight, std=1e-4)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.skip.weight)

    def message(self, h):
        b, n, d = h.shape
        return torch.sparse.mm(self.adj, h.permute(1, 0, 2).reshape(n, b*d)).reshape(n, b, d).permute(1, 0, 2)

    def forward(self, initial, spec):
        b = len(initial)
        nodes = torch.zeros(b, self.n+1, 7, dtype=torch.float32, device=initial.device)
        nodes[:, self.non, 0] = spec[..., 0].float()
        nodes[:, self.non, 1] = (spec[..., 1]*self.pq).float()
        nodes[:, self.non, 2] = (initial[..., 1]*(~self.pq)).float()
        nodes[:, self.non, 4] = self.pq.float()
        nodes[:, self.non, 5] = (~self.pq).float()
        nodes[:, self.slack, 2] = self.slack_state[1].float()
        nodes[:, self.slack, 3] = self.slack_state[0].float()
        nodes[:, self.slack, 6] = 1
        h = torch.nn.functional.silu(self.g1(self.message(nodes)))
        h = torch.nn.functional.silu(self.g2(self.message(h))).mean(1)
        raw = self.features(initial, spec)
        z = torch.cat((raw, h, self.slack_state.float().expand(b, 2)), -1)
        # Local Beta and a static-case output reference; neither uses solution labels.
        delta = .1*(self.output(self.fc(z))+self.skip(raw)).reshape(b, self.n, 2)
        return self.preserve(self.reference_angle+delta[..., 0], self.reference_vm+delta[..., 1], initial)

