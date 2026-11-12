from Proposed._baseline.paths import ROOT
import numpy as np
import torch
from torch import nn
from Proposed._baseline.common import Common

class AdaptiveReLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(width))

    def forward(self, z):
        return torch.relu(self.slope*z)


class PINN4PF(Common):
    """Two shared ReLU layers and four adaptive-ReLU layers per head."""
    def __init__(self, meta, train):
        super().__init__(meta)
        features = self.features(train['initial'], train['spec'])
        self.register_buffer('mean', features.mean(0))
        self.register_buffer('scale', features.std(0).clamp_min(1e-3))
        label = train.get('label')
        if label is None:
            center = torch.zeros(self.n, 2, dtype=torch.float64)
            scale = torch.ones_like(center)
        else:
            voltage = torch.polar(label[..., 1], label[..., 0])
            components = torch.view_as_real(voltage)
            center = components.mean(0)
            scale = components.std(0).clamp_min(1e-4)
        self.register_buffer('voltage_center', center)
        self.register_buffer('voltage_scale', scale)
        self.shared = nn.Sequential(nn.Linear(3*self.n, 2*self.n), nn.ReLU(), nn.Dropout(.001),
                                    nn.Linear(2*self.n, 2*self.n), nn.ReLU(), nn.Dropout(.001))
        def head():
            layers = []
            width = 2*self.n
            for _ in range(4):
                layers += [nn.Linear(width, self.n), AdaptiveReLU(self.n), nn.Dropout(.001)]
                width = self.n
            layers.append(nn.Linear(self.n, self.n))
            nn.init.normal_(layers[-1].weight, std=1e-4)
            nn.init.zeros_(layers[-1].bias)
            return nn.Sequential(*layers)
        self.real, self.imag = head(), head()

    def forward(self, initial, spec):
        h = self.shared((self.features(initial, spec)-self.mean)/self.scale)
        mu = self.voltage_center[:, 0] + self.voltage_scale[:, 0] * self.real(h).double()
        omega = self.voltage_center[:, 1] + self.voltage_scale[:, 1] * self.imag(h).double()
        return self.preserve(torch.atan2(omega, mu), torch.sqrt(mu.square()+omega.square()+1e-24), initial)

