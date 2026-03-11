"""피팅 모델 — Elbow (K-factor)"""
from __future__ import annotations
from dataclasses import dataclass
from fluid import FluidState
from components import Component


@dataclass
class Elbow(Component):
    name: str = "Elbow"
    area: float = 0.24
    angle_deg: float = 90.0
    r_over_D: float = 1.5

    def __post_init__(self):
        tbl = {0.5: 1.20, 0.75: 0.57, 1.0: 0.42, 1.5: 0.27, 2.0: 0.20, 2.5: 0.17}
        rds = sorted(tbl.keys())
        r = self.r_over_D
        if r <= rds[0]: K90 = tbl[rds[0]]
        elif r >= rds[-1]: K90 = tbl[rds[-1]]
        else:
            for i in range(len(rds) - 1):
                if rds[i] <= r <= rds[i + 1]:
                    f = (r - rds[i]) / (rds[i + 1] - rds[i])
                    K90 = tbl[rds[i]] * (1 - f) + tbl[rds[i + 1]] * f
                    break
        self.K = K90 * (self.angle_deg / 90.0)

    def pressure_drop(self, Q, fluid):
        v = Q / self.area if self.area > 0 else 0
        return self.K * 0.5 * fluid.rho * v**2

    def heat_transfer(self, Q, fluid): return 0.0
    def comp_type(self): return 'elbow'
