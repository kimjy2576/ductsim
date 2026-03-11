"""덕트 구성요소 — HX / Fan / Filter / Damper"""
from __future__ import annotations
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
from fluid import FluidState


class Component(ABC):
    name: str
    @abstractmethod
    def pressure_drop(self, Q: float, fluid: FluidState) -> float: ...
    @abstractmethod
    def heat_transfer(self, Q: float, fluid: FluidState) -> float: ...
    @abstractmethod
    def comp_type(self) -> str: ...


@dataclass
class HeatExchanger(Component):
    name: str = "HX"
    face_area: float = 0.25
    core_depth: float = 0.05
    sigma: float = 0.55
    Dh_air: float = 0.003
    UA: float = 500.0
    T_fluid_in: float = 280.15
    C_fluid: float = 2000.0
    f_core: float = 0.02

    def pressure_drop(self, Q, fluid):
        G = fluid.rho * Q / (self.face_area * self.sigma)
        A_Ac = 4.0 * self.core_depth / self.Dh_air
        Kc = 0.42 * (1 - self.sigma**2)
        Ke = (1 - self.sigma**2)
        dp = (G**2 / (2 * fluid.rho)) * (Kc + (1 - self.sigma**2) + self.f_core * A_Ac - Ke)
        return max(dp, 0.0)

    def heat_transfer(self, Q, fluid):
        m_dot = fluid.rho * Q
        C_air = m_dot * fluid.cp
        C_min = min(C_air, self.C_fluid)
        C_max = max(C_air, self.C_fluid)
        if C_min <= 0: return 0.0
        C_r = C_min / C_max
        NTU = self.UA / C_min
        if C_r == 0:
            eps = 1 - math.exp(-NTU)
        else:
            eps = 1 - math.exp((NTU**0.22 / C_r) * (math.exp(-C_r * NTU**0.78) - 1))
        return eps * C_min * abs(fluid.T - self.T_fluid_in)

    def comp_type(self): return 'hx'


@dataclass
class Fan(Component):
    name: str = "Fan"
    curve_coeffs: List[float] = field(default_factory=lambda: [500.0, 0.0, -2000.0])
    rpm_rated: float = 1450.0
    rpm: float = 1450.0
    eta_total: float = 0.65

    def pressure_drop(self, Q, fluid):
        ratio = self.rpm / self.rpm_rated if self.rpm_rated > 0 else 1.0
        Q_corr = Q / ratio if ratio > 0 else Q
        dp_rated = sum(c * Q_corr**i for i, c in enumerate(self.curve_coeffs))
        return -max(dp_rated * ratio**2, 0.0)

    def fan_dp_positive(self, Q):
        ratio = self.rpm / self.rpm_rated if self.rpm_rated > 0 else 1.0
        Q_corr = Q / ratio if ratio > 0 else Q
        dp_rated = sum(c * Q_corr**i for i, c in enumerate(self.curve_coeffs))
        return max(dp_rated * ratio**2, 0.0)

    def heat_transfer(self, Q, fluid):
        dp = abs(self.pressure_drop(Q, fluid))
        W = dp * Q / self.eta_total if self.eta_total > 0 else 0
        return W * (1 - self.eta_total)

    @property
    def max_flow(self):
        if len(self.curve_coeffs) >= 3:
            a0, a1, a2 = self.curve_coeffs[:3]
            disc = a1**2 - 4 * a2 * a0
            if disc >= 0 and a2 != 0:
                return max((-a1 + math.sqrt(disc)) / (2 * a2),
                           (-a1 - math.sqrt(disc)) / (2 * a2), 0.01)
        return 1.0

    def comp_type(self): return 'fan'


@dataclass
class Filter(Component):
    name: str = "Filter"
    face_area: float = 0.25
    C_resistance: float = 50.0
    n_exponent: float = 1.8
    loading_factor: float = 1.0

    def pressure_drop(self, Q, fluid):
        v = Q / self.face_area if self.face_area > 0 else 0
        return self.C_resistance * self.loading_factor * abs(v)**self.n_exponent
    def heat_transfer(self, Q, fluid): return 0.0
    def comp_type(self): return 'filter'


@dataclass
class Damper(Component):
    name: str = "Damper"
    area: float = 0.24
    opening_deg: float = 90.0
    K_curve: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 1e6), (10, 200), (20, 52), (30, 18), (45, 6.5), (60, 2.6), (75, 0.8), (90, 0.2)])

    def _K(self):
        return float(np.interp(self.opening_deg,
                               [p[0] for p in self.K_curve], [p[1] for p in self.K_curve]))
    def pressure_drop(self, Q, fluid):
        v = Q / self.area if self.area > 0 else 0
        return self._K() * 0.5 * fluid.rho * v**2
    def heat_transfer(self, Q, fluid): return 0.0
    def comp_type(self): return 'damper'
