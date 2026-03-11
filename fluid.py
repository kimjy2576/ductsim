"""유체 상태 모델 — 공기 물성치 (CoolProp / fallback 간이식)"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False


@dataclass
class FluidState:
    T: float          # [K]
    P: float          # [Pa]
    RH: float = 0.0
    _rho: Optional[float] = field(default=None, repr=False)
    _mu:  Optional[float] = field(default=None, repr=False)
    _cp:  Optional[float] = field(default=None, repr=False)
    _k:   Optional[float] = field(default=None, repr=False)

    def _invalidate(self):
        self._rho = self._mu = self._cp = self._k = None

    @property
    def rho(self):
        if self._rho is None:
            if HAS_COOLPROP:
                self._rho = CP.PropsSI('D', 'T', self.T, 'P', self.P, 'Air')
            else:
                self._rho = self.P / (287.058 * self.T)
        return self._rho

    @property
    def mu(self):
        if self._mu is None:
            if HAS_COOLPROP:
                self._mu = CP.PropsSI('V', 'T', self.T, 'P', self.P, 'Air')
            else:
                T_ref, mu_ref, S = 291.15, 1.827e-5, 120.0
                self._mu = mu_ref * (self.T / T_ref)**1.5 * (T_ref + S) / (self.T + S)
        return self._mu

    @property
    def cp(self):
        if self._cp is None:
            if HAS_COOLPROP:
                self._cp = CP.PropsSI('C', 'T', self.T, 'P', self.P, 'Air')
            else:
                self._cp = 1006.0
        return self._cp

    @property
    def k(self):
        if self._k is None:
            if HAS_COOLPROP:
                self._k = CP.PropsSI('L', 'T', self.T, 'P', self.P, 'Air')
            else:
                self._k = 0.0241 + 7.5e-5 * (self.T - 273.15)
        return self._k

    @property
    def nu(self): return self.mu / self.rho
    @property
    def Pr(self): return self.mu * self.cp / self.k

    def update(self, T=None, P=None, RH=None):
        if T is not None: self.T = T
        if P is not None: self.P = P
        if RH is not None: self.RH = RH
        self._invalidate()

    def copy(self):
        return FluidState(T=self.T, P=self.P, RH=self.RH)

    @classmethod
    def standard_air(cls, T_C=20.0, P_atm=101325.0):
        return cls(T=T_C + 273.15, P=P_atm)
