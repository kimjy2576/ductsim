"""덕트 세그먼트 (Edge) — 직관 마찰 + 부착 컴포넌트 직렬 ΔP"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional
from fluid import FluidState
from geometry import CrossSection
from components import Component, Fan


@dataclass
class DuctSegment:
    id: str
    section: CrossSection
    length: float
    roughness: float = 0.0003
    components: List[Component] = field(default_factory=list)
    label: str = ""
    Q: float = 0.0
    fluid_in: Optional[FluidState] = None
    fluid_out: Optional[FluidState] = None

    def friction_factor(self, Re):
        if Re <= 0: return 0.0
        if Re < 2300: return 64.0 / Re
        e_D = self.roughness / self.section.Dh if self.section.Dh > 0 else 0
        return 0.25 / (math.log10(e_D / 3.7 + 5.74 / Re**0.9))**2

    def friction_dp(self, Q, fluid):
        v = self.section.velocity(Q)
        Re = self.section.Re(Q, fluid.rho, fluid.mu)
        f = self.friction_factor(Re)
        Dh = self.section.Dh
        return f * (self.length / Dh) * 0.5 * fluid.rho * v**2 if Dh > 0 else 0.0

    def total_pressure_drop(self, Q, fluid):
        dp = self.friction_dp(Q, fluid)
        for c in self.components:
            dp += c.pressure_drop(Q, fluid)
        return dp

    def system_dp_no_fan(self, Q, fluid):
        dp = self.friction_dp(Q, fluid)
        for c in self.components:
            if not isinstance(c, Fan):
                dp += c.pressure_drop(Q, fluid)
        return dp

    def total_heat_transfer(self, Q, fluid):
        return sum(c.heat_transfer(Q, fluid) for c in self.components)

    def calc_outlet_temp(self, Q, fluid_in):
        Q_heat = self.total_heat_transfer(Q, fluid_in)
        m_dot = fluid_in.rho * Q
        dT = Q_heat / (m_dot * fluid_in.cp) if m_dot > 1e-10 else 0.0
        return fluid_in.T + dT
