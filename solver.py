"""단일 경로 솔버 — 직렬 ΔP + Fan-System 매칭"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from fluid import FluidState
from components import Fan
from network import DuctNetwork

try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class SinglePathResult:
    Q: float
    total_dp: float
    node_pressures: Dict[str, float] = field(default_factory=dict)
    edge_dp: Dict[str, float] = field(default_factory=dict)
    edge_velocity: Dict[str, float] = field(default_factory=dict)
    edge_Re: Dict[str, float] = field(default_factory=dict)
    node_temperatures: Dict[str, float] = field(default_factory=dict)
    edge_friction_dp: Dict[str, float] = field(default_factory=dict)
    edge_comp_dp: Dict[str, List] = field(default_factory=dict)


def solve_single_path(network, Q, P_inlet=101325.0, T_inlet=293.15):
    path = network.get_path_order()
    result = SinglePathResult(Q=Q, total_dp=0.0)
    fluid = FluidState(T=T_inlet, P=P_inlet)
    inlet = network.inlet_nodes[0]
    P_cur = P_inlet
    result.node_pressures[inlet.id] = P_cur
    result.node_temperatures[inlet.id] = T_inlet

    for eid in path:
        seg = network.edges[eid]
        seg.Q = Q
        seg.fluid_in = fluid.copy()
        dp_fric = seg.friction_dp(Q, fluid)
        result.edge_friction_dp[eid] = dp_fric
        comp_details, dp_comps = [], 0.0
        for c in seg.components:
            cdp = c.pressure_drop(Q, fluid)
            comp_details.append({'name': c.name, 'type': c.comp_type(), 'dp': cdp})
            dp_comps += cdp
        result.edge_comp_dp[eid] = comp_details
        dp_total = dp_fric + dp_comps
        result.edge_dp[eid] = dp_total
        result.edge_velocity[eid] = seg.section.velocity(Q)
        result.edge_Re[eid] = seg.section.Re(Q, fluid.rho, fluid.mu)
        T_out = seg.calc_outlet_temp(Q, fluid)
        P_cur -= dp_total
        result.total_dp += dp_total
        fluid.update(T=T_out, P=P_cur)
        seg.fluid_out = fluid.copy()
        dn_id = network.connectivity[eid][1]
        result.node_pressures[dn_id] = P_cur
        result.node_temperatures[dn_id] = T_out
    return result


def find_operating_point(network, P_inlet=101325.0, T_inlet=293.15, Q_range=(0.01, 5.0)):
    fan = None
    for seg in network.edges.values():
        for c in seg.components:
            if isinstance(c, Fan):
                fan = c; break
        if fan: break
    if fan is None:
        return solve_single_path(network, Q_range[0], P_inlet, T_inlet)

    def residual(Q):
        fluid = FluidState(T=T_inlet, P=P_inlet)
        dp_sys = sum(network.edges[eid].system_dp_no_fan(Q, fluid)
                     for eid in network.get_path_order())
        return dp_sys - fan.fan_dp_positive(Q)

    if HAS_SCIPY:
        try:
            Q_op = brentq(residual, Q_range[0], Q_range[1], xtol=1e-6)
        except ValueError:
            r0, r1 = abs(residual(Q_range[0])), abs(residual(Q_range[1]))
            Q_op = Q_range[0] if r0 < r1 else Q_range[1]
    else:
        lo, hi = Q_range
        for _ in range(60):
            mid = (lo + hi) / 2
            if residual(mid) > 0: lo = mid
            else: hi = mid
        Q_op = (lo + hi) / 2

    return solve_single_path(network, Q_op, P_inlet, T_inlet)


def calc_system_curve(network, Q_array, T_inlet=293.15, P_inlet=101325.0):
    dp_arr = np.zeros_like(Q_array)
    for i, Q in enumerate(Q_array):
        if Q <= 0: continue
        fluid = FluidState(T=T_inlet, P=P_inlet)
        dp_arr[i] = sum(network.edges[eid].system_dp_no_fan(Q, fluid)
                        for eid in network.get_path_order())
    return dp_arr
