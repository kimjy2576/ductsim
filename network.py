"""덕트 네트워크 — Node-Edge 그래프"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from fluid import FluidState
from duct_segment import DuctSegment


class NodeType(Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    JUNCTION = "junction"


@dataclass
class Node:
    id: str
    node_type: NodeType = NodeType.JUNCTION
    P_boundary: Optional[float] = None
    T_boundary: Optional[float] = None
    fluid: FluidState = field(default_factory=lambda: FluidState.standard_air())


@dataclass
class DuctNetwork:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, DuctSegment] = field(default_factory=dict)
    connectivity: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, seg, from_id, to_id):
        self.edges[seg.id] = seg
        self.connectivity[seg.id] = (from_id, to_id)

    @property
    def inlet_nodes(self):
        return [n for n in self.nodes.values() if n.node_type == NodeType.INLET]

    def get_path_order(self):
        inlets = self.inlet_nodes
        if not inlets: return list(self.edges.keys())
        path, current, visited = [], inlets[0].id, set()
        while True:
            out_edges = [eid for eid, (u, _) in self.connectivity.items()
                         if u == current and eid not in visited]
            if not out_edges: break
            eid = out_edges[0]
            path.append(eid)
            visited.add(eid)
            current = self.connectivity[eid][1]
        return path
