"""덕트 단면 기하형상 — 원형 / 직사각형"""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


class CrossSection(ABC):
    @property
    @abstractmethod
    def area(self) -> float: ...
    @property
    @abstractmethod
    def perimeter(self) -> float: ...
    @property
    def Dh(self): return 4.0 * self.area / self.perimeter
    def velocity(self, Q): return Q / self.area if self.area > 0 else 0.0
    def Re(self, Q, rho, mu): return rho * self.velocity(Q) * self.Dh / mu if mu > 0 else 0.0

@dataclass
class CircularSection(CrossSection):
    D: float
    @property
    def area(self): return math.pi * self.D**2 / 4
    @property
    def perimeter(self): return math.pi * self.D

@dataclass
class RectangularSection(CrossSection):
    W: float
    H: float
    @property
    def area(self): return self.W * self.H
    @property
    def perimeter(self): return 2 * (self.W + self.H)
