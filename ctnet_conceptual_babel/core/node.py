from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


def stable_seed(text: str) -> int:
    h = 2166136261
    for b in text.encode('utf-8', errors='replace'):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def rng_for(text: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(text))


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x.copy()
    return x / n


def rmsnorm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x) + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(np.ravel(a), np.ravel(b)) / den)


@dataclass
class ConceptNode:
    name: str
    state: np.ndarray
    complexity_level: int = 0
    charts: Dict[str, str] = field(default_factory=dict)
    memory_trace: float = 0.0
    potential_cardinality: str = 'unbounded'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> 'ConceptNode':
        return ConceptNode(self.name, self.state.copy(), self.complexity_level, dict(self.charts), float(self.memory_trace), self.potential_cardinality, json.loads(json.dumps(self.metadata, ensure_ascii=False)))

    def as_serializable(self) -> Dict[str, Any]:
        return {'name': self.name, 'state': self.state.tolist(), 'complexity_level': self.complexity_level, 'charts': self.charts, 'memory_trace': self.memory_trace, 'potential_cardinality': self.potential_cardinality, 'metadata': self.metadata}

    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> 'ConceptNode':
        return ConceptNode(d['name'], np.asarray(d['state'], dtype=float), int(d.get('complexity_level', 0)), dict(d.get('charts', {})), float(d.get('memory_trace', 0.0)), d.get('potential_cardinality', 'unbounded'), dict(d.get('metadata', {})))

    def refine(self, context: str, max_children: int = 3) -> List['ConceptNode']:
        rng = rng_for(f'refine::{self.name}::{context}::{self.complexity_level}')
        children: List[ConceptNode] = []
        for idx, child_name in enumerate(self._child_names(context, max_children)):
            perturb = rng.normal(0.0, 0.055 + 0.015 * self.complexity_level, self.state.shape)
            gate = 0.72 + 0.06 * math.sin(idx + self.complexity_level)
            child_state = normalize(gate * self.state + perturb)
            children.append(ConceptNode(child_name, child_state, self.complexity_level + 1, {'spanish': self._spanish_projection(child_name), 'symbolic': child_name}, 0.8 * self.memory_trace, 'unbounded', {'parent': self.name, 'refinement_context': context}))
        return children

    def _child_names(self, context: str, max_children: int) -> List[str]:
        presets = {'biblioteca_babel': ['potencial_expresivo_total', 'proyeccion_textual_fraccionaria', 'clausura_nodal_relacional'], 'tensor_coherencia': ['metrica_de_error', 'masa_de_cierre', 'ponderacion_direccional'], 'u_p': ['actuacion', 'inercia_formal', 'reciprocidad_estructural'], 'nodo_conceptual': ['estado_interno', 'valencia_relacional', 'proyecciones_posibles'], 'relacion_infinita': ['composicion', 'implicacion', 'compatibilidad']}
        base = presets.get(self.name, [f'{self.name}_aspecto_{i+1}' for i in range(max_children)])
        return base[:max_children]

    @staticmethod
    def _spanish_projection(name: str) -> str:
        return name.replace('_', ' ')


@dataclass
class FractionNode(ConceptNode):
    fraction: str = ''
    chart_name: str = 'surface'

    def clone(self) -> 'FractionNode':
        return FractionNode(self.name, self.state.copy(), self.complexity_level, dict(self.charts), float(self.memory_trace), self.potential_cardinality, json.loads(json.dumps(self.metadata, ensure_ascii=False)), self.fraction, self.chart_name)
