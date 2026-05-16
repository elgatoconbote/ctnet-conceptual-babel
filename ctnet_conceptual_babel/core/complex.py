from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

from .node import ConceptNode
from .relation import RelationOperator


@dataclass
class ConceptComplex:
    nodes: Dict[str, ConceptNode] = field(default_factory=dict)
    relations: List[RelationOperator] = field(default_factory=list)
    regime: str = 'technical'
    closure_goal: str = 'explain'
    history: List[str] = field(default_factory=list)
    potential_cardinality: str = 'unbounded'
    relation_compositions: List[str] = field(default_factory=list)
    projection_chart: str = 'text'
    closure_state: str = 'open'
    u_p_residual: float = 0.0
    coherence_energy: float = 0.0
    coherence_mass: float = 0.0
    generated_one_shot: bool = False

    def clone(self) -> 'ConceptComplex':
        return ConceptComplex({k: v.clone() for k, v in self.nodes.items()}, [r.clone() for r in self.relations], self.regime, self.closure_goal, list(self.history), self.potential_cardinality, list(self.relation_compositions), self.projection_chart, self.closure_state, self.u_p_residual, self.coherence_energy, self.coherence_mass, self.generated_one_shot)

    def active_state(self) -> np.ndarray:
        return np.zeros(1) if not self.nodes else np.mean([n.state for n in self.nodes.values()], axis=0)

    def signature(self) -> str:
        ns = ','.join(sorted(self.nodes.keys()))
        rs = ','.join(f'{r.source}->{r.target}:{r.relation_type}' for r in self.relations)
        return f'{self.regime}|{self.closure_goal}|{ns}|{rs}'

    def as_serializable(self) -> Dict[str, Any]:
        return {'nodes': {k: v.as_serializable() for k, v in self.nodes.items()}, 'relations': [r.as_serializable() for r in self.relations], 'regime': self.regime, 'closure_goal': self.closure_goal, 'history': self.history, 'potential_cardinality': self.potential_cardinality, 'relation_compositions': self.relation_compositions, 'projection_chart': self.projection_chart, 'closure_state': self.closure_state, 'u_p_residual': self.u_p_residual, 'coherence_energy': self.coherence_energy, 'coherence_mass': self.coherence_mass, 'generated_one_shot': self.generated_one_shot}

    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> 'ConceptComplex':
        return ConceptComplex(nodes={k: ConceptNode.from_serializable(v) for k, v in d.get('nodes', {}).items()}, relations=[RelationOperator.from_serializable(r) for r in d.get('relations', [])], regime=d.get('regime', 'technical'), closure_goal=d.get('closure_goal', 'explain'), history=list(d.get('history', [])), potential_cardinality=d.get('potential_cardinality', 'unbounded'), relation_compositions=list(d.get('relation_compositions', [])), projection_chart=d.get('projection_chart', 'text'), closure_state=d.get('closure_state', 'open'), u_p_residual=float(d.get('u_p_residual', 0.0)), coherence_energy=float(d.get('coherence_energy', 0.0)), coherence_mass=float(d.get('coherence_mass', 0.0)), generated_one_shot=bool(d.get('generated_one_shot', False)))
