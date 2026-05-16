from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .node import ConceptNode
from .relation import RelationOperator


@dataclass
class ConceptComplex:
    nodes: Dict[str, ConceptNode] = field(default_factory=dict)
    relations: List[RelationOperator] = field(default_factory=list)
    regime: str = "technical"
    closure_goal: str = "explain"
    history: List[str] = field(default_factory=list)
    potential_cardinality: str = "unbounded"

    def clone(self) -> "ConceptComplex":
        return ConceptComplex(
            nodes={k: v.clone() for k, v in self.nodes.items()},
            relations=[r.clone() for r in self.relations],
            regime=self.regime,
            closure_goal=self.closure_goal,
            history=list(self.history),
            potential_cardinality=self.potential_cardinality,
        )

    def active_state(self) -> np.ndarray:
        return np.zeros(1) if not self.nodes else np.mean([n.state for n in self.nodes.values()], axis=0)

    def signature(self) -> str:
        ns = ",".join(sorted(self.nodes.keys()))
        rs = ",".join(f"{r.source}->{r.target}:{r.relation_type}" for r in self.relations)
        return f"{self.regime}|{self.closure_goal}|{ns}|{rs}"

    def as_serializable(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.as_serializable() for k, v in self.nodes.items()},
            "relations": [r.as_serializable() for r in self.relations],
            "regime": self.regime,
            "closure_goal": self.closure_goal,
            "history": self.history,
            "potential_cardinality": self.potential_cardinality,
        }

    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> "ConceptComplex":
        return ConceptComplex(
            nodes={k: ConceptNode.from_serializable(v) for k, v in d.get("nodes", {}).items()},
            relations=[RelationOperator.from_serializable(r) for r in d.get("relations", [])],
            regime=d.get("regime", "technical"),
            closure_goal=d.get("closure_goal", "explain"),
            history=list(d.get("history", [])),
            potential_cardinality=d.get("potential_cardinality", "unbounded"),
        )


@dataclass
class ActiveNodalComplex(ConceptComplex):
    relation_compositions: List[RelationOperator] = field(default_factory=list)
    projection_chart: str = "text"
    closure_state: float = 0.0
    u_p_residual: List[float] = field(default_factory=list)
    coherence_energy: float = 0.0
    coherence_mass: float = 0.0
    generated_one_shot: bool = True
    closed_complex_id: Optional[str] = None
    reentry_applied: bool = False
    surface_emission: str = ""

    def clone(self) -> "ActiveNodalComplex":
        return ActiveNodalComplex(
            nodes={k: v.clone() for k, v in self.nodes.items()},
            relations=[r.clone() for r in self.relations],
            regime=self.regime,
            closure_goal=self.closure_goal,
            history=list(self.history),
            potential_cardinality=self.potential_cardinality,
            relation_compositions=[r.clone() for r in self.relation_compositions],
            projection_chart=self.projection_chart,
            closure_state=float(self.closure_state),
            u_p_residual=list(self.u_p_residual),
            coherence_energy=float(self.coherence_energy),
            coherence_mass=float(self.coherence_mass),
            generated_one_shot=bool(self.generated_one_shot),
            closed_complex_id=self.closed_complex_id,
            reentry_applied=bool(self.reentry_applied),
            surface_emission=self.surface_emission,
        )

    def as_serializable(self) -> Dict[str, Any]:
        base = super().as_serializable()
        base.update(
            {
                "relation_compositions": [r.as_serializable() for r in self.relation_compositions],
                "projection_chart": self.projection_chart,
                "closure_state": self.closure_state,
                "u_p_residual": self.u_p_residual,
                "coherence_energy": self.coherence_energy,
                "coherence_mass": self.coherence_mass,
                "generated_one_shot": self.generated_one_shot,
                "closed_complex_id": self.closed_complex_id,
                "reentry_applied": self.reentry_applied,
                "surface_emission": self.surface_emission,
            }
        )
        return base

    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> "ActiveNodalComplex":
        base = ConceptComplex.from_serializable(d)
        return ActiveNodalComplex(
            nodes=base.nodes,
            relations=base.relations,
            regime=base.regime,
            closure_goal=base.closure_goal,
            history=base.history,
            potential_cardinality=base.potential_cardinality,
            relation_compositions=[RelationOperator.from_serializable(r) for r in d.get("relation_compositions", [])],
            projection_chart=d.get("projection_chart", "text"),
            closure_state=float(d.get("closure_state", 0.0)),
            u_p_residual=list(d.get("u_p_residual", [])),
            coherence_energy=float(d.get("coherence_energy", 0.0)),
            coherence_mass=float(d.get("coherence_mass", 0.0)),
            generated_one_shot=bool(d.get("generated_one_shot", True)),
            closed_complex_id=d.get("closed_complex_id"),
            reentry_applied=bool(d.get("reentry_applied", False)),
            surface_emission=d.get("surface_emission", ""),
        )
