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
    regime: str = "technical"
    closure_goal: str = "explain"
    history: List[str] = field(default_factory=list)
    potential_cardinality: str = "unbounded"
    def clone(self)->"ConceptComplex":
        return ConceptComplex(nodes={k:v.clone() for k,v in self.nodes.items()},relations=[r.clone() for r in self.relations],regime=self.regime,closure_goal=self.closure_goal,history=list(self.history),potential_cardinality=self.potential_cardinality)
    def active_state(self)->np.ndarray:
        if not self.nodes: return np.zeros(1)
        return np.mean([n.state for n in self.nodes.values()],axis=0)
    def signature(self)->str:
        ns=",".join(sorted(self.nodes.keys())); rs=",".join(f"{r.source}->{r.target}:{r.relation_type}" for r in self.relations)
        return f"{self.regime}|{self.closure_goal}|{ns}|{rs}"
    def as_serializable(self)->Dict[str,Any]:
        return {"nodes":{k:v.as_serializable() for k,v in self.nodes.items()},"relations":[r.as_serializable() for r in self.relations],"regime":self.regime,"closure_goal":self.closure_goal,"history":self.history,"potential_cardinality":self.potential_cardinality}
    @staticmethod
    def from_serializable(d:Dict[str,Any])->"ConceptComplex":
        return ConceptComplex(nodes={k:ConceptNode.from_serializable(v) for k,v in d.get("nodes",{}).items()},relations=[RelationOperator.from_serializable(r) for r in d.get("relations",[])],regime=d.get("regime","technical"),closure_goal=d.get("closure_goal","explain"),history=list(d.get("history",[])),potential_cardinality=d.get("potential_cardinality","unbounded"))
