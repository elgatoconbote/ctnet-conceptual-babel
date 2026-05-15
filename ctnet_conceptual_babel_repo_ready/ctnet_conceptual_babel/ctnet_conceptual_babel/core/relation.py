from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np
from .node import rng_for

@dataclass
class RelationOperator:
    source: str
    target: str
    relation_type: str
    matrix: np.ndarray
    weight: float = 1.0
    expected_alignment: float = 0.72
    metadata: Dict[str, Any] = field(default_factory=dict)
    def clone(self) -> "RelationOperator":
        return RelationOperator(self.source,self.target,self.relation_type,self.matrix.copy(),float(self.weight),float(self.expected_alignment),json.loads(json.dumps(self.metadata, ensure_ascii=False)))
    def as_serializable(self)->Dict[str,Any]:
        return {"source":self.source,"target":self.target,"relation_type":self.relation_type,"matrix":self.matrix.tolist(),"weight":self.weight,"expected_alignment":self.expected_alignment,"metadata":self.metadata}
    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> "RelationOperator":
        return RelationOperator(d["source"],d["target"],d["relation_type"],np.asarray(d["matrix"],dtype=float),float(d.get("weight",1.0)),float(d.get("expected_alignment",0.72)),dict(d.get("metadata",{})))

def make_relation(source:str,target:str,relation_type:str,d:int,weight:float=1.0)->RelationOperator:
    rng=rng_for(f"relation::{source}->{target}:{relation_type}")
    a=rng.normal(0.0,0.025,(d,d))
    return RelationOperator(source,target,relation_type,np.eye(d)+a,weight=weight)
