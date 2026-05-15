from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .complex import ConceptComplex
from .node import normalize, rng_for
from .relation import make_relation
from .up import UPBundle


def softmax_neg_energy(energies: Sequence[float], temp: float = 1.0) -> np.ndarray:
    e = -np.asarray(energies, dtype=float) / max(temp, 1e-8)
    e = e - np.max(e)
    w = np.exp(e)
    return w / (np.sum(w) + 1e-12)

# add cosine util fallback

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a=np.ravel(a); b=np.ravel(b); den=np.linalg.norm(a)*np.linalg.norm(b)+1e-12
    return float(np.dot(a,b)/den)

class CoherenceTensor:
    def __init__(self,error_dim:int,rank:int=8,seed:str="H"):
        self.error_dim=error_dim; self.rank=rank; rng=rng_for(seed)
        self.base_D=0.75+0.5*rng.random(error_dim); self.base_L=rng.normal(0.0,0.055,(error_dim,rank))
    def instantiate(self, complex_: ConceptComplex):
        regime_gain={"conceptual":1.00,"formal":1.18,"implementation":1.32,"critical":1.45}.get(complex_.regime,1.0)
        D=self.base_D*regime_gain*(1.0+0.025*len(complex_.nodes)); L=self.base_L*math.sqrt(1.0+0.012*len(complex_.relations)); return D,L
    @staticmethod
    def energy(error,D,L): return float(np.sum(D*error*error)+np.sum((L.T@error)**2))

class ConceptualEnergy:
    def __init__(self,d:int,max_nodes:int=12,max_relations:int=32):
        self.d=d; self.max_nodes=max_nodes; self.max_relations=max_relations; self.up=UPBundle(d)
        self.error_dim=d*max_nodes+d*max_relations+8; self.tensor=CoherenceTensor(self.error_dim,rank=10)
    def _neighborhood_state(self,name,complex_):
        if name not in complex_.nodes: return np.zeros(self.d)
        related=[]
        for rel in complex_.relations:
            if rel.source==name and rel.target in complex_.nodes: related.append(rel.matrix@complex_.nodes[rel.target].state)
            elif rel.target==name and rel.source in complex_.nodes: related.append(rel.matrix.T@complex_.nodes[rel.source].state)
        if not related: return complex_.nodes[name].state
        return normalize(np.mean(related,axis=0))
    @staticmethod
    def _projection_readiness(complex_): return sum(1 for n in complex_.nodes.values() if "spanish" in n.charts)/len(complex_.nodes) if complex_.nodes else 0.0
    @staticmethod
    def _contradiction_pressure(complex_):
        names=set(complex_.nodes.keys()); p=0.0
        if "fraccion_superficial" in names and "nodo_conceptual" in names: p-=0.05
        if "token_ontology" in names and "nodo_conceptual" in names: p+=0.8
        return max(0.0,p)
    def error_field(self,complex_,memory=None):
        memory=np.zeros(self.d) if memory is None else memory; errors=[]
        node_items=list(complex_.nodes.items())[:self.max_nodes]
        for name,node in node_items: errors.append(self.up.residual_node(node,self._neighborhood_state(name,complex_)))
        errors += [np.zeros(self.d)]*(self.max_nodes-len(node_items))
        rel_items=complex_.relations[:self.max_relations]
        for rel in rel_items: errors.append(self.up.residual_relation(rel,complex_.nodes))
        errors += [np.zeros(self.d)]*(self.max_relations-len(rel_items))
        active=complex_.active_state(); mem_err=1.0-cosine(active,memory) if np.linalg.norm(memory)>1e-9 else 0.2
        rd=len(complex_.relations)/max(len(complex_.nodes),1); cx=np.mean([n.complexity_level for n in complex_.nodes.values()]) if complex_.nodes else 0.0
        pr=self._projection_readiness(complex_); cd=self._contradiction_pressure(complex_); op=1.0 if complex_.potential_cardinality=="unbounded" else 0.0
        ge=np.array([mem_err,max(0.0,1.0-pr),1.0/(1.0+rd+pr),cd,0.05*cx,0.1/(1.0+rd),0.15 if complex_.regime in {"conceptual","implementation","formal"} else 0.4,0.01*op])
        return np.concatenate(errors+[ge])
    def energy(self,complex_,memory=None):
        e=self.error_field(complex_,memory); D,L=self.tensor.instantiate(complex_); return self.tensor.energy(e,D,L),e,D,L
    def closure(self,complex_,memory=None):
        E,_,_,_=self.energy(complex_,memory); rd=len(complex_.relations)/max(len(complex_.nodes),1); pr=self._projection_readiness(complex_)
        return float(np.exp(-E/(self.error_dim+1e-9))*(0.65+0.20*pr+0.15*min(rd/3.0,1.0)))
