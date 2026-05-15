from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from .node import ConceptNode, normalize, rng_for
from .relation import RelationOperator

class UPBundle:
    def __init__(self,d:int):
        self.d=d; self.half=d//2; rng=rng_for('UPBundle'); self.F=rng.normal(0.0,0.14,(self.half,self.half)); self.G=rng.normal(0.0,0.14,(self.half,self.half))
    def split(self,x:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        if len(x)!=self.d: raise ValueError(f'Expected dimension {self.d}, got {len(x)}')
        return x[:self.half],x[self.half:]
    def residual_node(self,node:ConceptNode, neighborhood:np.ndarray)->np.ndarray:
        x=normalize(0.82*node.state+0.18*neighborhood);u,p=self.split(x);pred_u=np.tanh(p@self.F);pred_p=np.tanh(u@self.G);return np.concatenate([u-pred_u,p-pred_p])
    def residual_relation(self,rel:RelationOperator,nodes:Dict[str,ConceptNode])->np.ndarray:
        if rel.source not in nodes or rel.target not in nodes:return np.zeros(self.d)
        src,tgt=nodes[rel.source].state,nodes[rel.target].state; transported=normalize(rel.matrix@tgt); expected=normalize(src)
        return rel.weight*(transported-rel.expected_alignment*expected)
