from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .charts.text import TextChart
from .core.coherence import ConceptualEnergy
from .core.complex import ActiveNodalComplex, ConceptComplex
from .core.node import normalize
from .core.relation import compose_relations


class ConceptualBabel:
    """Raw unbounded Babel potential."""

    def __init__(self, d: int):
        self.d = d


class CoherenceConditionedBabelGenerator:
    def __init__(self, d: int):
        self.d = d
        self.energy_model = ConceptualEnergy(d)
        self.raw_babel = ConceptualBabel(d)
        self._last_complex: Optional[ActiveNodalComplex] = None
        self._closed_counter = 0

    def _compose_relations(self, base: ConceptComplex):
        composed = []
        rels = base.relations
        for i in range(len(rels)):
            for j in range(len(rels)):
                if rels[i].target == rels[j].source and rels[i].source != rels[j].target:
                    composed.append(compose_relations(rels[i], rels[j], relation_type=f"{rels[i].relation_type}_o_{rels[j].relation_type}"))
                    if len(composed) >= 12:
                        return composed
        return composed

    def _condition_state(self, surface: str, base: ConceptComplex, state: np.ndarray, memory: np.ndarray, regime: str) -> np.ndarray:
        surface_vec = TextChart(self.d).concept_vector(surface.strip() or 'continuidad_cerrada')
        regime_vec = TextChart(self.d).concept_vector(f'regime::{regime}')
        active = base.active_state() if base.nodes else np.zeros(self.d)
        return normalize(0.32 * active + 0.26 * state + 0.22 * memory + 0.12 * surface_vec + 0.08 * regime_vec)

    def generate_closed_complex(self, surface: str, state: np.ndarray, memory: np.ndarray, regime: str) -> ActiveNodalComplex:
        chart = TextChart(self.d)
        base = chart.lift(surface) if surface.strip() else (self._last_complex.clone() if self._last_complex is not None else chart.lift('continuidad conceptual'))
        base.regime = regime or base.regime
        conditioned_state = self._condition_state(surface, base, state, memory, base.regime)

        # u/p + H conditioning BEFORE emission
        E0, e, D, L = self.energy_model.energy(base, memory=memory)
        H_diag = D + np.sum(L * L, axis=1)
        residual_norm = float(np.linalg.norm(e))
        coherence_push = float(np.tanh(np.mean(H_diag) / (1.0 + E0)))

        for node in base.nodes.values():
            node.state = normalize(0.76 * node.state + 0.24 * conditioned_state * (1.0 + 0.15 * coherence_push))

        relation_compositions = self._compose_relations(base)
        closure = self.energy_model.closure(base, memory=memory)
        E1, e1, _, _ = self.energy_model.energy(base, memory=memory)
        coherence_mass = float(1.0 / (1.0 + E1 / (self.energy_model.error_dim + 1e-9)))

        self._closed_counter += 1
        out = ActiveNodalComplex(
            nodes=base.nodes,
            relations=base.relations,
            regime=base.regime,
            closure_goal=base.closure_goal,
            history=base.history + [f'one_shot_closed:{self._closed_counter}'],
            potential_cardinality='unbounded',
            relation_compositions=relation_compositions,
            projection_chart='text',
            closure_state=float(closure),
            u_p_residual=e1[: self.d].tolist(),
            coherence_energy=float(E1),
            coherence_mass=coherence_mass,
            generated_one_shot=True,
            closed_complex_id=f'closed-{self._closed_counter}',
            reentry_applied=False,
        )
        self._last_complex = out.clone()
        return out


class CoherenceFlow:
    def __init__(self, d: int):
        self.d = d
        self.energy_model = ConceptualEnergy(d)

    def run(self, initial: ConceptComplex, context: str, memory: Optional[np.ndarray] = None):
        E, _, _, _ = self.energy_model.energy(initial, memory=memory)
        closure = self.energy_model.closure(initial, memory=memory)
        trace = {'generated_one_shot': True, 'coherence_energy': float(E), 'closure': float(closure)}
        return initial.clone(), trace


class ConceptualMemory:
    def __init__(self, d: int): self.d=d; self.memory=np.zeros(d); self.episodes=[]
    def read(self): return self.memory.copy()
    def retrieve(self, query_complex: ConceptComplex, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.episodes:
            return []
        q = query_complex.active_state(); out = []
        for ep in self.episodes:
            ev = np.asarray(ep.get('state', np.zeros(self.d)), dtype=float)
            qn = np.linalg.norm(q); en = np.linalg.norm(ev)
            sim = float(np.dot(q, ev) / (qn * en + 1e-12)) if qn > 1e-12 and en > 1e-12 else 0.0
            influence = max(0.0, sim) * (0.35 + 0.65 * float(ep.get('closure', 0.5)))
            out.append({'episode_id': ep.get('episode_id'), 'similarity': sim, 'influence': influence, 'nodes': list(ep.get('nodes', []))[:10]})
        return sorted(out, key=lambda x: x['influence'], reverse=True)[:top_k]
    def influence_vector(self, retrieved: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not retrieved: return np.zeros(self.d)
        acc = np.zeros(self.d); idx = {ep.get('episode_id'): ep for ep in self.episodes}
        for item in retrieved:
            ep = idx.get(item.get('episode_id'))
            if ep is not None: acc += float(item.get('influence', 0.0)) * np.asarray(ep.get('state', np.zeros(self.d)), dtype=float)
        return normalize(acc) if np.linalg.norm(acc) > 1e-9 else acc
    def reenter(self, complex_, surface, trace):
        active=complex_.active_state(); influence=float(trace.get('memory_retrieval',{}).get('total_influence',0.0))
        if np.linalg.norm(self.memory) < 1e-9: self.memory = active
        else:
            reentry_gain=min(0.32, 0.14 + 0.22 * influence)
            self.memory = normalize((1.0-reentry_gain)*self.memory+reentry_gain*active)
        for n in complex_.nodes.values(): n.memory_trace=min(1.0,n.memory_trace+0.04)
        self.episodes.append({'episode_id': len(self.episodes) + 1, 'surface':surface,'nodes':sorted(list(complex_.nodes.keys())),'relations':len(complex_.relations),'energy':trace.get('coherence_energy'),'closure':trace.get('closure'),'state':active.tolist(),'reentry_gain': float(trace.get('reentry_gain', 0.0))}); self.episodes=self.episodes[-200:]
    def as_serializable(self): return {'memory': self.memory.tolist(), 'episodes': self.episodes}
    @staticmethod
    def from_serializable(d:int,data:Dict[str,Any]): m=ConceptualMemory(d); m.memory=np.asarray(data.get('memory',m.memory),dtype=float); m.episodes=list(data.get('episodes',[])); return m


class ConceptualBabelRuntime:
    def __init__(self, d: int = 48, beam: int = 7, steps: int = 6, state_path: Optional[str] = None):
        if d % 2 != 0: raise ValueError('d must be even for u/p split')
        self.d=d; self.chart=TextChart(d); self.memory=ConceptualMemory(d); self.generator=CoherenceConditionedBabelGenerator(d); self.flow=CoherenceFlow(d); self.state_path=Path(state_path) if state_path else None
        if self.state_path and self.state_path.exists(): self.load(self.state_path)

    def respond(self, surface: str) -> Dict[str, Any]:
        complex0=self.chart.lift(surface if surface.strip() else 'continuidad conceptual')
        retrieved=self.memory.retrieve(complex0,top_k=4)
        influence_vec=self.memory.influence_vector(retrieved)
        base_memory=self.memory.read()
        flow_memory=normalize(0.80 * base_memory + 0.20 * influence_vec) if np.linalg.norm(influence_vec) > 1e-9 else base_memory

        closed = self.generator.generate_closed_complex(surface=surface, state=complex0.active_state(), memory=flow_memory, regime=complex0.regime)
        total_influence=float(sum(x['influence'] for x in retrieved))
        trace = {
            'generated_one_shot': True,
            'conditioning_operator': 'u_p_H',
            'babel_generator_conditioned': True,
            'u_p_residual_norm': float(np.linalg.norm(np.asarray(closed.u_p_residual, dtype=float))),
            'coherence_energy': closed.coherence_energy,
            'coherence_mass': closed.coherence_mass,
            'projection_chart': closed.projection_chart,
            'closed_complex_id': closed.closed_complex_id,
            'reentry_applied': False,
            'closure': closed.closure_state,
            'memory_retrieval': {'retrieved':retrieved,'total_influence':total_influence},
            'reentry_gain': min(0.32, 0.14 + 0.22 * total_influence),
        }
        response=self.chart.project(closed,trace)
        closed.reentry_applied = True
        trace['reentry_applied'] = True
        self.memory.reenter(closed,response,trace)
        if self.state_path: self.save(self.state_path)
        return {'input':surface,'response':response,'complex':closed.as_serializable(),'trace':trace,'memory_episodes':len(self.memory.episodes)}

    def save(self, path: Path): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({'d':self.d,'memory':self.memory.as_serializable()},ensure_ascii=False,indent=2),encoding='utf-8')
    def load(self, path: Path):
        data=json.loads(path.read_text(encoding='utf-8'))
        if int(data.get('d',self.d)) != self.d: raise ValueError('Saved runtime dimension mismatch')
        self.memory = ConceptualMemory.from_serializable(self.d, data.get('memory', {}))
