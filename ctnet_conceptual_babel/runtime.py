from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .charts.text import TextChart
from .core.coherence import ConceptualEnergy
from .core.complex import ConceptComplex
from .core.node import ConceptNode, normalize
from .core.relation import compose_relations, make_relation


class ConceptualBabel:
    def __init__(self, d: int):
        self.d = d

    def expand(self, complex_: ConceptComplex, context: str) -> List[ConceptComplex]:
        expansions = self._refine_high_potential_nodes(complex_, context) + self._add_missing_core_relations(complex_) + self._add_projection_node_if_needed(complex_) + self._add_closure_node_if_needed(complex_)
        out, seen = [], set()
        for c in expansions:
            c.potential_cardinality = 'unbounded'
            if c.signature() not in seen:
                seen.add(c.signature()); out.append(c)
        return out

    def _refine_high_potential_nodes(self, complex_, context):
        ex=[]
        for node in sorted(complex_.nodes.values(), key=lambda n:(n.memory_trace,-n.complexity_level,n.name), reverse=True)[:3]:
            if node.complexity_level>=2: continue
            c=complex_.clone()
            for child in node.refine(context,3):
                c.nodes[child.name]=child; c.relations.append(make_relation(node.name,child.name,'refina',self.d,0.55)); c.relations.append(make_relation(child.name,node.name,'pertenece_a',self.d,0.35))
            c.history.append(f'refine:{node.name}'); ex.append(c)
        return ex

    def _add_missing_core_relations(self, complex_):
        names=set(complex_.nodes); desired=[]
        if {'biblioteca_babel','tensor_coherencia'}.issubset(names): desired.append(('tensor_coherencia','biblioteca_babel','metriza'))
        if {'u_p','tensor_coherencia'}.issubset(names): desired.append(('u_p','tensor_coherencia','produce_error_para'))
        if {'nodo_conceptual','relacion_infinita'}.issubset(names): desired.append(('nodo_conceptual','relacion_infinita','se_define_por'))
        if {'fraccion_superficial','nodo_conceptual'}.issubset(names): desired.append(('fraccion_superficial','nodo_conceptual','se_eleva_a'))
        if {'proyeccion_textual','nodo_conceptual'}.issubset(names): desired.append(('nodo_conceptual','proyeccion_textual','se_proyecta_en'))
        existing={(r.source,r.target,r.relation_type) for r in complex_.relations}; out=[]
        for src,tgt,typ in desired:
            if (src,tgt,typ) not in existing:
                c=complex_.clone(); c.relations.append(make_relation(src,tgt,typ,self.d,0.95)); c.history.append(f'relate:{src}->{tgt}:{typ}'); out.append(c)
        return out

    def _add_projection_node_if_needed(self, complex_):
        if 'proyeccion_textual' in complex_.nodes: return []
        c=complex_.clone(); chart=TextChart(self.d)
        c.nodes['proyeccion_textual']=ConceptNode('proyeccion_textual', chart.concept_vector('proyeccion_textual'), charts={'spanish':'proyección textual','symbolic':'π_text'}, metadata={'role':'surface_projection_chart'})
        for name in list(complex_.nodes.keys())[:6]: c.relations.append(make_relation(name,'proyeccion_textual','puede_proyectarse',self.d,0.30))
        c.history.append('add:proyeccion_textual'); return [c]

    def _add_closure_node_if_needed(self, complex_):
        if 'cierre_estructural' in complex_.nodes: return []
        c=complex_.clone(); chart=TextChart(self.d)
        c.nodes['cierre_estructural']=ConceptNode('cierre_estructural', chart.concept_vector('cierre_estructural'), charts={'spanish':'cierre estructural','symbolic':'CLOSE'}, metadata={'role':'closure_condition'})
        for name in list(complex_.nodes.keys())[:6]: c.relations.append(make_relation(name,'cierre_estructural','debe_cerrar_en',self.d,0.25))
        c.history.append('add:cierre_estructural'); return [c]


class CoherenceConditionedBabelGenerator:
    def __init__(self, d: int):
        self.d = d
        self.babel = ConceptualBabel(d)
        self.energy_model = ConceptualEnergy(d)

    def generate_closed_complex(self, surface: str, state: Optional[ConceptComplex], memory: Optional[np.ndarray], regime: str = 'conceptual') -> ConceptComplex:
        conditioned = self._conditioned_seed(surface, state, memory, regime)
        if state is not None and state.relations and conditioned.relations:
            for r_prev in state.relations[:3]:
                for r_new in conditioned.relations[:3]:
                    if r_prev.target == r_new.source:
                        conditioned.relations.append(compose_relations(r_prev, r_new, relation_type='reentry_composition'))
                        conditioned.relation_compositions.append(f'{r_prev.source}->{r_new.target}')
                        break
        E, e, D, L = self.energy_model.energy(conditioned, memory)
        closure = self.energy_model.closure(conditioned, memory)
        conditioned.closure_state = 'closed' if closure >= 0.08 else 'open'
        conditioned.u_p_residual = float(np.linalg.norm(e))
        conditioned.coherence_energy = float(E)
        conditioned.coherence_mass = float(np.linalg.norm(D) + np.linalg.norm(L))
        conditioned.generated_one_shot = True
        conditioned.history.append('one_shot_conditioned_generation')
        return conditioned

    def _conditioned_seed(self, surface: str, state: Optional[ConceptComplex], memory: Optional[np.ndarray], regime: str) -> ConceptComplex:
        chart = TextChart(self.d)
        lifted = chart.lift(surface if surface.strip() else 'continuidad_conceptual')
        lifted.regime = regime
        if state is not None:
            for k, v in list(state.nodes.items())[:4]:
                if k not in lifted.nodes:
                    lifted.nodes[k] = v.clone()
            lifted.history.append('state_continuation')
        expansions = self.babel.expand(lifted, surface)
        base = expansions[0] if expansions else lifted
        if memory is not None and np.linalg.norm(memory) > 1e-9:
            for node in base.nodes.values():
                node.state = normalize(0.88 * node.state + 0.12 * memory)
            base.history.append('u_p_H_conditioned')
        return base


class CoherenceFlow:
    def __init__(self, d: int, beam: int = 6, steps: int = 5, temp: float = 1.0):
        self.d = d
        self.energy_model = ConceptualEnergy(d)
        self.generator = CoherenceConditionedBabelGenerator(d)

    def run(self, initial: ConceptComplex, context: str, memory: Optional[np.ndarray] = None):
        closed = self.generator.generate_closed_complex(context, initial, memory, regime=initial.regime)
        closure = self.energy_model.closure(closed, memory)
        trace = {'steps': [{'step': 0, 'generated_one_shot': True}], 'energy': float(closed.coherence_energy), 'closure': float(closure), 'nodes': sorted(list(closed.nodes.keys())), 'relations': len(closed.relations), 'potential': closed.potential_cardinality}
        return closed, trace


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
        closure_val = trace.get('closure', 0.5)
        if closure_val is None:
            closure_val = 0.5
        self.episodes.append({'episode_id': len(self.episodes) + 1, 'surface':surface,'nodes':sorted(list(complex_.nodes.keys())),'relations':len(complex_.relations),'energy':trace.get('coherence_energy'),'closure':float(closure_val),'state':active.tolist(),'reentry_gain': float(trace.get('reentry_gain', 0.0))}); self.episodes=self.episodes[-200:]
    def as_serializable(self): return {'memory': self.memory.tolist(), 'episodes': self.episodes}
    @staticmethod
    def from_serializable(d:int,data:Dict[str,Any]): m=ConceptualMemory(d); m.memory=np.asarray(data.get('memory',m.memory),dtype=float); m.episodes=list(data.get('episodes',[])); return m


class ConceptualBabelRuntime:
    def __init__(self, d: int = 48, beam: int = 7, steps: int = 6, state_path: Optional[str] = None):
        if d % 2 != 0: raise ValueError('d must be even for u/p split')
        self.d=d; self.chart=TextChart(d); self.generator=CoherenceConditionedBabelGenerator(d); self.flow=CoherenceFlow(d, beam=beam, steps=steps, temp=1.0); self.memory=ConceptualMemory(d); self.state_path=Path(state_path) if state_path else None; self.last_complex: Optional[ConceptComplex] = None
        if self.state_path and self.state_path.exists(): self.load(self.state_path)
    def respond(self, surface: str) -> Dict[str, Any]:
        complex0 = self.chart.lift(surface if surface.strip() else 'continuidad_conceptual')
        retrieved=self.memory.retrieve(complex0,top_k=4)
        influence_vec=self.memory.influence_vector(retrieved)
        base_memory=self.memory.read()
        flow_memory=normalize(0.80 * base_memory + 0.20 * influence_vec) if np.linalg.norm(influence_vec) > 1e-9 else base_memory
        closed = self.generator.generate_closed_complex(surface, self.last_complex, flow_memory, regime=complex0.regime)
        response=self.chart.project(closed, {'energy': closed.coherence_energy, 'closure': closed.coherence_mass/(closed.coherence_mass+1.0), 'memory_retrieval': {'retrieved': retrieved, 'total_influence': float(sum(x['influence'] for x in retrieved))}, 'reentry_gain': min(0.32, 0.14 + 0.22 * float(sum(x['influence'] for x in retrieved)))})
        closure = self.generator.energy_model.closure(closed, flow_memory)
        trace = {
            'generated_one_shot': True,
            'conditioning_operator': 'u_p_H',
            'babel_generator_conditioned': True,
            'u_p_residual_norm': float(closed.u_p_residual),
            'coherence_energy': float(closed.coherence_energy),
            'coherence_mass': float(closed.coherence_mass),
            'projection_chart': 'text',
            'closed_complex_id': closed.signature(),
            'reentry_applied': True,
            'closure': float(closure),
            'memory_retrieval': {'retrieved': retrieved, 'total_influence': float(sum(x['influence'] for x in retrieved))},
            'reentry_gain': min(0.32, 0.14 + 0.22 * float(sum(x['influence'] for x in retrieved))),
        }
        self.memory.reenter(closed,response,trace)
        self.last_complex = closed.clone()
        if self.state_path: self.save(self.state_path)
        return {'input':surface,'response':response,'complex':closed.as_serializable(),'trace':trace,'memory_episodes':len(self.memory.episodes)}
    def save(self, path: Path): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({'d':self.d,'memory':self.memory.as_serializable(),'last_complex': self.last_complex.as_serializable() if self.last_complex else None},ensure_ascii=False,indent=2),encoding='utf-8')
    def load(self, path: Path):
        data=json.loads(path.read_text(encoding='utf-8'))
        if int(data.get('d',self.d)) != self.d: raise ValueError('Saved runtime dimension mismatch')
        self.memory = ConceptualMemory.from_serializable(self.d, data.get('memory', {}))
        if data.get('last_complex'): self.last_complex = ConceptComplex.from_serializable(data['last_complex'])
