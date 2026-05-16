from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .charts.text import TextChart
from .core.coherence import ConceptualEnergy
from .core.complex import ConceptComplex
from .core.node import ConceptNode, cosine, normalize
from .core.relation import make_relation


def softmax_neg_energy(energies: Sequence[float], temp: float = 1.0) -> np.ndarray:
    e = -np.asarray(energies, dtype=float) / max(temp, 1e-8)
    e = e - np.max(e)
    w = np.exp(e)
    return w / (np.sum(w) + 1e-12)


class ConceptualBabel:
    def __init__(self, d: int): self.d = d
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


class CoherenceFlow:
    def __init__(self, d: int, beam: int = 6, steps: int = 5, temp: float = 1.0):
        self.d, self.beam, self.steps, self.temp = d, beam, steps, temp
        self.babel = ConceptualBabel(d); self.energy_model = ConceptualEnergy(d)
    def run(self, initial: ConceptComplex, context: str, memory: Optional[np.ndarray] = None):
        frontier=[(initial.clone(),1.0)]; trace={'steps':[]}
        best_complex=initial.clone(); best_energy,_,_,_=self.energy_model.energy(best_complex,memory); best_closure=self.energy_model.closure(best_complex,memory)
        for step in range(self.steps):
            candidates=[]
            for c,_ in frontier: candidates.append(c); candidates.extend(self.babel.expand(c,context))
            scored=[]
            for c in candidates:
                E,_,_,_=self.energy_model.energy(c,memory); closure=self.energy_model.closure(c,memory); scored.append((E-0.25*self.energy_model.error_dim*closure,E,closure,c))
            kept=sorted(scored,key=lambda x:x[0])[:self.beam]; w=softmax_neg_energy([x[0] for x in kept],self.temp); frontier=[(kept[i][3],float(w[i])) for i in range(len(kept))]
            if kept[0][1] < best_energy or kept[0][2] > best_closure: best_energy, best_closure, best_complex = kept[0][1], kept[0][2], kept[0][3].clone()
            trace['steps'].append({'step':step,'candidates':len(candidates),'best_objective':float(kept[0][0]),'best_energy':float(kept[0][1]),'best_closure':float(kept[0][2]),'best_nodes':sorted(list(kept[0][3].nodes.keys()))[:12],'best_history_tail':kept[0][3].history[-4:]})
        trace.update({'energy':float(best_energy),'closure':float(best_closure),'nodes':sorted(list(best_complex.nodes.keys())),'relations':len(best_complex.relations),'potential':best_complex.potential_cardinality})
        return best_complex, trace


class ConceptualMemory:
    def __init__(self, d: int): self.d=d; self.memory=np.zeros(d); self.episodes=[]
    def read(self): return self.memory.copy()
    def retrieve(self, query_complex: ConceptComplex, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.episodes:
            return []
        q = query_complex.active_state()
        scored = []
        for ep in self.episodes:
            v = np.asarray(ep.get('state', np.zeros(self.d)), dtype=float)
            if v.shape[0] != self.d:
                continue
            sim = 0.0 if np.linalg.norm(v) < 1e-9 else max(-1.0, min(1.0, cosine(q, v)))
            scored.append((sim, ep))
        out = []
        for sim, ep in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]:
            influence = max(0.0, min(1.0, 0.5 * (sim + 1.0)))
            out.append({'episode_id': ep['episode_id'], 'similarity': float(sim), 'influence': float(influence), 'nodes': list(ep.get('nodes', []))[:8]})
        return out
    def influence_memory(self, retrieval: List[Dict[str, Any]]) -> np.ndarray:
        if not retrieval:
            return self.read()
        total = sum(item['influence'] for item in retrieval) + 1e-9
        vec = np.zeros(self.d)
        by_id = {ep['episode_id']: ep for ep in self.episodes}
        for item in retrieval:
            ep = by_id.get(item['episode_id'])
            if not ep:
                continue
            vec += (item['influence'] / total) * np.asarray(ep.get('state', np.zeros(self.d)), dtype=float)
        return normalize(0.65 * self.memory + 0.35 * vec) if np.linalg.norm(vec) > 1e-9 else self.read()
    def reenter(self, complex_, surface, trace):
        active=complex_.active_state(); self.memory = active if np.linalg.norm(self.memory) < 1e-9 else normalize(0.86*self.memory+0.14*active)
        for n in complex_.nodes.values(): n.memory_trace=min(1.0,n.memory_trace+0.04)
        self.episodes.append({'episode_id': len(self.episodes), 'surface':surface, 'state': active.tolist(), 'nodes':sorted(list(complex_.nodes.keys())),'relations':len(complex_.relations),'energy':trace.get('energy'),'closure':trace.get('closure')}); self.episodes=self.episodes[-200:]
    def as_serializable(self): return {'memory': self.memory.tolist(), 'episodes': self.episodes}
    @staticmethod
    def from_serializable(d:int,data:Dict[str,Any]): m=ConceptualMemory(d); m.memory=np.asarray(data.get('memory',m.memory),dtype=float); m.episodes=list(data.get('episodes',[])); return m


class ConceptualBabelRuntime:
    def __init__(self, d: int = 48, beam: int = 7, steps: int = 6, state_path: Optional[str] = None):
        if d % 2 != 0: raise ValueError('d must be even for u/p split')
        self.d=d; self.chart=TextChart(d); self.flow=CoherenceFlow(d,beam=beam,steps=steps,temp=1.0); self.memory=ConceptualMemory(d); self.state_path=Path(state_path) if state_path else None
        if self.state_path and self.state_path.exists(): self.load(self.state_path)
    def respond(self, surface: str) -> Dict[str, Any]:
        complex0=self.chart.lift(surface)
        retrieval = self.memory.retrieve(complex0)
        memory_state = self.memory.influence_memory(retrieval)
        best,trace=self.flow.run(complex0,context=surface,memory=memory_state)
        trace['memory_retrieval'] = retrieval
        trace['reentry'] = {'prior_episode_count': len(self.memory.episodes), 'memory_norm_before': float(np.linalg.norm(self.memory.read())), 'influence_total': float(sum(x['influence'] for x in retrieval))}
        response=self.chart.project(best,trace)
        self.memory.reenter(best,response,trace)
        trace['reentry'].update({'posterior_episode_count': len(self.memory.episodes), 'memory_norm_after': float(np.linalg.norm(self.memory.read()))})
        if self.state_path: self.save(self.state_path)
        return {'input':surface,'response':response,'complex':best.as_serializable(),'trace':trace,'memory_episodes':len(self.memory.episodes)}
    def save(self, path: Path): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({'d':self.d,'memory':self.memory.as_serializable()},ensure_ascii=False,indent=2),encoding='utf-8')
    def load(self, path: Path):
        data=json.loads(path.read_text(encoding='utf-8'))
        if int(data.get('d',self.d)) != self.d: raise ValueError('Saved runtime dimension mismatch')
        self.memory = ConceptualMemory.from_serializable(self.d, data.get('memory', {}))
