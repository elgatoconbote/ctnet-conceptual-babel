from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .charts.text import TextChart
from .core.complex import ConceptComplex
from .core.coherence import ConceptualEnergy, softmax_neg_energy
from .core.node import ConceptNode, normalize
from .core.relation import make_relation

class ConceptualBabel:
    def __init__(self,d:int): self.d=d
    def expand(self,complex_,context):
        ex=[]; ex.extend(self._refine_high_potential_nodes(complex_,context)); ex.extend(self._add_missing_core_relations(complex_)); ex.extend(self._add_projection_node_if_needed(complex_)); ex.extend(self._add_closure_node_if_needed(complex_));
        seen=set(); out=[]
        for c in ex:
            c.potential_cardinality='unbounded'; s=c.signature();
            if s not in seen: out.append(c); seen.add(s)
        return out
    def _refine_high_potential_nodes(self,complex_,context):
        expansions=[]; candidates=sorted(complex_.nodes.values(),key=lambda n:(n.memory_trace,-n.complexity_level,n.name),reverse=True)[:3]
        for node in candidates:
            if node.complexity_level>=2: continue
            c=complex_.clone();
            for child in node.refine(context,3):
                c.nodes[child.name]=child; c.relations.append(make_relation(node.name,child.name,'refina',self.d,0.55)); c.relations.append(make_relation(child.name,node.name,'pertenece_a',self.d,0.35))
            c.history.append(f'refine:{node.name}'); expansions.append(c)
        return expansions
    def _add_missing_core_relations(self,complex_):
        names=set(complex_.nodes.keys()); desired=[]
        if {'biblioteca_babel','tensor_coherencia'}.issubset(names): desired.append(('tensor_coherencia','biblioteca_babel','metriza'))
        if {'u_p','tensor_coherencia'}.issubset(names): desired.append(('u_p','tensor_coherencia','produce_error_para'))
        if {'nodo_conceptual','relacion_infinita'}.issubset(names): desired.append(('nodo_conceptual','relacion_infinita','se_define_por'))
        if {'fraccion_superficial','nodo_conceptual'}.issubset(names): desired.append(('fraccion_superficial','nodo_conceptual','se_eleva_a'))
        existing={(r.source,r.target,r.relation_type) for r in complex_.relations}; out=[]
        for s,t,ty in desired:
            if (s,t,ty) not in existing: c=complex_.clone(); c.relations.append(make_relation(s,t,ty,self.d,0.95)); c.history.append(f'relate:{s}->{t}:{ty}'); out.append(c)
        return out
    def _add_projection_node_if_needed(self,complex_):
        if 'proyeccion_textual' in complex_.nodes:return []
        c=complex_.clone();chart=TextChart(self.d); c.nodes['proyeccion_textual']=ConceptNode('proyeccion_textual',chart.concept_vector('proyeccion_textual'),charts={'spanish':'proyección textual','symbolic':'π_text'},metadata={'role':'surface_projection_chart'})
        for n in list(complex_.nodes.keys())[:6]: c.relations.append(make_relation(n,'proyeccion_textual','puede_proyectarse',self.d,0.30))
        c.history.append('add:proyeccion_textual'); return [c]
    def _add_closure_node_if_needed(self,complex_):
        if 'cierre_estructural' in complex_.nodes:return []
        c=complex_.clone();chart=TextChart(self.d); c.nodes['cierre_estructural']=ConceptNode('cierre_estructural',chart.concept_vector('cierre_estructural'),charts={'spanish':'cierre estructural','symbolic':'CLOSE'},metadata={'role':'closure_condition'})
        for n in list(complex_.nodes.keys())[:6]: c.relations.append(make_relation(n,'cierre_estructural','debe_cerrar_en',self.d,0.25))
        c.history.append('add:cierre_estructural'); return [c]

class CoherenceFlow:
    def __init__(self,d:int,beam:int=6,steps:int=5,temp:float=1.0): self.d=d; self.beam=beam; self.steps=steps; self.temp=temp; self.babel=ConceptualBabel(d); self.energy_model=ConceptualEnergy(d)
    def run(self,initial,context,memory=None):
        frontier=[(initial.clone(),1.0)]; trace={'steps':[]}; best=initial.clone(); bestE,_,_,_=self.energy_model.energy(best,memory); bestC=self.energy_model.closure(best,memory)
        for step in range(self.steps):
            cand=[]
            for c,_w in frontier: cand.append(c); cand.extend(self.babel.expand(c,context))
            scored=[]
            for c in cand:
                E,_,_,_=self.energy_model.energy(c,memory); cl=self.energy_model.closure(c,memory); obj=E-0.25*self.energy_model.error_dim*cl; scored.append((obj,E,cl,c))
            scored.sort(key=lambda x:x[0]); kept=scored[:self.beam]; w=softmax_neg_energy([x[0] for x in kept],temp=self.temp); frontier=[(kept[i][3],float(w[i])) for i in range(len(kept))]
            if kept[0][1]<bestE or kept[0][2]>bestC: bestE,bestC,best=kept[0][1],kept[0][2],kept[0][3].clone()
            trace['steps'].append({'step':step,'candidates':len(cand),'best_objective':float(kept[0][0]),'best_energy':float(kept[0][1]),'best_closure':float(kept[0][2]),'best_nodes':sorted(list(kept[0][3].nodes.keys()))[:12],'best_history_tail':kept[0][3].history[-4:]})
        trace.update({'energy':float(bestE),'closure':float(bestC),'nodes':sorted(list(best.nodes.keys())),'relations':len(best.relations),'potential':best.potential_cardinality}); return best,trace

class ConceptualMemory:
    def __init__(self,d:int): self.d=d; self.memory=np.zeros(d); self.episodes=[]
    def read(self): return self.memory.copy()
    def reenter(self,complex_,surface,trace):
        active=complex_.active_state(); self.memory=active if np.linalg.norm(self.memory)<1e-9 else normalize(0.86*self.memory+0.14*active)
        for n in complex_.nodes.values(): n.memory_trace=min(1.0,n.memory_trace+0.04)
        self.episodes.append({'surface':surface,'nodes':sorted(list(complex_.nodes.keys())),'relations':len(complex_.relations),'energy':trace.get('energy'),'closure':trace.get('closure')}); self.episodes=self.episodes[-200:]
    def as_serializable(self): return {'memory':self.memory.tolist(),'episodes':self.episodes}
    @staticmethod
    def from_serializable(d:int,data): m=ConceptualMemory(d); m.memory=np.asarray(data.get('memory',m.memory),dtype=float); m.episodes=list(data.get('episodes',[])); return m

class ConceptualBabelRuntime:
    def __init__(self,d=48,beam=7,steps=6,state_path=None):
        if d%2!=0: raise ValueError('d must be even for u/p split')
        self.d=d; self.chart=TextChart(d); self.flow=CoherenceFlow(d,beam=beam,steps=steps,temp=1.0); self.memory=ConceptualMemory(d); self.state_path=Path(state_path) if state_path else None
        if self.state_path and self.state_path.exists(): self.load(self.state_path)
    def respond(self,surface):
        c0=self.chart.lift(surface); best,trace=self.flow.run(c0,context=surface,memory=self.memory.read()); response=self.chart.project(best,trace); self.memory.reenter(best,response,trace)
        if self.state_path: self.save(self.state_path)
        return {'input':surface,'response':response,'complex':best.as_serializable(),'trace':trace,'memory_episodes':len(self.memory.episodes)}
    def save(self,path): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps({'d':self.d,'memory':self.memory.as_serializable()},ensure_ascii=False,indent=2),encoding='utf-8')
    def load(self,path): data=json.loads(path.read_text(encoding='utf-8')); self.memory=ConceptualMemory.from_serializable(self.d,data.get('memory',{}))

def demo():
    runtime=ConceptualBabelRuntime(d=48,beam=7,steps=6); prompts=["La Biblioteca de Babel no son caracteres: son nodos conceptuales relacionales.","El tensor de coherencia y u/p deben actuar sobre ese potencial completo.","Hazlo realidad en código sin tratar tokens como fundamento."]
    return {'outputs':[runtime.respond(p) for p in prompts],'episodes':runtime.memory.episodes}
