from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from ..core.complex import ConceptComplex
from ..core.node import ConceptNode, FractionNode, normalize, rng_for, stable_seed
from ..core.relation import RelationOperator, make_relation

class TextChart:
    def __init__(self,d:int):
        self.d=d
        self.lexicon={"babel":"biblioteca_babel","biblioteca":"biblioteca_babel","infinito":"potencial_infinito","potencial":"potencial_infinito","tensor":"tensor_coherencia","coherencia":"tensor_coherencia","u/p":"u_p","up":"u_p","nodo":"nodo_conceptual","nodos":"nodo_conceptual","concepto":"nodo_conceptual","conceptual":"nodo_conceptual","relación":"relacion_infinita","relaciones":"relacion_infinita","relacional":"relacion_infinita","carácter":"fraccion_superficial","caracter":"fraccion_superficial","token":"fraccion_superficial","tokens":"fraccion_superficial","texto":"proyeccion_textual","proyección":"proyeccion_textual","proyeccion":"proyeccion_textual","memoria":"memoria_topologica","cierre":"cierre_estructural","codex":"realizacion_codigo","código":"realizacion_codigo","codigo":"realizacion_codigo"}
    def concept_vector(self,name:str)->np.ndarray:
        rng=rng_for("concept::"+name); v=rng.normal(0.0,1.0,self.d); j=np.arange(1,self.d+1); ph=(stable_seed(name)%997)+1
        v += 0.25*np.sin(j*ph/17.0)+0.15*np.cos(j*ph/31.0); return normalize(v)
    def lift(self,surface:str)->ConceptComplex:
        lower=surface.lower(); names=[]
        for key,concept in self.lexicon.items():
            if key in lower and concept not in names: names.append(concept)
        if not names: names.append("mensaje_abierto")
        if surface.strip(): names.append("fraccion_superficial")
        nodes={}
        for name in names:
            vec=self.concept_vector(name)
            if name=="fraccion_superficial":
                nodes[name]=FractionNode(name=name,state=vec,charts={"spanish":"fracción superficial","surface":surface[:64]},fraction=surface[:64],chart_name="text",metadata={"lifted_from_surface":True})
            else:
                nodes[name]=ConceptNode(name=name,state=vec,charts={"spanish":name.replace("_"," "),"symbolic":name},metadata={"lifted_from_surface":True})
        return ConceptComplex(nodes=nodes,relations=self._initial_relations(nodes),regime=self._infer_regime(lower),closure_goal=self._infer_goal(lower),history=[surface])
    def _initial_relations(self,nodes):
        names=list(nodes.keys()); rels=[]
        for i,a in enumerate(names):
            for b in names[i+1:]:
                rels.append(make_relation(a,b,self._relation_type(a,b),self.d,weight=0.7)); rels.append(make_relation(b,a,"co-implica",self.d,weight=0.35))
        return rels
    @staticmethod
    def _relation_type(a,b):
        pair={a,b}
        if "fraccion_superficial" in pair and "nodo_conceptual" in pair: return "fraccion_elevable"
        if "biblioteca_babel" in pair and "tensor_coherencia" in pair: return "metriza_potencial"
        if "u_p" in pair and "tensor_coherencia" in pair: return "mide_reciprocidad"
        if "proyeccion_textual" in pair: return "proyecta"
        return "relaciona"
    @staticmethod
    def _infer_regime(lower):
        if any(w in lower for w in ["código","codigo","codex","python","implementa","hazlo"]): return "implementation"
        if any(w in lower for w in ["ecuación","ecuacion","formal","matem"]): return "formal"
        return "conceptual"
    @staticmethod
    def _infer_goal(lower):
        if "hazlo" in lower or "crear" in lower or "implement" in lower: return "realize"
        if "por qué" in lower or "porque" in lower: return "explain_cause"
        return "stabilize"
    def project(self,complex_,trace=None):
        names=list(complex_.nodes.keys()); has=set(names); dominant=names[:6]; energy=trace.get("energy") if trace else None; closure=trace.get("closure") if trace else None
        if "biblioteca_babel" in has and "tensor_coherencia" in has and "u_p" in has:
            core="La Biblioteca de Babel queda tratada como campo potencial de complejos conceptuales, no como lista de caracteres. u/p abre la tensión entre actuación y forma; el tensor de coherencia pesa esa tensión y curva el potencial para que aparezca un complejo proyectable."
        elif "nodo_conceptual" in has and "relacion_infinita" in has:
            core="El centro activo es un complejo de nodos conceptuales relacionales. Cada nodo conserva una sección finita, pero su potencial queda abierto por expansión, relación, refinamiento y proyección."
        elif "realizacion_codigo" in has:
            core="La realización correcta es construir un runtime donde el texto solo sea una carta: se eleva a nodos, se estabiliza por energía u/p bajo H, y luego se proyecta en español."
        else:
            core="Activo el campo conceptual recibido, estabilizo las relaciones dominantes y proyecto solo la parte que tiene cierre suficiente."
        if energy is not None and closure is not None: core += f" Energía final={energy:.4f}; cierre={closure:.4f}."
        core += " Nodos dominantes: " + ", ".join(d.replace("_", " ") for d in dominant) + "."
        return core
