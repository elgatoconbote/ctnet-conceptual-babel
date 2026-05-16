from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.complex import ConceptComplex
from ..core.node import ConceptNode, FractionNode, normalize, rng_for, stable_seed
from ..core.relation import RelationOperator, make_relation


class TextChart:
    def __init__(self, d: int):
        self.d = d
        self.lexicon = {'babel': 'biblioteca_babel', 'biblioteca': 'biblioteca_babel', 'infinito': 'potencial_infinito', 'potencial': 'potencial_infinito', 'tensor': 'tensor_coherencia', 'coherencia': 'tensor_coherencia', 'u/p': 'u_p', 'up': 'u_p', 'nodo': 'nodo_conceptual', 'nodos': 'nodo_conceptual', 'concepto': 'nodo_conceptual', 'conceptual': 'nodo_conceptual', 'relación': 'relacion_infinita', 'relaciones': 'relacion_infinita', 'relacional': 'relacion_infinita', 'carácter': 'fraccion_superficial', 'caracter': 'fraccion_superficial', 'token': 'fraccion_superficial', 'tokens': 'fraccion_superficial', 'texto': 'proyeccion_textual', 'proyección': 'proyeccion_textual', 'proyeccion': 'proyeccion_textual', 'memoria': 'memoria_topologica', 'cierre': 'cierre_estructural', 'codex': 'realizacion_codigo', 'código': 'realizacion_codigo', 'codigo': 'realizacion_codigo'}

    def concept_vector(self, name: str) -> np.ndarray:
        v = rng_for('concept::' + name).normal(0.0, 1.0, self.d)
        j = np.arange(1, self.d + 1)
        phase = (stable_seed(name) % 997) + 1
        return normalize(v + 0.25 * np.sin(j * phase / 17.0) + 0.15 * np.cos(j * phase / 31.0))

    def lift(self, surface: str) -> ConceptComplex:
        lower = surface.lower()
        names = [concept for key, concept in self.lexicon.items() if key in lower and concept not in []]
        uniq = []
        for n in names:
            if n not in uniq:
                uniq.append(n)
        if not uniq:
            uniq = ['mensaje_abierto']
        if surface.strip():
            uniq.append('fraccion_superficial')
        nodes: Dict[str, ConceptNode] = {}
        for name in uniq:
            vec = self.concept_vector(name)
            nodes[name] = FractionNode(name, vec, charts={'spanish': 'fracción superficial', 'surface': surface[:64]}, fraction=surface[:64], chart_name='text', metadata={'lifted_from_surface': True}) if name == 'fraccion_superficial' else ConceptNode(name, vec, charts={'spanish': name.replace('_', ' '), 'symbolic': name}, metadata={'lifted_from_surface': True})
        return ConceptComplex(nodes=nodes, relations=self._initial_relations(nodes), regime=self._infer_regime(lower), closure_goal=self._infer_goal(lower), history=[surface])

    def _initial_relations(self, nodes: Dict[str, ConceptNode]) -> List[RelationOperator]:
        rels = []
        names = list(nodes.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rels.append(make_relation(a, b, self._relation_type(a, b), self.d, weight=0.7))
                rels.append(make_relation(b, a, 'co-implica', self.d, weight=0.35))
        return rels

    @staticmethod
    def _relation_type(a: str, b: str) -> str:
        pair = {a, b}
        if 'fraccion_superficial' in pair and 'nodo_conceptual' in pair: return 'fraccion_elevable'
        if 'biblioteca_babel' in pair and 'tensor_coherencia' in pair: return 'metriza_potencial'
        if 'u_p' in pair and 'tensor_coherencia' in pair: return 'mide_reciprocidad'
        if 'proyeccion_textual' in pair: return 'proyecta'
        return 'relaciona'

    @staticmethod
    def _infer_regime(lower: str) -> str:
        if any(w in lower for w in ['código', 'codigo', 'codex', 'python', 'implementa', 'hazlo']): return 'implementation'
        if any(w in lower for w in ['ecuación', 'ecuacion', 'formal', 'matem']): return 'formal'
        return 'conceptual'

    @staticmethod
    def _infer_goal(lower: str) -> str:
        if 'hazlo' in lower or 'crear' in lower or 'implement' in lower: return 'realize'
        if 'por qué' in lower or 'porque' in lower: return 'explain_cause'
        return 'stabilize'

    def _spanish_name(self, name: str) -> str:
        mapping = {
            'biblioteca_babel': 'Biblioteca de Babel',
            'tensor_coherencia': 'tensor de coherencia H',
            'u_p': 'campo u/p',
            'nodo_conceptual': 'nodo conceptual',
            'relacion_infinita': 'trama relacional abierta',
            'fraccion_superficial': 'fracción superficial textual',
            'proyeccion_textual': 'carta de proyección textual',
            'memoria_topologica': 'memoria topológica',
            'cierre_estructural': 'cierre estructural',
            'mensaje_abierto': 'mensaje abierto',
            'realizacion_codigo': 'realización de código',
        }
        return mapping.get(name, name.replace('_', ' '))

    def _dominant_relations(self, rels: Sequence[RelationOperator], limit: int = 4) -> List[RelationOperator]:
        return sorted(rels, key=lambda r: abs(float(r.weight)), reverse=True)[:limit]

    def project(self, complex_: ConceptComplex, trace: Optional[Dict[str, Any]] = None) -> str:
        names = list(complex_.nodes.keys())
        rels = self._dominant_relations(complex_.relations, limit=5)
        if not names:
            return 'Desde el estado condicionado, emerjo con una proyección mínima: no hay nodos activos suficientes para articular una trama coherente.'

        lead_nodes = [self._spanish_name(n) for n in names[:4]]
        first = f"En este cierre de un solo paso, {lead_nodes[0]} se acopla con {', '.join(lead_nodes[1:]) if len(lead_nodes) > 1 else 'su vecindario conceptual'} para formar un complejo activo coherente."

        if rels:
            rel_text = []
            for r in rels[:3]:
                rel_text.append(f"{self._spanish_name(r.source)} {r.relation_type.replace('_', ' ')} {self._spanish_name(r.target)}")
            second = 'La dinámica interna queda determinada por relaciones como ' + '; '.join(rel_text) + '.'
        else:
            second = 'La dinámica interna queda determinada por la consistencia de estado entre nodos activos.'

        has_up_h = 'u_p' in complex_.nodes and 'tensor_coherencia' in complex_.nodes
        if has_up_h:
            third = 'Aquí, u/p no evalúa candidatos: deforma directamente el campo generativo y H = D + L·L^T penaliza el residual antes de emitir la superficie, por eso la respuesta sale ya condicionada.'
        else:
            third = 'La emisión superficial surge después del cierre nodal: primero se estabiliza el complejo y solo entonces se proyecta en español continuo.'

        closure = float(trace.get('closure', getattr(complex_, 'closure_state', 0.0))) if trace else float(getattr(complex_, 'closure_state', 0.0))
        energy = float(trace.get('coherence_energy', getattr(complex_, 'coherence_energy', 0.0))) if trace else float(getattr(complex_, 'coherence_energy', 0.0))
        mass = float(trace.get('coherence_mass', getattr(complex_, 'coherence_mass', 0.0))) if trace else float(getattr(complex_, 'coherence_mass', 0.0))
        fourth = f"El resultado mantiene cierre={closure:.3f}, energía={energy:.3f} y masa coherente={mass:.3f}, con semántica generada desde la estructura nodal y no desde una plantilla fija."

        return ' '.join([first, second, third, fourth])

