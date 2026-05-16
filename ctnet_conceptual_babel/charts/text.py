from __future__ import annotations

import re
from typing import Dict, List, Optional, Any

import numpy as np

from ..core.complex import ConceptComplex
from ..core.node import ConceptNode, FractionNode, normalize, rng_for, stable_seed
from ..core.relation import RelationOperator, make_relation


class TextChart:
    def __init__(self, d: int):
        self.d = d
        self.lexicon = {
            "babel": "biblioteca_babel",
            "biblioteca": "biblioteca_babel",
            "generador": "generador_babel",
            "genera": "generador_babel",
            "generar": "generador_babel",
            "emite": "emision_directa",
            "emitir": "emision_directa",
            "tirón": "emision_directa",
            "tiron": "emision_directa",
            "directa": "emision_directa",
            "filtra": "no_filtrado_posterior",
            "filtro": "no_filtrado_posterior",
            "después": "no_filtrado_posterior",
            "despues": "no_filtrado_posterior",
            "infinito": "potencial_infinito",
            "potencial": "potencial_infinito",
            "información": "informacion_coherente",
            "informacion": "informacion_coherente",
            "coherente": "informacion_coherente",
            "coherencia": "tensor_coherencia",
            "tensor": "tensor_coherencia",
            "h": "tensor_coherencia",
            "u/p": "u_p",
            "up": "u_p",
            "nodo": "nodo_conceptual",
            "nodos": "nodo_conceptual",
            "concepto": "nodo_conceptual",
            "conceptual": "nodo_conceptual",
            "relación": "relacion_infinita",
            "relaciones": "relacion_infinita",
            "relacional": "relacion_infinita",
            "carácter": "fraccion_superficial",
            "caracter": "fraccion_superficial",
            "token": "fraccion_superficial",
            "tokens": "fraccion_superficial",
            "texto": "proyeccion_textual",
            "proyección": "proyeccion_textual",
            "proyeccion": "proyeccion_textual",
            "memoria": "memoria_topologica",
            "cierre": "cierre_estructural",
            "codex": "realizacion_codigo",
            "código": "realizacion_codigo",
            "codigo": "realizacion_codigo",
        }

    def concept_vector(self, name: str) -> np.ndarray:
        v = rng_for("concept::" + name).normal(0.0, 1.0, self.d)
        j = np.arange(1, self.d + 1)
        phase = (stable_seed(name) % 997) + 1
        return normalize(v + 0.25 * np.sin(j * phase / 17.0) + 0.15 * np.cos(j * phase / 31.0))

    def lift(self, surface: str) -> ConceptComplex:
        lower = surface.lower()
        tokens = re.findall(r"[a-záéíóúüñ/]+", lower, flags=re.UNICODE)

        uniq: List[str] = []
        for tok in tokens:
            concept = self.lexicon.get(tok)
            if concept and concept not in uniq:
                uniq.append(concept)

        if "u/p" in lower and "u_p" not in uniq:
            uniq.append("u_p")
        if re.search(r"(?<![a-záéíóúüñ])h(?![a-záéíóúüñ])", lower) and "tensor_coherencia" not in uniq:
            uniq.append("tensor_coherencia")

        if not uniq:
            uniq = ["mensaje_abierto"]

        if surface.strip() and "fraccion_superficial" not in uniq:
            uniq.append("fraccion_superficial")

        nodes: Dict[str, ConceptNode] = {}
        for name in uniq:
            vec = self.concept_vector(name)
            if name == "fraccion_superficial":
                nodes[name] = FractionNode(
                    name,
                    vec,
                    charts={"spanish": "fracción superficial", "surface": surface[:128]},
                    fraction=surface[:128],
                    chart_name="text",
                    metadata={"lifted_from_surface": True},
                )
            else:
                nodes[name] = ConceptNode(
                    name,
                    vec,
                    charts={"spanish": name.replace("_", " "), "symbolic": name},
                    metadata={"lifted_from_surface": True},
                )

        return ConceptComplex(
            nodes=nodes,
            relations=self._initial_relations(nodes),
            regime=self._infer_regime(lower),
            closure_goal=self._infer_goal(lower),
            history=[surface],
        )

    def _initial_relations(self, nodes: Dict[str, ConceptNode]) -> List[RelationOperator]:
        rels: List[RelationOperator] = []
        names = list(nodes.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rels.append(make_relation(a, b, self._relation_type(a, b), self.d, weight=0.7))
                rels.append(make_relation(b, a, "co-implica", self.d, weight=0.35))
        return rels

    @staticmethod
    def _relation_type(a: str, b: str) -> str:
        pair = {a, b}
        if {"biblioteca_babel", "generador_babel"}.issubset(pair):
            return "genera_desde"
        if {"generador_babel", "informacion_coherente"}.issubset(pair):
            return "emite_coherencia"
        if {"generador_babel", "u_p"}.issubset(pair):
            return "condicionado_por"
        if {"generador_babel", "tensor_coherencia"}.issubset(pair):
            return "curvado_por"
        if {"u_p", "tensor_coherencia"}.issubset(pair):
            return "mide_reciprocidad"
        if {"emision_directa", "no_filtrado_posterior"}.issubset(pair):
            return "niega_filtrado"
        if "fraccion_superficial" in pair and "nodo_conceptual" in pair:
            return "fraccion_elevable"
        if "proyeccion_textual" in pair:
            return "proyecta"
        return "relaciona"

    @staticmethod
    def _infer_regime(lower: str) -> str:
        if any(w in lower for w in ["código", "codigo", "codex", "python", "implementa", "hazlo"]):
            return "implementation"
        if any(w in lower for w in ["ecuación", "ecuacion", "formal", "matem"]):
            return "formal"
        return "conceptual"

    @staticmethod
    def _infer_goal(lower: str) -> str:
        if any(w in lower for w in ["hazlo", "crear", "implement"]):
            return "realize"
        if any(w in lower for w in ["por qué", "porque", "implica"]):
            return "explain_cause"
        return "stabilize"

    def project(self, complex_: ConceptComplex, trace: Optional[Dict[str, Any]] = None) -> str:
        from ..surface import CoherenceForcedBabelTextGenerator

        return CoherenceForcedBabelTextGenerator(self.d).emit(complex_, trace or {})
