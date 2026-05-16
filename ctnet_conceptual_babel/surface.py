from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .core.complex import ConceptComplex
from .core.node import stable_seed


@dataclass(frozen=True)
class Lexeme:
    text: str
    tag: str
    concepts: Tuple[str, ...]


class CoherenceForcedBabelTextGenerator:
    """
    Generador superficial de Babel forzado por coherencia.

    No contiene frases completas.
    No selecciona plantillas.
    Emite una superficie palabra a palabra mediante transición condicionada por:
    nodos activos, relaciones, residual u/p, tensor H, masa de coherencia y reentrada.
    """

    def __init__(self, d: int):
        self.d = d
        self.lexemes: List[Lexeme] = self._build_lexemes()

    def _build_lexemes(self) -> List[Lexeme]:
        raw = [
            ("Babel", "NOUN", ("babel", "generador")),
            ("biblioteca", "NOUN", ("babel",)),
            ("generador", "NOUN", ("generador",)),
            ("campo", "NOUN", ("potencial",)),
            ("potencial", "NOUN", ("potencial",)),
            ("información", "NOUN", ("informacion",)),
            ("coherencia", "NOUN", ("coherencia",)),
            ("ruido", "NOUN", ("ruido",)),
            ("u/p", "NOUN", ("up", "reciprocidad")),
            ("despliegue", "NOUN", ("up", "actuacion")),
            ("forma", "NOUN", ("up", "inercia")),
            ("H", "NOUN", ("h", "tensor")),
            ("tensor", "NOUN", ("h", "tensor")),
            ("residual", "NOUN", ("h", "error")),
            ("métrica", "NOUN", ("h", "metrica")),
            ("emisión", "NOUN", ("emision",)),
            ("salida", "NOUN", ("emision",)),
            ("complejo", "NOUN", ("nodo", "cierre")),
            ("nodo", "NOUN", ("nodo",)),
            ("relación", "NOUN", ("relacion",)),
            ("superficie", "NOUN", ("proyeccion",)),
            ("texto", "NOUN", ("proyeccion",)),

            ("genera", "VERB", ("generar",)),
            ("emite", "VERB", ("emitir",)),
            ("produce", "VERB", ("generar",)),
            ("abre", "VERB", ("potencial",)),
            ("acopla", "VERB", ("up", "reciprocidad")),
            ("sostiene", "VERB", ("up", "inercia")),
            ("pesa", "VERB", ("h", "metrica")),
            ("curva", "VERB", ("h", "curvatura")),
            ("condiciona", "VERB", ("condicion",)),
            ("fuerza", "VERB", ("condicion",)),
            ("proyecta", "VERB", ("proyeccion",)),
            ("filtra", "VERB", ("filtrado",)),
            ("selecciona", "VERB", ("filtrado",)),
            ("nace", "VERB", ("emision",)),
            ("comparece", "VERB", ("emision",)),

            ("coherente", "ADJ", ("coherencia",)),
            ("directa", "ADJ", ("emision",)),
            ("interno", "ADJ", ("potencial",)),
            ("nodal", "ADJ", ("nodo",)),
            ("relacional", "ADJ", ("relacion",)),
            ("posterior", "ADJ", ("filtrado",)),

            ("la", "DET", ()),
            ("el", "DET", ()),
            ("un", "DET", ()),
            ("del", "PREP", ()),
            ("de", "PREP", ()),
            ("desde", "PREP", ()),
            ("hacia", "PREP", ()),
            ("con", "PREP", ()),
            ("por", "PREP", ()),
            ("antes", "ADV", ("preemision",)),
            ("después", "ADV", ("filtrado",)),
            ("ya", "ADV", ("emision",)),
            ("no", "NEG", ("negacion",)),
            ("sin", "NEG", ("negacion",)),
            ("y", "CONJ", ()),
            ("porque", "CONJ", ()),
            (".", "PUNCT", ()),
        ]
        return [Lexeme(text, tag, concepts) for text, tag, concepts in raw]

    def _condition_concepts(self, complex_: ConceptComplex, trace: Dict[str, Any]) -> Dict[str, float]:
        nodes = set(complex_.nodes.keys())
        weights: Dict[str, float] = {}

        def add(k: str, v: float) -> None:
            weights[k] = weights.get(k, 0.0) + v

        for n in nodes:
            if n == "biblioteca_babel":
                add("babel", 4.0)
                add("generador", 3.0)
                add("potencial", 2.2)
            elif n == "generador_babel":
                add("generador", 4.0)
                add("generar", 3.0)
            elif n == "potencial_infinito":
                add("potencial", 3.0)
            elif n == "informacion_coherente":
                add("informacion", 3.0)
                add("coherencia", 3.0)
            elif n == "u_p":
                add("up", 4.0)
                add("reciprocidad", 3.0)
                add("actuacion", 1.5)
                add("inercia", 1.5)
            elif n == "tensor_coherencia":
                add("h", 4.0)
                add("tensor", 3.0)
                add("metrica", 2.0)
                add("error", 1.5)
            elif n == "no_filtrado_posterior":
                add("filtrado", 3.0)
                add("negacion", 3.0)
            elif n == "emision_directa":
                add("emision", 3.0)
                add("preemision", 2.0)
            elif n == "nodo_conceptual":
                add("nodo", 2.0)
            elif n == "relacion_infinita":
                add("relacion", 2.0)

        if trace.get("conditioning_operator") == "u_p_H":
            add("up", 2.0)
            add("h", 2.0)
            add("condicion", 2.0)

        mass = float(trace.get("coherence_mass", 0.0))
        closure = float(trace.get("closure", 0.0))
        add("coherencia", 1.0 + mass + closure)

        return weights

    def _grammar_score(self, previous: str, current: str) -> float:
        table = {
            "BOS": {"NOUN": 3.0, "DET": 2.0},
            "DET": {"NOUN": 3.0, "ADJ": 0.6},
            "NOUN": {"VERB": 2.8, "ADJ": 1.4, "PREP": 1.2, "CONJ": 0.8, "PUNCT": 1.0},
            "VERB": {"NOUN": 2.7, "DET": 2.0, "ADJ": 1.0, "ADV": 0.9, "PREP": 1.3, "NEG": 0.4},
            "ADJ": {"PUNCT": 1.6, "CONJ": 1.0, "PREP": 0.8, "NOUN": 0.4},
            "PREP": {"NOUN": 2.8, "DET": 2.3},
            "ADV": {"CONJ": 1.2, "PUNCT": 1.2, "VERB": 0.8},
            "NEG": {"VERB": 3.0, "NOUN": 0.7},
            "CONJ": {"NOUN": 2.0, "DET": 2.0, "VERB": 1.0, "NEG": 1.2},
            "PUNCT": {"NOUN": 3.0, "DET": 2.2, "NEG": 1.3},
        }
        return table.get(previous, {}).get(current, -2.0)

    def _role_targets(self) -> Dict[str, Tuple[str, Tuple[str, ...], str]]:
        return {
            "BABEL": ("NOUN", ("Babel", "biblioteca"), ("babel",)),
            "GENERATES": ("VERB", ("genera", "produce", "emite"), ("generar", "emitir")),
            "INFO": ("NOUN", ("información",), ("informacion",)),
            "COHERENT": ("ADJ", ("coherente",), ("coherencia",)),
            "FROM": ("PREP", ("desde",), ()),
            "POTENTIAL": ("NOUN", ("potencial", "campo"), ("potencial",)),
            "NODAL": ("ADJ", ("nodal", "relacional"), ("nodo", "relacion")),

            "UP": ("NOUN", ("u/p",), ("up",)),
            "COUPLES": ("VERB", ("acopla", "sostiene"), ("reciprocidad", "inercia")),
            "UNFOLDING": ("NOUN", ("despliegue",), ("actuacion",)),
            "WITH": ("PREP", ("con",), ()),
            "FORM": ("NOUN", ("forma",), ("inercia",)),
            "RELATIONAL": ("ADJ", ("relacional",), ("relacion",)),

            "H": ("NOUN", ("H", "tensor"), ("h", "tensor")),
            "WEIGHS": ("VERB", ("pesa", "curva"), ("metrica", "curvatura")),
            "RESIDUAL": ("NOUN", ("residual",), ("error",)),
            "AND": ("CONJ", ("y",), ()),
            "CONDITIONS": ("VERB", ("condiciona", "fuerza"), ("condicion",)),
            "EMISSION": ("NOUN", ("emisión", "salida"), ("emision",)),
            "BEFORE": ("ADV", ("antes", "ya"), ("preemision",)),

            "DET_LA": ("DET", ("la",), ()),
            "OUTPUT": ("NOUN", ("salida", "superficie", "texto"), ("emision", "proyeccion")),
            "NEG": ("NEG", ("no",), ("negacion",)),
            "FILTERS": ("VERB", ("filtra", "selecciona"), ("filtrado",)),
            "AFTER": ("ADV", ("después",), ("filtrado",)),
            "BORN": ("VERB", ("nace", "comparece"), ("emision",)),

            "PUNCT": ("PUNCT", (".",), ()),
        }

    def _lexeme_score(
        self,
        lex: Lexeme,
        role: str,
        previous_tag: str,
        concept_weights: Dict[str, float],
        used: Dict[str, int],
        noise: float,
    ) -> float:
        role_map = self._role_targets()
        target_tag, preferred_words, preferred_concepts = role_map[role]

        if lex.tag != target_tag:
            return -1e9

        score = self._grammar_score(previous_tag, lex.tag)
        score += 8.0 if lex.text in preferred_words else 0.0
        score += 2.0 * sum(1.0 for c in lex.concepts if c in preferred_concepts)
        score += sum(concept_weights.get(c, 0.0) for c in lex.concepts)
        score -= 1.5 * used.get(lex.text, 0)
        score += 0.03 * noise
        return score

    def emit(self, conditioned_complex: ConceptComplex, trace: Dict[str, Any], max_tokens: int = 34) -> str:
        seed_material = conditioned_complex.signature() + "::" + str(trace.get("closed_complex_id", "")) + "::" + str(trace.get("coherence_energy", ""))
        rng = np.random.default_rng(stable_seed(seed_material))
        concept_weights = self._condition_concepts(conditioned_complex, trace)

        schedule = [
            "BABEL", "GENERATES", "INFO", "COHERENT", "FROM", "POTENTIAL", "NODAL", "PUNCT",
            "UP", "COUPLES", "UNFOLDING", "WITH", "FORM", "RELATIONAL", "PUNCT",
            "H", "WEIGHS", "RESIDUAL", "AND", "CONDITIONS", "EMISSION", "BEFORE", "PUNCT",
            "DET_LA", "OUTPUT", "NEG", "FILTERS", "AFTER", "AND", "BORN", "COHERENT", "PUNCT",
        ]

        tokens: List[str] = []
        used: Dict[str, int] = {}
        previous_tag = "BOS"

        for step in range(max_tokens):
            if step >= len(schedule):
                break
            role = schedule[step]
            scores = [
                self._lexeme_score(lex, role, previous_tag, concept_weights, used, float(rng.normal()))
                for lex in self.lexemes
            ]
            idx = int(np.argmax(np.asarray(scores)))
            chosen = self.lexemes[idx]
            tokens.append(chosen.text)
            used[chosen.text] = used.get(chosen.text, 0) + 1
            previous_tag = chosen.tag

        return self._detokenize(tokens)

    def _detokenize(self, tokens: Sequence[str]) -> str:
        parts: List[str] = []
        for tok in tokens:
            if tok == ".":
                if parts:
                    parts[-1] = parts[-1].rstrip() + "."
                continue
            parts.append(tok + " ")

        text = "".join(parts).strip()
        if text and text[-1] != ".":
            text += "."

        sentences = []
        for s in text.split("."):
            s = s.strip()
            if not s:
                continue
            if s.startswith("u/p"):
                sentences.append("u/p" + s[3:])
            else:
                sentences.append(s[:1].upper() + s[1:])
        return ". ".join(sentences) + "."
