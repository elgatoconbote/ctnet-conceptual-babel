#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet Conceptual Babel
======================

A proof-of-concept runtime where the primitive is not a token/character.
The primitive is a conceptual-information-relational node. Text is only a
projection chart: it can lift surface fragments into conceptual fields and
project stabilized conceptual complexes back to Spanish.

Core equation implemented by the runtime:

    E_X(S) = < e(S), H_X e(S) >

where:
    S        = conceptual-relational complex, not a string.
    e(S)    = combined u/p + relational + compositional + projection error.
    H_X     = coherence tensor D + L L^T induced by current state.

The infinite Babel is represented implicitly: each ConceptNode contains open
operators expand/project/lift/relate. The executable state only keeps a finite
active section of an unbounded conceptual potential.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Numeric utilities
# ---------------------------------------------------------------------------


def stable_seed(text: str) -> int:
    h = 2166136261
    for b in text.encode("utf-8", errors="replace"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def rng_for(text: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(text))


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x.copy()
    return x / n


def rmsnorm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x) + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a)
    b = np.ravel(b)
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / den)


def softmax_neg_energy(energies: Sequence[float], temp: float = 1.0) -> np.ndarray:
    e = -np.asarray(energies, dtype=float) / max(temp, 1e-8)
    e = e - np.max(e)
    w = np.exp(e)
    return w / (np.sum(w) + 1e-12)


# ---------------------------------------------------------------------------
# Conceptual nodes: primitive ontology
# ---------------------------------------------------------------------------


@dataclass
class ConceptNode:
    """
    Conceptual-information-relational node.

    The finite vector is not the node itself. It is only the active chart of a
    node whose potential is unbounded through expansion/refinement operators.
    """

    name: str
    state: np.ndarray
    complexity_level: int = 0
    charts: Dict[str, str] = field(default_factory=dict)
    memory_trace: float = 0.0
    potential_cardinality: str = "unbounded"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ConceptNode":
        return ConceptNode(
            name=self.name,
            state=self.state.copy(),
            complexity_level=self.complexity_level,
            charts=dict(self.charts),
            memory_trace=float(self.memory_trace),
            potential_cardinality=self.potential_cardinality,
            metadata=json.loads(json.dumps(self.metadata, ensure_ascii=False)),
        )

    def as_serializable(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.tolist(),
            "complexity_level": self.complexity_level,
            "charts": self.charts,
            "memory_trace": self.memory_trace,
            "potential_cardinality": self.potential_cardinality,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> "ConceptNode":
        return ConceptNode(
            name=d["name"],
            state=np.asarray(d["state"], dtype=float),
            complexity_level=int(d.get("complexity_level", 0)),
            charts=dict(d.get("charts", {})),
            memory_trace=float(d.get("memory_trace", 0.0)),
            potential_cardinality=d.get("potential_cardinality", "unbounded"),
            metadata=dict(d.get("metadata", {})),
        )

    def refine(self, context: str, max_children: int = 3) -> List["ConceptNode"]:
        """
        Opens the internal potential of the node into subnodes.

        This is not tokenization. It is conceptual refinement: a node unfolds
        into relational aspects when the coherence field needs more resolution.
        """
        rng = rng_for(f"refine::{self.name}::{context}::{self.complexity_level}")
        child_names = self._child_names(context, max_children)
        children: List[ConceptNode] = []
        for idx, child_name in enumerate(child_names):
            perturb = rng.normal(0.0, 0.055 + 0.015 * self.complexity_level, self.state.shape)
            gate = 0.72 + 0.06 * math.sin(idx + self.complexity_level)
            child_state = normalize(gate * self.state + perturb)
            children.append(
                ConceptNode(
                    name=child_name,
                    state=child_state,
                    complexity_level=self.complexity_level + 1,
                    charts={
                        "spanish": self._spanish_projection(child_name),
                        "symbolic": child_name,
                    },
                    memory_trace=0.80 * self.memory_trace,
                    potential_cardinality="unbounded",
                    metadata={"parent": self.name, "refinement_context": context},
                )
            )
        return children

    def _child_names(self, context: str, max_children: int) -> List[str]:
        presets = {
            "biblioteca_babel": [
                "potencial_expresivo_total",
                "proyeccion_textual_fraccionaria",
                "clausura_nodal_relacional",
            ],
            "tensor_coherencia": [
                "metrica_de_error",
                "masa_de_cierre",
                "ponderacion_direccional",
            ],
            "u_p": [
                "actuacion",
                "inercia_formal",
                "reciprocidad_estructural",
            ],
            "nodo_conceptual": [
                "estado_interno",
                "valencia_relacional",
                "proyecciones_posibles",
            ],
            "relacion_infinita": [
                "composicion",
                "implicacion",
                "compatibilidad",
            ],
        }
        base = presets.get(self.name)
        if base is None:
            base = [f"{self.name}_aspecto_{i+1}" for i in range(max_children)]
        return base[:max_children]

    @staticmethod
    def _spanish_projection(name: str) -> str:
        return name.replace("_", " ")


@dataclass
class FractionNode(ConceptNode):
    """
    A surface fraction that is also a conceptual node.

    This models the bidirectionality requested: a character/word/surface mark is
    not an ontological atom, but it can be lifted as a node under a chart; and a
    node can project back into fractions.
    """

    fraction: str = ""
    chart_name: str = "surface"

    def clone(self) -> "FractionNode":
        return FractionNode(
            name=self.name,
            state=self.state.copy(),
            complexity_level=self.complexity_level,
            charts=dict(self.charts),
            memory_trace=float(self.memory_trace),
            potential_cardinality=self.potential_cardinality,
            metadata=json.loads(json.dumps(self.metadata, ensure_ascii=False)),
            fraction=self.fraction,
            chart_name=self.chart_name,
        )


@dataclass
class RelationOperator:
    """Operatorial relation between conceptual nodes."""

    source: str
    target: str
    relation_type: str
    matrix: np.ndarray
    weight: float = 1.0
    expected_alignment: float = 0.72
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "RelationOperator":
        return RelationOperator(
            source=self.source,
            target=self.target,
            relation_type=self.relation_type,
            matrix=self.matrix.copy(),
            weight=float(self.weight),
            expected_alignment=float(self.expected_alignment),
            metadata=json.loads(json.dumps(self.metadata, ensure_ascii=False)),
        )

    def as_serializable(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "matrix": self.matrix.tolist(),
            "weight": self.weight,
            "expected_alignment": self.expected_alignment,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> "RelationOperator":
        return RelationOperator(
            source=d["source"],
            target=d["target"],
            relation_type=d["relation_type"],
            matrix=np.asarray(d["matrix"], dtype=float),
            weight=float(d.get("weight", 1.0)),
            expected_alignment=float(d.get("expected_alignment", 0.72)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class ConceptComplex:
    """Finite active section of the unbounded conceptual Babel."""

    nodes: Dict[str, ConceptNode] = field(default_factory=dict)
    relations: List[RelationOperator] = field(default_factory=list)
    regime: str = "technical"
    closure_goal: str = "explain"
    history: List[str] = field(default_factory=list)
    potential_cardinality: str = "unbounded"

    def clone(self) -> "ConceptComplex":
        return ConceptComplex(
            nodes={k: v.clone() for k, v in self.nodes.items()},
            relations=[r.clone() for r in self.relations],
            regime=self.regime,
            closure_goal=self.closure_goal,
            history=list(self.history),
            potential_cardinality=self.potential_cardinality,
        )

    def active_state(self) -> np.ndarray:
        if not self.nodes:
            return np.zeros(1)
        return np.mean([n.state for n in self.nodes.values()], axis=0)

    def signature(self) -> str:
        ns = ",".join(sorted(self.nodes.keys()))
        rs = ",".join(f"{r.source}->{r.target}:{r.relation_type}" for r in self.relations)
        return f"{self.regime}|{self.closure_goal}|{ns}|{rs}"

    def as_serializable(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.as_serializable() for k, v in self.nodes.items()},
            "relations": [r.as_serializable() for r in self.relations],
            "regime": self.regime,
            "closure_goal": self.closure_goal,
            "history": self.history,
            "potential_cardinality": self.potential_cardinality,
        }

    @staticmethod
    def from_serializable(d: Dict[str, Any]) -> "ConceptComplex":
        return ConceptComplex(
            nodes={k: ConceptNode.from_serializable(v) for k, v in d.get("nodes", {}).items()},
            relations=[RelationOperator.from_serializable(r) for r in d.get("relations", [])],
            regime=d.get("regime", "technical"),
            closure_goal=d.get("closure_goal", "explain"),
            history=list(d.get("history", [])),
            potential_cardinality=d.get("potential_cardinality", "unbounded"),
        )


# ---------------------------------------------------------------------------
# Charts: text is a projection/elevation chart, not the ontology
# ---------------------------------------------------------------------------


class TextChart:
    """Surface chart: lift text into conceptual field and project complexes."""

    def __init__(self, d: int):
        self.d = d
        self.lexicon = {
            "babel": "biblioteca_babel",
            "biblioteca": "biblioteca_babel",
            "infinito": "potencial_infinito",
            "potencial": "potencial_infinito",
            "tensor": "tensor_coherencia",
            "coherencia": "tensor_coherencia",
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
        rng = rng_for("concept::" + name)
        v = rng.normal(0.0, 1.0, self.d)
        # Adds a structured harmonic component, not based on token ontology.
        j = np.arange(1, self.d + 1)
        phase = (stable_seed(name) % 997) + 1
        v += 0.25 * np.sin(j * phase / 17.0) + 0.15 * np.cos(j * phase / 31.0)
        return normalize(v)

    def lift(self, surface: str) -> ConceptComplex:
        """
        surface -> conceptual field.

        The raw text is only evidence. It is not the internal primitive.
        """
        lower = surface.lower()
        names: List[str] = []
        for key, concept in self.lexicon.items():
            if key in lower and concept not in names:
                names.append(concept)

        if not names:
            names.append("mensaje_abierto")

        # Always add a surface-fraction node as a reminder that the surface is a
        # fraction that can itself be lifted as a node.
        if surface.strip():
            names.append("fraccion_superficial")

        nodes: Dict[str, ConceptNode] = {}
        for name in names:
            vec = self.concept_vector(name)
            if name == "fraccion_superficial":
                nodes[name] = FractionNode(
                    name=name,
                    state=vec,
                    charts={"spanish": "fracción superficial", "surface": surface[:64]},
                    fraction=surface[:64],
                    chart_name="text",
                    metadata={"lifted_from_surface": True},
                )
            else:
                nodes[name] = ConceptNode(
                    name=name,
                    state=vec,
                    charts={"spanish": name.replace("_", " "), "symbolic": name},
                    metadata={"lifted_from_surface": True},
                )

        relations = self._initial_relations(nodes)
        regime = self._infer_regime(lower)
        goal = self._infer_goal(lower)
        return ConceptComplex(nodes=nodes, relations=relations, regime=regime, closure_goal=goal, history=[surface])

    def _initial_relations(self, nodes: Dict[str, ConceptNode]) -> List[RelationOperator]:
        names = list(nodes.keys())
        rels: List[RelationOperator] = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rel_type = self._relation_type(a, b)
                rels.append(make_relation(a, b, rel_type, self.d, weight=0.7))
                rels.append(make_relation(b, a, "co-implica", self.d, weight=0.35))
        return rels

    @staticmethod
    def _relation_type(a: str, b: str) -> str:
        pair = {a, b}
        if "fraccion_superficial" in pair and "nodo_conceptual" in pair:
            return "fraccion_elevable"
        if "biblioteca_babel" in pair and "tensor_coherencia" in pair:
            return "metriza_potencial"
        if "u_p" in pair and "tensor_coherencia" in pair:
            return "mide_reciprocidad"
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
        if "hazlo" in lower or "crear" in lower or "implement" in lower:
            return "realize"
        if "por qué" in lower or "porque" in lower:
            return "explain_cause"
        return "stabilize"

    def project(self, complex_: ConceptComplex, trace: Optional[Dict[str, Any]] = None) -> str:
        names = list(complex_.nodes.keys())
        has = set(names)
        dominant = names[:6]
        energy = trace.get("energy", None) if trace else None
        closure = trace.get("closure", None) if trace else None

        if "biblioteca_babel" in has and "tensor_coherencia" in has and "u_p" in has:
            core = (
                "La Biblioteca de Babel queda tratada como campo potencial de complejos conceptuales, "
                "no como lista de caracteres. u/p abre la tensión entre actuación y forma; "
                "el tensor de coherencia pesa esa tensión y curva el potencial para que aparezca "
                "un complejo proyectable."
            )
        elif "nodo_conceptual" in has and "relacion_infinita" in has:
            core = (
                "El centro activo es un complejo de nodos conceptuales relacionales. "
                "Cada nodo conserva una sección finita, pero su potencial queda abierto por expansión, "
                "relación, refinamiento y proyección."
            )
        elif "realizacion_codigo" in has:
            core = (
                "La realización correcta es construir un runtime donde el texto solo sea una carta: "
                "se eleva a nodos, se estabiliza por energía u/p bajo H, y luego se proyecta en español."
            )
        else:
            core = (
                "Activo el campo conceptual recibido, estabilizo las relaciones dominantes y proyecto "
                "solo la parte que tiene cierre suficiente."
            )

        refinements = sorted(
            [n for n in names if "aspecto" in n or n in {
                "potencial_expresivo_total", "proyeccion_textual_fraccionaria", "clausura_nodal_relacional",
                "metrica_de_error", "masa_de_cierre", "ponderacion_direccional",
                "actuacion", "inercia_formal", "reciprocidad_estructural",
                "estado_interno", "valencia_relacional", "proyecciones_posibles",
            }]
        )
        if refinements:
            core += " Refinamientos abiertos: " + ", ".join(r.replace("_", " ") for r in refinements[:7]) + "."

        if energy is not None and closure is not None:
            core += f" Energía final={energy:.4f}; cierre={closure:.4f}."
        core += " Nodos dominantes: " + ", ".join(d.replace("_", " ") for d in dominant) + "."
        return core


# ---------------------------------------------------------------------------
# Relations and coherence machinery
# ---------------------------------------------------------------------------


def make_relation(source: str, target: str, relation_type: str, d: int, weight: float = 1.0) -> RelationOperator:
    rng = rng_for(f"relation::{source}->{target}:{relation_type}")
    a = rng.normal(0.0, 0.025, (d, d))
    # near-identity relation operator: a relation transforms while preserving interpretability.
    matrix = np.eye(d) + a
    return RelationOperator(source=source, target=target, relation_type=relation_type, matrix=matrix, weight=weight)


class UPBundle:
    """u/p reciprocity over nodes and relation contexts."""

    def __init__(self, d: int):
        self.d = d
        self.half = d // 2
        rng = rng_for("UPBundle")
        self.F = rng.normal(0.0, 0.14, (self.half, self.half))
        self.G = rng.normal(0.0, 0.14, (self.half, self.half))

    def split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(x) != self.d:
            raise ValueError(f"Expected dimension {self.d}, got {len(x)}")
        return x[: self.half], x[self.half :]

    def residual_node(self, node: ConceptNode, neighborhood: np.ndarray) -> np.ndarray:
        # The neighborhood makes u/p relational, not a local vector trick.
        x = normalize(0.82 * node.state + 0.18 * neighborhood)
        u, p = self.split(x)
        pred_u = np.tanh(p @ self.F)
        pred_p = np.tanh(u @ self.G)
        e_u = u - pred_u
        e_p = p - pred_p
        return np.concatenate([e_u, e_p])

    def residual_relation(self, rel: RelationOperator, nodes: Dict[str, ConceptNode]) -> np.ndarray:
        if rel.source not in nodes or rel.target not in nodes:
            return np.zeros(self.d)
        src = nodes[rel.source].state
        tgt = nodes[rel.target].state
        transported = normalize(rel.matrix @ tgt)
        expected = normalize(src)
        return rel.weight * (transported - rel.expected_alignment * expected)


class CoherenceTensor:
    """H = D + L L^T, acting on the full error field."""

    def __init__(self, error_dim: int, rank: int = 8, seed: str = "H"):
        self.error_dim = error_dim
        self.rank = rank
        rng = rng_for(seed)
        self.base_D = 0.75 + 0.5 * rng.random(error_dim)
        self.base_L = rng.normal(0.0, 0.055, (error_dim, rank))

    def instantiate(self, complex_: ConceptComplex) -> Tuple[np.ndarray, np.ndarray]:
        # Regime shifts tensor anisotropy without changing ontology.
        regime_gain = {
            "conceptual": 1.00,
            "formal": 1.18,
            "implementation": 1.32,
            "critical": 1.45,
        }.get(complex_.regime, 1.0)
        node_gain = 1.0 + 0.025 * len(complex_.nodes)
        relation_gain = 1.0 + 0.012 * len(complex_.relations)
        D = self.base_D * regime_gain * node_gain
        L = self.base_L * math.sqrt(relation_gain)
        return D, L

    @staticmethod
    def energy(error: np.ndarray, D: np.ndarray, L: np.ndarray) -> float:
        if len(error) != len(D):
            raise ValueError(f"Error field dimension mismatch: {len(error)} vs {len(D)}")
        diag_part = float(np.sum(D * error * error))
        low_part = float(np.sum((L.T @ error) ** 2))
        return diag_part + low_part


class ConceptualEnergy:
    """Builds total error field and computes E_X(S)."""

    def __init__(self, d: int, max_nodes: int = 12, max_relations: int = 32):
        self.d = d
        self.max_nodes = max_nodes
        self.max_relations = max_relations
        self.up = UPBundle(d)
        # error field = node residual slots + relation residual slots + global slots
        self.error_dim = d * max_nodes + d * max_relations + 8
        self.tensor = CoherenceTensor(self.error_dim, rank=10)

    def error_field(self, complex_: ConceptComplex, memory: Optional[np.ndarray] = None) -> np.ndarray:
        memory = np.zeros(self.d) if memory is None else memory
        errors: List[np.ndarray] = []

        # Node residuals: each node is measured in its relational neighborhood.
        node_items = list(complex_.nodes.items())[: self.max_nodes]
        for name, node in node_items:
            neighborhood = self._neighborhood_state(name, complex_)
            errors.append(self.up.residual_node(node, neighborhood))
        for _ in range(self.max_nodes - len(node_items)):
            errors.append(np.zeros(self.d))

        # Relation residuals.
        rel_items = complex_.relations[: self.max_relations]
        for rel in rel_items:
            errors.append(self.up.residual_relation(rel, complex_.nodes))
        for _ in range(self.max_relations - len(rel_items)):
            errors.append(np.zeros(self.d))

        # Global errors: memory, projection readiness, closure tension, complexity pressure.
        active = complex_.active_state()
        mem_err = 1.0 - cosine(active, memory) if np.linalg.norm(memory) > 1e-9 else 0.2
        relation_density = len(complex_.relations) / max(len(complex_.nodes), 1)
        complexity = np.mean([n.complexity_level for n in complex_.nodes.values()]) if complex_.nodes else 0.0
        projection_ready = self._projection_readiness(complex_)
        contradiction = self._contradiction_pressure(complex_)
        open_potential = 1.0 if complex_.potential_cardinality == "unbounded" else 0.0
        closure_need = 1.0 / (1.0 + relation_density + projection_ready)
        regime_pressure = 0.15 if complex_.regime in {"conceptual", "implementation", "formal"} else 0.4
        global_err = np.array([
            mem_err,
            max(0.0, 1.0 - projection_ready),
            closure_need,
            contradiction,
            0.05 * complexity,
            0.1 / (1.0 + relation_density),
            regime_pressure,
            0.01 * open_potential,
        ])
        return np.concatenate(errors + [global_err])

    def energy(self, complex_: ConceptComplex, memory: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        e = self.error_field(complex_, memory)
        D, L = self.tensor.instantiate(complex_)
        return self.tensor.energy(e, D, L), e, D, L

    def closure(self, complex_: ConceptComplex, memory: Optional[np.ndarray] = None) -> float:
        E, _, _, _ = self.energy(complex_, memory)
        relation_density = len(complex_.relations) / max(len(complex_.nodes), 1)
        projection_ready = self._projection_readiness(complex_)
        # Exponential inverse energy plus projection/relation bonuses.
        return float(np.exp(-E / (self.error_dim + 1e-9)) * (0.65 + 0.20 * projection_ready + 0.15 * min(relation_density / 3.0, 1.0)))

    def _neighborhood_state(self, name: str, complex_: ConceptComplex) -> np.ndarray:
        if name not in complex_.nodes:
            return np.zeros(self.d)
        related = []
        for rel in complex_.relations:
            if rel.source == name and rel.target in complex_.nodes:
                related.append(rel.matrix @ complex_.nodes[rel.target].state)
            elif rel.target == name and rel.source in complex_.nodes:
                related.append(rel.matrix.T @ complex_.nodes[rel.source].state)
        if not related:
            return complex_.nodes[name].state
        return normalize(np.mean(related, axis=0))

    @staticmethod
    def _projection_readiness(complex_: ConceptComplex) -> float:
        if not complex_.nodes:
            return 0.0
        ready = sum(1 for n in complex_.nodes.values() if "spanish" in n.charts)
        return ready / len(complex_.nodes)

    @staticmethod
    def _contradiction_pressure(complex_: ConceptComplex) -> float:
        names = set(complex_.nodes.keys())
        pressure = 0.0
        if "fraccion_superficial" in names and "nodo_conceptual" in names:
            # This is not contradiction; it is explicitly allowed by the ontology.
            pressure -= 0.05
        if "token_ontology" in names and "nodo_conceptual" in names:
            pressure += 0.8
        return max(0.0, pressure)


# ---------------------------------------------------------------------------
# Implicit conceptual Babel: unbounded expansion over nodes/relations
# ---------------------------------------------------------------------------


class ConceptualBabel:
    """
    Implicit potential over conceptual-relational complexes.

    It never stores all complexes. It defines universal expansion operators.
    """

    def __init__(self, d: int):
        self.d = d

    def expand(self, complex_: ConceptComplex, context: str) -> List[ConceptComplex]:
        expansions: List[ConceptComplex] = []
        expansions.extend(self._refine_high_potential_nodes(complex_, context))
        expansions.extend(self._add_missing_core_relations(complex_))
        expansions.extend(self._add_projection_node_if_needed(complex_))
        expansions.extend(self._add_closure_node_if_needed(complex_))

        # Deduplicate by signature, keep active unbounded potential marker.
        seen = set()
        unique: List[ConceptComplex] = []
        for c in expansions:
            c.potential_cardinality = "unbounded"
            sig = c.signature()
            if sig not in seen:
                unique.append(c)
                seen.add(sig)
        return unique

    def _refine_high_potential_nodes(self, complex_: ConceptComplex, context: str) -> List[ConceptComplex]:
        expansions: List[ConceptComplex] = []
        # Refine only a few nodes at each step; the operator itself is unbounded.
        candidates = sorted(
            complex_.nodes.values(),
            key=lambda n: (n.memory_trace, -n.complexity_level, n.name),
            reverse=True,
        )[:3]
        for node in candidates:
            if node.complexity_level >= 2:
                continue
            c = complex_.clone()
            children = node.refine(context, max_children=3)
            for child in children:
                c.nodes[child.name] = child
                c.relations.append(make_relation(node.name, child.name, "refina", self.d, weight=0.55))
                c.relations.append(make_relation(child.name, node.name, "pertenece_a", self.d, weight=0.35))
            c.history.append(f"refine:{node.name}")
            expansions.append(c)
        return expansions

    def _add_missing_core_relations(self, complex_: ConceptComplex) -> List[ConceptComplex]:
        names = set(complex_.nodes.keys())
        desired = []
        if {"biblioteca_babel", "tensor_coherencia"}.issubset(names):
            desired.append(("tensor_coherencia", "biblioteca_babel", "metriza"))
        if {"u_p", "tensor_coherencia"}.issubset(names):
            desired.append(("u_p", "tensor_coherencia", "produce_error_para"))
        if {"nodo_conceptual", "relacion_infinita"}.issubset(names):
            desired.append(("nodo_conceptual", "relacion_infinita", "se_define_por"))
        if {"fraccion_superficial", "nodo_conceptual"}.issubset(names):
            desired.append(("fraccion_superficial", "nodo_conceptual", "se_eleva_a"))
        if {"proyeccion_textual", "nodo_conceptual"}.issubset(names):
            desired.append(("nodo_conceptual", "proyeccion_textual", "se_proyecta_en"))

        out = []
        existing = {(r.source, r.target, r.relation_type) for r in complex_.relations}
        for src, tgt, typ in desired:
            if (src, tgt, typ) not in existing:
                c = complex_.clone()
                c.relations.append(make_relation(src, tgt, typ, self.d, weight=0.95))
                c.history.append(f"relate:{src}->{tgt}:{typ}")
                out.append(c)
        return out

    def _add_projection_node_if_needed(self, complex_: ConceptComplex) -> List[ConceptComplex]:
        if "proyeccion_textual" in complex_.nodes:
            return []
        c = complex_.clone()
        chart = TextChart(self.d)
        c.nodes["proyeccion_textual"] = ConceptNode(
            name="proyeccion_textual",
            state=chart.concept_vector("proyeccion_textual"),
            charts={"spanish": "proyección textual", "symbolic": "π_text"},
            metadata={"role": "surface_projection_chart"},
        )
        for name in list(complex_.nodes.keys())[:6]:
            c.relations.append(make_relation(name, "proyeccion_textual", "puede_proyectarse", self.d, weight=0.30))
        c.history.append("add:proyeccion_textual")
        return [c]

    def _add_closure_node_if_needed(self, complex_: ConceptComplex) -> List[ConceptComplex]:
        if "cierre_estructural" in complex_.nodes:
            return []
        c = complex_.clone()
        chart = TextChart(self.d)
        c.nodes["cierre_estructural"] = ConceptNode(
            name="cierre_estructural",
            state=chart.concept_vector("cierre_estructural"),
            charts={"spanish": "cierre estructural", "symbolic": "CLOSE"},
            metadata={"role": "closure_condition"},
        )
        for name in list(complex_.nodes.keys())[:6]:
            c.relations.append(make_relation(name, "cierre_estructural", "debe_cerrar_en", self.d, weight=0.25))
        c.history.append("add:cierre_estructural")
        return [c]


class CoherenceFlow:
    """
    Finite approximation of a simultaneous field over the infinite potential.

    The frontier is not Babel. It is a computable section of the distribution
    μ_X(S) ∝ exp(-E_X(S)/τ) over conceptual complexes.
    """

    def __init__(self, d: int, beam: int = 6, steps: int = 5, temp: float = 1.0):
        self.d = d
        self.beam = beam
        self.steps = steps
        self.temp = temp
        self.babel = ConceptualBabel(d)
        self.energy_model = ConceptualEnergy(d)

    def run(self, initial: ConceptComplex, context: str, memory: Optional[np.ndarray] = None) -> Tuple[ConceptComplex, Dict[str, Any]]:
        frontier: List[Tuple[ConceptComplex, float]] = [(initial.clone(), 1.0)]
        trace: Dict[str, Any] = {"steps": []}

        best_complex = initial.clone()
        best_energy, _, _, _ = self.energy_model.energy(best_complex, memory)
        best_closure = self.energy_model.closure(best_complex, memory)

        for step in range(self.steps):
            candidates: List[ConceptComplex] = []
            for complex_, _weight in frontier:
                candidates.append(complex_)
                candidates.extend(self.babel.expand(complex_, context))

            scored = []
            for c in candidates:
                E, err, D, L = self.energy_model.energy(c, memory)
                closure = self.energy_model.closure(c, memory)
                # Energy reduced by closure; more coherent projection wins.
                objective = E - 0.25 * self.energy_model.error_dim * closure
                scored.append((objective, E, closure, c))

            scored.sort(key=lambda x: x[0])
            kept = scored[: self.beam]
            weights = softmax_neg_energy([x[0] for x in kept], temp=self.temp)
            frontier = [(kept[i][3], float(weights[i])) for i in range(len(kept))]

            if kept[0][1] < best_energy or kept[0][2] > best_closure:
                best_energy = kept[0][1]
                best_closure = kept[0][2]
                best_complex = kept[0][3].clone()

            trace["steps"].append({
                "step": step,
                "candidates": len(candidates),
                "best_objective": float(kept[0][0]),
                "best_energy": float(kept[0][1]),
                "best_closure": float(kept[0][2]),
                "best_nodes": sorted(list(kept[0][3].nodes.keys()))[:12],
                "best_history_tail": kept[0][3].history[-4:],
            })

        trace["energy"] = float(best_energy)
        trace["closure"] = float(best_closure)
        trace["nodes"] = sorted(list(best_complex.nodes.keys()))
        trace["relations"] = len(best_complex.relations)
        trace["potential"] = best_complex.potential_cardinality
        return best_complex, trace


# ---------------------------------------------------------------------------
# Runtime: conversation over conceptual fields
# ---------------------------------------------------------------------------


class ConceptualMemory:
    def __init__(self, d: int):
        self.d = d
        self.memory = np.zeros(d)
        self.episodes: List[Dict[str, Any]] = []

    def read(self) -> np.ndarray:
        return self.memory.copy()

    def reenter(self, complex_: ConceptComplex, surface: str, trace: Dict[str, Any]) -> None:
        active = complex_.active_state()
        if np.linalg.norm(self.memory) < 1e-9:
            self.memory = active
        else:
            self.memory = normalize(0.86 * self.memory + 0.14 * active)
        for n in complex_.nodes.values():
            n.memory_trace = min(1.0, n.memory_trace + 0.04)
        self.episodes.append({
            "surface": surface,
            "nodes": sorted(list(complex_.nodes.keys())),
            "relations": len(complex_.relations),
            "energy": trace.get("energy"),
            "closure": trace.get("closure"),
        })
        self.episodes = self.episodes[-200:]

    def as_serializable(self) -> Dict[str, Any]:
        return {"memory": self.memory.tolist(), "episodes": self.episodes}

    @staticmethod
    def from_serializable(d: int, data: Dict[str, Any]) -> "ConceptualMemory":
        m = ConceptualMemory(d)
        if "memory" in data:
            m.memory = np.asarray(data["memory"], dtype=float)
        m.episodes = list(data.get("episodes", []))
        return m


class ConceptualBabelRuntime:
    """Conversational runtime: lift -> flow -> project -> reenter."""

    def __init__(self, d: int = 48, beam: int = 7, steps: int = 6, state_path: Optional[str] = None):
        if d % 2 != 0:
            raise ValueError("d must be even for u/p split")
        self.d = d
        self.chart = TextChart(d)
        self.flow = CoherenceFlow(d, beam=beam, steps=steps, temp=1.0)
        self.memory = ConceptualMemory(d)
        self.state_path = Path(state_path) if state_path else None
        if self.state_path and self.state_path.exists():
            self.load(self.state_path)

    def respond(self, surface: str) -> Dict[str, Any]:
        complex0 = self.chart.lift(surface)
        best, trace = self.flow.run(complex0, context=surface, memory=self.memory.read())
        response = self.chart.project(best, trace)
        self.memory.reenter(best, response, trace)
        if self.state_path:
            self.save(self.state_path)
        return {
            "input": surface,
            "response": response,
            "complex": best.as_serializable(),
            "trace": trace,
            "memory_episodes": len(self.memory.episodes),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"d": self.d, "memory": self.memory.as_serializable()}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("d", self.d)) != self.d:
            raise ValueError("Saved runtime dimension mismatch")
        self.memory = ConceptualMemory.from_serializable(self.d, data.get("memory", {}))


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def demo() -> Dict[str, Any]:
    runtime = ConceptualBabelRuntime(d=48, beam=7, steps=6)
    prompts = [
        "La Biblioteca de Babel no son caracteres: son nodos conceptuales relacionales.",
        "El tensor de coherencia y u/p deben actuar sobre ese potencial completo.",
        "Hazlo realidad en código sin tratar tokens como fundamento.",
    ]
    outputs = [runtime.respond(p) for p in prompts]
    return {"outputs": outputs, "episodes": runtime.memory.episodes}


if __name__ == "__main__":
    result = demo()
    print(json.dumps(result, ensure_ascii=False, indent=2)[:8000])
