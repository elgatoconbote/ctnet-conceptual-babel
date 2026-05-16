from __future__ import annotations

from .charts.text import TextChart
from .core.coherence import ConceptualEnergy, CoherenceTensor
from .core.complex import ActiveNodalComplex, ConceptComplex
from .core.node import ConceptNode, FractionNode
from .core.relation import RelationOperator, compose_relations, make_relation, project_relation
from .runtime import ConceptualBabel, ConceptualBabelRuntime, ConceptualMemory, CoherenceConditionedBabelGenerator, CoherenceFlow
from .surface import CoherenceForcedBabelTextGenerator


def demo(message: str = "La Biblioteca de Babel no son tokens: son nodos conceptuales relacionales.") -> str:
    runtime = ConceptualBabelRuntime()
    return runtime.respond(message)["response"]


__all__ = [
    "TextChart",
    "ConceptualEnergy",
    "CoherenceTensor",
    "ConceptComplex",
    "ActiveNodalComplex",
    "ConceptNode",
    "FractionNode",
    "RelationOperator",
    "make_relation",
    "compose_relations",
    "project_relation",
    "ConceptualBabel",
    "ConceptualBabelRuntime",
    "ConceptualMemory",
    "CoherenceConditionedBabelGenerator",
    "CoherenceForcedBabelTextGenerator",
    "CoherenceFlow",
    "demo",
]
