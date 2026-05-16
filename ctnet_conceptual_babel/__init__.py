from __future__ import annotations

from .charts.text import TextChart
from .core.coherence import ConceptualEnergy, CoherenceTensor
from .core.complex import ConceptComplex, ActiveNodalComplex
from .core.node import ConceptNode, FractionNode
from .core.relation import RelationOperator, compose_relations, make_relation, project_relation
from .runtime import ConceptualBabel, ConceptualBabelRuntime, ConceptualMemory, CoherenceConditionedBabelGenerator


def demo(message: str = 'La Biblioteca de Babel no son tokens: son nodos conceptuales relacionales.') -> str:
    """Backward-compatible demo entrypoint returning one response string."""
    runtime = ConceptualBabelRuntime()
    return runtime.respond(message)['response']


__all__ = [
    'TextChart', 'ConceptualEnergy', 'CoherenceTensor', 'ConceptComplex', 'ActiveNodalComplex',
    'ConceptNode', 'FractionNode', 'RelationOperator', 'make_relation', 'compose_relations', 'project_relation',
    'ConceptualBabel', 'ConceptualBabelRuntime', 'ConceptualMemory', 'CoherenceConditionedBabelGenerator', 'demo'
]
