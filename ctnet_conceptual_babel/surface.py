from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .core.complex import ConceptComplex
from .core.node import stable_seed


class BabelSurfaceGenerator:
    """Deterministic one-shot surface generator conditioned by nodal coherence state."""

    def __init__(self, d: int):
        self.d = d

    def _seed_from_state(self, complex_: ConceptComplex, trace: Dict[str, Any]) -> int:
        names = '|'.join(sorted(complex_.nodes.keys()))
        rels = '|'.join(sorted(f"{r.source}->{r.target}:{r.relation_type}" for r in complex_.relations[:16]))
        cid = str(trace.get('closed_complex_id', 'closed-0'))
        energy = float(trace.get('coherence_energy', 0.0))
        mass = float(trace.get('coherence_mass', 0.0))
        residual = float(trace.get('u_p_residual_norm', 0.0))
        key = f"{names}::{rels}::{cid}::{energy:.5f}::{mass:.5f}::{residual:.5f}::{complex_.regime}"
        return stable_seed(key)

    def emit(self, conditioned_complex: ConceptComplex, trace: Dict[str, Any], max_chars: Optional[int] = None) -> str:
        seed = self._seed_from_state(conditioned_complex, trace)
        rng = np.random.default_rng(seed)

        node_names = set(conditioned_complex.nodes.keys())
        has_babel = 'biblioteca_babel' in node_names
        has_up = 'u_p' in node_names
        has_h = 'tensor_coherencia' in node_names

        opener_pool = [
            'La Biblioteca de Babel aquí funciona como un generador de información potencial y no como un archivo estático de frases.',
            'En este estado, Babel opera como fuente generativa de complejos informacionales que luego se vuelven lenguaje.',
            'Babel aparece como un motor de potencial semántico: produce estructura antes de producir palabras.',
        ]
        noise_pool = [
            'Si se deja sola, esa potencia puede derramarse en ruido, deriva o fragmentos sin cierre suficiente.',
            'Sin condicionamiento, el flujo puede abrir demasiadas ramas y devolver texto disperso.',
            'Sin control recíproco, el campo puede emitir material verbalmente activo pero conceptualmente inestable.',
        ]
        up_pool = [
            'El acoplamiento u/p corrige eso porque enlaza el despliegue activo con la forma que sostiene el complejo.',
            'u/p fija la reciprocidad entre lo que el campo despliega y la forma que lo mantiene consistente.',
            'La condición u/p obliga a que cada avance de contenido conserve estructura y no solo expansión.',
        ]
        h_pool = [
            'Luego H = D + L·L^T pondera el residual de esa reciprocidad y curva la emisión hacia estados de mayor coherencia.',
            'Después, el tensor H pesa el error recíproco y reequilibra la dinámica antes de que aparezca la superficie textual.',
            'Con H, el residual se valora geométricamente y el generador queda precondicionado antes de emitir.',
        ]
        close_pool = [
            'Por eso la respuesta nace coherente en un solo disparo: no se generan candidatos para filtrarlos después.',
            'El texto sale ya condicionado y no por selección posterior: no filtra después, produce directamente coherencia.',
            'La salida no proviene de ranking posterior; emerge en una pasada como proyección de un complejo ya cerrado.',
        ]

        sentences: List[str] = []
        if has_babel:
            sentences.append(opener_pool[int(rng.integers(0, len(opener_pool)))])
        else:
            sentences.append('El campo activo se comporta como generador de información potencial organizada en nodos y relaciones.')
        sentences.append(noise_pool[int(rng.integers(0, len(noise_pool)))])
        if has_up:
            sentences.append(up_pool[int(rng.integers(0, len(up_pool)))])
        if has_h or trace.get('conditioning_operator') == 'u_p_H':
            sentences.append(h_pool[int(rng.integers(0, len(h_pool)))])

        # relation-conditioned clause
        if conditioned_complex.relations:
            top = sorted(conditioned_complex.relations, key=lambda r: abs(float(r.weight)), reverse=True)[:2]
            rel_clause = ' En este cierre pesan relaciones como ' + ' y '.join(
                f"{r.source.replace('_', ' ')} {r.relation_type.replace('_', ' ')} {r.target.replace('_', ' ')}" for r in top
            ) + '.'
            sentences.append(rel_clause.strip())

        sentences.append(close_pool[int(rng.integers(0, len(close_pool)))])
        out = ' '.join(sentences)
        if max_chars is not None and len(out) > max_chars:
            out = out[: max_chars - 1].rstrip() + '…'
        return out
