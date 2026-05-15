#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from ctnet_conceptual_babel import (
    ConceptualBabelRuntime,
    TextChart,
    ConceptualEnergy,
    ConceptualBabel,
    FractionNode,
)
from ctnet_conceptual_babel.core.relation import make_relation


def test_text_lifts_to_nodes_not_tokens():
    chart = TextChart(48)
    c = chart.lift("Babel tensor u/p nodo relación")
    assert "biblioteca_babel" in c.nodes
    assert "tensor_coherencia" in c.nodes


def test_fraction_is_bidirectional_node():
    chart = TextChart(48)
    c = chart.lift("a")
    assert isinstance(c.nodes["fraccion_superficial"], FractionNode)


def test_conceptual_babel_expands_nodes_and_relations():
    chart = TextChart(48)
    c = chart.lift("Biblioteca de Babel tensor coherencia u/p")
    ex = ConceptualBabel(48).expand(c, "contexto")
    assert ex and any(len(x.nodes) > len(c.nodes) for x in ex)


def test_persistence_and_reentry(tmp_path):
    state = tmp_path / "state.json"
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state))
    rt.respond("Haz real CTNet Babel con nodos conceptuales, no tokens")
    rt.respond("reanuda con coherencia")
    assert state.exists()
    rt2 = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state))
    assert len(rt2.memory.episodes) >= 2


def test_conceptual_lifting_and_projection():
    chart = TextChart(48)
    c = chart.lift("nodo conceptual relacional")
    out = chart.project(c, {"energy": 1.2, "closure": 0.4})
    assert "nodos" in out.lower() or "nodo" in out.lower()


def test_coherence_energy_defined_on_complex():
    c = TextChart(48).lift("Nodos conceptuales relacionales con tensor")
    E, e, D, L = ConceptualEnergy(48).energy(c)
    assert E > 0 and len(e) == len(D) and L.shape[0] == len(D)


def test_relation_composition_and_projection():
    d = 48
    chart = TextChart(d)
    c = chart.lift("babel tensor")
    r1 = make_relation("biblioteca_babel", "tensor_coherencia", "metriza_potencial", d)
    r2 = make_relation("tensor_coherencia", "biblioteca_babel", "co-implica", d)
    composed = r1.matrix @ r2.matrix
    assert composed.shape == (d, d)
    vec = c.nodes["biblioteca_babel"].state
    projected = composed @ vec
    assert projected.shape == vec.shape

