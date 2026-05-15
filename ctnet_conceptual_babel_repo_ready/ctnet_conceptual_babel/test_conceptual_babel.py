#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np

from ctnet_conceptual_babel import (
    ConceptualBabelRuntime,
    TextChart,
    ConceptualEnergy,
    ConceptualBabel,
    FractionNode,
)


def test_text_lifts_to_nodes_not_tokens():
    chart = TextChart(48)
    c = chart.lift("Babel tensor u/p nodo relación")
    assert "biblioteca_babel" in c.nodes
    assert "tensor_coherencia" in c.nodes
    assert "u_p" in c.nodes
    assert "nodo_conceptual" in c.nodes
    assert "fraccion_superficial" in c.nodes
    assert isinstance(c.nodes["fraccion_superficial"], FractionNode)
    assert c.potential_cardinality == "unbounded"


def test_fraction_is_bidirectional_node():
    chart = TextChart(48)
    c = chart.lift("a")
    f = c.nodes["fraccion_superficial"]
    assert isinstance(f, FractionNode)
    assert f.fraction == "a"
    assert "surface" in f.charts
    assert f.potential_cardinality == "unbounded"


def test_conceptual_babel_expands_nodes_and_relations():
    chart = TextChart(48)
    c = chart.lift("Biblioteca de Babel tensor coherencia u/p")
    babel = ConceptualBabel(48)
    ex = babel.expand(c, "contexto")
    assert ex
    assert any(len(x.nodes) > len(c.nodes) for x in ex)
    assert all(x.potential_cardinality == "unbounded" for x in ex)


def test_tensor_up_energy_defined_on_complex():
    chart = TextChart(48)
    c = chart.lift("Nodos conceptuales relacionales con tensor")
    model = ConceptualEnergy(48)
    E, e, D, L = model.energy(c)
    assert E > 0
    assert len(e) == len(D)
    assert L.shape[0] == len(D)


def test_flow_reduces_or_stabilizes_energy():
    rt = ConceptualBabelRuntime(d=48, beam=7, steps=6)
    c0 = rt.chart.lift("La biblioteca de babel son nodos y relaciones bajo tensor de coherencia")
    E0, _, _, _ = rt.flow.energy_model.energy(c0, rt.memory.read())
    best, trace = rt.flow.run(c0, context="test", memory=rt.memory.read())
    E1 = trace["energy"]
    assert E1 <= E0 or trace["closure"] >= rt.flow.energy_model.closure(c0, rt.memory.read())
    assert "nodes" in trace and len(trace["nodes"]) >= len(c0.nodes)


def test_runtime_chat_persists_state(tmp_path):
    state = tmp_path / "state.json"
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state))
    out = rt.respond("Haz real CTNet Babel con nodos conceptuales, no tokens")
    assert "response" in out
    assert state.exists()
    rt2 = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state))
    assert len(rt2.memory.episodes) >= 1


def run_all():
    test_text_lifts_to_nodes_not_tokens()
    test_fraction_is_bidirectional_node()
    test_conceptual_babel_expands_nodes_and_relations()
    test_tensor_up_energy_defined_on_complex()
    test_flow_reduces_or_stabilizes_energy()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_runtime_chat_persists_state(Path(d))
    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    run_all()
