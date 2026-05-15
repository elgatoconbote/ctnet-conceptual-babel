#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ctnet_conceptual_babel import (
    ConceptualBabelRuntime, TextChart, ConceptualEnergy, ConceptualBabel, FractionNode,
    compose_relations, project_relation, make_relation,
)


def test_text_lifts_to_nodes_not_tokens():
    c = TextChart(48).lift("Babel tensor u/p nodo relación")
    assert all(k in c.nodes for k in ["biblioteca_babel", "tensor_coherencia", "u_p", "nodo_conceptual", "fraccion_superficial"])
    assert isinstance(c.nodes["fraccion_superficial"], FractionNode)


def test_persistence_and_reentry(tmp_path):
    state = tmp_path / "state.json"
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state))
    out1 = rt.respond("Haz real CTNet Babel con nodos conceptuales, no tokens")
    out2 = rt.respond("Reingresa al campo conceptual")
    assert out2["memory_episodes"] == out1["memory_episodes"] + 1
    mem_norm = np.linalg.norm(rt.memory.read())
    assert mem_norm > 0
    rt2 = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state))
    assert len(rt2.memory.episodes) >= 2


def test_conceptual_lifting_and_flow_energy():
    rt = ConceptualBabelRuntime(d=48, beam=7, steps=6)
    c0 = rt.chart.lift("La biblioteca de babel son nodos y relaciones bajo tensor de coherencia")
    E0, _, _, _ = rt.flow.energy_model.energy(c0, rt.memory.read())
    best, trace = rt.flow.run(c0, context="test", memory=rt.memory.read())
    assert trace["energy"] <= E0 or trace["closure"] >= rt.flow.energy_model.closure(c0, rt.memory.read())
    assert len(best.nodes) >= len(c0.nodes)


def test_coherence_energy_defined_and_positive():
    c = TextChart(48).lift("Nodos conceptuales relacionales con tensor")
    E, e, D, L = ConceptualEnergy(48).energy(c)
    assert E > 0 and len(e) == len(D) and L.shape[0] == len(D)


def test_relation_composition_and_projection():
    d = 48
    r1 = make_relation("a", "b", "rel1", d, weight=0.6)
    r2 = make_relation("b", "c", "rel2", d, weight=0.8)
    rc = compose_relations(r1, r2)
    x = np.ones(d)
    y_seq = project_relation(r2, project_relation(r1, x))
    y_comp = project_relation(rc, x)
    assert rc.source == "a" and rc.target == "c"
    assert np.allclose(y_seq, y_comp)


def test_babel_expands_nodes_and_relations():
    c = TextChart(48).lift("Biblioteca de Babel tensor coherencia u/p")
    ex = ConceptualBabel(48).expand(c, "contexto")
    assert ex and any(len(x.nodes) > len(c.nodes) for x in ex)

