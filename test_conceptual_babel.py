#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

from ctnet_conceptual_babel import ConceptualBabelRuntime, TextChart, ConceptualEnergy, ConceptualBabel, FractionNode
from ctnet_conceptual_babel.core.relation import compose_relations, make_relation, project_relation


def test_text_lifts_to_nodes_not_tokens():
    chart = TextChart(48)
    c = chart.lift('Babel tensor u/p nodo relación')
    assert {'biblioteca_babel', 'tensor_coherencia', 'u_p', 'nodo_conceptual', 'fraccion_superficial'}.issubset(c.nodes)
    assert isinstance(c.nodes['fraccion_superficial'], FractionNode)


def test_fraction_is_bidirectional_node():
    f = TextChart(48).lift('a').nodes['fraccion_superficial']
    assert isinstance(f, FractionNode) and f.fraction == 'a' and 'surface' in f.charts


def test_conceptual_babel_expands_nodes_and_relations():
    c = TextChart(48).lift('Biblioteca de Babel tensor coherencia u/p')
    ex = ConceptualBabel(48).expand(c, 'contexto')
    assert ex and any(len(x.nodes) > len(c.nodes) for x in ex)


def test_tensor_up_energy_defined_on_complex():
    E, e, D, L = ConceptualEnergy(48).energy(TextChart(48).lift('Nodos conceptuales relacionales con tensor'))
    assert E > 0 and len(e) == len(D) and L.shape[0] == len(D)


def test_flow_reduces_or_stabilizes_energy():
    rt = ConceptualBabelRuntime(d=48, beam=7, steps=6)
    c0 = rt.chart.lift('La biblioteca de babel son nodos y relaciones bajo tensor de coherencia')
    E0, _, _, _ = rt.flow.energy_model.energy(c0, rt.memory.read())
    _, trace = rt.flow.run(c0, context='test', memory=rt.memory.read())
    assert trace['energy'] <= E0 or trace['closure'] >= rt.flow.energy_model.closure(c0, rt.memory.read())


def test_runtime_chat_persists_state(tmp_path):
    state = tmp_path / 'state.json'
    out = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state)).respond('Haz real CTNet Babel con nodos conceptuales, no tokens')
    assert 'response' in out and state.exists()
    assert len(ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(state)).memory.episodes) >= 1


def test_reentry_updates_memory_and_episode_count():
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=3)
    rt.respond('primer mensaje conceptual')
    m1 = rt.memory.read()
    rt.respond('segundo mensaje conceptual')
    assert len(rt.memory.episodes) == 2
    assert not np.allclose(m1, rt.memory.read())


def test_conceptual_lifting_adds_surface_fraction_node():
    c = TextChart(48).lift('texto breve')
    assert 'fraccion_superficial' in c.nodes and isinstance(c.nodes['fraccion_superficial'], FractionNode)


def test_coherence_energy_and_closure_are_bounded():
    c = TextChart(48).lift('babel tensor coherencia')
    model = ConceptualEnergy(48)
    E, *_ = model.energy(c)
    clos = model.closure(c)
    assert E > 0 and 0.0 < clos <= 1.0


def test_relation_composition_is_explicit_and_node_relational():
    r1 = make_relation('nodo_conceptual', 'tensor_coherencia', 'influye_en', 48)
    r2 = make_relation('tensor_coherencia', 'biblioteca_babel', 'metriza', 48)
    composed = compose_relations(r1, r2, relation_type='influye_y_metriza')
    assert composed.source == 'nodo_conceptual'
    assert composed.target == 'biblioteca_babel'
    assert composed.relation_type == 'influye_y_metriza'
    assert composed.matrix.shape == (48, 48)


def test_relation_projection_maps_node_state_through_operator():
    chart = TextChart(48)
    c = chart.lift('nodo conceptual relación')
    rel = make_relation('nodo_conceptual', 'relacion_infinita', 'se_define_por', 48)
    source = c.nodes['nodo_conceptual'].state
    projected = project_relation(rel, source)
    assert projected.shape == (48,)
    assert np.linalg.norm(projected) > 0


def test_multi_turn_memory_accumulates_and_traces_retrieval():
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4)
    a = rt.respond('Babel como campo conceptual relacional, no lista de cadenas.')
    b = rt.respond('Retoma Babel y tensor de coherencia en continuidad conceptual.')
    assert a['memory_episodes'] == 1 and b['memory_episodes'] == 2
    assert 'memory_retrieval' in b['trace']
    assert isinstance(b['trace']['memory_retrieval']['retrieved'], list)


def test_episode_retrieval_has_ids_similarity_and_influence():
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4)
    rt.respond('primer episodio: nodo conceptual y relación infinita')
    out = rt.respond('segundo episodio: conecta nodo conceptual con relación')
    retrieved = out['trace']['memory_retrieval']['retrieved']
    assert retrieved
    top = retrieved[0]
    assert {'episode_id', 'similarity', 'influence', 'nodes'}.issubset(top.keys())


def test_reentry_changes_later_trace_or_response():
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4)
    r1 = rt.respond('Babel tensor coherencia y u/p en campo conceptual')
    r2 = rt.respond('Babel tensor coherencia y u/p en campo conceptual')
    assert r2['trace']['reentry_gain'] >= r1['trace']['reentry_gain']
    assert r1['response'] != r2['response'] or r1['trace']['memory_retrieval'] != r2['trace']['memory_retrieval']


def test_save_load_preserves_memory_and_episode_vectors(tmp_path):
    p = tmp_path / 'rt.json'
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(p))
    rt.respond('memoria topológica para babel conceptual')
    rt.save(p)
    rt2 = ConceptualBabelRuntime(d=48, beam=5, steps=4, state_path=str(p))
    assert len(rt2.memory.episodes) == 1
    assert 'state' in rt2.memory.episodes[0]


def test_projection_not_static_under_state_change():
    rt = ConceptualBabelRuntime(d=48, beam=5, steps=4)
    prompt = 'La biblioteca de Babel opera como campo conceptual.'
    r1 = rt.respond(prompt)['response']
    rt.respond('mensaje intermedio sobre memoria y reentrada conceptual')
    r3 = rt.respond(prompt)['response']
    assert r1 != r3


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__]))
