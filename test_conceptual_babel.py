#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

from ctnet_conceptual_babel import ConceptualBabelRuntime, TextChart, ConceptualEnergy, ConceptualBabel, FractionNode
from ctnet_conceptual_babel.core.relation import make_relation


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


def test_relation_composition_and_projection_paths_present_after_expand():
    c = TextChart(48).lift('nodo conceptual relación texto')
    ex = ConceptualBabel(48).expand(c, 'ctx')
    relation_types = {r.relation_type for cc in ex for r in cc.relations}
    assert 'se_define_por' in relation_types or 'relaciona' in relation_types
    assert 'puede_proyectarse' in relation_types or 'proyecta' in relation_types


def test_relation_operator_projection_residual_shape():
    chart = TextChart(48)
    c = chart.lift('nodo conceptual relación')
    rel = make_relation('nodo_conceptual', 'relacion_infinita', 'se_define_por', 48)
    c.relations.append(rel)
    residual = ConceptualEnergy(48).up.residual_relation(rel, c.nodes)
    assert residual.shape == (48,)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__]))
