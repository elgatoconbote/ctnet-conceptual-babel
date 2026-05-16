#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

from ctnet_conceptual_babel import ConceptualBabelRuntime, TextChart, ConceptualEnergy, ConceptualBabel, FractionNode
from ctnet_conceptual_babel.runtime import CoherenceConditionedBabelGenerator
from ctnet_conceptual_babel.core.relation import compose_relations, make_relation, project_relation


def test_generator_exposes_generate_closed_complex():
    gen = CoherenceConditionedBabelGenerator(48)
    c = gen.generate_closed_complex('babel tensor coherencia', np.zeros(48), np.zeros(48), 'conceptual')
    assert c.generated_one_shot is True


def test_trace_has_required_one_shot_fields():
    rt = ConceptualBabelRuntime(d=48)
    out = rt.respond('Babel como campo conceptual')
    t = out['trace']
    assert t['generated_one_shot'] is True and t['babel_generator_conditioned'] is True


def test_no_candidate_ranking_api_in_runtime_respond():
    src = Path('ctnet_conceptual_babel/runtime.py').read_text(encoding='utf-8')
    assert 'softmax_neg_energy' not in src
    assert '.expand(' not in src


def test_output_complex_exists_before_projection_and_projection_uses_complex():
    rt = ConceptualBabelRuntime(d=48)
    out = rt.respond('nodo conceptual relacional')
    assert 'complex' in out and out['complex']['generated_one_shot'] is True


def test_conditioning_changes_generated_complex_with_memory_change():
    rt = ConceptualBabelRuntime(d=48)
    a = rt.respond('Babel tensor coherencia')['complex']['closed_complex_id']
    rt.respond('mensaje intermedio que altera memoria')
    b = rt.respond('Babel tensor coherencia')['complex']['closed_complex_id']
    assert a != b


def test_empty_input_continues_prior_closed_complex():
    rt = ConceptualBabelRuntime(d=48)
    r1 = rt.respond('Babel continuidad nodal')
    r2 = rt.respond('')
    assert r1['complex']['nodes']
    assert r2['trace']['generated_one_shot'] is True


def test_text_lifts_to_nodes_not_tokens():
    chart = TextChart(48)
    c = chart.lift('Babel tensor u/p nodo relación')
    assert {'biblioteca_babel', 'tensor_coherencia', 'u_p', 'nodo_conceptual', 'fraccion_superficial'}.issubset(c.nodes)
    assert isinstance(c.nodes['fraccion_superficial'], FractionNode)


def test_fraction_is_bidirectional_node():
    f = TextChart(48).lift('a').nodes['fraccion_superficial']
    assert isinstance(f, FractionNode) and f.fraction == 'a' and 'surface' in f.charts


def test_conceptual_babel_exists():
    assert ConceptualBabel(48).d == 48


def test_tensor_up_energy_defined_on_complex():
    E, e, D, L = ConceptualEnergy(48).energy(TextChart(48).lift('Nodos conceptuales relacionales con tensor'))
    assert E > 0 and len(e) == len(D) and L.shape[0] == len(D)


def test_runtime_chat_persists_state(tmp_path):
    state = tmp_path / 'state.json'
    out = ConceptualBabelRuntime(d=48, state_path=str(state)).respond('Haz real CTNet Babel con nodos conceptuales, no tokens')
    assert 'response' in out and state.exists()
    assert len(ConceptualBabelRuntime(d=48, state_path=str(state)).memory.episodes) >= 1


def test_relation_composition_is_explicit_and_node_relational():
    r1 = make_relation('nodo_conceptual', 'tensor_coherencia', 'influye_en', 48)
    r2 = make_relation('tensor_coherencia', 'biblioteca_babel', 'metriza', 48)
    composed = compose_relations(r1, r2, relation_type='influye_y_metriza')
    assert composed.source == 'nodo_conceptual'


def test_relation_projection_maps_node_state_through_operator():
    chart = TextChart(48)
    c = chart.lift('nodo conceptual relación')
    rel = make_relation('nodo_conceptual', 'relacion_infinita', 'se_define_por', 48)
    projected = project_relation(rel, c.nodes['nodo_conceptual'].state)
    assert projected.shape == (48,)
