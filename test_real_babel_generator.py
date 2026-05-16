from pathlib import Path

from ctnet_conceptual_babel import ConceptualBabelRuntime, CoherenceForcedBabelTextGenerator, TextChart


def test_surface_generator_has_no_sentence_pools():
    src = Path("ctnet_conceptual_babel/surface.py").read_text(encoding="utf-8")
    banned = [
        "opener_pool",
        "noise_pool",
        "up_pool",
        "h_pool",
        "close_pool",
        "En este cierre de un solo paso",
        "La dinámica interna queda determinada",
        "El resultado mantiene cierre",
        "Activo el campo conceptual recibido",
    ]
    for item in banned:
        assert item not in src


def test_surface_generator_is_transition_emitter():
    src = Path("ctnet_conceptual_babel/surface.py").read_text(encoding="utf-8")
    assert "for step in range(max_tokens)" in src
    assert "_lexeme_score" in src
    assert "_grammar_score" in src
    assert "np.argmax" in src


def test_prompt_generates_substantive_babel_answer():
    rt = ConceptualBabelRuntime(d=48)
    out = rt.respond("Babel genera información coherente del tirón bajo u/p y H.")
    response = out["response"]

    assert "Babel" in response
    assert "u/p" in response
    assert "H" in response or "tensor" in response
    assert "coher" in response.lower()
    assert "filtra" in response.lower() or "después" in response.lower()
    assert not response.startswith("Activo el campo conceptual recibido")
    assert "En este cierre de un solo paso" not in response
    assert "El resultado mantiene cierre" not in response


def test_trace_marks_real_transition_generator():
    rt = ConceptualBabelRuntime(d=48)
    out = rt.respond("Babel genera información coherente del tirón bajo u/p y H.")
    trace = out["trace"]

    assert trace["generated_one_shot"] is True
    assert trace["babel_generator_conditioned"] is True
    assert trace["surface_generator"] == "transition_field_not_template"
    assert trace["template_sentence_pools"] is False


def test_text_chart_is_adapter_not_template():
    c = TextChart(48).lift("Babel u/p H coherencia")
    response = TextChart(48).project(
        c,
        {
            "conditioning_operator": "u_p_H",
            "coherence_mass": 0.99,
            "coherence_energy": 1.0,
            "closure": 0.9,
            "closed_complex_id": "test",
        },
    )
    assert "Activo el campo conceptual recibido" not in response
    assert "En este cierre de un solo paso" not in response
