from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .charts.text import TextChart
from .core.coherence import ConceptualEnergy
from .core.complex import ActiveNodalComplex, ConceptComplex
from .core.node import ConceptNode, normalize, stable_seed
from .core.relation import compose_relations, make_relation
from .surface import CoherenceForcedBabelTextGenerator


class ConceptualBabel:
    """
    Babel no se almacena.
    Babel es el generador potencial: el runtime produce una sección activa finita.

    expand() queda solo como compatibilidad con v0.1/v0.2 tests.
    No es el órgano generativo de v0.3.
    """
    def __init__(self, d: int):
        self.d = d

    def expand(self, complex_: ConceptComplex, context: str) -> List[ConceptComplex]:
        out: List[ConceptComplex] = []

        refined = complex_.clone()
        added = False
        for node in list(refined.nodes.values())[:2]:
            if node.complexity_level >= 2:
                continue
            for child in node.refine(context, max_children=2):
                if child.name not in refined.nodes:
                    refined.nodes[child.name] = child
                    refined.relations.append(make_relation(node.name, child.name, "refina", self.d, 0.55))
                    refined.relations.append(make_relation(child.name, node.name, "pertenece_a", self.d, 0.35))
                    added = True

        if added:
            refined.history.append("compat_expand_refine")
            refined.potential_cardinality = "unbounded"
            out.append(refined)

        closed = complex_.clone()
        if "cierre_estructural" not in closed.nodes:
            chart = TextChart(self.d)
            closed.nodes["cierre_estructural"] = ConceptNode(
                "cierre_estructural",
                chart.concept_vector("cierre_estructural"),
                charts={"spanish": "cierre estructural", "symbolic": "CLOSE"},
                metadata={"role": "compat_closure_node"},
            )
            for name in list(complex_.nodes.keys())[:4]:
                closed.relations.append(make_relation(name, "cierre_estructural", "debe_cerrar_en", self.d, 0.25))
            closed.history.append("compat_expand_close")
            closed.potential_cardinality = "unbounded"
            out.append(closed)

        return out


class CoherenceConditionedBabelGenerator:
    """
    Generador one-shot.

    No genera candidatos.
    No selecciona entre expansiones.
    No rankea salidas.
    Condiciona el estado generativo con u/p + H y emite un complejo nodal cerrado.
    """

    def __init__(self, d: int):
        self.d = d
        self.energy_model = ConceptualEnergy(d)
        self.raw_babel = ConceptualBabel(d)
        self.chart = TextChart(d)
        self.surface_generator = CoherenceForcedBabelTextGenerator(d)
        self.last_closed_complex: Optional[ActiveNodalComplex] = None
        self.counter = 0

    def _relation_closure(self, c: ConceptComplex) -> List[Any]:
        out = []
        rels = c.relations
        for a in rels:
            for b in rels:
                if a.target == b.source and a.source != b.target:
                    out.append(compose_relations(a, b, relation_type=f"{a.relation_type}_o_{b.relation_type}"))
                    if len(out) >= 8:
                        return out
        return out

    def _add_required_relations(self, c: ConceptComplex) -> None:
        names = set(c.nodes)

        def add(src: str, tgt: str, typ: str, weight: float = 0.9) -> None:
            if src in names and tgt in names:
                exists = any(r.source == src and r.target == tgt and r.relation_type == typ for r in c.relations)
                if not exists:
                    c.relations.append(make_relation(src, tgt, typ, self.d, weight=weight))

        add("biblioteca_babel", "generador_babel", "se_realiza_como_generador")
        add("generador_babel", "informacion_coherente", "emite")
        add("u_p", "generador_babel", "condiciona_antes_de_emitir")
        add("tensor_coherencia", "generador_babel", "curva_antes_de_emitir")
        add("u_p", "tensor_coherencia", "produce_residual_para")
        add("emision_directa", "no_filtrado_posterior", "niega")
        add("generador_babel", "emision_directa", "produce_del_tiron")

    def _condition_nodes(self, c: ConceptComplex, memory: np.ndarray, regime: str) -> ConceptComplex:
        conditioned = c.clone()
        conditioned.regime = regime or c.regime
        self._add_required_relations(conditioned)

        active = conditioned.active_state()
        if active.shape[0] != self.d:
            active = np.zeros(self.d)

        E0, e, D, L = self.energy_model.energy(conditioned, memory)
        H_diag = D[: self.d] + np.sum(L[: self.d, :] * L[: self.d, :], axis=1)
        residual = e[: self.d]
        drive = normalize(active + 0.35 * memory[: self.d] - 0.18 * residual + 0.22 * H_diag)

        for node in conditioned.nodes.values():
            node.state = normalize(0.62 * node.state + 0.38 * drive)
            node.memory_trace = min(1.0, node.memory_trace + 0.03)

        conditioned.history.append("babel_generator_conditioned_by_u_p_H_before_surface_emission")
        return conditioned

    def generate_closed_complex(
        self,
        surface: str,
        state: Optional[ConceptComplex],
        memory: np.ndarray,
        regime: str,
    ) -> ActiveNodalComplex:
        if surface.strip():
            base = self.chart.lift(surface)
        elif self.last_closed_complex is not None:
            base = self.last_closed_complex.clone()
            base.history.append("continued_from_previous_closed_complex")
        elif state is not None:
            base = state.clone()
        else:
            base = self.chart.lift("continuidad conceptual")

        if state is not None and state.nodes:
            for k, v in list(state.nodes.items())[:4]:
                if k not in base.nodes:
                    base.nodes[k] = v.clone()

        conditioned = self._condition_nodes(base, memory, regime)
        relation_compositions = self._relation_closure(conditioned)

        E, e, _, _ = self.energy_model.energy(conditioned, memory)
        closure = self.energy_model.closure(conditioned, memory)
        mass = float(1.0 / (1.0 + E / (self.energy_model.error_dim + 1e-9)))

        self.counter += 1
        closed_id_material = conditioned.signature() + "::" + str(self.counter) + "::" + str(float(E))
        closed_id = f"closed-{stable_seed(closed_id_material) % 1000000:06d}"

        out = ActiveNodalComplex(
            nodes=conditioned.nodes,
            relations=conditioned.relations,
            regime=conditioned.regime,
            closure_goal=conditioned.closure_goal,
            history=conditioned.history + ["one_shot_babel_surface_ready"],
            potential_cardinality="unbounded",
            relation_compositions=relation_compositions,
            projection_chart="text",
            closure_state=float(closure),
            u_p_residual=e[: self.d].tolist(),
            coherence_energy=float(E),
            coherence_mass=mass,
            generated_one_shot=True,
            closed_complex_id=closed_id,
            reentry_applied=False,
        )

        self.last_closed_complex = out.clone()
        return out

    def emit_surface(self, closed: ActiveNodalComplex, trace: Dict[str, Any]) -> str:
        emitted = self.surface_generator.emit(closed, trace)
        closed.surface_emission = emitted
        return emitted


class CoherenceFlow:
    """
    Compatibilidad con tests antiguos. Ya no es el órgano de generación.
    """
    def __init__(self, d: int, beam: int = 6, steps: int = 5, temp: float = 1.0):
        self.d = d
        self.energy_model = ConceptualEnergy(d)

    def run(self, initial: ConceptComplex, context: str, memory: Optional[np.ndarray] = None):
        mem = np.zeros(self.d) if memory is None else memory
        E, _, _, _ = self.energy_model.energy(initial, mem)
        closure = self.energy_model.closure(initial, mem)
        trace = {
            "energy": float(E),
            "closure": float(closure),
            "nodes": sorted(list(initial.nodes.keys())),
            "relations": len(initial.relations),
            "potential": initial.potential_cardinality,
            "generated_one_shot": True,
        }
        return initial.clone(), trace


class ConceptualMemory:
    def __init__(self, d: int):
        self.d = d
        self.memory = np.zeros(d)
        self.episodes = []

    def read(self):
        return self.memory.copy()

    def retrieve(self, query_complex: ConceptComplex, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.episodes:
            return []
        q = query_complex.active_state()
        out = []
        for ep in self.episodes:
            ev = np.asarray(ep.get("state", np.zeros(self.d)), dtype=float)
            qn = np.linalg.norm(q)
            en = np.linalg.norm(ev)
            sim = float(np.dot(q, ev) / (qn * en + 1e-12)) if qn > 1e-12 and en > 1e-12 else 0.0
            influence = max(0.0, sim) * (0.35 + 0.65 * float(ep.get("closure", 0.5)))
            out.append(
                {
                    "episode_id": ep.get("episode_id"),
                    "similarity": sim,
                    "influence": influence,
                    "nodes": list(ep.get("nodes", []))[:10],
                }
            )
        return sorted(out, key=lambda x: x["influence"], reverse=True)[:top_k]

    def influence_vector(self, retrieved: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not retrieved:
            return np.zeros(self.d)
        acc = np.zeros(self.d)
        idx = {ep.get("episode_id"): ep for ep in self.episodes}
        for item in retrieved:
            ep = idx.get(item.get("episode_id"))
            if ep is not None:
                acc += float(item.get("influence", 0.0)) * np.asarray(ep.get("state", np.zeros(self.d)), dtype=float)
        return normalize(acc) if np.linalg.norm(acc) > 1e-9 else acc

    def reenter(self, complex_, surface, trace):
        active = complex_.active_state()
        influence = float(trace.get("memory_retrieval", {}).get("total_influence", 0.0))
        if np.linalg.norm(self.memory) < 1e-9:
            self.memory = active
        else:
            reentry_gain = min(0.32, 0.14 + 0.22 * influence)
            self.memory = normalize((1.0 - reentry_gain) * self.memory + reentry_gain * active)
        for n in complex_.nodes.values():
            n.memory_trace = min(1.0, n.memory_trace + 0.04)
        self.episodes.append(
            {
                "episode_id": len(self.episodes) + 1,
                "surface": surface,
                "nodes": sorted(list(complex_.nodes.keys())),
                "relations": len(complex_.relations),
                "energy": trace.get("coherence_energy"),
                "closure": trace.get("closure"),
                "state": active.tolist(),
                "reentry_gain": float(trace.get("reentry_gain", 0.0)),
                "closed_complex_id": trace.get("closed_complex_id"),
            }
        )
        self.episodes = self.episodes[-200:]

    def as_serializable(self):
        return {"memory": self.memory.tolist(), "episodes": self.episodes}

    @staticmethod
    def from_serializable(d: int, data: Dict[str, Any]):
        m = ConceptualMemory(d)
        m.memory = np.asarray(data.get("memory", m.memory), dtype=float)
        m.episodes = list(data.get("episodes", []))
        return m


class ConceptualBabelRuntime:
    def __init__(self, d: int = 48, beam: int = 7, steps: int = 6, state_path: Optional[str] = None):
        if d % 2 != 0:
            raise ValueError("d must be even for u/p split")
        self.d = d
        self.chart = TextChart(d)
        self.generator = CoherenceConditionedBabelGenerator(d)
        self.flow = CoherenceFlow(d, beam=beam, steps=steps, temp=1.0)
        self.memory = ConceptualMemory(d)
        self.state_path = Path(state_path) if state_path else None
        self.last_closed_complex: Optional[ActiveNodalComplex] = None
        if self.state_path and self.state_path.exists():
            self.load(self.state_path)

    def respond(self, surface: str) -> Dict[str, Any]:
        lifted = self.chart.lift(surface)
        retrieved = self.memory.retrieve(lifted, top_k=4)
        influence_vec = self.memory.influence_vector(retrieved)
        base_memory = self.memory.read()
        flow_memory = normalize(0.80 * base_memory + 0.20 * influence_vec) if np.linalg.norm(influence_vec) > 1e-9 else base_memory

        closed = self.generator.generate_closed_complex(
            surface=surface,
            state=self.last_closed_complex or lifted,
            memory=flow_memory,
            regime=lifted.regime,
        )

        total_influence = float(sum(x["influence"] for x in retrieved))
        trace = {
            "generated_one_shot": True,
            "conditioning_operator": "u_p_H",
            "babel_generator_conditioned": True,
            "surface_generator": "transition_field_not_template",
            "template_sentence_pools": False,
            "u_p_residual_norm": float(np.linalg.norm(np.asarray(closed.u_p_residual, dtype=float))),
            "coherence_energy": float(closed.coherence_energy),
            "coherence_mass": float(closed.coherence_mass),
            "projection_chart": closed.projection_chart,
            "closed_complex_id": closed.closed_complex_id,
            "closure": float(closed.closure_state),
            "memory_retrieval": {"retrieved": retrieved, "total_influence": total_influence},
            "reentry_gain": min(0.32, 0.14 + 0.22 * total_influence),
            "reentry_applied": False,
        }

        response = self.generator.emit_surface(closed, trace)

        closed.reentry_applied = True
        trace["reentry_applied"] = True
        self.memory.reenter(closed, response, trace)
        self.last_closed_complex = closed.clone()

        if self.state_path:
            self.save(self.state_path)

        return {
            "input": surface,
            "response": response,
            "complex": closed.as_serializable(),
            "trace": trace,
            "memory_episodes": len(self.memory.episodes),
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "d": self.d,
                    "memory": self.memory.as_serializable(),
                    "last_closed_complex": self.last_closed_complex.as_serializable() if self.last_closed_complex else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("d", self.d)) != self.d:
            raise ValueError("Saved runtime dimension mismatch")
        self.memory = ConceptualMemory.from_serializable(self.d, data.get("memory", {}))
        if data.get("last_closed_complex"):
            self.last_closed_complex = ActiveNodalComplex.from_serializable(data["last_closed_complex"])
            self.generator.last_closed_complex = self.last_closed_complex.clone()
