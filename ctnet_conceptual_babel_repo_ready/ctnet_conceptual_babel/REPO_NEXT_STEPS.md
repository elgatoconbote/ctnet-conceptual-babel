# Siguiente paso del repositorio

Este repositorio debe contener el runtime conceptual, no los PDFs ni los experimentos anteriores como núcleo.

## Núcleo

- `ctnet_conceptual_babel.py`: implementación principal.
- `demo_chat.py`: interfaz CLI conversacional mínima.
- `test_conceptual_babel.py`: pruebas de invariantes.
- `README.md`: descripción y uso.
- `CODEX_HANDOFF.md`: instrucciones para continuar con Codex.

## No meter como núcleo

Los PDFs, benchmarks viejos y resultados generados deben quedar fuera del primer commit o ir en `docs/` / `experiments/` después.

## Primer objetivo con Codex

Convertir el prototipo en paquete:

1. separar módulos:
   - `core/node.py`
   - `core/relation.py`
   - `core/complex.py`
   - `core/up.py`
   - `core/coherence.py`
   - `runtime.py`
   - `charts/text.py`
2. añadir persistencia de estado;
3. mejorar `TextChart` para elevar texto a campo de nodos;
4. añadir pruebas de energía, refinamiento, reentrada y memoria;
5. crear benchmark de conversación.
