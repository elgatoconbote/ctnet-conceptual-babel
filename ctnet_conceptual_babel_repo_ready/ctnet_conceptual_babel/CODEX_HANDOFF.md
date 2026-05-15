# Codex handoff: CTNet Conceptual Babel

Objetivo de continuación:

Construir un runtime conversacional donde el fundamento no sean tokens ni caracteres, sino nodos conceptuales-informacionales-relacionales de complejidad abierta. El texto debe ser una carta de proyección/elevación, no la ontología primaria.

Comandos de verificación:

```bash
cd /mnt/data/ctnet_conceptual_babel
python3 test_conceptual_babel.py
python3 demo_chat.py --once "Haz real CTNet Babel con nodos conceptuales y tensor de coherencia" --trace
```

Invariantes que no deben romperse:

1. No introducir un `vocab` o `alphabet` textual como fundamento del sistema.
2. Mantener `ConceptNode` como primitiva.
3. Mantener `FractionNode` como fracción superficial elevable a nodo.
4. Mantener `E_X(S) = <e(S), H_X e(S)>` como criterio de estabilización.
5. Mantener `H = D + LL^T` sin matriz densa obligatoria.
6. Hacer que el texto sea `TextChart.lift()` y `TextChart.project()`, no generación base.
7. La memoria debe reentrar complejos nodales, no solo strings.

Siguiente desarrollo recomendado:

- Añadir `MathChart`, `CodeChart`, `DiagramChart`.
- Añadir composición categorial de relaciones `R_ik ≈ R_jk ∘ R_ij`.
- Añadir refinamiento adaptativo por tensión local.
- Añadir persistencia completa de complejos, no solo memoria global.
- Añadir interfaz CLI de inspección del grafo conceptual activo.
