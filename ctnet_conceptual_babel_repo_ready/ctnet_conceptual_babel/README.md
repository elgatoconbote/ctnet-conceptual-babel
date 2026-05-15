# CTNet Conceptual Babel Runtime

Este proyecto convierte la formulación correcta en código:

- La primitiva no es carácter ni token.
- La primitiva es el nodo conceptual-informacional-relacional.
- Los caracteres, palabras y texto son fracciones/proyecciones de nodos, y también pueden elevarse como nodos bajo una carta.
- La Biblioteca de Babel se representa como potencial conceptual-relacional implícito, no como lista de cadenas.
- u/p + tensor de coherencia actúan sobre complejos nodales-relacionales mediante una energía:

```text
E_X(S) = < e(S), H_X e(S) >
H_X = D_X + L_X L_X^T
```

Donde `S` es un complejo conceptual, `e(S)` contiene errores u/p, relacionales, composicionales, de proyección y cierre, y `H_X` pesa qué tensiones importan.

## Ejecutar demo

```bash
cd /mnt/data/ctnet_conceptual_babel
python3 ctnet_conceptual_babel.py
```

## Chat terminal

```bash
cd /mnt/data/ctnet_conceptual_babel
python3 demo_chat.py --once "La Biblioteca de Babel no son tokens: son nodos conceptuales relacionales" --trace
```

Modo interactivo:

```bash
cd /mnt/data/ctnet_conceptual_babel
python3 demo_chat.py --trace
```

## Tests

```bash
cd /mnt/data/ctnet_conceptual_babel
python3 test_conceptual_babel.py
```

## Estructura

- `ConceptNode`: nodo conceptual con potencial abierto.
- `FractionNode`: fracción superficial que también puede elevarse como nodo.
- `RelationOperator`: relación como operador entre nodos.
- `ConceptComplex`: sección activa finita del potencial infinito.
- `TextChart`: carta de elevación/proyección textual.
- `UPBundle`: reciprocidad actuación/inercia sobre nodos y relaciones.
- `CoherenceTensor`: tensor `D + LLᵀ` sobre el campo de errores.
- `ConceptualBabel`: operadores implícitos de expansión/refinamiento/relación/proyección.
- `CoherenceFlow`: aproximación finita de la distribución sobre el potencial infinito.
- `ConceptualBabelRuntime`: bucle conversacional lift → flow → project → reentry.
