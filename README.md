# CTNet Conceptual Babel Runtime

CTNet Conceptual Babel modela conversación como dinámica **conceptual-relacional**.
La ontología base **no** son tokens ni cadenas; la primitiva es el **nodo conceptual-informacional-relacional**.

## Arquitectura (definitiva)

- **ConceptNode**: sección activa finita de un nodo conceptual no acotado.
- **FractionNode**: fracción superficial (texto) que también es elevable como nodo conceptual.
- **RelationOperator**: operador relacional entre nodos.
- **ConceptComplex**: complejo activo de nodos+relaciones dentro del campo Babel implícito.
- **UPBundle**: reciprocidad u/p sobre nodos y vecindarios relacionales.
- **CoherenceTensor**: forma de bajo rango `H = D + L L^T`.
- **ConceptualEnergy**: energía `E_X(S) = <e(S), H_X e(S)>` sobre un complejo `S`.
- **ConceptualBabel**: potencial implícito no acotado por expansión/refinamiento/relación/proyección.
- **TextChart**: carta superficial de lift/project (texto no primario ontológico).
- **ConceptualBabelRuntime**: ciclo `lift → cierre relacional → residual u/p → flujo coherente → proyección → reentrada`.

## Ecuaciones clave

```text
E_X(S) = < e(S), H_X e(S) >
H_X = D_X + L_X L_X^T
```

Aquí `S` es un complejo conceptual-relacional, no una secuencia de tokens.

## Ejecutar tests

```bash
pytest -q
```

## Ejecutar demo CLI

Una sola interacción:

```bash
python demo_chat.py --once "La Biblioteca de Babel no son tokens: son nodos conceptuales relacionales bajo u/p y tensor de coherencia." --trace
```

Modo interactivo:

```bash
python demo_chat.py --trace
```

## Compatibilidad

El paquete mantiene símbolos de uso común en nivel superior (`ConceptualBabelRuntime`, `make_relation`, `demo`).

> Nota: este repositorio no usa ningún contenedor legado `ctnet_conceptual_babel_repo_ready/`.
