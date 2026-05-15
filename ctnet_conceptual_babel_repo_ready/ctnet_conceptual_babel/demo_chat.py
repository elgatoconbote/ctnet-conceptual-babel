#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal terminal chat for the conceptual CTNet-Babel runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctnet_conceptual_babel import ConceptualBabelRuntime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="/mnt/data/ctnet_conceptual_babel/runtime_state.json")
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--beam", type=int, default=7)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--once", default=None, help="Run one message and exit")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    rt = ConceptualBabelRuntime(d=args.d, beam=args.beam, steps=args.steps, state_path=args.state)

    if args.once is not None:
        out = rt.respond(args.once)
        print(out["response"])
        if args.trace:
            print(json.dumps(out["trace"], ensure_ascii=False, indent=2))
        return

    print("CTNet Conceptual Babel runtime. Escribe 'salir' para terminar.")
    while True:
        try:
            msg = input("Tú> ").strip()
        except EOFError:
            break
        if msg.lower() in {"salir", "exit", "quit"}:
            break
        if not msg:
            continue
        out = rt.respond(msg)
        print("CTNet> " + out["response"])
        if args.trace:
            print(json.dumps(out["trace"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
