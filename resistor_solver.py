#!/usr/bin/env python3

import argparse
import itertools
from math import isclose

def read_resistors(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.append(float(line))
    return vals

def parallel(resistors):
    return 1.0 / sum(1.0/r for r in resistors)

def evaluate_combo(combo):
    results = []

    # All series
    series_val = sum(combo)
    results.append(("series", combo, series_val))

    # All parallel
    if all(r > 0 for r in combo):
        try:
            p = parallel(combo)
            results.append(("parallel", combo, p))
        except ZeroDivisionError:
            pass

    # Mixed: split into two groups: (A series) in parallel with (B series)
    if len(combo) >= 2:
        for i in range(1, len(combo)):
            A = combo[:i]
            B = combo[i:]
            valA = sum(A)
            valB = sum(B)
            try:
                mixed = 1.0 / (1.0/valA + 1.0/valB)
                results.append(("series||series", (A, B), mixed))
            except ZeroDivisionError:
                pass

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=float, help="Target resistance (ohms)")
    parser.add_argument("--file", required=True, help="Resistor list file")
    parser.add_argument("--max", type=int, default=3, help="Max resistor count")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")

    args = parser.parse_args()

    resistors = read_resistors(args.file)
    target = args.target

    best = []

    for n in range(1, args.max + 1):
        for combo in itertools.combinations_with_replacement(resistors, n):
            for kind, structure, value in evaluate_combo(combo):
                error = abs(value - target) / target
                best.append((error, value, kind, structure))

    best.sort(key=lambda x: x[0])

    for error, value, kind, structure in best[:args.top]:
        pct = error * 100
        print(f"{value:.2f} Ω  ({pct:.4f}% error)  [{kind}]  {structure}")

if __name__ == "__main__":
    main()
