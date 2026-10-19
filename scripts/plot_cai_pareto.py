#!/usr/bin/env python3
"""Draw the comparison-addition tradeoff of the shared-hub family.

Cai, arXiv:2609.04825, Theorem 6.1: on the graph `H_k` — a hub `x`, spokes
`v_1..v_k`, arcs `s→x`, `s→v_i`, `v_i→x`, `x→v_i`, so `m = 3k + 1` and
`n = k + 2` — a deterministic comparison-addition program that makes at most
`k + r` additions makes `Θ(k + λ(k, r))` comparisons in the worst case, where

    λ(k, r) = log2( k! / ( r! · (r + 1)^(k − r) ) ),      0 ≤ r ≤ k.

`k` additions is the minimum on this topology (Theorem 3.1), and at `r = 0`
the comparison count is `log2(k!)`, which is sorting scale on a graph whose
output is a labelled distance vector and not an order.

What is drawn is the shape of the law across the slack `r`, which the theorem
states but does not picture: `λ(k, r) / k` against `r / k`, one line per `k`.
The question it answers is where on the slack axis the sorting scale lives —
whether a program that is allowed a constant factor more additions than the
minimum is still forced to sort, or whether the entropy term is confined to
a vanishing fraction of the axis. The right edge is the addition count of a
program that relaxes every arc once, `3k + 1`, which every shortest-path
algorithm in the target model makes at least.

Prints the table the plot is drawn from, so a recorded run carries the
numbers as well as the picture.

Usage: plot_cai_pareto.py [<out.png>]
"""

from __future__ import annotations

import sys
from math import lgamma, log, log2
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def lam(k: int, r: int) -> float:
    """log2( k! / (r! (r+1)^(k-r)) ), through log-gamma so k can be large."""
    return (lgamma(k + 1) - lgamma(r + 1) - (k - r) * log(r + 1)) / log(2)


def lower_per_spoke(k: int, r: int) -> float:
    """The Theorem 6.1 lower bound max{k, λ(k, r)}, per spoke."""
    return max(k, lam(k, r)) / k


KS = [2**10, 2**14, 2**18, 2**22, 2**26]
SLACKS = [(1, "1"), (2, "2"), (4, "4"), (16, "16"), (256, "256")]


def table() -> str:
    lines = []
    lines.append("lambda(k, r) / k  --  entropy term of Cai Thm 6.1 per spoke, by addition slack r")
    lines.append("")
    head = f"{'k':>10} {'log2 k':>7} | " + " ".join(f"{'r=k/'+s:>9}" for _, s in SLACKS)
    head += f" | {'r=k/log2k':>10} {'r=1':>8} {'r=0':>8}"
    lines.append(head)
    for k in KS:
        cells = [f"{lam(k, k // d) / k:9.3f}" for d, _ in SLACKS]
        rl = lam(k, int(k / log2(k))) / k
        lines.append(
            f"{k:>10} {log2(k):7.0f} | "
            + " ".join(cells)
            + f" | {rl:10.3f} {lam(k, 1) / k:8.3f} {lam(k, 0) / k:8.3f}"
        )
    lines.append("")
    lines.append("r = k/d rows converge in k to a constant that depends on d alone;")
    lines.append("r = k/log2 k grows like log2 log2 k; r = 0 and r = 1 grow like log2 k.")
    lines.append("")
    lines.append("comparison lower bound per spoke, max{k, lambda} / k, at the addition counts of")
    lines.append("named programs on H_k (m = 3k + 1 arcs):")
    for k in [2**14, 2**22]:
        rows = [
            ("addition minimum, r = 0", 0),
            ("r = 1", 1),
            ("r = k / log2 k", int(k / log2(k))),
            ("r = k / 16", k // 16),
            ("r = k / 2", k // 2),
            ("r = k  (2k additions)", k),
        ]
        for name, r in rows:
            lines.append(f"  k = {k:>8}  {name:<28} {lower_per_spoke(k, r):8.3f}")
        lines.append(
            f"  k = {k:>8}  {'one relaxation per arc, 3k+1':<28} {'1.000':>8}"
            "   (r > k: lambda is 0 by definition, bound is the linear k)"
        )
    return "\n".join(lines)


def draw(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in KS:
        rs = np.unique(np.concatenate([[0, 1, 2, 3], np.geomspace(4, k, 400).astype(int)]))
        xs = (rs + 1) / k
        ys = [lower_per_spoke(k, int(r)) for r in rs]
        ax.plot(xs, ys, label=f"k = 2^{int(log2(k))}")
    ax.axvline(1.0, color="0.5", linestyle=":", linewidth=1)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, " r = k: 2k additions", va="top", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("addition slack (r + 1) / k   [k additions is the minimum on H_k]")
    ax.set_ylabel("comparison lower bound per spoke,  max{k, λ(k,r)} / k")
    ax.set_title("Cai 2026 Thm 6.1 on the shared-hub family: where the sorting scale lives")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cai_pareto.png")
    print(table())
    draw(out)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
