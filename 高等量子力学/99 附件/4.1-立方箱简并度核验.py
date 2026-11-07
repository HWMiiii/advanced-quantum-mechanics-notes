"""核验教材习题4.1：可区分、等质量、无相互作用的自旋1/2粒子。

能量单位 epsilon = pi**2 * hbar**2 / (2*m*L**2)。
直接枚举每个带标签粒子的空间量子数和自旋，而非仅再次套用组合公式。
只需 n_x,n_y,n_z in {1,2}：出现 n_i >= 3 的总能量至少为3*N+8，
高于前三能级的上界3*N+6。没有施加玻色或费米交换限制。
"""
from collections import Counter
from itertools import product
from pathlib import Path
import json


def count_lowest_levels(particle_count):
    # 每一个元组均标记一个正交单粒子态；末位为两种自旋投影的标签。
    states = list(product((1, 2), (1, 2), (1, 2), (-1, 1)))
    counts = Counter()
    for configuration in product(states, repeat=particle_count):
        energy = sum(nx * nx + ny * ny + nz * nz for nx, ny, nz, spin in configuration)
        counts[energy] += 1
    levels = sorted(counts.items())[:3]
    expected = [
        (3 * particle_count, 2**particle_count),
        (3 * particle_count + 3, 3 * particle_count * 2**particle_count),
        (3 * particle_count + 6,
         (3 * particle_count + 9 * particle_count * (particle_count - 1) // 2) * 2**particle_count),
    ]
    assert levels == expected, (particle_count, levels, expected)
    return {
        "N": particle_count,
        "enumerated_product_states": len(states) ** particle_count,
        "levels": [{"E_over_epsilon": e, "degeneracy": d} for e, d in levels],
        "agrees_with_analytic_count": levels == expected,
    }


if __name__ == "__main__":
    result = {
        "exercise": "4.1(a)(b)",
        "source": "PDF302 / printed280",
        "energy_unit": "pi^2 hbar^2 / (2 m L^2)",
        "assumptions": ["distinguishable", "equal masses", "noninteracting", "spin 1/2", "hard-wall cubic box"],
        "cutoff_proof": "Any spatial quantum number >=3 gives E/epsilon >=3N+8, above the third level 3N+6.",
        "results": [count_lowest_levels(n) for n in (3, 4)],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
