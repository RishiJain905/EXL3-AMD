"""Experimental AsterKV grids for normalized, Hadamard-rotated groups of 32.

The default cubic approximates a symmetric Lloyd-Max fit with pinned endpoints
on bounded K/V calibration
samples from public code/documentation prompts. Validation prompts were held
out. This model-specific pilot grid is not a universal quality guarantee.
See docs/KVCache-Research for provenance, attribution and measured limits.
"""

ASTER5_LLOYD_CENTROIDS = (
    -1.0, -0.9068740285851389, -0.8238543967412472, -0.7479173334247744,
    -0.6772382348197568, -0.6107654575853732, -0.5470800725559749, -0.4855662464177768,
    -0.42571228508545816, -0.367226117540165, -0.3097467544052727, -0.2525577412180148,
    -0.19590521731555247, -0.139690273431487, -0.08375087863880441, -0.027901667450546043,
    0.027901667450546043, 0.08375087863880441, 0.139690273431487, 0.19590521731555247,
    0.2525577412180148, 0.3097467544052727, 0.367226117540165, 0.42571228508545816,
    0.4855662464177768, 0.5470800725559749, 0.6107654575853732, 0.6772382348197568,
    0.7479173334247744, 0.8238543967412472, 0.9068740285851389, 1.0,
)

# Cubic fit to the calibration-derived grid, constrained to p(-1)=-1, p(1)=1.
# This removes indexed codebook loads from attention. The original Lloyd
# grid remains an explicit research control. Both are experimental.
ASTER5_CUBIC = (0.8307890996269471, 0.16921090037305286)
ASTER5_CENTROIDS = tuple(
    x * (ASTER5_CUBIC[0] + ASTER5_CUBIC[1] * x * x)
    for x in ((2 * i - 31) / 31 for i in range(32))
)
