"""Compare C++ edge determinant signs against exact integer determinants.

Every finite binary32 number is an integer multiple of 2^-149. Decoding the
input bits onto that common integer lattice gives an independent exact oracle;
this test does not reproduce the solver's floating-point expansion arithmetic.
"""

from __future__ import annotations

import random
import struct
import subprocess
import sys


def integer_value(bits: int) -> int:
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7fffff
    if exponent == 255:
        raise ValueError("Nonfinite test input")
    if exponent:
        mantissa = (mantissa | 0x800000) << (exponent - 1)
    return -mantissa if bits >> 31 else mantissa


def exact_sign(bits: list[int]) -> int:
    a, b, c, d, e, f, g, h, i = map(integer_value, bits)
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return (determinant > 0) - (determinant < 0)


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def cases() -> list[list[int]]:
    rng = random.Random(0xB11E_A2)
    result: list[list[int]] = []
    for _ in range(4000):
        # The full finite binary32 exponent range includes subnormals and
        # extreme exponent separation, not just well-conditioned unit inputs.
        row = [(rng.getrandbits(1) << 31) | (rng.randrange(255) << 23)
               | rng.getrandbits(23) for _ in range(9)]
        result.append(row)
        result.append(row[:3] + row[3:6] + row[:3])  # Exact coplanarity.
    for _ in range(2000):
        # Unit-vector-scale inputs with severe triple-product cancellation.
        result.append([0x3f800000 + rng.randrange(-256, 257) for _ in range(9)])
    for shift in (-100, -20, 0, 20, 100):
        # A nonzero integer determinant of exactly one, then exact power-of-two
        # scaling. The entries contain up to 24 significant bits.
        n = 1 << 23
        integer_rows = (n, n - 1, n - 2, n + 1, n, n - 1, n, n - 1, n - 1)
        row = [float_bits(value * 2.0 ** (shift - 23)) for value in integer_rows]
        result.extend((row, row[:3] + row[6:] + row[3:6]))
    # Exact-zero boundaries and smallest/largest finite input triples.
    special = (0, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000)
    for _ in range(1000):
        result.append([rng.choice(special) for _ in range(9)])
    return result


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_bilinear_boundary_exact.py <boundary-test-executable>")
    rows = cases()
    encoded = "".join(" ".join(f"{b:08x}" for b in row) + "\n" for row in rows)
    run = subprocess.run([sys.argv[1], "--batch-predicates"], input=encoded,
                         text=True, capture_output=True, check=True)
    actual = [int(line) for line in run.stdout.splitlines()]
    if len(actual) != len(rows):
        raise AssertionError(f"Expected {len(rows)} results; received {len(actual)}")
    for index, (row, got) in enumerate(zip(rows, actual)):
        expected = exact_sign(row)
        if got != expected:
            raise AssertionError(f"Predicate {index}: {got} != {expected}; bits={row}")
    print(f"Exact binary32 determinant oracle: {len(rows)} integer-reference cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
