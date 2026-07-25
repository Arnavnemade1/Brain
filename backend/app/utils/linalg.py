"""Linear algebra helpers that work around a NumPy/Accelerate defect.

On macOS builds of NumPy 2.x linked against Apple's Accelerate framework,
every BLAS-backed ``matmul`` leaves the floating-point status register dirty.
NumPy reads those flags afterwards and raises ``divide by zero``, ``overflow``,
``underflow`` and ``invalid value`` RuntimeWarnings — on *every* call, for
completely ordinary inputs. The computed result is bit-for-bit correct
(verified against ``einsum``), so the warnings are pure noise, but they bury
genuine numerical problems in log spam.

``safe_matmul`` suppresses the spurious flags and then checks the result is
actually finite, so a real numerical failure still surfaces loudly.
"""

from __future__ import annotations

import numpy as np


class NonFiniteResult(ValueError):
    """Raised when a matrix product genuinely produces NaN or infinity."""


def safe_matmul(a: np.ndarray, b: np.ndarray, *, context: str = "matmul") -> np.ndarray:
    """``a @ b`` without the platform's spurious FP warnings.

    Raises :class:`NonFiniteResult` if the product is not finite — that would
    be a real error, not a status-flag artifact.
    """
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = np.matmul(a, b)

    if not np.all(np.isfinite(result)):
        raise NonFiniteResult(
            "{0} produced non-finite values (shapes {1} @ {2})".format(
                context, a.shape, b.shape
            )
        )
    return result
