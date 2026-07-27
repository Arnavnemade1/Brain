"""Riemannian tangent-space features.

Covariance matrices do not live in a flat space. The set of symmetric
positive-definite matrices is a curved manifold, and the straight-line
distance between two of them — which is what a linear model on raw covariance
entries implicitly uses — is the wrong distance. Two covariances can be close
in Euclidean terms while describing quite different spatial patterns.

The fix is standard in motor-imagery decoding and it is the one thing that
reliably beats band power here: whiten every covariance by a reference point,
take the matrix logarithm to flatten the manifold around it, and read off the
result as an ordinary vector. Distances in that tangent space approximate
distances on the manifold, so a linear classifier finally sees the geometry
the data actually has.

Band power is the diagonal of a covariance. This uses the whole matrix,
including how pairs of electrodes co-vary — which is precisely where the
left-versus-right contrast lives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("mindscape.motor.riemann")

#: Shrinkage applied to every covariance before it is used. One second of
#: band-passed data does not estimate a 64x64 covariance well, and the matrix
#: logarithm is undefined the moment an eigenvalue reaches zero.
SHRINKAGE = 0.05


def regularise(covariances: np.ndarray, amount: float = SHRINKAGE) -> np.ndarray:
    """Pull covariances toward a scaled identity, keeping them invertible."""
    n = covariances.shape[-1]
    trace = np.trace(covariances, axis1=-2, axis2=-1)[..., None, None] / n
    eye = np.eye(n, dtype=covariances.dtype)
    return (1.0 - amount) * covariances + amount * trace * eye


def _sym_eig(matrices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition for a stack of symmetric matrices."""
    # Symmetrise first: accumulated floating error is enough to make `eigh`
    # and `eig` disagree, and everything downstream assumes symmetry.
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    return np.linalg.eigh(symmetric)


def _power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    """Matrix power of one symmetric positive-definite matrix."""
    values, vectors = _sym_eig(matrix)
    values = np.clip(values, 1e-12, None)
    return (vectors * values**exponent) @ vectors.T


def log_map(covariances: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Matrix logarithm of each covariance, whitened by ``reference``."""
    whitener = _power(reference, -0.5)
    whitened = whitener @ covariances @ whitener

    values, vectors = _sym_eig(whitened)
    values = np.clip(values, 1e-12, None)
    logged = np.einsum(
        "...ij,...j,...kj->...ik", vectors, np.log(values), vectors, optimize=True
    )
    return logged


def _upper_triangle(matrices: np.ndarray) -> np.ndarray:
    """Vectorise symmetric matrices, weighting off-diagonal terms.

    The sqrt(2) keeps the Euclidean norm of the vector equal to the Frobenius
    norm of the matrix, so distances in the vector space mean what they mean
    on the manifold.
    """
    n = matrices.shape[-1]
    rows, cols = np.triu_indices(n)
    weights = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return matrices[..., rows, cols] * weights


@dataclass
class TangentSpace:
    """Reference points, one per band, learned from training covariances."""

    references: List[np.ndarray]

    @classmethod
    def fit(
        cls,
        covariances: np.ndarray,
        iterations: int = 3,
        sample: int = 400,
        seed: int = 0,
    ) -> "TangentSpace":
        """Learn a reference covariance per band.

        The reference is the Riemannian geometric mean, approached by
        repeatedly averaging in the tangent space and stepping back onto the
        manifold. The Euclidean mean is a common shortcut, but it sits off the
        manifold in a way that biases every projection taken around it.
        """
        # The reference is a single central point; estimating it from every
        # training window costs an eigendecomposition per window per iteration
        # and buys nothing. A few hundred sampled windows fix it just as well.
        rng = np.random.default_rng(seed)
        if covariances.shape[0] > sample:
            picks = rng.choice(covariances.shape[0], sample, replace=False)
        else:
            picks = np.arange(covariances.shape[0])

        references = []
        for band in range(covariances.shape[1]):
            block = regularise(covariances[picks, band].astype(np.float64))
            reference = block.mean(axis=0)

            for _ in range(iterations):
                tangent = log_map(block, reference).mean(axis=0)
                # Step: reference^(1/2) expm(tangent) reference^(1/2).
                root = _power(reference, 0.5)
                values, vectors = _sym_eig(tangent)
                exp_tangent = (vectors * np.exp(values)) @ vectors.T
                reference = root @ exp_tangent @ root
                reference = 0.5 * (reference + reference.T)

            references.append(reference)

        return cls(references=references)

    def transform(self, covariances: np.ndarray) -> np.ndarray:
        """Tangent-space vectors for every window, bands concatenated."""
        parts = []
        for band, reference in enumerate(self.references):
            block = regularise(covariances[:, band].astype(np.float64))
            parts.append(_upper_triangle(log_map(block, reference)))
        return np.hstack(parts)

    @property
    def n_features(self) -> int:
        n = self.references[0].shape[0]
        return len(self.references) * n * (n + 1) // 2
