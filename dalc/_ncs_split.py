from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp


BUCKET_NAMES = ("hard", "ambiguous", "easy")
BUCKET_STRATEGIES = ("fixed", "quantile", "beta_mix", "beta_mix_guarded")


def assign_ncs_buckets(
    scores: Sequence[float],
    easy_threshold: float = 0.8,
    ambiguous_threshold: float = 0.6,
    strategy: str = "fixed",
) -> List[str]:
    if strategy == "fixed":
        return _assign_by_thresholds(scores, easy_threshold, ambiguous_threshold)
    if strategy == "quantile":
        easy_t, amb_t = _quantile_thresholds(scores)
        return _assign_by_thresholds(scores, easy_t, amb_t)
    if strategy == "beta_mix":
        return _assign_by_beta_mixture(scores)
    if strategy == "beta_mix_guarded":
        return _assign_by_guarded_beta_mixture(scores, easy_threshold, ambiguous_threshold)
    raise ValueError(f"Unknown bucket strategy: {strategy!r}; expected one of {BUCKET_STRATEGIES}")


def _assign_by_thresholds(
    scores: Sequence[float],
    easy_threshold: float,
    ambiguous_threshold: float,
) -> List[str]:
    if easy_threshold <= ambiguous_threshold:
        raise ValueError("easy_threshold must be greater than ambiguous_threshold")
    return [
        "easy" if s >= easy_threshold else "ambiguous" if s >= ambiguous_threshold else "hard"
        for s in scores
    ]


def _quantile_thresholds(scores: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.8, 0.6
    amb_t = float(np.quantile(arr, 1.0 / 3.0))
    easy_t = float(np.quantile(arr, 2.0 / 3.0))
    if easy_t <= amb_t:
        # Degenerate distribution (e.g. mass concentrated at one value).
        easy_t = amb_t + 1e-3
    return easy_t, amb_t


def _assign_by_beta_mixture(
    scores: Sequence[float],
    n_components: int = 3,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> List[str]:
    arr_raw = np.asarray(scores, dtype=np.float64)
    if arr_raw.size < 2 * n_components:
        easy_t, amb_t = _quantile_thresholds(scores)
        return _assign_by_thresholds(scores, easy_t, amb_t)
    arr = np.clip(arr_raw, 1e-4, 1.0 - 1e-4)
    n = arr.size
    K = n_components

    init_quantiles = np.quantile(arr, np.linspace(1.0 / (2 * K), 1.0 - 1.0 / (2 * K), K))
    means = init_quantiles.copy()
    overall_var = max(float(arr.var()), 1e-3)
    vars_ = np.full(K, overall_var / K)
    weights = np.full(K, 1.0 / K)
    alphas, betas = _moments_to_beta(means, vars_)

    log_x = np.log(arr)
    log_1mx = np.log(1.0 - arr)
    prev_ll = -np.inf
    log_resp = np.zeros((n, K))
    for _ in range(max_iter):
        log_dens = (
            (alphas - 1.0)[None, :] * log_x[:, None]
            + (betas - 1.0)[None, :] * log_1mx[:, None]
            - (gammaln(alphas) + gammaln(betas) - gammaln(alphas + betas))[None, :]
        )
        log_weighted = log_dens + np.log(weights)[None, :]
        log_norm = logsumexp(log_weighted, axis=1, keepdims=True)
        log_resp = log_weighted - log_norm
        resp = np.exp(log_resp)

        Nk = resp.sum(axis=0).clip(min=1e-6)
        weights = Nk / n
        means = (resp * arr[:, None]).sum(axis=0) / Nk
        diffs2 = (arr[:, None] - means[None, :]) ** 2
        vars_ = np.clip((resp * diffs2).sum(axis=0) / Nk, 1e-6, None)
        alphas, betas = _moments_to_beta(means, vars_)

        ll = float(log_norm.sum())
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Argmax-responsibility assignment. A component with low fitted weight
    # may end up with zero members; that is a meaningful signal (the data
    # only supports 2 modes) rather than a bug, and downstream code handles
    # empty buckets gracefully.
    component_argmax = log_resp.argmax(axis=1)
    order = np.argsort(means)
    bucket_idx = np.empty(n, dtype=np.int64)
    for new_idx, k in enumerate(order):
        bucket_idx[component_argmax == k] = new_idx
    return [BUCKET_NAMES[i] for i in bucket_idx]


def _assign_by_guarded_beta_mixture(
    scores: Sequence[float],
    easy_threshold: float,
    ambiguous_threshold: float,
    min_fraction: float = 0.05,
    min_count: int = 1,
) -> List[str]:
    """Use Beta-mixture buckets only when all downstream roles are identifiable.

    NCS is discrete for fixed top-k neighborhoods (e.g. top_k=5 gives multiples
    of 0.2), so an unconstrained continuous mixture often collapses one role.
    DaLC uses easy/ambiguous/hard as structural roles, not just distribution
    clusters; a missing role disables teacher transfer or hard/easy handling.
    """
    beta_buckets = _assign_by_beta_mixture(scores)
    if _has_min_bucket_support(beta_buckets, min_fraction=min_fraction, min_count=min_count):
        return beta_buckets

    fixed_buckets = _assign_by_thresholds(scores, easy_threshold, ambiguous_threshold)
    if _has_min_bucket_support(fixed_buckets, min_fraction=min_fraction, min_count=min_count):
        return fixed_buckets

    easy_t, amb_t = _quantile_thresholds(scores)
    return _assign_by_thresholds(scores, easy_t, amb_t)


def _has_min_bucket_support(
    buckets: Sequence[str],
    min_fraction: float,
    min_count: int,
) -> bool:
    n = len(buckets)
    if n == 0:
        return False
    required = max(int(min_count), int(np.ceil(float(min_fraction) * n)))
    counts = {name: 0 for name in BUCKET_NAMES}
    for bucket in buckets:
        counts[bucket] = counts.get(bucket, 0) + 1
    return all(counts[name] >= required for name in BUCKET_NAMES)


def _moments_to_beta(means: np.ndarray, vars_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.clip(means, 1e-3, 1.0 - 1e-3)
    vars_ = np.clip(vars_, 1e-6, None)
    common = np.clip(means * (1.0 - means) / vars_ - 1.0, 0.5, None)
    return means * common, (1.0 - means) * common
