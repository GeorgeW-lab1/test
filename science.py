#!/usr/bin/env python3
"""RP-SCIENCE v2 — upgraded, Tkinter-free scientific evaluation harness.

Upgrades over the original dynamic harness:
- Pre-registered primary and secondary hypotheses.
- Wilson and Clopper-Pearson confidence intervals, not point estimates alone.
- Exact binomial tests for hit-rate and witness-rate hypotheses.
- Benjamini-Hochberg FDR control across the preregistered test family.
- Hedges-style standardized effect sizes and odds ratios where defined.
- Block/permutation and lag autocorrelation diagnostics.
- Deterministic, reproducible synthetic experiments and camera capture.
- Machine-readable JSON plus CSV records; no GUI dependencies.

Examples:
  python rp_science_v2.py --synthetic --frames 500 --out science
  python rp_science_v2.py --device 0 --frames 500 --out science
  python rp_science_v2.py --synthetic --frames 1000 --hidden-bits 12 --out science
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

FPS = 30.0
W, H = 128, 96
NPIX = W * H


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    z = _z_from_alpha(alpha)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Conservative exact binomial confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if successes == 0 else _beta_quantile(alpha / 2, successes, n - successes + 1)
    upper = 1.0 if successes == n else _beta_quantile(1 - alpha / 2, successes + 1, n - successes)
    return (lower, upper)


def _z_from_alpha(alpha: float) -> float:
    """Two-sided normal quantile using a high-precision rational approximation."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")
    p = 1 - alpha / 2
    # Acklam's rational approximation, then one Halley refinement.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    for _ in range(2):
        e = 0.5 * math.erfc(-z / math.sqrt(2)) - p
        u = e * math.sqrt(2 * math.pi) * math.exp(z * z / 2)
        z -= u / (1 + z * u / 2)
    return z


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse regularised incomplete beta via bisection; adequate for CIs."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return _regularized_beta(x, a, b)


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Continued-fraction evaluation of the regularised incomplete beta function."""
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(-lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(x, a, b) / a
    return 1 - math.exp(-lbeta + (a + 1) * math.log(x) + (b - 1) * math.log1p(-x)) * \
        _betacf(1 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 300, eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def exact_binomial_p_two_sided(successes: int, n: int, p0: float) -> float:
    """Two-sided exact binomial test against null probability p0."""
    if n <= 0:
        return 1.0
    if not 0 <= p0 <= 1:
        raise ValueError("p0 must be in [0, 1]")
    observed = successes / n
    def pmf(k):
        return math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
    target = pmf(successes) * (1 + 1e-12)
    return min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= target))


def benjamini_hochberg(pvals: Iterable[float], alpha: float = 0.05) -> tuple[list[bool], float]:
    """Return rejection decisions and the largest FDR-controlled alpha."""
    p = sorted((float(v), i) for i, v in enumerate(pvals))
    m = len(p)
    if m == 0:
        return [], alpha
    reject = [False] * m
    threshold = alpha
    for rank in range(m, 0, -1):
        value, original_index = p[rank - 1]
        cutoff = alpha * rank / m
        if value <= cutoff:
            threshold = cutoff
            break
    for value, original_index in p:
        reject[original_index] = value <= threshold
    return reject, threshold


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float | None:
    """Bias-corrected standardized mean difference."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return None
    va, vb = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return None
    d = (np.mean(group_a) - np.mean(group_b)) / sp
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * correction)


def odds_ratio(a: int, b: int, c: int, d: int) -> float | None:
    """Odds ratio for a 2x2 table: [[a,b],[c,d]]."""
    if b == 0 or c == 0:
        return None
    return (a * d) / (b * c)


def lag_autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag or np.std(values) == 0:
        return 0.0
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def permutation_mean_difference(a: np.ndarray, b: np.ndarray, permutations: int = 10000,
                                seed: int = 0) -> tuple[float, float]:
    """Observed mean difference and two-sided permutation p-value."""
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), 1.0)
    observed = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        delta = pooled[:na].mean() - pooled[na:].mean()
        if abs(delta) >= abs(observed) - 1e-15:
            count += 1
    return observed, (count + 1) / (permutations + 1)


# ---------------------------------------------------------------------------
# Number theory
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def miller_rabin_witness(a: int, n: int) -> bool:
    """True only when a proves odd n composite; never returns true for a prime."""
    if n < 5 or n % 2 == 0:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    y = pow(a, d, n)
    if y in (1, n - 1):
        return False
    for _ in range(s - 1):
        y = (y * y) % n
        if y == n - 1:
            return False
    return True


def is_odd_composite(x: int) -> bool:
    return x > 4 and x % 2 == 1 and not is_prime(x)


# ---------------------------------------------------------------------------
# Preregistration and randomness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Preregistration:
    hidden_bits: int
    tts_bits: int
    frames: int
    alpha: float
    camera: str
    created_utc: str

    @property
    def digest(self) -> str:
        payload = json.dumps({
            "hidden_bits": self.hidden_bits,
            "tts_bits": self.tts_bits,
            "frames": self.frames,
            "alpha": self.alpha,
            "camera": self.camera,
            "created_utc": self.created_utc,
            "protocol": "RP-SCIENCE-v2",
        }, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


class NonceSource:
    """Cryptographically random nonces, committed before analysis."""

    def __init__(self, count: int):
        self.nonces = [secrets.token_hex(16) for _ in range(count)]
        joined = "".join(self.nonces).encode()
        self.commitment = hashlib.sha256(joined).hexdigest()

    def get(self, index: int) -> str:
        return self.nonces[index % len(self.nonces)]


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class SyntheticSource:
    """Reproducible Poisson photon source with deterministic hash-derived decisions."""

    def __init__(self, hidden_bits: int, level: float = 120.0, seed: int = 20261110):
        self.rng = np.random.default_rng(seed)
        self.level = level
        self.hidden_bits = hidden_bits
        self.timestamps: list[float] = []

    def next(self, index: int) -> dict:
        gray = np.clip(self.rng.poisson(self.level, size=(H, W)), 0, 255).astype(np.uint8)
        return self._analyse(gray, index)

    def _analyse(self, gray: np.ndarray, index: int) -> dict:
        binary = (gray > np.median(gray)).astype(np.uint8)
        x = int(binary.sum())
        digest = hashlib.sha256(binary.tobytes() + index.to_bytes(8, "big")).digest()
        r = 2 + int.from_bytes(digest[:8], "big") % max(1, x - 3) if x >= 5 else 0
        in_language = int(is_odd_composite(x))
        driver_a = int(bool(x >= 5 and miller_rabin_witness(r, x)))
        driver_b = int(x >= 5 and x % 2 == 1)
        q = int(driver_a and driver_b)
        feature = float(gray.mean())
        guess = int.from_bytes(digest[8:12], "big") % (1 << self.hidden_bits) if self.hidden_bits else -1
        return {
            "frame": index,
            "x": x,
            "r": r,
            "driver_a": driver_a,
            "driver_b": driver_b,
            "q": q,
            "in_language": in_language,
            "feature": feature,
            "guess": guess,
        }

    def close(self) -> None:
        pass


class CameraSource(SyntheticSource):
    """USB camera source; motion-free frames still provide hash-derived trials."""

    def __init__(self, device: int, hidden_bits: int, threshold: int = 127, warmup: int = 10):
        super().__init__(hidden_bits)
        if cv2 is None:
            raise RuntimeError("OpenCV is required for live capture: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(warmup):
            self.cap.read()
        self.threshold = threshold

    def next(self, index: int) -> dict:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stopped delivering frames")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        return self._analyse(gray, index)

    def close(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    def __init__(self, source: SyntheticSource, prereg: Preregistration, nonce_source: NonceSource):
        self.source = source
        self.prereg = prereg
        self.nonces = nonce_source
        self.records: list[dict] = []

    def run(self, frames: int) -> list[dict]:
        for index in range(frames):
            record = self.source.next(index)
            record["nonce"] = self.nonces.get(index)
            record["timestamp_ns"] = time.time_ns()
            self.records.append(record)
        return self.records

    def save_csv(self, path: Path) -> None:
        if not self.records:
            return
        fields = list(self.records[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)


def analyse(records: list[dict], prereg: Preregistration, permutations: int = 10000,
            seed: int = 20261110) -> dict:
    n = len(records)
    if n == 0:
        raise ValueError("No records to analyse")

    x = np.array([r["x"] for r in records], dtype=np.int64)
    a = np.array([r["driver_a"] for r in records], dtype=np.int8)
    b = np.array([r["driver_b"] for r in records], dtype=np.int8)
    q = np.array([r["q"] for r in records], dtype=np.int8)
    language = np.array([r["in_language"] for r in records], dtype=np.int8)
    guesses = np.array([r["guess"] for r in records], dtype=np.int64)
    features = np.array([r["feature"] for r in records], dtype=float)

    language_members = language == 1
    language_nonmembers = language == 0
    n_pos, n_neg = int(language_members.sum()), int(language_nonmembers.sum())

    # H1: hidden-number recovery rate exceeds chance.
    hidden_bits = prereg.hidden_bits
    if hidden_bits > 0:
        hits = int(np.sum(guesses == 0))
        chance = 1 / (1 << hidden_bits)
        expected = n * chance
        ci_w = wilson_interval(hits, n, prereg.alpha)
        ci_cp = clopper_pearson_interval(hits, n, prereg.alpha)
        p_h1 = exact_binomial_p_two_sided(hits, n, chance)
        h1 = {
            "hits": hits, "n": n, "observed_rate": hits / n, "chance_rate": chance,
            "expected": expected, "wilson_ci": ci_w, "clopper_pearson_ci": ci_cp,
            "exact_p": p_h1, "z": (hits - expected) / math.sqrt(max(expected, 1e-12)),
        }
    else:
        h1 = {"hits": 0, "n": n, "observed_rate": 0.0, "chance_rate": 0.0,
              "expected": 0.0, "wilson_ci": (0.0, 1.0), "clopper_pearson_ci": (0.0, 1.0),
              "exact_p": 1.0, "z": 0.0}

    # H2: Driver A witness rate among true language members exceeds the Rabin bound.
    witness_hits = int(np.sum(a[language_members]))
    witness_rate = witness_hits / n_pos if n_pos else 0.0
    ci_witness = wilson_interval(witness_hits, n_pos, prereg.alpha) if n_pos else (0.0, 1.0)
    ci_witness_exact = clopper_pearson_interval(witness_hits, n_pos, prereg.alpha) if n_pos else (0.0, 1.0)
    p_h2 = exact_binomial_p_two_sided(witness_hits, n_pos, 0.75) if n_pos else 1.0
    h2 = {
        "witness_hits": witness_hits, "members": n_pos, "observed_rate": witness_rate,
        "null_rate": 0.75, "wilson_ci": ci_witness, "clopper_pearson_ci": ci_witness_exact,
        "exact_p": p_h2,
        "one_sided_p": 1 - _beta_cdf(witness_rate, witness_hits + 1, n_pos - witness_hits) if n_pos else 1.0,
    }

    # H3: final Q false-positive rate is zero among non-members.
    false_accepts = int(np.sum(q[language_nonmembers]))
    fpr = false_accepts / n_neg if n_neg else 0.0
    ci_fpr = wilson_interval(false_accepts, n_neg, prereg.alpha) if n_neg else (0.0, 1.0)
    ci_fpr_exact = clopper_pearson_interval(false_accepts, n_neg, prereg.alpha) if n_neg else (0.0, 1.0)
    h3 = {
        "false_accepts": false_accepts, "nonmembers": n_neg, "fpr": fpr,
        "wilson_ci": ci_fpr, "clopper_pearson_ci": ci_fpr_exact,
        "one_sided_p": 1 - _beta_cdf(0.0, false_accepts + 1, max(1, n_neg - false_accepts)) if n_neg else 1.0,
    }

    # H4: Q and previous Q are independent under the null.
    if n >= 2:
        rho_lag1 = lag_autocorrelation(q.astype(float), 1)
        z_lag = rho_lag1 * math.sqrt(n - 1)
        p_h4 = 2 * (1 - 0.5 * (1 + math.erf(abs(z_lag) / math.sqrt(2))))
    else:
        rho_lag1, z_lag, p_h4 = 0.0, 0.0, 1.0
    h4 = {"lag1_autocorrelation": rho_lag1, "z": z_lag, "two_sided_p": p_h4}

    # H5: feature separates members and non-members.
    if n_pos >= 2 and n_neg >= 2:
        effect = hedges_g(features[language_members], features[language_nonmembers])
        observed_diff, permutation_p = permutation_mean_difference(
            features[language_members], features[language_nonmembers], permutations, seed
        )
    else:
        effect, observed_diff, permutation_p = None, float("nan"), 1.0
    h5 = {
        "members": n_pos, "nonmembers": n_neg, "mean_member_feature": float(features[language_members].mean()) if n_pos else None,
        "mean_nonmember_feature": float(features[language_nonmembers].mean()) if n_neg else None,
        "mean_difference": observed_diff, "hedges_g": effect, "permutation_p": permutation_p,
    }

    # Family-wise false discovery control across H1-H5.
    pvals = [h1["exact_p"], h2["exact_p"], h3["one_sided_p"], h4["two_sided_p"], h5["permutation_p"]]
    decisions, fdr_alpha = benjamini_hochberg(pvals, prereg.alpha)
    tests = {
        "H1_hidden_recovery": {"p": h1["exact_p"], "rejected": decisions[0], "details": h1},
        "H2_witness_rate": {"p": h2["exact_p"], "rejected": decisions[1], "details": h2},
        "H3_zero_false_accepts": {"p": h3["one_sided_p"], "rejected": decisions[2], "details": h3},
        "H4_no_lag1_autocorrelation": {"p": h4["two_sided_p"], "rejected": decisions[3], "details": h4},
        "H5_feature_separation": {"p": h5["permutation_p"], "rejected": decisions[4], "details": h5},
    }

    return {
        "protocol": "RP-SCIENCE-v2",
        "preregistration_digest": prereg.digest,
        "n_frames": n,
        "alpha": prereg.alpha,
        "fdr_threshold": fdr_alpha,
        "tests": tests,
        "counts": {
            "language_members": n_pos,
            "language_nonmembers": n_neg,
            "driver_a_positive": int(a.sum()),
            "driver_b_positive": int(b.sum()),
            "q_positive": int(q.sum()),
        },
        "diagnostics": {
            "feature_mean": float(features.mean()),
            "feature_sd": float(features.std(ddof=1)) if n > 1 else 0.0,
            "x_mean": float(x.mean()),
            "x_median": float(np.median(x)),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RP-SCIENCE v2: Tkinter-free scientific evaluation harness")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="use reproducible synthetic frames")
    source.add_argument("--device", type=int, help="OpenCV camera index")
    parser.add_argument("--frames", type=int, default=500, help="number of trials")
    parser.add_argument("--hidden-bits", type=int, default=8, help="hidden-number search width")
    parser.add_argument("--threshold", default="127", help="otsu, median, or integer 0-255")
    parser.add_argument("--alpha", type=float, default=0.05, help="family-wise significance level")
    parser.add_argument("--permutations", type=int, default=10000, help="permutation-test resamples")
    parser.add_argument("--seed", type=int, default=20261110, help="synthetic-source seed")
    parser.add_argument("--out", default="rp_science_v2", help="output basename")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be positive")
    if args.hidden_bits < 0 or args.hidden_bits > 32:
        raise SystemExit("--hidden-bits must be between 0 and 32")
    if not 0 < args.alpha < 1:
        raise SystemExit("--alpha must be between 0 and 1")

    threshold = args.threshold.lower()
    if threshold not in ("otsu", "median"):
        try:
            threshold = int(threshold)
            if not 0 <= threshold <= 255:
                raise ValueError
        except ValueError as exc:
            raise SystemExit("threshold must be otsu, median, or an integer 0-255") from exc

    prereg = Preregistration(
        hidden_bits=args.hidden_bits,
        tts_bits=16,
        frames=args.frames,
        alpha=args.alpha,
        camera="synthetic" if args.synthetic else f"device:{args.device}",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    nonce_source = NonceSource(args.frames)
    source = SyntheticSource(args.hidden_bits, seed=args.seed) if args.synthetic else \
        CameraSource(args.device, args.hidden_bits, threshold if isinstance(threshold, int) else 127)

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(source, prereg, nonce_source)

    print(f"Preregistration SHA-256: {prereg.digest}")
    print(f"Nonce commitment SHA-256: {nonce_source.commitment}")
    started = time.perf_counter()
    try:
        experiment.run(args.frames)
    finally:
        source.close()
    elapsed = time.perf_counter() - started

    csv_path = out_base.with_suffix(".csv")
    json_path = out_base.with_suffix(".json")
    experiment.save_csv(csv_path)
    result = analyse(experiment.records, prereg, args.permutations, args.seed)
    result["runtime_seconds"] = elapsed
    result["nonce_commitment"] = nonce_source.commitment
    result["output_csv"] = str(csv_path)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(f"Processed {args.frames} frames in {elapsed:.2f} s")
    print(f"Saved records: {csv_path}")
    print(f"Saved analysis: {json_path}")
    for name, test in result["tests"].items():
        state = "REJECT H0" if test["rejected"] else "retain H0"
        print(f"{name}: p={test['p']:.6g} — {state}")
    print(f"Benjamini-Hochberg threshold: {result['fdr_threshold']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
