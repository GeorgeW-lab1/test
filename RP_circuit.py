"""
RP-style optoelectronic decision circuit driven by a USB camera  (single file)

    pip install numpy opencv-python
    python rp_circuit_camera.py

Every camera frame is one trial of the circuit:

  1. Motion detection: the frame is shrunk to 128x96, blurred and differenced
     against the previous frame. Pixels that changed by more than a threshold
     form the motion mask.
  2. Instance x = number of moving pixels. (No motion -> no instance, Q = 0.)
     The language is L = { odd composite numbers }.
  3. Driver A (randomized witness test, Detector A = motion energy):
        the same camera frame feeds both detectors and no sensor noise is
        used. r is derived from the motion pattern itself (the size of the
        change at every moving pixel, hashed with SHA-256), so it is
        unpredictable the way the movement is. A = 1 if r is a Miller-Rabin
        witness that x is composite. A = 0 for every prime, so the error is
        one-sided.
  4. Driver B (deterministic check, Detector B = motion mask):
        c(x) = "x is odd and > 2". It drives capacitor C through R for one
        frame period; the NPN emitter follower gives V(B) = V(C) - 0.7 V and
        B is a valid logic HIGH at >= 3.5 V.
  5. Gates: OR and AND (tied outputs read as AND), 10 uF node (low-pass),
     time code Y (which frames may pass), 74HC74 latches D on the clock edge.
     Here the clock edge is the end of each camera frame.
  6. SELECT = "any Q = 1 in the last k frames" (the repetition step).

Driver analysis (button "Driver analysis"): every instance frame is scored against
the truth (is x really an odd composite?). For Driver A, Driver B and the final Q
it reports TPR/FPR/PPV/NPV/accuracy and mutual information with "x in L", checks
the RP properties the bounds rest on (A never fires outside L, A's witness rate
p >= 1/2, B never misses a member of L), calibrates A against the exact witness
density of every x, and tests the independence the (1-p)^N bound assumes.

Time code Y (proved in time_code_proof_check.py, shown live in the window):
  * Gating reading, D = A.B.Y: x not in L is never accepted for ANY Y (Thm 1).
    For x in L the false-reject probability is exactly (1-p)^N, N = open bins
    in the k-frame window (Thm 2), so Y sets the error exponent, not the sidedness.
  * Y accepts arithmetic: "1000|0011" OR-adds codes (N never drops, disjoint
    codes multiply their errors), "1111&1011" ANDs them (N never rises),
    "1011+0001" adds them as binary numbers: popcount(a+b) = popcount(a) +
    popcount(b) - carries, so each carry loses one open bin (Thm 3).
  * "forces HIGH" read as OR (checkbox) lets a forced bin accept any x, so the
    one-sided guarantee is lost (Thm 4). Use it to watch false accepts appear.

Concurrency: the camera read runs in a worker thread; Driver A and Driver B
process each frame concurrently as asyncio tasks; the flip-flop stage runs
once both are finished. The whole pipeline lives in a background thread with
its own asyncio loop and talks to Tk through a queue.Queue.

Two sequential even states: a hash is mined, and the hidden number is guessed, only on a
frame whose x is even AND whose previous frame's x was also even. An odd x or a no-motion
frame in between breaks the run, so the next even frame only starts a new one.

Higher/lower search (separate from both): a secret of N bits (default 1000) and a
higher/lower oracle. On every armed frame (two sequential evens) one query yields one bit
of the secret, which is POOLED; no equality check is made per query. Only when all N bits
are pooled is the candidate compared with the secret, once. A 1000-bit secret falls in
1000 pooled queries plus 1 final check instead of about 2^1000 blind guesses. A found secret earns
one time-code bin and a new secret is hidden.

Hidden-number test (separate from the bit search): a secret integer is drawn in
0..range-1. A separate clock thread cycles a candidate through that range at a frequency
that you set in the GUI (1 Hz to 2 kHz, default 17 Hz). The thread never sees the
secret. Only when two sequential EVEN states have arrived does the frame loop spot-check the candidate AT
THAT MOMENT; if it equals the secret, one time-code bin is added (same mechanism as a found
secret). After EVERY spot check, hit or miss, a new secret is hidden. Expected hits = checks / range.
"New secret" (and changing the range) draws a fresh secret and zeroes the spot checks,
hits and luck factor. The luck factor is hits / expected hits in percent (100% = average),
shown with P(>= hits by chance), the binomial tail probability.

Factoriser (separate source, own panel): N is factored through the same higher/lower bit search.
The oracle holds the smallest prime factor p of N and answers only "p >= trial?". Each armed frame
pools one bit; one final check runs when the pool is full: 1 < p < N, N % p == 0 and the full prime
factorisation multiplies back to N. A time-code bin is earned ONLY after that verified solve.
Typed N is really factored by a background engine (trial division, Fermat, Pollard rho, p-1, ECM,
quadratic sieve; pip install gmpy2 for speed); the oracle is not ready until the engine finishes.
Random semiprimes keep their factors inside the oracle, which shows the mechanism only. A
higher/lower oracle for a factor cannot exist without already knowing it, so the search does not
make factoring itself cheaper. Camera motion hashes seed the engine's randomness.

Needs a real USB camera and OpenCV: motion comes only from what the camera
sees. There is no simulated source; if the camera can't be opened, it says so.
"""
import asyncio
import hashlib
import json
import math
import multiprocessing as mp
import queue
import secrets
import sys
import threading
import time
import urllib.request
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import ttk, messagebox, filedialog

import numpy as np

try:
    import cv2
except ImportError:                      # reported when you press Start
    cv2 = None

# ------------------------------------------------------------------ constants
SW, SH = 128, 96                  # processing size
PW, PH = 288, 216                 # live preview size
WARMUP = 8                        # frames ignored while auto-exposure settles
VS = 5.0                          # driver B output voltage
V_BE = 0.7                        # NPN turn-on
V_IH = 3.5                        # 74HC input-high at 5 V
RC_B = 0.004                      # s, IF/THEN link RC (must settle within a frame)
TAU_NODE = 0.004                  # s, 10 uF node low-pass
LOGIC_TIME = RC_B * math.log(VS / (VS - V_IH - V_BE))   # 1.83 RC for these values


# ------------------------------------------------------------------ maths
def miller_rabin_witness(a: int, n: int) -> bool:
    """True if a proves n composite. Never True for a prime n."""
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    y = pow(a, d, n)
    if y in (1, n - 1):
        return False
    for _ in range(s - 1):
        y = y * y % n
        if y == n - 1:
            return False
    return True


def is_odd_composite(x: int) -> bool:
    if x < 9 or x % 2 == 0:
        return False
    return any(x % d == 0 for d in range(3, int(x ** 0.5) + 1, 2))


# ------------------------------------------------------------------ big-number engine setup
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)     # allow numbers with more than 4300 digits


# ------------------------------------------------------------------ big-number factoriser
# Ladder: trial division -> perfect power -> Fermat -> Pollard rho (Brent) -> Pollard p-1
#         -> ECM (Montgomery curves, stage 1 + stage 2) -> quadratic sieve.
# Optional speed-up:  pip install gmpy2   (used automatically when present)
try:
    import gmpy2
    _mpz = gmpy2.mpz
    _gcd = gmpy2.gcd
    _isqrt = gmpy2.isqrt
    _invert = gmpy2.invert
    HAVE_GMP = True
except ImportError:                                   # plain Python ints are fine, just slower
    gmpy2 = None
    _mpz = int
    _gcd = math.gcd
    _isqrt = math.isqrt
    _invert = lambda a, n: pow(a, -1, n)
    HAVE_GMP = False


class Cancelled(Exception):
    pass


class EntropyPool:
    """Randomness for the factoriser. Camera motion frames are hashed into the pool when the
    circuit runs; the operating-system generator is always mixed in, so it works with no camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = secrets.token_bytes(32)
        self.frames = 0

    def feed(self, data: bytes):
        with self._lock:
            self._state = hashlib.sha256(self._state + data).digest()
            self.frames += 1

    def rand_bits(self, bits: int) -> int:
        out, ctr = b"", 0
        with self._lock:
            seed = self._state
            self._state = hashlib.sha256(self._state + b"\x01").digest()
        need = (bits + 7) // 8
        while len(out) < need:
            out += hashlib.sha256(seed + secrets.token_bytes(16) + ctr.to_bytes(4, "big")).digest()
            ctr += 1
        return int.from_bytes(out[:need], "big") & ((1 << bits) - 1)

    def randrange(self, lo: int, hi: int) -> int:
        span = hi - lo
        return lo + self.rand_bits(span.bit_length() + 32) % span


def _primes_upto(limit: int):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    sieve[4::2] = False
    for i in range(3, int(limit ** 0.5) + 1, 2):
        if sieve[i]:
            sieve[i * i::2 * i] = False
    return np.nonzero(sieve)[0]


_SMALL_PRIMES = _primes_upto(200_000)


def is_probable_prime(n: int, rounds: int = 24) -> bool:
    n = int(n)
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    if gmpy2 is not None:
        return bool(gmpy2.is_prime(_mpz(n), rounds))
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n.bit_length() <= 1024:
        bases += [2 + secrets.randbelow(n - 3) for _ in range(max(0, rounds - len(bases)))]
    for a in bases:
        a %= n
        if a < 2:
            continue
        y = pow(a, d, n)
        if y in (1, n - 1):
            continue
        for _ in range(s - 1):
            y = y * y % n
            if y == n - 1:
                break
        else:
            return False
    return True


def _iroot(n: int, k: int) -> int:
    """floor(n ** (1/k)) by Newton iteration."""
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + k - 1) // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _perfect_power(n: int):
    """(base, exp) with base ** exp == n and exp > 1, else None."""
    for k in range(2, min(n.bit_length(), 128) + 1):
        r = _iroot(n, k)
        if r < 2:
            break
        if r ** k == n:
            return r, k
    return None


class Factoriser:
    def __init__(self, pool: EntropyPool, cancel, log, progress=None):
        self.pool = pool
        self.cancel = cancel                  # callable -> bool
        self.log = log
        self.progress = progress or (lambda *_: None)

    def check(self):
        if self.cancel():
            raise Cancelled()

    # ------------------------------------------------------------ top level
    def factor(self, n: int):
        """Return sorted list of prime factors (with repetition)."""
        n = int(n)
        if n < 2:
            return [n] if n else [0]
        out = []
        # trial division by small primes
        for p in _SMALL_PRIMES[:20000]:
            p = int(p)
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    out.append(p)
                    n //= p
        self.check()
        if n > 1:
            self.log(f"testing {len(str(n))}-digit N (primality, then factor search)")
            self._split(n, out)
        out.sort()
        return out

    def _split(self, n: int, out: list):
        self.check()
        if n == 1:
            return
        if is_probable_prime(n):
            out.append(n)
            return
        pp = _perfect_power(n)
        if pp:
            b, k = pp
            sub = []
            self._split(b, sub)
            out.extend(sub * k)
            return
        d = self.find_factor(n)
        self._split(d, out)
        self._split(n // d, out)

    # ------------------------------------------------------------ method ladder
    def find_factor(self, n: int) -> int:
        digits = len(str(n))
        self.log(f"composite, {digits} digits: searching for a factor")
        for name, fn in (("Fermat", self.fermat), ("Pollard rho", self.rho),
                         ("Pollard p-1", self.pm1)):
            self.check()
            d = fn(n)
            if d and 1 < d < n:
                self.log(f"  {name} found a factor ({len(str(d))} digits)")
                return int(d)
        # ECM through the t25 level first, then QS if the number is within its reach
        schedule = [(2_000, 25), (11_000, 90), (50_000, 220), (250_000, 450), (1_000_000, 900),
                    (3_000_000, 2300), (11_000_000, 4500)]
        qs_ok = digits <= 85
        for li, (b1, curves) in enumerate(schedule):
            for c in range(curves):
                self.check()
                self.progress(f"ECM B1={b1:,}  curve {c + 1}/{curves}",
                              (c + 1) / curves)
                d = self.ecm_curve(n, b1)
                if d and 1 < d < n:
                    self.log(f"  ECM (B1={b1:,}, curve {c + 1}) found a {len(str(d))}-digit factor")
                    return int(d)
            self.log(f"  no factor at ECM level B1={b1:,} ({curves} curves)")
            if qs_ok and b1 >= 50_000:
                self.log("  switching to the quadratic sieve")
                d = self.qs(n)
                if d and 1 < d < n:
                    return int(d)
                qs_ok = False
        raise RuntimeError("ran out of methods")

    # ------------------------------------------------------------ Fermat
    def fermat(self, n: int, iters: int = 20000):
        a = _isqrt(_mpz(n))
        if a * a < n:
            a += 1
        b2 = a * a - n
        for _ in range(iters):
            b = _isqrt(b2)
            if b * b == b2:
                return int(a - b)
            a += 1
            b2 += 2 * a - 1
        return None

    # ------------------------------------------------------------ Pollard rho, Brent variant
    def rho(self, n: int, max_iter: int = 400_000):
        n = _mpz(n)
        for attempt in range(3):
            y = _mpz(self.pool.randrange(1, int(n)))
            c = _mpz(self.pool.randrange(1, int(n)))
            m = 128
            g = r = q = _mpz(1)
            x = ys = y
            total = 0
            while g == 1 and total < max_iter:
                self.check()
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = _gcd(q, n)
                    k += m
                total += r
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
            if 1 < g < n:
                return int(g)
        return None

    # ------------------------------------------------------------ Pollard p-1
    def pm1(self, n: int, b1: int = 200_000):
        n = _mpz(n)
        a = _mpz(2)
        for p in _SMALL_PRIMES:
            p = int(p)
            if p > b1:
                break
            pe = p
            while pe * p <= b1:
                pe *= p
            a = pow(a, pe, n)
        g = _gcd(a - 1, n)
        return int(g) if 1 < g < n else None

    # ------------------------------------------------------------ ECM
    _stage1_cache = {}
    _prime_cache = {}

    @classmethod
    def _stage1_scalar(cls, b1):
        if b1 not in cls._stage1_cache:
            k = 1
            for p in _primes_upto(b1):
                p = int(p)
                pe = p
                while pe * p <= b1:
                    pe *= p
                k *= pe
            cls._stage1_cache[b1] = k
        return cls._stage1_cache[b1]

    @classmethod
    def _primes_for(cls, b2):
        if b2 not in cls._prime_cache:
            cls._prime_cache.clear()
            cls._prime_cache[b2] = _primes_upto(b2)
        return cls._prime_cache[b2]

    def ecm_curve(self, n: int, b1: int):
        """One Montgomery curve (Suyama parametrisation), stage 1 then stage 2."""
        n = _mpz(n)
        sigma = _mpz(self.pool.randrange(6, 1 << 62))
        u = (sigma * sigma - 5) % n
        v = 4 * sigma % n
        x0 = pow(u, 3, n)
        z0 = pow(v, 3, n)
        denom = 16 * x0 * v % n
        g = _gcd(denom, n)
        if 1 < g < n:
            return int(g)
        if g == n:
            return None
        a24 = pow(v - u, 3, n) * (3 * u + v) % n * _invert(denom, n) % n

        def dbl(x, z):
            s = (x + z) ** 2 % n
            d = (x - z) ** 2 % n
            t = s - d
            return s * d % n, t * (d + a24 * t) % n

        def add(xp, zp, xq, zq, xd, zd):
            uu = (xp - zp) * (xq + zq) % n
            vv = (xp + zp) * (xq - zq) % n
            s = uu + vv
            d = uu - vv
            return zd * (s * s % n) % n, xd * (d * d % n) % n

        def mul(k, x, z):
            """[k](x:z) by the Montgomery ladder."""
            r0 = (x, z)
            r1 = dbl(x, z)
            for bit in bin(k)[3:]:
                if bit == "1":
                    r0 = add(r0[0], r0[1], r1[0], r1[1], x, z)
                    r1 = dbl(r1[0], r1[1])
                else:
                    r1 = add(r0[0], r0[1], r1[0], r1[1], x, z)
                    r0 = dbl(r0[0], r0[1])
            return r0

        # ---- stage 1
        qx, qz = mul(self._stage1_scalar(b1), x0, z0)
        g = _gcd(qz, n)
        if 1 < g < n:
            return int(g)
        if g == n:
            return None

        # ---- stage 2 (baby-step / giant-step over primes in (B1, B2])
        b2 = min(50 * b1, 20_000_000)
        D = 2310
        half = D // 2
        # baby steps [j]Q for 1 <= j <= D/2, kept when gcd(j, D) == 1
        baby = {}
        px, pz = qx, qz                       # [1]Q
        baby[1] = (px, pz)
        p2x, p2z = dbl(qx, qz)                # [2]Q
        prev = (px, pz)
        cur = (p2x, p2z)
        if math.gcd(2, D) == 1:
            baby[2] = cur
        for j in range(3, half + 1):
            nxt = add(cur[0], cur[1], qx, qz, prev[0], prev[1])
            prev, cur = cur, nxt
            if math.gcd(j, D) == 1:
                baby[j] = cur
        # giant steps [mD]Q, each new one = previous + [D]Q with the one before as difference
        dx, dz = mul(D, qx, qz)
        m = max(2, b1 // D)
        r_prev = mul((m - 1) * D, qx, qz)
        rx, rz = mul(m * D, qx, qz)
        primes = self._primes_for(b2)
        pi = int(np.searchsorted(primes, b1, side="right"))
        np_ = len(primes)
        acc = _mpz(1)
        while pi < np_:
            self.check()
            c = m * D
            while pi < np_ and primes[pi] <= c + half:
                p = int(primes[pi])
                j = abs(p - c)
                if p > 11 and j in baby:
                    bx, bz = baby[j]
                    acc = acc * ((rx * bz - bx * rz) % n) % n
                pi += 1
            nx, nz = add(rx, rz, dx, dz, r_prev[0], r_prev[1])
            r_prev, (rx, rz) = (rx, rz), (nx, nz)
            m += 1
        g = _gcd(acc, n)
        if 1 < g < n:
            return int(g)
        return None

    # ------------------------------------------------------------ quadratic sieve
    def qs(self, n: int):
        n = int(n)
        digits = len(str(n))
        ln_n = math.log(n)
        bound = int(math.exp(0.5 * math.sqrt(ln_n * math.log(ln_n))))
        bound = max(bound, 300)
        # factor base: primes p with n a quadratic residue mod p
        fb, roots = [], []
        for p in _primes_upto(bound):
            p = int(p)
            if p == 2:
                fb.append(2)
                roots.append(1)
                continue
            if pow(n % p, (p - 1) // 2, p) == 1:
                fb.append(p)
                roots.append(_tonelli(n % p, p))
        nfb = len(fb)
        need = nfb + 1 + 24                  # +1 for the sign column
        self.log(f"  QS: {digits} digits, factor base {nfb} primes (bound {bound:,}), "
                 f"need {need} relations")
        s = math.isqrt(n) + 1
        c0 = s * s - n                       # Q(x) = x*(x + 2s) + c0
        fb_np = np.array(fb, dtype=np.int64)
        roots_np = np.array(roots, dtype=np.int64)
        s_mod = np.array([s % p for p in fb], dtype=np.int64)
        r1 = np.array([(r - s) % p for r, p in zip(roots, fb)], dtype=np.int64)       # x = r - s
        r2 = np.array([(-r - s) % p for r, p in zip(roots, fb)], dtype=np.int64)
        logp = np.log2(fb_np.astype(np.float64)).astype(np.float32)
        sieve_from = 6                                    # skip the tiniest primes while sieving
        BS = 1 << 16 if digits < 45 else 1 << 17
        slack = 1.4 * math.log2(fb[-1]) + 6
        piv = {}                       # high bit -> (vector, history)
        rels = []                      # (s + x, {prime_index: exponent}, sign)
        block = 0
        found = None
        s_f = float(s)
        while found is None:
            for sign_blk in (1, -1):
                self.check()
                start = block * BS if sign_blk == 1 else -(block + 1) * BS
                xs = np.arange(start, start + BS, dtype=np.int64)
                xf = xs.astype(np.float64)
                qf = np.abs(xf * (xf + 2 * s_f) + float(c0))
                qf[qf < 1] = 1
                target = np.log2(qf).astype(np.float32) - slack
                acc = np.zeros(BS, dtype=np.float32)
                for i in range(sieve_from, nfb):
                    p = fb[i]
                    lg = logp[i]
                    for r in ((int(r1[i]), int(r2[i])) if r1[i] != r2[i] else (int(r1[i]),)):
                        off = (r - start) % p
                        acc[off::p] += lg
                cand = np.nonzero(acc >= target)[0]
                for ci in cand:
                    x = int(start + ci)
                    val = x * (x + 2 * s) + c0
                    sign = 1 if val < 0 else 0
                    v = abs(val)
                    if v == 0:
                        continue
                    # which FB primes divide this value: x + s == +-root (mod p)
                    xm = (x + s_mod) % fb_np
                    hit = np.nonzero((xm == roots_np) | (xm == (fb_np - roots_np) % fb_np))[0]
                    expo = {}
                    for i in hit:
                        p = fb[int(i)]
                        e = 0
                        while v % p == 0:
                            v //= p
                            e += 1
                        if e:
                            expo[int(i)] = e
                    if v != 1:
                        continue
                    vec = sign
                    for i, e in expo.items():
                        if e & 1:
                            vec |= 1 << (i + 1)
                    rels.append((x + s, expo, sign))
                    idx = len(rels) - 1
                    hist = 1 << idx
                    while vec:
                        hb = vec.bit_length() - 1
                        if hb in piv:
                            pv, ph = piv[hb]
                            vec ^= pv
                            hist ^= ph
                        else:
                            piv[hb] = (vec, hist)
                            break
                    if vec == 0:
                        d = self._qs_try(n, rels, hist, fb)
                        if d:
                            return d
                self.progress(f"QS  {len(rels)}/{need} relations", min(1.0, len(rels) / need))
            block += 1
        return None

    def _qs_try(self, n, rels, hist, fb):
        x = 1
        tot = {}
        sign_count = 0
        i = 0
        while hist:
            if hist & 1:
                xv, expo, sg = rels[i]
                x = x * xv % n
                sign_count += sg
                for k, e in expo.items():
                    tot[k] = tot.get(k, 0) + e
            hist >>= 1
            i += 1
        y = 1
        for k, e in tot.items():
            y = y * pow(fb[k], e // 2, n) % n
        g = math.gcd(x - y, n)
        return g if 1 < g < n else None


def _tonelli(a: int, p: int) -> int:
    """Square root of a mod an odd prime p (a is a quadratic residue)."""
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


# ------------------------------------------------------------------ hidden-number guesser
HIDDEN_RANGE_DEFAULT = 100
HIDDEN_RANGE_MAX = 1_000_000
GUESS_HZ_MIN, GUESS_HZ_MAX = 20.0, 2000.0     # the clock wanders between these
GUESS_RETUNE_S = 0.3                          # how often the clock re-reads the frequency setting
GUESS_HZ_DEFAULT = 17.0                       # starting clock frequency (adjustable in the GUI)
GUESS_HZ_LIMITS = (1.0, 2000.0)               # allowed range for the frequency box


def prob_at_least(hits: int, n: int, p: float) -> float:
    """P(X >= hits) for X ~ Binomial(n, p): the chance of being at least this lucky."""
    if hits <= 0:
        return 1.0
    if hits > n:
        return 0.0
    s = 0.0
    for i in range(hits):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                      + i * math.log(p) + (n - i) * math.log(1 - p))
    return min(1.0, max(0.0, 1.0 - s))


class HiddenNumber:
    """A secret integer in [0, size) and a separate thread that clock-cycles a candidate
    through the same range at a randomly varying frequency.

    The thread only counts. It never compares against the secret. The comparison happens
    in spot_check(), which the frame loop calls when an even state arrives: it reads the
    candidate AT THAT INSTANT (no scanning, no waiting) and reports whether it equals the
    secret. So a hit is pure timing luck, with probability about 1/size per check."""

    def __init__(self, size=HIDDEN_RANGE_DEFAULT, rng=None):
        self._rng = rng or np.random.default_rng()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.size = 1
        self.candidate = 0
        self.hz = GUESS_HZ_DEFAULT
        self.target_hz = GUESS_HZ_DEFAULT
        self._wake = threading.Event()      # interrupts the clock's wait on a change
        self.ticks = 0
        self.checks = 0
        self.hits = 0
        self.rounds = 0                     # how many secrets have been guessed so far
        self.last_check = None              # (candidate, hit)
        self.last_found = None              # (number, checks it took) of the latest hit
        self._secret = 0
        self.revealed = False
        self.set_range(size)
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ---- control
    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def set_hz(self, hz: float):
        """Change the clock frequency (takes effect immediately)."""
        lo, hi = GUESS_HZ_LIMITS
        hz = max(lo, min(hi, float(hz)))
        with self._lock:
            self.target_hz = hz
        self._wake.set()

    def _reset_counts_locked(self):
        """Zero the statistics (caller holds the lock)."""
        self.checks = 0
        self.hits = 0
        self.rounds = 0
        self.last_check = None
        self.last_found = None

    def set_range(self, size: int):
        """Change the range, draw a fresh secret and restart the counts."""
        size = max(2, min(HIDDEN_RANGE_MAX, int(size)))
        with self._lock:
            self.size = size
            self.candidate %= size
            self._secret = int(self._rng.integers(0, size))
            self.revealed = False
            self._reset_counts_locked()

    def new_secret(self):
        """Draw a fresh secret and restart the counts (and the luck factor)."""
        with self._lock:
            self._secret = int(self._rng.integers(0, self.size))
            self.revealed = False
            self._reset_counts_locked()

    def reveal(self):
        """Show the secret to the user (never used by the clock thread)."""
        with self._lock:
            self.revealed = True
            return self._secret

    # ---- the separate clock thread: cycles the candidate, frequency varies
    def _run(self):
        hz = self.target_hz
        period = 1.0 / hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                new_hz = self.target_hz
            if new_hz != hz:                       # frequency was changed from the GUI
                hz = new_hz
                period = 1.0 / hz
                nxt = time.perf_counter()
            with self._lock:
                self.hz = hz
                self.candidate = (self.candidate + 1) % self.size
                self.ticks += 1
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                self._wake.wait(delay)             # wakes early on stop or frequency change
                self._wake.clear()
            elif delay < -0.05:                    # fell behind: do not burst to catch up
                nxt = time.perf_counter()

    # ---- the spot check, called when an even-state frame arrives
    def spot_check(self) -> bool:
        with self._lock:
            cand = self.candidate                  # the value on the clock right now
            hit = cand == self._secret
            self.checks += 1
            self.last_check = (cand, hit)
            if hit:
                self.hits += 1
                self.rounds += 1
                self.last_found = (cand, self.checks)   # the correct number, as hidden
            # hit or miss, the number is thrown away and a new one is hidden after every check
            self._secret = int(self._rng.integers(0, self.size))
            self.revealed = False
            return hit

    def snapshot(self) -> dict:
        with self._lock:
            p = 1.0 / self.size
            expected = self.checks * p
            luck = (self.hits / expected * 100.0) if expected > 0 else None
            tail = prob_at_least(self.hits, self.checks, p) if self.checks else None
            return dict(size=self.size, candidate=self.candidate, hz=self.hz,
                        ticks=self.ticks, checks=self.checks, hits=self.hits,
                        rounds=self.rounds, last_check=self.last_check,
                        last_found=self.last_found, expected=expected, luck=luck, luck_tail=tail,
                        secret=self._secret if self.revealed else None)


# ------------------------------------------------------------------ higher/lower (binary) search
SEARCH_BITS_DEFAULT = 1000
SEARCH_BITS_MAX = 4096


def short_int(n: int, keep: int = 20) -> str:
    """Hex form of a big integer, shortened to its first and last digits."""
    h = format(n, "x")
    return h if len(h) <= 2 * keep + 3 else f"{h[:keep]}...{h[-keep:]} ({len(h) * 4} bit hex)"


class BinarySearch:
    """A secret integer of `bits` bits and a higher/lower oracle, with POOLED bits.

    Each guess() is one oracle query and yields exactly one bit of the secret
    (most significant bit first): "is secret >= pooled-so-far with this bit set?".
    The bit is added to a pool and NO equality check is made on that query. Only
    when all `bits` bits have been pooled is the pooled candidate compared with the
    secret, once. A secret of n bits therefore costs n pooled queries plus one
    final check, instead of an independent equality check after every guess.
    When the final check passes one time-code bin is earned and a new secret is
    hidden."""

    def __init__(self, bits=SEARCH_BITS_DEFAULT):
        self._lock = threading.Lock()
        self.bits = 2
        self.guesses = 0                    # all oracle queries since the last reset
        self.final_checks = 0               # equality checks made (one per full pool)
        self.solves = 0                     # secrets found since the last reset
        self.solved_guesses = 0             # queries spent on the solved secrets
        self.round_guesses = 0              # queries spent on the current secret
        self.last = None                    # (verdict, guess_bits)
        self.last_found = None              # (number, guesses it took) of the latest find
        self.revealed = False
        self.set_bits(bits)

    def _new_round_locked(self):
        self._secret = secrets.randbelow(1 << self.bits)
        self.pooled = 0                     # bits collected so far, MSB first
        self.pooled_n = 0                   # how many bits are in the pool
        self.round_guesses = 0
        self.revealed = False

    def _reset_counts_locked(self):
        self.guesses = self.solves = self.solved_guesses = self.final_checks = 0
        self.last = None
        self.last_found = None

    def set_bits(self, bits: int):
        """Change the size of the secret, draw a new one and restart the counts."""
        with self._lock:
            self.bits = max(2, min(SEARCH_BITS_MAX, int(bits)))
            self._new_round_locked()
            self._reset_counts_locked()

    def new_secret(self):
        with self._lock:
            self._new_round_locked()
            self._reset_counts_locked()

    def reveal(self):
        with self._lock:
            self.revealed = True
            return self._secret

    def guess(self) -> bool:
        """One oracle query. Pools one bit; only the query that completes the pool
        runs the single final equality check. True when that check finds the secret."""
        with self._lock:
            self.guesses += 1
            self.round_guesses += 1
            shift = self.bits - 1 - self.pooled_n
            trial = self.pooled | (1 << shift)
            bit = 1 if self._secret >= trial else 0      # oracle: higher/lower, no equality
            self.pooled |= bit << shift
            self.pooled_n += 1
            if self.pooled_n < self.bits:
                self.last = (f"pooled bit {self.pooled_n}/{self.bits} = {bit}", self.pooled_n)
                return False

            # pool is full: the ONE final check
            self.final_checks += 1
            if self.pooled == self._secret:
                self.solves += 1
                self.solved_guesses += self.round_guesses
                self.last_found = (self.pooled, self.round_guesses)
                self.last = ("FOUND (final check passed)", self.bits)
                self._new_round_locked()
                return True
            self.last = ("final check failed, pool cleared", self.bits)
            self.pooled = 0
            self.pooled_n = 0
            return False

    def snapshot(self) -> dict:
        with self._lock:
            return dict(bits=self.bits, guesses=self.guesses, solves=self.solves,
                        round_guesses=self.round_guesses,
                        pooled_n=self.pooled_n, final_checks=self.final_checks,
                        interval_bits=self.bits - self.pooled_n,
                        avg=(self.solved_guesses / self.solves if self.solves else None),
                        last=self.last, last_found=self.last_found,
                        secret=short_int(self._secret) if self.revealed else None)



# ------------------------------------------------------------------ factoring by higher/lower search
FACTOR_BITS_DEFAULT = 256
FACTOR_BITS_MAX = 4096


def _engine_proc(n: int, seed: bytes, q):
    """Runs in a CHILD PROCESS: big-int pow() never releases the GIL, so doing this work in a
    thread froze the GUI and the camera loop. Results come back through the queue."""
    try:
        pool = EntropyPool()
        pool.feed(seed)
        fz = Factoriser(pool, lambda: False,
                        lambda s: q.put(("log", s)), lambda t, f: q.put(("prog", t, f)))
        fs = [int(f) for f in fz.factor(n)]
        ok = math.prod(fs) == n and all(is_probable_prime(f) for f in fs)
        q.put(("done", fs, ok))
    except BaseException as e:                                   # report, never hang the parent
        q.put(("err", f"{type(e).__name__}: {e}"))


def short_dec(n: int, keep: int = 14) -> str:
    s = str(n)
    return s if len(s) <= 2 * keep + 3 else f"{s[:keep]}...{s[-keep:]} ({len(s)} digits)"


def parse_number(text: str) -> int:
    """Plain integer (spaces, commas, underscores allowed) or a form like 2^127-1."""
    import re
    t = re.sub(r"[\s,_]", "", text).lstrip("+")
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", t):
        return int(t, 16)
    t = re.sub(r"[lLnN]$", "", t) if re.fullmatch(r"\d+[lLnN]", t) else t
    m = re.fullmatch(r"(\d+)\^(\d+)([+-]\d+)?", t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if a < 2 or b * math.log2(a) > 20000:
            raise ValueError("power too large (limit about 20000 bits)")
        return a ** b + c
    if t.isdigit():
        return int(t)
    raise ValueError("type digits, or a form like 2^127-1")


class FactorSearch:
    """Factoring with the higher/lower bit search, and a time code ONLY for a verified factorisation.

    N is the number to factor. The oracle holds a prime factor p of N (the smallest one) and answers
    only "is p >= this trial value?". Each armed frame (two sequential evens) is one query and pools
    one bit of p, most significant bit first. No equality test is made per query. When all bits are
    pooled, ONE final check runs: 1 < p < N, N % p == 0, and the full prime factorisation multiplies
    back to N with every factor a probable prime. Only then is a time-code bin earned.

    Where the oracle gets p:
      * typed N : a background thread really factors N (trial division, Fermat, Pollard rho, p-1,
                  ECM, quadratic sieve). Until it finishes the oracle is not ready, no bits are
                  pooled and no time code can be earned.
      * generated N : a random semiprime is built and its factors are kept inside the oracle. This
                  shows the mechanism; the search does not factor N, the oracle already knows p.
    A higher/lower oracle for a factor of N cannot exist without already knowing that factor, so the
    pooling is a mechanism, not a shortcut: the cost of factoring is paid by the engine thread."""

    def __init__(self, pool):
        self._lock = threading.Lock()
        self.pool = pool
        self._run = 0
        self.gen_bits = FACTOR_BITS_DEFAULT
        self.state = "idle"             # idle generating factoring ready solved prime failed cancelled
        self.mode = ""
        self.n = 0
        self.width = 0
        self._secret = 0
        self.factors = []
        self.verified = False
        self.highlow = False            # True: typed N is searched directly, no engine, no factor checks
        self.progress = ""
        self.msgs = []
        self.revealed = False
        self.pooled = self.pooled_n = 0
        self.round_guesses = 0
        self.guesses = self.final_checks = self.solves = 0
        self.last = None
        self.last_found = None          # (p, q, queries)
        self.history = []               # every number the engine finished / every verified solve
        self.t_start = 0.0
        self.engine_s = None

    # ---------------------------------------------------------- bookkeeping
    def _say(self, run, s):
        with self._lock:
            if run == self._run:
                self.msgs.append(s)

    def _prog(self, run, text, frac):
        with self._lock:
            if run == self._run:
                self.progress = f"{text}  {frac:.0%}"

    def drain(self):
        with self._lock:
            m, self.msgs = self.msgs, []
            return m

    def _begin(self, mode, state, n=0, keep_counts=False):
        with self._lock:
            self._run += 1
            self.mode, self.state, self.n = mode, state, n
            self.width = max(1, math.isqrt(n).bit_length()) if n else 0
            self._secret, self.factors, self.verified, self.highlow = 0, [], False, False
            self.pooled = self.pooled_n = self.round_guesses = 0
            self.progress, self.revealed, self.engine_s = "", False, None
            self.t_start = time.time()
            if not keep_counts:
                self.guesses = self.final_checks = self.solves = 0
                self.last = self.last_found = None
            return self._run

    def _record(self, mode, n, factors, status, **extra):
        """Caller holds self._lock. Big ints are stored as decimal strings (JSON-safe)."""
        self.history.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, status=status,
                                 n=str(n), bits=int(n).bit_length(),
                                 factors=[str(f) for f in factors],
                                 **{k: (str(v) if isinstance(v, int) and v > 2**53 else v)
                                    for k, v in extra.items()}))

    def export(self) -> dict:
        with self._lock:
            cur = dict(state=self.state, mode=self.mode, n=str(self.n) if self.n else None,
                       factors=[str(f) for f in self.factors],
                       secret=str(self._secret) if self.revealed and self._secret else None)
            return dict(saved=time.strftime("%Y-%m-%d %H:%M:%S"), current=cur,
                        verified_solves=self.solves, queries=self.guesses,
                        history=list(self.history))

    def cancel(self):
        with self._lock:
            self._run += 1
            if self.state in ("generating", "factoring"):
                self.state = "cancelled"
                self.progress = ""
                self.msgs.append("# factoriser: cancelled")

    def reveal(self):
        with self._lock:
            self.revealed = True
            return self._secret

    # ---------------------------------------------------------- typed number
    def search_number(self, n: int):
        """High/low only: no primality test, no factoring engine, no factor verification.
        The oracle holds the typed number itself and answers only "is it >= trial?". Each armed frame
        pools one bit (most significant first); after the last bit ONE equality check runs."""
        n = int(n)
        with self._lock:
            self._run += 1
            self.mode, self.n, self.state, self.highlow = "typed (high/low only)", n, "ready", True
            self.width = max(1, n.bit_length())
            self._secret, self.factors, self.verified = n, [n], True
            self.pooled = self.pooled_n = self.round_guesses = 0
            self.guesses = self.final_checks = self.solves = 0
            self.progress, self.revealed, self.engine_s = "", False, None
            self.last = self.last_found = None
            self.msgs.append(f"# high/low only: N = {short_dec(n)} ({n.bit_length()} bit), no checks; "
                             f"pooling {self.width} bits on armed frames")

    def factor_number(self, n: int):
        n = int(n)
        run = self._begin("typed", "factoring", n)
        if n < 4:
            with self._lock:
                self.state = "failed"
                self.progress = "N must be at least 4"
            return
        self._say(run, f"# factoriser: N = {short_dec(n)} ({n.bit_length()} bit), "
                       f"{'gmpy2' if HAVE_GMP else 'plain Python ints'}")
        threading.Thread(target=self._work_typed, args=(run, n), daemon=True).start()

    def _work_typed(self, run, n):
        t0 = time.time()
        q = mp.Queue()
        proc = mp.Process(target=_engine_proc, args=(n, self.pool.rand_bits(256).to_bytes(32, "big"), q),
                          daemon=True)
        proc.start()
        fs = ok = None
        try:
            while True:
                if self._run != run:                              # cancelled or a new N was entered
                    return
                try:
                    msg = q.get(timeout=0.1)
                except queue.Empty:
                    if not proc.is_alive():
                        try:
                            msg = q.get(timeout=0.2)              # result may still be in flight
                        except queue.Empty:
                            msg = ("err", "engine process stopped unexpectedly")
                    else:
                        continue
                if msg[0] == "log":
                    self._say(run, msg[1])
                elif msg[0] == "prog":
                    self._prog(run, msg[1], msg[2])
                elif msg[0] == "err":
                    with self._lock:
                        if run == self._run:
                            self.state, self.progress = "failed", f"engine error: {msg[1]}"
                    return
                else:
                    fs, ok = msg[1], msg[2]
                    break
        finally:
            if proc.is_alive():
                proc.terminate()
            q.close()
        with self._lock:
            if run != self._run:
                return
            self.engine_s = time.time() - t0
            self.progress = ""
            if not ok:
                self.state, self.progress = "failed", "factorisation did not verify"
            elif len(fs) == 1:
                self.state = "prime"
                self._record("typed", n, fs, "prime")
                self.msgs.append(f"# factoriser: N = {short_dec(n)} is (probably) prime ({n.bit_length()} bit). "
                                 "A prime has no factor p with 1 < p < N, so there is no oracle and no "
                                 "time code to earn. Type a composite N, or press Generate.")
            else:
                self.factors, self._secret, self.verified = fs, fs[0], True
                self.state = "ready"
                self._record("typed", n, fs, "factored")
                self.msgs.append("# factoriser: engine finished in "
                                 f"{self.engine_s:.1f} s, oracle ready ({len(fs)} prime factors); "
                                 "pooling bits on armed frames")

    # ---------------------------------------------------------- generated semiprime
    def generate(self, bits: int, keep_counts=False):
        bits = max(16, min(FACTOR_BITS_MAX, int(bits)))
        self.gen_bits = bits
        run = self._begin("generated", "generating", 0, keep_counts)
        threading.Thread(target=self._work_gen, args=(run, bits), daemon=True).start()

    def _work_gen(self, run, bits):
        def prime(b):
            while True:
                if self._run != run:
                    raise Cancelled()
                c = self.pool.rand_bits(b) | (1 << (b - 1)) | 1
                if is_probable_prime(c):
                    return c
        try:
            p = prime(bits // 2)
            q = prime(bits - bits // 2)
            while q == p:
                q = prime(bits - bits // 2)
        except Cancelled:
            return
        n, fs = p * q, sorted((p, q))
        with self._lock:
            if run != self._run:
                return
            self.n, self.width = n, max(1, math.isqrt(n).bit_length())
            self.factors, self._secret, self.verified = fs, fs[0], True
            self.state = "ready"
            self.msgs.append(f"# factoriser: new semiprime N = {short_dec(n)} ({n.bit_length()} bit); "
                             "its factors are held by the oracle")

    # ---------------------------------------------------------- one armed frame = one oracle query
    def guess(self) -> bool:
        """Pools one bit. Only the query that completes the pool runs the single final check.
        True only when that check verifies a correct factorisation."""
        regen = False
        with self._lock:
            if self.state != "ready":
                return False
            self.guesses += 1
            self.round_guesses += 1
            shift = self.width - 1 - self.pooled_n
            trial = self.pooled | (1 << shift)
            bit = 1 if self._secret >= trial else 0            # higher/lower, no equality
            self.pooled |= bit << shift
            self.pooled_n += 1
            if self.pooled_n < self.width:
                self.last = f"pooled bit {self.pooled_n}/{self.width} = {bit}"
                return False
            self.final_checks += 1
            c = self.pooled
            if self.highlow:                                   # one equality check, nothing else
                ok = (c == self._secret)
            else:
                ok = (1 < c < self.n and self.n % c == 0 and self.verified
                      and math.prod(self.factors) == self.n)
            if not ok:
                self.last = "final check failed, pool cleared"
                self.pooled = self.pooled_n = 0
                return False
            self.solves += 1
            self.last_found = (c, 1 if self.highlow else self.n // c, self.round_guesses)
            self._record(self.mode, self.n, self.factors, "solved", p=c, queries=self.round_guesses)
            self.last = ("FOUND: pooled bits equal the number" if self.highlow
                         else "FOUND: N % p == 0 and the factorisation multiplies back to N")
            self.state = "solved"
            regen = self.mode == "generated"
        if regen:
            self.generate(self.gen_bits, keep_counts=True)      # next semiprime, same size
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return dict(state=self.state, mode=self.mode, n=self.n, width=self.width,
                        pooled_n=self.pooled_n, guesses=self.guesses, solves=self.solves,
                        final_checks=self.final_checks, round_guesses=self.round_guesses,
                        progress=self.progress, last=self.last, last_found=self.last_found,
                        engine_s=self.engine_s, factors=list(self.factors) if self.state == "solved" else [],
                        secret=short_dec(self._secret) if self.revealed and self._secret else None)


# ------------------------------------------------------------------ time-code maths
def carries(a: int, b: int, w: int) -> int:
    """Number of carries (including the carry out of the top) when adding a + b."""
    c = n = 0
    for i in range(w):
        t = ((a >> i) & 1) + ((b >> i) & 1) + c
        c = t >> 1
        n += c
    return n


def parse_code(expr: str):
    """Evaluate a time-code expression: '1011', '1000|0011', '1111&1011', '1011+0001'.

    Operands are bit strings read as binary numbers (written order = bin order).
    |  OR-addition   &  AND   +  integer addition (wraps at the code width).
    Returns (Y, info). Raises ValueError on a malformed expression.
    """
    e = expr.replace(" ", "")
    toks, cur = [], ""
    for ch in e:
        if ch in "|&+":
            toks += [cur, ch]
            cur = ""
        else:
            cur += ch
    toks.append(cur)
    operands, ops = toks[0::2], toks[1::2]
    for o in operands:
        if not o or set(o) - {"0", "1"}:
            raise ValueError(f"bad operand '{o}'")
    W = max(len(o) for o in operands)
    mask = (1 << W) - 1
    val, car, wrapped = int(operands[0], 2), 0, 0
    for op, o in zip(ops, operands[1:]):
        b = int(o, 2)
        if op == "|":
            val |= b
        elif op == "&":
            val &= b
        else:
            car += carries(val, b, W)
            wrapped += int(val + b > mask)
            val = (val + b) & mask
    return format(val, f"0{W}b"), dict(carries=car, wrapped=wrapped, expr=e)


def open_in_window(Y: str, k: int, start: int) -> int:
    return sum(int(Y[(start + t) % len(Y)]) for t in range(k))


def window_stats(Y: str, k: int):
    """(worst, best) number of open bins over every k-frame window of the periodic code."""
    counts = [open_in_window(Y, k, s) for s in range(len(Y))]
    return min(counts), max(counts)


def frames_needed(Y: str, n_open: int):
    """Smallest k whose worst-phase window holds n_open open bins (None if Y is all 0)."""
    if "1" not in Y:
        return None
    k = n_open
    while window_stats(Y, k)[0] < n_open:
        k += 1
    return k


# ------------------------------------------------------------------ driver analysis
_WD: dict = {}


def witness_density(x: int) -> float:
    """Exact fraction of r in [2, x-2] that are Miller-Rabin witnesses for x
    (0 for primes, even numbers and x < 5). This is Driver A's true success
    probability on instance x. Rabin: >= 3/4 for every odd composite."""
    if x not in _WD:
        if x < 5 or x % 2 == 0:
            _WD[x] = 0.0
        else:
            _WD[x] = sum(miller_rabin_witness(a, x) for a in range(2, x - 1)) / (x - 3)
    return _WD[x]


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mutual_info(cm: dict) -> float:
    """I(driver output ; x in L) in bits from a 2x2 confusion matrix."""
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    N = tp + fp + fn + tn
    if N == 0:
        return 0.0
    pl = (tp + fn) / N
    pd = (tp + fp) / N
    mi = 0.0
    for cnt, d, l in ((tp, pd, pl), (fp, pd, 1 - pl), (fn, 1 - pd, pl), (tn, 1 - pd, 1 - pl)):
        if cnt and d > 0 and l > 0:
            mi += cnt / N * math.log2((cnt / N) / (d * l))
    return mi


def _div(a, b):
    return a / b if b else None


def _f(v, w=5):
    return " " * (w - 1) + "-" if v is None else f"{v:{w}.2f}"


class DriverAnalysis:
    """Predictive properties of Driver A, Driver B and the final Q, scored on
    every instance frame against the ground truth x in L."""

    NAMES = (("A", "Driver A (random)"), ("B", "Driver B (determ.)"), ("Q", "Q  (A.B.Y latch)"))

    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = {n: dict(tp=0, fp=0, fn=0, tn=0) for n, _ in self.NAMES}
        self.n_noinst = 0
        self.link_miss = 0                  # c(x)=1 but B stayed LOW (timing)
        self.q_checked = self.q_mismatch = 0
        self.a_seq, self.w_seq, self.r_seq = [], [], []     # in-L frames, in order
        self.min_w = 1.0

    def update(self, rec: dict, w: float):
        A, B, Q, y = rec["A"], rec["B"], rec["Q"], rec["y_bin"]
        pred = ((A & B) | y) if rec["force"] else (A & B & y)    # what Q should be
        self.q_checked += 1
        self.q_mismatch += int(pred != Q)
        if not rec["motion"]:
            self.n_noinst += 1
            return
        pos = rec["in_L"]
        for name, val in (("A", A), ("B", B), ("Q", Q)):
            key = ("tp" if val else "fn") if pos else ("fp" if val else "tn")
            self.cm[name][key] += 1
        if rec["c_of_x"] and not B:
            self.link_miss += 1
        if pos:
            self.a_seq.append(A)
            self.w_seq.append(w)
            self.r_seq.append(rec["r"])
            self.min_w = min(self.min_w, w)
            if len(self.a_seq) > 4000:
                del self.a_seq[:1000], self.w_seq[:1000], self.r_seq[:1000]

    def lines(self, k: int, force: bool = False):
        a = self.cm["A"]
        n_in = a["tp"] + a["fn"]
        n_out = a["fp"] + a["tn"]
        n = n_in + n_out
        L = [f"DRIVER ANALYSIS   instances n={n}  (x in L: {n_in}, not in L: {n_out});"
             f" no-motion frames: {self.n_noinst}"]
        if n == 0:
            return L + ["waiting for motion ..."]
        base = n_in / n
        H = -sum(q * math.log2(q) for q in (base, 1 - base) if q > 0)
        L.append(f"truth: P(x in L)={base:.2f}   H(L)={H:.2f} bit   (L = odd composite)")
        L.append("")
        L.append(f"{'':20s}{'TPR':>6}{'FPR':>6}{'PPV':>6}{'NPV':>6}{'acc':>6}{'I(D;L)':>9}")
        for name, label in self.NAMES:
            c = self.cm[name]
            tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
            L.append(f"{label:20s}{_f(_div(tp, tp + fn), 6)}{_f(_div(fp, fp + tn), 6)}"
                     f"{_f(_div(tp, tp + fp), 6)}{_f(_div(tn, tn + fn), 6)}"
                     f"{_f(_div(tp + tn, n), 6)}{mutual_info(c):7.2f} b")
        L.append("")
        L.append("RP properties the bounds rely on")
        fa = a["fp"]
        lo, hi = wilson(fa, n_out)
        L.append(f" A fires only inside L (FP=0): {'PASS' if fa == 0 else 'FAIL'}   "
                 f"{fa}/{n_out}, FPR <= {hi:.1%} (95%)")
        lo, hi = wilson(a["tp"], n_in)
        verdict = "PASS" if lo >= 0.5 else ("FAIL" if hi < 0.5 else "inconclusive")
        L.append(f" A witness rate p >= 1/2     : {verdict}   p={_div(a['tp'], n_in) or 0:.3f}"
                 f"  [{lo:.3f}, {hi:.3f}]")
        b = self.cm["B"]
        L.append(f" B never misses L (FN=0)     : {'PASS' if b['fn'] == 0 else 'FAIL'}   "
                 f"{b['fn']}/{n_in}   link timing misses: {self.link_miss}")
        q = self.cm["Q"]
        L.append(f" Q one-sided (FP=0)          : {'PASS' if q['fp'] == 0 else 'FAIL'}   "
                 f"{q['fp']}/{n_out}" + ("   (force mode breaks it, Thm 4)" if force else ""))
        L.append("")
        L.append("Calibration of A (exact witness density w(x) of each member x)")
        m = len(self.a_seq)
        if m >= 10:
            obs = sum(self.a_seq)
            exp = sum(self.w_seq)
            var = sum(w * (1 - w) for w in self.w_seq)
            z = (obs - exp) / math.sqrt(var) if var > 0 else None
            tag = ("n/a" if z is None else ("calibrated" if abs(z) < 2.5 else "MISCALIBRATED"))
            L.append(f" predicted mean {exp / m:.3f}   observed {obs / m:.3f}"
                     f"   z={'-' if z is None else f'{z:+.1f}'}   {tag}")
            L.append(f" lowest density seen {self.min_w:.3f}   "
                     f"(Rabin bound 0.750: {'ok' if self.min_w >= 0.75 else 'VIOLATED'})")
        else:
            L.append(" need >= 10 members of L")
        L.append("")
        L.append("Independence of A across trials (assumed by (1-p)^N)")
        if m >= 10:
            pairs = list(zip(self.a_seq, self.a_seq[1:]))
            n11 = sum(1 for u, v in pairs if u and v)
            n10 = sum(1 for u, v in pairs if u and not v)
            n01 = sum(1 for u, v in pairs if not u and v)
            n00 = sum(1 for u, v in pairs if not u and not v)
            den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            phi = (n11 * n00 - n10 * n01) / den if den else None
            L.append(f" lag-1 phi of A on successive members: "
                     f"{'n/a (A nearly constant)' if phi is None else f'{phi:+.2f}'}")
            L.append(f" distinct witnesses r: {len(set(self.r_seq)) / m:.0%} of members")
            nb = m // k
            if nb >= 1:
                zeros = exp_z = 0.0
                for i in range(nb):
                    seg = range(i * k, (i + 1) * k)
                    zeros += int(not any(self.a_seq[j] for j in seg))
                    pr = 1.0
                    for j in seg:
                        pr *= 1 - self.w_seq[j]
                    exp_z += pr
                L.append(f" blocks of k={k} members with no witness: observed {int(zeros)}, "
                         f"expected {exp_z:.2f}  (of {nb})")
        else:
            L.append(" need >= 10 members of L")
        L.append("")
        L.append(f"Logic check  Q == " + ("(A.B) OR Y" if force else "A.B.Y") +
                 f" : {self.q_mismatch} mismatches in {self.q_checked} frames")
        return L


# ------------------------------------------------------------------ frame source
class Source:
    """Real USB camera via OpenCV."""

    def __init__(self, index: int):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed.  Run: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            raise RuntimeError(
                f"Cannot open camera {index}. Check that it is connected, not in use "
                f"by another app, and that the camera number is right.")
        self.cap = cap
        self.rgb = None                  # latest colour frame (for the preview)

    def read(self):
        """Return a full-resolution uint8 gray frame, or None."""
        ok, bgr = self.cap.read()
        if not ok:
            return None
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()


def downscale(gray: np.ndarray) -> np.ndarray:
    """Shrink and blur so sensor grain does not count as motion."""
    small = cv2.resize(gray, (SW, SH), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0)


def make_preview(rgb: np.ndarray, mask: np.ndarray, overlay: bool) -> np.ndarray:
    """Colour camera frame at PW x PH, optionally with the motion mask in red."""
    h, w = rgb.shape[:2]
    pv = rgb[np.arange(PH) * h // PH][:, np.arange(PW) * w // PW].copy()
    if overlay and mask.any():
        big = mask[np.arange(PH) * SH // PH][:, np.arange(PW) * SW // PW]
        pv[big] = (pv[big] * 0.4 + np.array([255, 40, 40]) * 0.6).astype(np.uint8)
    return pv


# ------------------------------------------------------------------ pipeline
@dataclass
class Packet:
    idx: int
    dt: float
    x: int
    motion: bool
    rand_int: int


@dataclass
class St:
    A: int = 0
    r: int = 0
    B: int = 0
    c_of_x: int = 0
    vC: float = 0.0
    vB: float = 0.0
    traj: list = field(default_factory=list)
    node: float = 0.0
    orv: int = 0
    andv: int = 0
    gate: int = 0
    y_bin: int = 1
    Q: int = 0


class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out):
        super().__init__(daemon=True)
        self.cam_index, self.ctl, self.out = cam_index, ctl, out

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out = self.ctl, self.out
        hidden = ctl.get("hidden")
        search = ctl.get("search")
        fsearch = ctl.get("fsearch")
        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            out.put(("done", None))
            return
        out.put(("info", f"USB camera {self.cam_index} opened"))
        st = St()
        qa, qb, qc = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
        beta_tau = TAU_NODE

        async def driver_a():                          # randomized witness test
            while True:
                p = await qa.get()
                if p is None:
                    qa.task_done()
                    return
                st.r, st.A = 0, 0
                if p.motion and p.x > 4 and p.x % 2 == 1:
                    st.r = 2 + p.rand_int % (p.x - 3)           # r in [2, x-2]
                    st.A = int(miller_rabin_witness(st.r, p.x))
                qa.task_done()

        async def driver_b():                          # deterministic check + link
            while True:
                p = await qb.get()
                if p is None:
                    qb.task_done()
                    return
                st.c_of_x = int(p.motion and p.x > 2 and p.x % 2 == 1)
                traj, n = [], 24
                for j in range(n + 1):                 # C charges from 0 each frame
                    t = p.dt * j / n
                    vC = VS * (1 - math.exp(-t / RC_B)) if st.c_of_x else 0.0
                    traj.append((t * 1000, vC, max(0.0, vC - V_BE)))
                st.traj = traj
                st.vC, st.vB = traj[-1][1], traj[-1][2]
                st.B = int(st.vB >= V_IH)
                qb.task_done()

        async def clock_ff():                          # gates, node, Y, 74HC74
            while True:
                p = await qc.get()
                if p is None:
                    qc.task_done()
                    return
                st.orv = st.A | st.B
                st.andv = st.A & st.B
                st.gate = st.orv & st.andv                       # tied -> AND
                st.node = st.gate * (1 - math.exp(-p.dt / beta_tau))   # 10 uF node
                y = ctl["y"]
                added = 1 if ctl.get("timecode_pending", 0) > 0 else 0
                st.y_bin = int(y[p.idx % len(y)]) | added
                if added:                                # one found number = one added bin
                    ctl["timecode_pending"] -= 1
                d_in = int(st.node > 0.5)
                if ctl.get("force"):                             # Y OR-ed in (Thm 4 demo)
                    st.Q = d_in | st.y_bin
                else:                                            # Y gates the result
                    st.Q = d_in & st.y_bin                       # latch on frame edge
                qc.task_done()

        qn = asyncio.Queue()
        AN = DriverAnalysis()

        # Shared statistics. Defined BEFORE the tasks so it never races the closures.
        S = dict(frames=0, motion=0, in_L=0, acc_in_L=0, false_acc=0, accepts=0,
                 open_inL=0, acc_open_inL=0,
                 mined_even=0, last_x=0, last_even=False, last_armed=False,
                 timecodes_added=0, hidden_timecodes=0, search_timecodes=0,
                 factor_timecodes=0)

        def codes_snapshot():
            """Time-code statistics from the hidden-number and bit searches."""
            return dict(
                circuit=bool(ctl.get("circuit", True)),
                last_x=S["last_x"], last_even=S["last_even"], last_armed=S["last_armed"],
                armed_frames=S["mined_even"],
                timecodes_added=S["timecodes_added"],
                hidden_timecodes=S["hidden_timecodes"],
                search_timecodes=S["search_timecodes"],
                factor_timecodes=S["factor_timecodes"],
                search=S.get("search"),
                fsearch=S.get("fsearch"),
                hidden=S.get("hidden"),
                timecode_pending=int(ctl.get("timecode_pending", 0)))

        async def analyst():                           # runs beside the drivers
            seen = 0
            while True:
                rec = await qn.get()
                if rec is None:
                    out.put(("analysis", AN.lines(ctl["k"], ctl.get("force", False))))
                    return
                if ctl.get("reset_an"):
                    AN.reset()
                    ctl["reset_an"] = False
                w = 0.0
                if rec["in_L"]:                        # exact density, off the loop
                    w = _WD.get(rec["x"])
                    if w is None:
                        w = await asyncio.to_thread(witness_density, rec["x"])
                AN.update(rec, w)
                seen += 1
                if seen % 5 == 0 or qn.empty():
                    out.put(("analysis", AN.lines(ctl["k"], rec["force"])))

        tasks = [asyncio.create_task(f())
                 for f in (driver_a, driver_b, clock_ff, analyst)]

        prev = None
        prev_even = False                  # was the previous frame an even state?
        t_prev = time.perf_counter()
        idx = 0
        recent_q = deque(maxlen=60)
        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break
                now = time.perf_counter()
                dt = min(max(now - t_prev, 1e-3), 0.25)
                t_prev = now

                # ---- motion detection
                small = downscale(gray)
                if prev is None or idx < WARMUP:
                    diff = np.zeros((SH, SW), np.int16)
                    mask = np.zeros((SH, SW), bool)
                else:
                    diff = np.abs(small.astype(np.int16) - prev.astype(np.int16))
                    mask = diff > ctl["thr"]
                prev = small
                count = int(mask.sum())
                motion = count >= ctl["min_px"]
                x = count if motion else 0

                # ---- witness source: the motion pattern (same camera, no noise)
                energy = np.where(mask, np.clip(diff, 0, 255), 0).astype(np.uint8)
                digest = hashlib.sha256(energy.tobytes()).digest()
                if ctl.get("pool") is not None:
                    ctl["pool"].feed(digest)            # camera motion seeds the factoriser randomness
                pkt = Packet(idx, dt, x, motion, int.from_bytes(digest, "big"))

                # A hidden-number spot check or a search guess is only allowed after TWO SEQUENTIAL even
                # states: this frame and the one right before it must both be even. A frame
                # with no motion (x = 0) or an odd x breaks the run.
                even_now = x > 0 and x % 2 == 0
                even_state = even_now and prev_even        # armed: two evens in a row
                prev_even = even_now
                S["last_x"], S["last_even"], S["last_armed"] = x, even_now, even_state
                if even_state:
                    S["mined_even"] += 1                   # armed frames seen

                # BTC high/low oracle: one armed release is consumed on the next even state.
                if even_now and ctl.get("btc_release_armed", False):
                    ctl["btc_release_armed"] = False
                    ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + 1
                    S["btc_timecodes"] = S.get("btc_timecodes", 0) + 1
                    S["timecodes_added"] = S.get("timecodes_added", 0) + 1
                    btc_call = ctl.get("btc_last_call", "--")
                    btc_price = ctl.get("btc_last_price", 0.0)
                    out.put(("log",
                        f"BTC HIGH/LOW ORACLE EVEN RELEASE: even frame x={x}; "
                        f"oracle call={btc_call}; BTC/USDT=${btc_price:,.2f}; "
                        f"one time-code bin released; pending={ctl['timecode_pending']}."
                    ))

                # Hidden-number test: only two sequential even states trigger a spot check, and it reads
                # the clock-cycled candidate at this very moment.
                hidden_hit = False
                if hidden is not None and ctl.get("hidden_on", True) and even_state:
                    hidden_hit = hidden.spot_check()
                    if hidden_hit:
                        ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + 1
                        S["hidden_timecodes"] += 1
                        S["timecodes_added"] += 1
                        out.put(("hidden_hit", dict(frame=idx, x=x, time=time.time(),
                                                    snap=hidden.snapshot())))
                if hidden is not None:
                    S["hidden"] = hidden.snapshot()

                # Higher/lower search: one oracle guess per armed frame (two sequential evens)
                search_hit = False
                if search is not None and ctl.get("search_on", True) and even_state:
                    search_hit = search.guess()
                    if search_hit:
                        ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + 1
                        S["search_timecodes"] += 1
                        S["timecodes_added"] += 1
                        sn = search.snapshot()
                        out.put(("search_hit", dict(frame=idx, x=x, bits=sn["bits"],
                                                    solves=sn["solves"], avg=sn["avg"],
                                                    found=sn["last_found"])))
                if search is not None:
                    S["search"] = search.snapshot()

                # Factoriser: one higher/lower query per armed frame; a time code is earned ONLY when the
                # single final check verifies a correct factorisation (separate source, own counter).
                factor_hit = False
                if fsearch is not None and ctl.get("fsearch_on", True) and even_state:
                    factor_hit = fsearch.guess()
                    if factor_hit:
                        ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + 1
                        S["factor_timecodes"] += 1
                        S["timecodes_added"] += 1
                        fn = fsearch.snapshot()
                        out.put(("factor_hit", dict(frame=idx, x=x, found=fn["last_found"],
                                                    solves=fn["solves"])))
                if fsearch is not None:
                    S["fsearch"] = fsearch.snapshot()
                # ---- drivers A and B on this frame, concurrently; then flip-flop
                run_circuit = ctl.get("circuit", True)
                if run_circuit:
                    await qa.put(pkt)
                    await qb.put(pkt)
                    await asyncio.gather(qa.join(), qb.join())
                    await qc.put(pkt)
                    await qc.join()
                else:                                  # preview only: circuit idle
                    for name, val in vars(St()).items():
                        setattr(st, name, val)

                out.put(("codes", codes_snapshot()))

                # ---- bookkeeping
                in_L = bool(motion and is_odd_composite(x))
                select = 0
                if run_circuit:
                    recent_q.append(st.Q)
                    select = int(any(list(recent_q)[-ctl["k"]:]))
                    S["frames"] += 1
                    if st.Q and not in_L:
                        S["false_acc"] += 1                # x not in L was accepted
                    if motion:
                        S["motion"] += 1
                        if in_L:
                            S["in_L"] += 1
                            S["acc_in_L"] += st.Q
                            if st.y_bin:                   # frames where Y is open
                                S["open_inL"] += 1
                                S["acc_open_inL"] += st.Q
                        S["accepts"] += st.Q

                if run_circuit:
                    qn.put_nowait(dict(x=x, motion=motion, in_L=in_L, A=st.A, B=st.B,
                                       c_of_x=st.c_of_x, Q=st.Q, y_bin=st.y_bin,
                                       r=st.r, force=bool(ctl.get("force"))))
                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                preview = make_preview(rgb, mask, ctl.get("overlay", True))
                out.put(("frame", dict(
                    circuit=run_circuit, preview=preview, idx=idx, dt=dt, x=x, motion=motion, in_L=in_L, small=small,
                    mask=mask, energy=energy, A=st.A, r=st.r, B=st.B,
                    c_of_x=st.c_of_x, vC=st.vC, vB=st.vB, traj=st.traj,
                    node=st.node, orv=st.orv, andv=st.andv, gate=st.gate,
                    y_bin=st.y_bin, Q=st.Q, select=select, stats=dict(S),
                    hidden_hit=hidden_hit, search_hit=search_hit, factor_hit=factor_hit)))
                idx += 1
        finally:
            for q in (qa, qb, qc, qn):
                await q.put(None)
            await asyncio.gather(*tasks)
            src.close()
            out.put(("done", None))


# ------------------------------------------------------------------ gui
OFF, ON, GOOD = "#f1efe8", "#ffd9a8", "#b9e6b9"
W_OFF, W_ON = "#999999", "#d9480f"


def ppm_photo(rgb: np.ndarray) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


def set_text(widget, text, **kw):
    """Update a Tk label only when its text/options really changed."""
    cache = getattr(widget, "_cache", None)
    key = (text, tuple(sorted(kw.items())))
    if cache != key:
        widget.config(text=text, **kw)
        widget._cache = key


def fixed_label(parent, width_px, height_px, **kw):
    """A label inside a fixed-size frame: its text can change without resizing the layout."""
    box = ttk.Frame(parent, width=width_px, height=height_px)
    box.pack_propagate(False)
    lab = ttk.Label(box, **kw)
    lab.pack(anchor="nw")
    return box, lab


FRAME_INTERVAL = 33                # ms between GUI redraws (~30 fps cap)
LOG_INTERVAL = 100                 # ms between log flushes
MINER_INTERVAL = 0.2               # s between miner panel refreshes



class BitcoinHighLowOracle:
    """Binance BTC/USDT HIGH/LOW oracle.

    Each check records a HIGH/LOW call in the log. A successful, driver-verified check
    arms one release; the circuit releases it only on its next even state.
    """

    URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    def __init__(self):
        self.last_price = None
        self.last_hash = None
        self.last_factual = None
        self.last_checks = None
        self.last_update = None
        self.pending_guess = None
        self.pending_price = None
        self.last_prediction = None
        self.correct = 0
        self.total = 0
        self.fresh_count = 0
        self.factual_count = 0
        self.rejected_count = 0
        self.released_count = 0
        self._lock = threading.Lock()

    @staticmethod
    def _driver_b_checks(raw, data, price):
        checks = {
            "payload_nonempty": isinstance(raw, bytes) and len(raw) > 0,
            "json_object": isinstance(data, dict),
            "symbol_key": isinstance(data, dict) and data.get("symbol") == "BTCUSDT",
            "price_key": isinstance(data, dict) and "price" in data,
            "price_numeric": isinstance(price, (int, float)) and not isinstance(price, bool),
            "price_finite": isinstance(price, (int, float)) and math.isfinite(float(price)),
            "price_positive": isinstance(price, (int, float)) and float(price) > 0,
        }
        checks["driver_b"] = all(checks.values())
        return checks

    @staticmethod
    def _driver_a_check(raw):
        digest = hashlib.sha256(raw).digest()
        x = int.from_bytes(digest[:8], "big") | 1
        base = int.from_bytes(digest[8:12], "big") % max(2, x - 2) + 2 if x > 3 else 2
        composite = is_odd_composite(x)
        witness = miller_rabin_witness(base, x) if composite else False
        checks = {
            "hash_reproducible": hashlib.sha256(raw).digest() == digest,
            "witness_consistent": (not composite) or witness,
        }
        checks["driver_a"] = all(checks.values())
        return {"digest": digest.hex(), "x": x, "composite": composite, **checks}

    def fetch_price(self):
        request = urllib.request.Request(
            self.URL, headers={"User-Agent": "rp-circuit-btc-oracle/1.0"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
        data = json.loads(raw.decode())
        price = float(data["price"])

        b_checks = self._driver_b_checks(raw, data, price)
        a_checks = self._driver_a_check(raw)
        factual = b_checks["driver_b"] and a_checks["driver_a"]

        with self._lock:
            changed = self.last_price != price
            self.last_price = price
            self.last_hash = a_checks["digest"]
            self.last_factual = factual
            self.last_checks = {
                "driver_a": a_checks, "driver_b": b_checks,
                "factual": factual, "changed": changed,
            }
            self.last_update = time.time()
            self.fresh_count += 1
            if factual:
                self.factual_count += 1
            else:
                self.rejected_count += 1

        prediction_result = None
        if self.pending_guess is not None and self.pending_price is not None:
            went_up = price > self.pending_price
            actual_direction = "HIGH" if went_up else "LOW"
            is_correct = (
                (self.pending_guess == "higher" and went_up) or
                (self.pending_guess == "lower" and not went_up)
            )
            self.total += 1
            if is_correct:
                self.correct += 1
            prediction_result = {
                "guess": self.pending_guess.upper(),
                "previous": self.pending_price,
                "actual": price,
                "actual_direction": actual_direction,
                "correct": is_correct,
            }

        next_guess = "higher" if (int(price * 100) % 2 == 0) else "lower"
        self.pending_guess = next_guess
        self.pending_price = price
        self.last_prediction = {
            "guess": next_guess.upper(),
            "made_at_price": price,
        }
        return price, prediction_result, factual, changed

    def guess(self, direction):
        if direction not in ("higher", "lower"):
            raise ValueError("direction must be 'higher' or 'lower'")
        if self.last_price is None:
            self.fetch_price()
        self.pending_guess = direction
        self.pending_price = self.last_price
        self.last_prediction = {"guess": direction.upper(), "made_at_price": self.last_price}


class ScrollFrame(ttk.Frame):
    """A frame whose content scrolls (wheel + bars) when the window is smaller than it."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_outer)
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", self._bind_wheel)
            w.bind("<Leave>", self._unbind_wheel)

    def _on_inner(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._toggle_bars()

    def _on_outer(self, _e=None):
        self._toggle_bars()

    def _toggle_bars(self):
        """Show a scrollbar only while the content really is bigger than the window."""
        need_w = self.inner.winfo_reqwidth()
        need_h = self.inner.winfo_reqheight()
        have_w = self.canvas.winfo_width()
        have_h = self.canvas.winfo_height()
        if need_h > have_h + 1:
            self.vbar.grid()
        else:
            self.vbar.grid_remove()
            self.canvas.yview_moveto(0)
        if need_w > have_w + 1:
            self.hbar.grid()
        else:
            self.hbar.grid_remove()
            self.canvas.xview_moveto(0)

    def _bind_wheel(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _e=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _wheel(self, e):
        # leave the wheel alone over spinboxes (it changes their value) and text boxes
        if e.widget.winfo_class() in ("TSpinbox", "Text"):
            return
        if not self.vbar.winfo_ismapped():
            return
        step = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
        self.canvas.yview_scroll(step * 3, "units")


class App:
    def __init__(self, window):
        self.window = window                          # the real Tk window
        window.title("RP circuit + hidden number, bit search and BTC HIGH/LOW even-frame oracle")
        self.scroller = ScrollFrame(window)
        self.scroller.pack(fill="both", expand=True)
        root = self.scroller.inner                    # every widget below lives in here
        self.root = root
        root.after = window.after                     # keep timers on the real window
        root.after_idle = window.after_idle
        root.protocol = window.protocol
        root.destroy = window.destroy
        self.q = queue.Queue()
        self.thread = None
        self.ctl = dict(stop=False, thr=18, min_px=30, k=5, y="1111", circuit=True,
                        overlay=True, force=False,
                        timecode_pending=0,
                        hidden_on=True, search_on=True)
        self.hidden = HiddenNumber(HIDDEN_RANGE_DEFAULT)
        self.hidden.start()
        self.ctl["hidden"] = self.hidden
        self.search = BinarySearch(SEARCH_BITS_DEFAULT)
        self.ctl["search"] = self.search
        self.pool = EntropyPool()
        self.fsearch = FactorSearch(self.pool)
        self.ctl.update(fsearch=self.fsearch, fsearch_on=True, pool=self.pool)
        self.btc = BitcoinHighLowOracle()
        self.photos = [None, None, None]
        self.btc_auto_job = None
        self.btc_auto_running = False
        self.q_hist = deque(maxlen=84)          # (Q, y_bin, motion)
        self.x_hist = deque(maxlen=150)
        self.last_traj = []
        self.err = False
        self.an_win = None
        self.an_lines = ["Press Start circuit, then move in front of the camera."]
        self.code_info = dict(carries=0, wrapped=0, expr="1111")
        self.y_ok = True

        window.resizable(True, True)
        root.grid_columnconfigure(0, weight=0, minsize=730)     # circuit diagram
        root.grid_columnconfigure(1, weight=0, minsize=440)     # plots + verdict
        root.grid_columnconfigure(2, weight=1, minsize=300)     # camera + analysis
        for r in range(3):
            root.grid_rowconfigure(r, weight=0)
        self._last_render = 0.0
        self._last_codes = 0.0
        self._pending_codes = None
        self._log_buf = []
        self._photo_cache = {}

        self._controls()
        self._miner_panel()
        self._btc_panel()
        self.c = tk.Canvas(root, width=710, height=410, bg="white",
                           highlightthickness=1, highlightbackground="#bbb")
        self.c.grid(row=1, column=0, padx=6, pady=4)
        self.boxes, self.wires = {}, {}
        self._diagram()
        self._right_panel()
        side = ttk.Frame(root)                         # column 2: camera over analysis
        side.grid(row=1, column=2, sticky="n", padx=6, pady=4)
        self.side = side
        self.cam = tk.Canvas(side, width=PW, height=PH, bg="black",
                             highlightthickness=1, highlightbackground="#bbb")
        self.cam.pack(anchor="w")
        self.cam_img = self.cam.create_image(0, 0, anchor="nw")
        self.cam_text = self.cam.create_text(
            8, 8, anchor="nw", fill="#ffd54a", font=("Consolas", 10, "bold"),
            text="Live camera preview\n(press Preview only or Start circuit)")
        self.cam_photo = None
        box, self.analysis = fixed_label(side, PW, 290, text="", font=("Consolas", 8),
                                         justify="left", wraplength=PW - 6)
        box.pack(anchor="w", pady=(6, 0))
        self.analysis_box = box
        # the log sits in the time-code row (see _miner_panel)
        self._y_changed()
        self._compact = False
        self.v_compact = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ctl_row2, text="Compact (small screens)", variable=self.v_compact,
                        command=self._apply_compact).pack(side="left", padx=(12, 0))
        window.after(200, self._auto_compact)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(15, self._poll)
        root.after(300, self._factor_tick)

    # ---------------------------------------------------------- compact mode
    def _apply_compact(self):
        """Hide the bulky read-only extras so the essentials fit a laptop screen:
        the IF/THEN link plot and the time-code analysis text."""
        on = self.v_compact.get()
        if on == self._compact:
            return
        self._compact = on
        if on:
            self.analysis_box.pack_forget()
            self.plot1.pack_forget()
            self.plot1_label.pack_forget()
            self.log.config(height=7)
            self.cam.config(width=PW, height=PH)
        else:
            self.plot1_label.pack(anchor="w", before=self.plot2_label_anchor)
            self.plot1.pack(before=self.plot2_label_anchor)
            self.analysis_box.pack(anchor="w", pady=(6, 0))
            self.log.config(height=14)
        self.root.update_idletasks()
        self.scroller._toggle_bars()

    def _auto_compact(self):
        """Switch to compact mode on its own when the window is clearly too small."""
        try:
            need_h = self.root.winfo_reqheight()
            have_h = self.window.winfo_screenheight() - 110
            if need_h > have_h and not self.v_compact.get():
                self.v_compact.set(True)
                self._apply_compact()
                self.window.update_idletasks()
                fit_window(self.window, self.root.winfo_reqwidth() + 24, self.root.winfo_reqheight() + 8)
        except Exception:
            pass

    # ---------------------------------------------------------- controls
    def _controls(self):
        outer = ttk.Frame(self.root)
        outer.grid(row=0, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
        f = ttk.Frame(outer)
        f.pack(anchor="w")
        f2 = ttk.Frame(outer)
        f2.pack(anchor="w", pady=(5, 0))
        self.ctl_row2 = f2
        ttk.Label(f, text="Camera:").pack(side="left")
        self.s_cam = ttk.Spinbox(f, from_=0, to=9, width=3)
        self.s_cam.set(0)
        self.s_cam.pack(side="left", padx=(2, 12))

        def scale(label, key, lo, hi, init):
            ttk.Label(f, text=label).pack(side="left")
            s = ttk.Scale(f, from_=lo, to=hi, value=init, length=90,
                          command=lambda v, key=key: (self.ctl.update({key: int(float(v))}),
                                                      self._refresh_analysis()))
            s.pack(side="left", padx=(2, 10))
        scale("pixel diff >", "thr", 4, 60, 4)
        scale("min moving px", "min_px", 5, 800, 30)
        scale("k frames", "k", 1, 30, 5)

        ttk.Label(f2, text="Time code Y:").pack(side="left")
        self.e_y = ttk.Entry(f2, width=22)
        self.e_y.insert(0, "1111")
        self.e_y.pack(side="left", padx=(2, 6))
        self.e_y.bind("<KeyRelease>", self._y_changed)
        self.l_y = ttk.Label(f2, text="", font=("Consolas", 9), width=26)
        self.l_y.pack(side="left")
        ttk.Label(f2, text="ops:  |  OR-add    &  AND    +  integer add",
                  foreground="#666").pack(side="left", padx=(0, 10))
        self.v_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(f2, text="'forces HIGH' as OR (breaks one-sidedness)",
                        variable=self.v_force,
                        command=lambda: (self.ctl.update(force=self.v_force.get()),
                                         self._refresh_analysis())).pack(side="left")
        self.b_start = ttk.Button(f, text="Start circuit", command=lambda: self.start(True))
        self.b_start.pack(side="left", padx=2)
        self.b_prev = ttk.Button(f, text="Preview only", command=lambda: self.start(False))
        self.b_prev.pack(side="left", padx=2)
        self.b_stop = ttk.Button(f, text="Stop", command=self.stop, state="disabled")
        self.b_stop.pack(side="left", padx=2)
        self.b_an = ttk.Button(f, text="Driver analysis", command=self._open_analysis)
        self.b_an.pack(side="left", padx=(10, 2))
        self.v_ov = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="show motion", variable=self.v_ov,
                        command=lambda: self.ctl.update(overlay=self.v_ov.get())
                        ).pack(side="left", padx=8)

    def _btc_panel(self):
        panel = ttk.LabelFrame(
            self.root,
            text="Bitcoin HIGH/LOW oracle — logs calls; releases on the next even frame"
        )
        panel.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

        self.btc_price_var = tk.StringVar(value="BTC/USDT: --")
        ttk.Label(panel, textvariable=self.btc_price_var,
                  font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=8, pady=6)

        ttk.Button(panel, text="Check now", command=self.btc_refresh).pack(side="left", padx=4)

        self.btc_call_var = tk.StringVar(value="After next check: --")
        ttk.Label(panel, textvariable=self.btc_call_var,
                  font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=10)

        self.btc_period_var = tk.StringVar(value="10")
        ttk.Label(panel, text="Check every").pack(side="left", padx=(12, 2))
        ttk.Spinbox(panel, from_=1, to=3600, width=5, textvariable=self.btc_period_var).pack(side="left")
        ttk.Label(panel, text="s").pack(side="left", padx=(2, 8))

        self.btc_auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Auto-check", variable=self.btc_auto_var,
                        command=self.btc_toggle_auto).pack(side="left", padx=4)

        self.btc_result_var = tk.StringVar(
            value="Auto-check off. Each factual check logs a HIGH/LOW call and arms one release for the next even frame."
        )
        ttk.Label(panel, textvariable=self.btc_result_var, wraplength=720).pack(
            side="left", padx=12, pady=6, fill="x", expand=True
        )

    def btc_toggle_auto(self):
        if self.btc_auto_var.get():
            self.btc_auto_running = True
            self._log("BTC HIGH/LOW oracle: automatic checking enabled.")
            self.btc_refresh()
            self.btc_schedule_auto()
        else:
            self.btc_auto_running = False
            if self.btc_auto_job is not None:
                try: self.root.after_cancel(self.btc_auto_job)
                except Exception: pass
            self.btc_auto_job = None
            self.ctl["btc_release_armed"] = False
            self._log("BTC HIGH/LOW oracle: automatic checking disabled; armed release cancelled.")
            self.btc_result_var.set("Auto-check stopped. No further oracle releases.")

    def btc_schedule_auto(self):
        if not self.btc_auto_running: return
        try: seconds = max(1, int(float(self.btc_period_var.get())))
        except Exception: seconds = 10
        self.btc_auto_job = self.root.after(seconds * 1000, self.btc_auto_tick)

    def btc_auto_tick(self):
        if not self.btc_auto_running: return
        self.btc_refresh()
        self.btc_schedule_auto()

    def btc_refresh(self):
        try:
            price, result, factual, changed = self.btc.fetch_price()
            self.btc_price_var.set(f"BTC/USDT: ${price:,.2f}")
            call = self.btc.last_prediction["guess"]
            self.btc_call_var.set(
                f"After next check: {call} "
                f"(from ${self.btc.last_prediction['made_at_price']:,.2f})"
            )
            self.ctl["btc_last_call"] = call
            self.ctl["btc_last_price"] = price

            checks = self.btc.last_checks
            a, b = checks["driver_a"], checks["driver_b"]
            change = "price changed" if changed else "price unchanged"

            if factual:
                self.ctl["btc_release_armed"] = True
                self.btc_result_var.set(
                    f"FACTUAL ({change}); Driver B={b['driver_b']}, Driver A={a['driver_a']}; "
                    f"oracle call {call}; release armed for next even frame."
                )
                self._log(
                    f"BTC HIGH/LOW ORACLE CHECK: ${price:,.2f} ({change}); "
                    f"Driver A={a['driver_a']}, Driver B={b['driver_b']}; "
                    f"oracle call={call}; release armed for next even frame."
                )
            else:
                self.ctl["btc_release_armed"] = False
                self.btc_result_var.set(
                    f"NOT FACTUAL ({change}); Driver B={b['driver_b']}, "
                    f"Driver A={a['driver_a']}; no oracle release armed."
                )
                self._log(
                    f"BTC HIGH/LOW ORACLE REJECTED: ${price:,.2f} ({change}); "
                    f"Driver A={a['driver_a']}, Driver B={b['driver_b']}; "
                    f"no release armed."
                )

            if result:
                outcome = "CORRECT" if result["correct"] else "WRONG"
                self._log(
                    f"BTC HIGH/LOW ORACLE RESULT: previous call {result['guess']}; "
                    f"price after check was {result['actual_direction']}; "
                    f"${result['previous']:,.2f} -> ${result['actual']:,.2f}; "
                    f"{outcome}; record {self.btc.correct}/{self.btc.total}"
                )

            self._log(
                f"BTC HIGH/LOW ORACLE NEXT CALL: {call} from "
                f"${self.btc.last_prediction['made_at_price']:,.2f}; "
                f"totals checks={self.btc.fresh_count}, factual={self.btc.factual_count}, "
                f"rejected={self.btc.rejected_count}, released={self.btc.released_count}, "
                f"calls={self.btc.correct}/{self.btc.total}."
            )
        except Exception as exc:
            self.ctl["btc_release_armed"] = False
            self.btc.last_factual = False
            self.btc_result_var.set(f"Binance check failed: {exc}; no oracle release armed.")
            self._log(f"BTC HIGH/LOW ORACLE ERROR: {exc}; no release armed.")

    def btc_guess(self, direction):
        try:
            self.btc.guess(direction)
            call = self.btc.last_prediction["guess"]
            self.btc_call_var.set(
                f"After next check: {call} "
                f"(from ${self.btc.last_prediction['made_at_price']:,.2f})"
            )
            self.ctl["btc_last_call"] = call
            self.ctl["btc_last_price"] = self.btc.pending_price
            self.btc_result_var.set(
                f"Manual oracle call: after the next check, BTC/USDT will be {call} "
                f"relative to ${self.btc.pending_price:,.2f}."
            )
            self._log(
                f"BTC HIGH/LOW ORACLE MANUAL CALL: {call} from "
                f"${self.btc.pending_price:,.2f}."
            )
        except Exception as exc:
            self.btc_result_var.set(f"Bitcoin guess failed: {exc}")
            self._log(f"BTC HIGH/LOW ORACLE ERROR: {exc}")

    def _miner_panel(self):
        panel = ttk.LabelFrame(self.root, text="Time-code sources")
        panel.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

        left = ttk.Frame(panel, width=380, height=270)
        left.pack_propagate(False)
        left.pack(side="left", anchor="n", padx=8, pady=8)

        self.l_miner = ttk.Label(
            left,
            text="Time codes added: 0 (pending 0)\n"
                 "Armed frames (2 evens in a row): 0\n"
                 "Now: waiting for the circuit",
            font=("Consolas", 9), justify="left", anchor="nw", width=52)
        self.l_miner.pack(anchor="w", pady=(4, 0))

        right = ttk.Frame(panel)
        right.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        logf = ttk.Frame(panel)
        logf.pack(side="left", fill="y", padx=(0, 8), pady=8)
        ttk.Label(logf, text="Log", font=("Arial", 9, "bold")).pack(anchor="w")
        self.log = tk.Text(logf, height=14, width=58, font=("Consolas", 8), state="disabled",
                           wrap="none")
        self.log.pack(fill="y", expand=True)
        self._hidden_panel(right)
        self._search_panel(right)
        self._factor_panel(right)

    def _hidden_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Hidden number (spot check after 2 sequential evens)")
        box.pack(fill="x", pady=(8, 0))
        self.hidden_box = box
        row = ttk.Frame(box)
        row.pack(anchor="w", padx=6, pady=(4, 0))
        self.v_hidden_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Enabled", variable=self.v_hidden_on,
                        command=lambda: self.ctl.update(hidden_on=self.v_hidden_on.get())
                        ).pack(side="left")
        ttk.Label(row, text="Range 0 to").pack(side="left", padx=(12, 2))
        self.v_hrange = tk.StringVar(value=str(HIDDEN_RANGE_DEFAULT))
        self.s_hrange = ttk.Spinbox(row, from_=2, to=HIDDEN_RANGE_MAX, increment=100, width=9,
                                    textvariable=self.v_hrange, command=self._hidden_range_changed)
        self.s_hrange.pack(side="left")
        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            self.s_hrange.bind(ev, lambda e: self.root.after_idle(self._hidden_range_changed), add="+")
        ttk.Button(row, text="New secret",
                   command=self._hidden_new_secret).pack(side="left", padx=(12, 2))
        ttk.Button(row, text="Reveal", command=self._hidden_reveal).pack(side="left", padx=2)
        ttk.Label(row, text="Freq Hz").pack(side="left", padx=(12, 2))
        self.v_hhz = tk.StringVar(value=f"{GUESS_HZ_DEFAULT:g}")
        self.s_hhz = ttk.Spinbox(row, from_=GUESS_HZ_LIMITS[0], to=GUESS_HZ_LIMITS[1],
                                 increment=1, width=7, textvariable=self.v_hhz,
                                 command=self._hidden_hz_changed)
        self.s_hhz.pack(side="left")
        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            self.s_hhz.bind(ev, lambda e: self.root.after_idle(self._hidden_hz_changed), add="+")
        hbox, self.l_hidden = fixed_label(box, 480, 92, text="", font=("Consolas", 9),
                                          justify="left", wraplength=480)
        hbox.pack(anchor="w", padx=6, pady=(2, 4))
        set_text(self.l_hidden, "clock idle")

    def _search_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Higher/lower search (binary, 1 guess per 2 sequential evens)")
        box.pack(fill="x", pady=(8, 0))
        self.search_box = box
        row = ttk.Frame(box)
        row.pack(anchor="w", padx=6, pady=(4, 0))
        self.v_search_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Enabled", variable=self.v_search_on,
                        command=lambda: self.ctl.update(search_on=self.v_search_on.get())
                        ).pack(side="left")
        ttk.Label(row, text="Secret bits").pack(side="left", padx=(12, 2))
        self.v_sbits = tk.StringVar(value=str(SEARCH_BITS_DEFAULT))
        self.s_sbits = ttk.Spinbox(row, from_=2, to=SEARCH_BITS_MAX, increment=8, width=6,
                                   textvariable=self.v_sbits, command=self._search_bits_changed)
        self.s_sbits.pack(side="left")
        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            self.s_sbits.bind(ev, lambda e: self.root.after_idle(self._search_bits_changed), add="+")
        ttk.Button(row, text="New secret", command=self._search_new_secret).pack(side="left", padx=(12, 2))
        ttk.Button(row, text="Reveal", command=self._search_reveal).pack(side="left", padx=2)
        sbox, self.l_search = fixed_label(box, 480, 106, text="", font=("Consolas", 9),
                                          justify="left", wraplength=480)
        sbox.pack(anchor="w", padx=6, pady=(2, 4))
        set_text(self.l_search, "waiting for two sequential even states")

    def _search_bits_changed(self, _e=None):
        try:
            n = int(self.v_sbits.get().strip())
        except (ValueError, tk.TclError):
            return
        n = max(2, min(SEARCH_BITS_MAX, n))
        if n != self.search.bits:
            self.search.set_bits(n)
            self._show_search(self.search.snapshot())
            self._log(f"# search: {n}-bit secret, new secret drawn, counts reset")

    def _search_new_secret(self):
        self.search.new_secret()
        self._show_search(self.search.snapshot())
        self._log("# search: new secret drawn, counts reset")

    def _search_reveal(self):
        v = self.search.reveal()
        self._log(f"# search secret revealed: {short_int(v, 40)}")
        self._show_search(self.search.snapshot())

    def _show_search(self, h):
        if not h:
            return
        last = "none yet" if h["last"] is None else f"{h['last'][0]}"
        avg = "n/a" if h["avg"] is None else f"{h['avg']:.1f}"
        sec = "hidden" if h["secret"] is None else h["secret"]
        lf = h["last_found"]
        found = ("none yet" if lf is None else
                 f"{short_int(lf[0], 16)} in {lf[1]} guesses")
        set_text(self.l_search, (
            f"secret: {h['bits']} bits   blind guessing would need ~2^{h['bits']} tries\n"
            f"guesses: {h['guesses']} total, {h['round_guesses']} on this secret   "
            f"found: {h['solves']}   avg guesses/found: {avg}\n"
            f"pooled: {h['pooled_n']}/{h['bits']} bits   final checks: {h['final_checks']}   last: {last}\n"
            f"last FOUND number: {found}\n"
            f"current secret: {sec}"))

    def _factor_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Factoriser (higher/lower bit search; time code only after a verified factorisation)")
        box.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(box)
        row.pack(anchor="w", padx=6, pady=(4, 0))
        self.v_fs_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Enabled", variable=self.v_fs_on,
                        command=lambda: self.ctl.update(fsearch_on=self.v_fs_on.get())).pack(side="left")
        ttk.Label(row, text="N =").pack(side="left", padx=(12, 2))
        self.e_fn = ttk.Entry(row, width=46)
        self.e_fn.pack(side="left")
        self.e_fn.bind("<Return>", lambda e: self._factor_go())
        ttk.Button(row, text="Factor N", command=self._factor_go).pack(side="left", padx=(8, 2))
        self.v_fhl = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="high/low only", variable=self.v_fhl).pack(side="left", padx=(4, 2))
        ttk.Button(row, text="Cancel", command=self._factor_cancel).pack(side="left", padx=2)
        ttk.Button(row, text="Save numbers", command=self._factor_save).pack(side="left", padx=2)
        row2 = ttk.Frame(box)
        row2.pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(row2, text="or random semiprime, bits").pack(side="left")
        self.v_fbits = tk.StringVar(value=str(FACTOR_BITS_DEFAULT))
        ttk.Spinbox(row2, from_=16, to=FACTOR_BITS_MAX, increment=64, width=6,
                    textvariable=self.v_fbits).pack(side="left", padx=(4, 6))
        ttk.Button(row2, text="Generate", command=self._factor_generate).pack(side="left", padx=2)
        ttk.Button(row2, text="Reveal factor", command=self._factor_reveal).pack(side="left", padx=2)
        fbox, self.l_factor = fixed_label(box, 480, 118, text="", font=("Consolas", 9),
                                          justify="left", wraplength=480)
        fbox.pack(anchor="w", padx=6, pady=(2, 4))
        set_text(self.l_factor, "type N and press Factor N, or generate a semiprime")

    def _factor_go(self):
        try:
            n = parse_number(self.e_fn.get())
        except ValueError as e:
            messagebox.showerror("Number", str(e))
            return
        if self.v_fhl.get():
            self.fsearch.search_number(n)
        else:
            self.fsearch.factor_number(n)

    def _factor_generate(self):
        try:
            bits = int(self.v_fbits.get().strip())
        except ValueError:
            return
        self.fsearch.generate(bits)

    def _factor_save(self):
        data = self.fsearch.export()
        if not data["history"] and not data["current"]["n"]:
            messagebox.showinfo("Save numbers", "No numbers yet: type N and press Factor N, or Generate.")
            return
        path = filedialog.asksaveasfilename(
            title="Save numbers", defaultextension=".json", initialfile="factor_numbers.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            messagebox.showerror("Save numbers", str(e))
            return
        self._log(f"# saved {len(data['history'])} number record(s) to {path}")

    def _factor_cancel(self):
        self.fsearch.cancel()

    def _factor_reveal(self):
        self.fsearch.reveal()
        v = self.fsearch.snapshot()["secret"]
        self._log(f"# factor revealed: {v}" if v else "# factor not known yet")

    def _factor_tick(self):
        for s in self.fsearch.drain():
            self._log(s)
        self._show_factor(self.fsearch.snapshot())
        self.root.after(300, self._factor_tick)

    def _show_factor(self, h):
        st = {"idle": "idle", "generating": "building a semiprime", "factoring": "ENGINE RUNNING: oracle not ready",
              "ready": "READY: pooling bits on armed frames", "solved": "SOLVED and verified",
              "prime": "N is PRIME: nothing to factor, no oracle (type a composite)", "failed": "failed", "cancelled": "cancelled"}.get(h["state"], h["state"])
        nn = "-" if not h["n"] else short_dec(h["n"], 12) + f", {h['n'].bit_length()} bit"
        lines = [f"state: {st}",
                 f"N: {nn}   mode: {h['mode'] or '-'}",
                 f"pooled {h['pooled_n']}/{h['width']} bits   queries {h['guesses']}   "
                 f"final checks {h['final_checks']}   verified solves {h['solves']}"]
        if h["progress"]:
            lines.append("engine: " + h["progress"])
        elif h["last"]:
            lines.append("last: " + h["last"])
        lf = h["last_found"]
        if lf:
            lines.append(f"p = {short_dec(lf[0], 12)}   q = {short_dec(lf[1], 12)}   ({lf[2]} queries)")
        if h["secret"]:
            lines.append("oracle factor: " + h["secret"])
        set_text(self.l_factor, "\n".join(lines))

    def _hidden_range_changed(self, _e=None):
        try:
            n = int(self.v_hrange.get().strip())
        except (ValueError, tk.TclError):
            return
        n = max(2, min(HIDDEN_RANGE_MAX, n))
        if n != self.hidden.size:
            self.hidden.set_range(n)
            self._show_hidden(self.hidden.snapshot())
            self._log(f"# hidden number: range 0..{n - 1}, new secret drawn, counts reset")

    def _hidden_hz_changed(self, _e=None):
        try:
            hz = float(self.v_hhz.get().strip())
        except (ValueError, tk.TclError):
            return
        lo, hi = GUESS_HZ_LIMITS
        hz = max(lo, min(hi, hz))
        if hz != self.hidden.target_hz:
            self.hidden.set_hz(hz)
            self._log(f"# hidden number: clock frequency set to {hz:g} Hz")

    def _hidden_new_secret(self):
        self.hidden.new_secret()
        self._show_hidden(self.hidden.snapshot())      # show the zeroed counts at once
        self._log("# hidden number: new secret drawn, counts and luck reset")

    def _hidden_reveal(self):
        v = self.hidden.reveal()
        self._log(f"# hidden number revealed: {v}")

    def _show_hidden(self, h):
        if not h:
            return
        lc = h["last_check"]
        last = "none yet" if lc is None else f"candidate {lc[0]} -> {'HIT' if lc[1] else 'miss'}"
        sec = "hidden" if h["secret"] is None else str(h["secret"])
        if h["luck"] is None:
            luck = "luck factor: n/a (no spot checks yet)"
        else:
            luck = (f"luck factor: {h['luck']:.0f}%  (100% = average)   "
                    f"P(>= {h['hits']} hits by chance) = {h['luck_tail']:.1%}")
        lf = h["last_found"]
        found = "none yet" if lf is None else f"{lf[0]} (guessed on check #{lf[1]})"
        set_text(self.l_hidden, (
            f"clock: candidate {h['candidate']:>7d} of 0..{h['size'] - 1}   {h['hz']:7.1f} Hz (set)\n"
            f"spot checks: {h['checks']}   hits: {h['hits']}   expected ~{h['expected']:.2f}\n"
            f"{luck}\n"
            f"last check: {last}   secret: {sec}\n"
            f"last FOUND number: {found}"))

    def _show_miner(self, m):
        self._show_hidden(m.get("hidden"))
        self._show_search(m.get("search"))
        set_text(self.l_miner, (
            f"Time codes added: {m['timecodes_added']} (hidden-number: {m['hidden_timecodes']}, "
            f"search: {m['search_timecodes']}, factor: {m['factor_timecodes']}, "
            f"pending {m['timecode_pending']})\n"
            f"Armed frames (2 evens in a row): {m['armed_frames']}\n"
            f"{'ARMED' if m['last_armed'] else 'not armed'}"
            f"{'' if m['circuit'] else '  [circuit OFF: press Start circuit]'}"))

    def _y_changed(self, _e=None):
        try:
            y, info = parse_code(self.e_y.get())
        except ValueError as err:
            self.l_y.config(text=f"invalid: {err}", foreground="#c33")
            self.y_ok = False
            return
        self.y_ok = True
        self.ctl["y"] = y
        self.code_info = info
        note = ""
        if info["carries"]:
            note = f"  ({info['carries']} carries" + (", top wrapped)" if info["wrapped"] else ")")
        self.l_y.config(text=f"= {y}{note}", foreground="#222")
        self._refresh_analysis()

    def _open_analysis(self):
        if self.an_win is not None:
            self.an_win.lift()
            return
        w = tk.Toplevel(self.root)
        w.title("Driver analysis: predictive properties of A, B and Q")
        self.an_label = ttk.Label(w, text="", font=("Consolas", 9), justify="left")
        self.an_label.pack(padx=10, pady=8, anchor="w")
        ttk.Button(w, text="Reset counters",
                   command=lambda: self.ctl.update(reset_an=True)).pack(pady=(0, 8))
        self.an_win = w
        w.protocol("WM_DELETE_WINDOW", self._close_analysis)
        self._show_analysis()

    def _close_analysis(self):
        if self.an_win is not None:
            self.an_win.destroy()
        self.an_win = None

    def _show_analysis(self):
        if self.an_win is not None and self.an_win.winfo_exists():
            set_text(self.an_label, "\n".join(self.an_lines))

    def _refresh_analysis(self, S=None):
        """Show what the time code does to the error bound (Thm 1-4)."""
        if not hasattr(self, "analysis"):
            return
        Y, k = self.ctl["y"], self.ctl["k"]
        m, ones = len(Y), Y.count("1")
        L = [f"Time code Y = {Y}   open {ones}/{m}  (density {ones / m:.2f})"]
        ci = self.code_info
        if ci["carries"]:
            L.append(f"integer add: {ci['carries']} carry(ies), each loses one open bin (Thm 3c)")
        if self.ctl.get("force"):
            L += ["FORCE mode: D = (A.B) OR Y.", "A forced bin gives Q=1 for ANY x, so x not in L",
                  "is accepted: the one-sided guarantee is lost (Thm 4).",
                  "Watch 'false accepts' in the stats."]
        else:
            nmin, nmax = window_stats(Y, k)
            L.append(f"window k={k}: open bins worst N={nmin}, best {nmax}")
            L.append("false accept: impossible for any Y (Thm 1)")
            if nmin == 0:
                L.append("false reject: some window has NO open bin, no bound")
            else:
                L.append(f"false reject <= (1-p)^N; p=1/2: 2^-{nmin} = {0.5 ** nmin:.2g}")
                if S and S["open_inL"] >= 20:
                    ph = S["acc_open_inL"] / S["open_inL"]
                    L.append(f"measured p={ph:.3f} (n={S['open_inL']}): (1-p)^N = {(1 - ph) ** nmin:.2g}")
            fn = frames_needed(Y, 20)
            L.append(f"frames for error <= 2^-20 at p=1/2: {fn if fn else 'never'}")
        set_text(self.analysis, "\n".join(L))

    # ---------------------------------------------------------- diagram
    def _box(self, key, x0, y0, x1, y1, title, sub=""):
        c = self.c
        r = c.create_rectangle(x0, y0, x1, y1, fill=OFF, outline="#666", width=1.5)
        c.create_text((x0 + x1) / 2, y0 + 11, text=title, font=("Arial", 9, "bold"))
        s = c.create_text((x0 + x1) / 2, y1 - 11, text=sub, font=("Arial", 8), fill="#444")
        self.boxes[key] = (r, s)

    def _wire(self, key, *pts, arrow=False):
        w = self.c.create_line(*pts, fill=W_OFF, width=2, arrow="last" if arrow else None)
        self.wires.setdefault(key, []).append(w)

    def _diagram(self):
        c = self.c
        c.create_text(250, 10, text="USB camera  \u2192  frame difference (motion)",
                      font=("Arial", 9, "bold"), fill="#a33")
        c.create_text(90, 24, text="Detector B: motion mask", font=("Arial", 8))
        c.create_text(250, 24, text="camera + motion", font=("Arial", 8))
        c.create_text(410, 24, text="Detector A: motion energy", font=("Arial", 8))
        self.img_b = c.create_image(26, 34, anchor="nw")
        self.img_c = c.create_image(186, 34, anchor="nw")
        self.img_a = c.create_image(346, 34, anchor="nw")
        c.create_line(186, 82, 154, 82, arrow="last", fill="#d9480f")
        c.create_line(314, 82, 346, 82, arrow="last", fill="#d9480f")

        self._box("drvB", 30, 180, 150, 230, "Driver B", "deterministic")
        self._box("drvA", 350, 180, 470, 230, "Driver A", "randomized")
        self.x_text = c.create_text(250, 198, text="x = 0", font=("Arial", 12, "bold"))
        c.create_text(250, 218, text="moving pixels", font=("Arial", 7), fill="#666")
        c.create_line(170, 205, 150, 205, arrow="last")
        c.create_line(330, 205, 350, 205, arrow="last")
        c.create_line(90, 130, 90, 180, arrow="last")
        c.create_line(410, 130, 410, 180, arrow="last")

        self._box("link", 30, 262, 190, 332, "IF/THEN  (C + NPN)", "V(C)=0.00")
        self._wire("B", 90, 230, 90, 262, arrow=True)
        self._box("or", 300, 262, 370, 296, "OR")
        self._box("and", 300, 310, 370, 344, "AND")
        self._wire("A", 410, 230, 410, 247, 285, 247, 285, 319)
        self._wire("A", 285, 272, 300, 272)
        self._wire("A", 285, 319, 300, 319)
        self._wire("B", 190, 297, 240, 297, 240, 286, 300, 286)
        self._wire("B", 240, 297, 240, 336, 300, 336)
        self._wire("gate", 370, 279, 392, 279, 392, 327, 370, 327)
        self._wire("gate", 392, 303, 415, 303)
        self._box("node", 415, 280, 515, 330, "10\u00b5F node", "0.00")
        self._wire("node", 465, 330, 465, 352, arrow=True)
        self._box("Y", 415, 352, 515, 400, "Y time code", "")
        self._wire("D", 515, 376, 540, 376, 540, 262, 560, 262, arrow=True)
        self._box("ff", 560, 232, 660, 332, "74HC74", "")
        c.create_text(575, 262, text="D", font=("Arial", 8))
        c.create_text(575, 312, text="CLK", font=("Arial", 8))
        self._box("clk", 560, 352, 660, 400, "Clock", "= camera frame")
        c.create_line(610, 352, 610, 332, arrow="last")
        c.create_text(676, 247, text="SELECT", font=("Arial", 8, "bold"))
        self.lamp = c.create_oval(668, 258, 686, 276, fill="#ddd", outline="#555")
        c.create_text(676, 290, text="Q", font=("Arial", 8))
        c.create_text(676, 312, text="Q\u0304", font=("Arial", 8), fill="#666")

    def _lit(self, key, on, sub=None, color=ON):
        r, s = self.boxes[key]
        fill = color if on else OFF
        if self._photo_cache.get(("fill", key)) != fill:
            self._photo_cache[("fill", key)] = fill
            self.c.itemconfig(r, fill=fill)
        if sub is not None and self._photo_cache.get(("sub", key)) != sub:
            self._photo_cache[("sub", key)] = sub
            self.c.itemconfig(s, text=sub)

    def _wire_state(self, key, on):
        on = bool(on)
        if self._photo_cache.get(("wire", key)) is on:
            return
        self._photo_cache[("wire", key)] = on
        for w in self.wires.get(key, []):
            self.c.itemconfig(w, fill=W_ON if on else W_OFF)

    # ---------------------------------------------------------- right panel
    def _right_panel(self):
        f = ttk.Frame(self.root)
        f.grid(row=1, column=1, sticky="n", padx=6, pady=4)
        self.right_col = f
        self.plot1_label = ttk.Label(f, text="IF/THEN link during the latest frame",
                                     font=("Arial", 9, "bold"))
        self.plot1_label.pack(anchor="w")
        self.plot1 = tk.Canvas(f, width=420, height=170, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.plot1.pack()
        self.plot2_label = ttk.Label(f, text="Moving pixels per frame (x)",
                                     font=("Arial", 9, "bold"))
        self.plot2_label.pack(anchor="w", pady=(6, 0))
        self.plot2_label_anchor = self.plot2_label
        self.plot2 = tk.Canvas(f, width=420, height=90, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.plot2.pack()
        ttk.Label(f, text="Q per frame: green=1, grey=0, hollow=no motion, red edge=Y blocked",
                  font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 0))
        self.strip = tk.Canvas(f, width=420, height=34, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.strip.pack()
        vbox, self.verdict = fixed_label(f, 420, 44, text="Idle", font=("Arial", 11, "bold"),
                                         wraplength=420)
        vbox.pack(anchor="w", pady=(6, 0))
        sbox, self.stats = fixed_label(f, 420, 96, text="", font=("Consolas", 9),
                                       justify="left", wraplength=420)
        sbox.pack(anchor="w")

    def _draw_traj(self):
        p = self.plot1
        key = ("traj", id(self.last_traj), len(self.last_traj), round(self.ctl["min_px"]))
        if self._photo_cache.get("traj_key") == key:
            return
        self._photo_cache["traj_key"] = key
        p.delete("all")
        W, H, m = 420, 170, 28
        top, bot = 8, H - 20
        sy = lambda v: bot - v / 5.3 * (bot - top)
        for v in range(6):
            p.create_line(m, sy(v), W - 6, sy(v), fill="#eee")
            p.create_text(14, sy(v), text=str(v), font=("Arial", 7))
        for v, col, lab in ((V_BE, "#888", "0.7 V NPN on"), (V_IH, "#c33", "3.5 V logic HIGH")):
            p.create_line(m, sy(v), W - 6, sy(v), fill=col, dash=(4, 3))
            p.create_text(W - 8, sy(v) - 6, text=lab, anchor="e", font=("Arial", 7), fill=col)
        tr = self.last_traj
        if len(tr) > 1:
            tmax = tr[-1][0]
            sx = lambda t: m + t / tmax * (W - m - 6)
            for k, col, lab in ((1, "#1f6fb5", "V(C)"), (2, "#2a9d3a", "V(B)")):
                pts = []
                for row in tr:
                    pts += [sx(row[0]), sy(row[k])]
                p.create_line(*pts, fill=col, width=2)
            p.create_text(m + 6, H - 8, text="\u25cf V(C)", anchor="w", fill="#1f6fb5", font=("Arial", 8))
            p.create_text(m + 70, H - 8, text="\u25cf V(B)", anchor="w", fill="#2a9d3a", font=("Arial", 8))
            p.create_text(W - 8, H - 8, text=f"frame period {tmax:.0f} ms; 1.83RC = {LOGIC_TIME*1000:.1f} ms",
                          anchor="e", font=("Arial", 8), fill="#555")

    def _draw_x(self):
        p = self.plot2
        p.delete("all")
        W, H, m = 420, 90, 6
        if len(self.x_hist) < 2:
            return
        ymax = max(100, max(self.x_hist) * 1.1)
        sy = lambda v: H - 6 - v / ymax * (H - 12)
        step = (W - 2 * m) / (self.x_hist.maxlen - 1)
        thr = self.ctl["min_px"]
        p.create_line(m, sy(thr), W - m, sy(thr), fill="#c33", dash=(4, 3))
        p.create_text(W - m, sy(thr) - 6, text=f"min {thr}", anchor="e", fill="#c33", font=("Arial", 7))
        pts = []
        for i, v in enumerate(self.x_hist):
            pts += [m + i * step, sy(v)]
        p.create_line(*pts, fill="#d9480f", width=2)
        p.create_text(m + 2, 8, text=f"max {int(ymax)}", anchor="w", font=("Arial", 7), fill="#666")

    def _draw_strip(self):
        s = self.strip
        key = tuple(self.q_hist)
        if self._photo_cache.get("strip_key") == key:
            return
        self._photo_cache["strip_key"] = key
        s.delete("all")
        for i, (q, y, mo) in enumerate(self.q_hist):
            row, col = divmod(i, 42)
            x0, y0 = 2 + col * 10, 4 + row * 14
            if not mo:
                s.create_rectangle(x0, y0, x0 + 8, y0 + 8, outline="#aaa")
                continue
            s.create_rectangle(x0, y0, x0 + 8, y0 + 8,
                               fill="#3cb043" if q else "#d8d8d8",
                               outline="#c33" if y == 0 else "#888")

    # ---------------------------------------------------------- run control
    def start(self, circuit=True):
        self._y_changed()
        if not self.y_ok:
            messagebox.showerror("Time code", "Fix the time code first (bits 0/1 with | & + only).")
            return
        self.ctl.update(stop=False, circuit=circuit, timecode_pending=0)
        self.q = queue.Queue()
        self.q_hist.clear()
        self.x_hist.clear()
        self.last_traj = []
        self.err = False
        self._log_clear()
        set_text(self.verdict, "Starting ...")
        self.thread = SimThread(int(self.s_cam.get()), self.ctl, self.q)
        self.thread.start()
        self.b_start.config(state="disabled")
        self.b_prev.config(state="disabled")
        self.b_stop.config(state="normal")

    def stop(self):
        self.ctl["stop"] = True

    def _close(self):
        self.ctl["stop"] = True
        self.fsearch.cancel()
        self.hidden.stop()
        self.root.after(150, self.root.destroy)

    # ---------------------------------------------------------- events
    def _poll(self):
        last = None
        try:
            for _ in range(400):                       # bounded drain per tick
                ev = self.q.get_nowait()
                kind = ev[0]
                if kind == "frame":
                    d = ev[1]
                    self.x_hist.append(d["x"])
                    if d["circuit"]:
                        self.q_hist.append((d["Q"], d["y_bin"], d["motion"]))
                        self.last_traj = d["traj"] or self.last_traj
                        if d["motion"]:
                            self._frame_log(d)
                    last = d                           # older frames are skipped, not drawn
                elif kind == "codes":
                    self._pending_codes = ev[1]        # keep only the newest
                elif kind == "hidden_hit":
                    hh = ev[1]
                    self._log(f"HIDDEN NUMBER GUESSED frame {hh['frame']:5d} x={hh['x']} "
                              f"candidate={hh['snap']['last_check'][0]} "
                              f"after {hh['snap']['checks']} checks; time code added")
                elif kind == "search_hit":
                    sh = ev[1]
                    avg = "n/a" if sh["avg"] is None else f"{sh['avg']:.1f}"
                    val, n = sh["found"]
                    self._log(f"SEARCH FOUND frame {sh['frame']:5d} x={sh['x']} "
                              f"{sh['bits']}-bit secret found in {n} guesses "
                              f"(avg {avg}); time code added")
                    self._log(f"  number (hex, {val.bit_length()} bit): {val:x}")
                elif kind == "factor_hit":
                    fh = ev[1]
                    p, q, n = fh["found"]
                    self._log(f"FACTORISATION VERIFIED frame {fh['frame']:5d} x={fh['x']} "
                              f"p found in {n} queries; time code added")
                    self._log(f"  p = {short_dec(p, 30)}")
                    self._log(f"  q = {short_dec(q, 30)}")
                elif kind == "analysis":
                    self.an_lines = ev[1]
                    self._show_analysis()
                elif kind == "error":
                    self.err = True
                    self._log("# ERROR: " + ev[1])
                    messagebox.showerror("Camera", ev[1])
                elif kind == "info":
                    self._log("# " + ev[1])
                elif kind == "done":
                    self.b_start.config(state="normal")
                    self.b_prev.config(state="normal")
                    self.b_stop.config(state="disabled")
                    set_text(self.verdict, "Camera not available" if self.err else "Stopped")
        except queue.Empty:
            pass

        now = time.perf_counter()
        if last is not None:
            self._latest = last
        # redraw at most ~30 times per second, always from the newest frame
        if getattr(self, "_latest", None) is not None and now - self._last_render >= FRAME_INTERVAL / 1000:
            d, self._latest = self._latest, None
            self._last_render = now
            self._render(d)
            self._draw_traj()
            self._draw_x()
            self._draw_strip()
        if self._pending_codes is not None and now - self._last_codes >= MINER_INTERVAL:
            self._show_miner(self._pending_codes)
            self._pending_codes = None
            self._last_codes = now
        self._flush_log()
        self.root.after(FRAME_INTERVAL // 2, self._poll)

    def _render(self, d):
        g = d["small"]
        prev_rgb = np.stack([g, g, g], axis=-1).copy()
        prev_rgb[d["mask"]] = (255, 40, 40)
        m = d["mask"].astype(np.uint8)
        mask_rgb = np.stack([m * 255, m * 115, m * 40], axis=-1)
        e = np.clip(d["energy"].astype(np.int16) * 4, 0, 255).astype(np.uint8)
        energy_rgb = np.stack([e, e, e], axis=-1)
        self.photos = [ppm_photo(mask_rgb), ppm_photo(prev_rgb), ppm_photo(energy_rgb)]
        self.c.itemconfig(self.img_b, image=self.photos[0])
        self.c.itemconfig(self.img_c, image=self.photos[1])
        self.c.itemconfig(self.img_a, image=self.photos[2])
        xt = f"x = {d['x']}" if d["motion"] else "no motion"
        if self._photo_cache.get("x_text") != xt:
            self._photo_cache["x_text"] = xt
            self.c.itemconfig(self.x_text, text=xt)

        self.cam_photo = ppm_photo(d["preview"])
        self.cam.itemconfig(self.cam_img, image=self.cam_photo)
        # fps is smoothed so the number does not flicker every frame
        fps = 1 / d["dt"]
        self._fps = fps if getattr(self, "_fps", None) is None else 0.9 * self._fps + 0.1 * fps
        ct = f"{self._fps:.0f} fps   " + (f"moving px: {d['x']}" if d["motion"] else "no motion")
        if self._photo_cache.get("cam_text") != ct:
            self._photo_cache["cam_text"] = ct
            self.cam.itemconfig(self.cam_text, text=ct)
        if not d["circuit"]:
            set_text(self.verdict, "Preview only: circuit idle. Adjust the sliders, then Start circuit.")
            return

        self._lit("drvA", d["A"], f"A={d['A']}   r={d['r']}")
        self._lit("drvB", d["c_of_x"], f"c(x) = {d['c_of_x']}")
        self._lit("link", d["B"], f"V(C)={d['vC']:.2f}  V(B)={d['vB']:.2f}", GOOD)
        self._lit("or", d["orv"])
        self._lit("and", d["andv"])
        self._lit("node", d["node"] > 0.5, f"{d['node']:.2f}")
        self._lit("Y", d["y_bin"],
                  f"[{self.ctl['y']}]  bin {d['idx'] % len(self.ctl['y'])}={d['y_bin']}"
                  + (" +HIDDEN" if d.get("hidden_hit") else "")
                  + (" +SEARCH" if d.get("search_hit") else "")
                  + (" +FACTOR" if d.get("factor_hit") else ""))
        self._lit("ff", d["Q"], f"Q = {d['Q']}", GOOD)
        self._lit("clk", False, f"frame {d['idx']}  {round(d['dt'] * 1000, -1):.0f} ms")
        lamp = "#3cb043" if d["select"] else "#ddd"
        if self._photo_cache.get("lamp") != lamp:
            self._photo_cache["lamp"] = lamp
            self.c.itemconfig(self.lamp, fill=lamp)
        self._wire_state("A", d["A"])
        self._wire_state("B", d["B"])
        self._wire_state("gate", d["gate"])
        self._wire_state("node", d["node"] > 0.5)
        self._wire_state("D", d["node"] > 0.5 and d["y_bin"])

        S = d["stats"]
        if d["select"]:
            set_text(self.verdict, f"SELECT = 1  ACCEPT  (witness found within last {self.ctl['k']} frames)")
        elif d["motion"]:
            set_text(self.verdict, f"SELECT = 0  (x = {d['x']}: no witness in last {self.ctl['k']} frames)")
        else:
            set_text(self.verdict, "SELECT = 0  (no motion, no instance)")
        self._refresh_analysis(S)
        rate = S["acc_in_L"] / S["in_L"] if S["in_L"] else 0.0
        set_text(self.stats, (
            f"frames {S['frames']}  with motion {S['motion']}\n"
            f"x in L (odd composite): {S['in_L']}  accepted: {S['acc_in_L']}  ({rate:.0%})\n"
            f"x not in L accepted (false accepts): {S['false_acc']}\n"
            f"time codes added: {S['timecodes_added']} (hidden {S['hidden_timecodes']}, "
            f"search {S['search_timecodes']}, factor {S['factor_timecodes']})"))

    def _frame_log(self, d):
        kind = "odd composite" if d["in_L"] else ("even" if d["x"] % 2 == 0 else "prime/small")
        self._log(f"frame {d['idx']:5d} x={d['x']:<6d} ({kind:<13s}) r={d['r']:<6d} "
                  f"A={d['A']} B={d['B']} node={d['node']:.2f} Y={d['y_bin']} -> Q={d['Q']}")

    # ---------------------------------------------------------- log helpers
    def _log(self, s):
        self._log_buf.append(s)

    def _flush_log(self):
        if not self._log_buf:
            return
        lines, self._log_buf = self._log_buf[-200:], []
        self.log.config(state="normal")
        at_bottom = self.log.yview()[1] >= 0.999
        self.log.insert("end", "\n".join(lines) + "\n")
        n = int(self.log.index("end-1c").split(".")[0])
        if n > 400:
            self.log.delete("1.0", f"{n - 300}.0")
        if at_bottom:
            self.log.see("end")
        self.log.config(state="disabled")

    def _log_clear(self):
        self._log_buf = []
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")


def fit_window(win, content_w=1180, content_h=1180):
    """Open at the content size, but never larger than the usable screen."""
    win.update_idletasks()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w = min(content_w, sw - 40)
    h = min(content_h, sh - 110)              # room for title bar + taskbar
    x = max(0, (sw - w) // 2)
    win.geometry(f"{w}x{h}+{x}+0")
    win.minsize(640, 420)


"""
rp_settings_pack_v2.py  -  ~150 settings + automation for rp_circuit_camera.py

Put next to rp_circuit_camera.py and run:   python rp_settings_pack_v2.py
(The original file is never modified; everything is patched in at run time.)
Replaces rp_settings_pack.py (v1); the settings file is shared and stays compatible.

Tabs
  Motion      thresholds, k, blur size, overlay
  Image       blur type, median, gamma/contrast/brightness, CLAHE/equalize, flicker
              compensation, temporal smoothing, quantize, sharpen, edge-only mode, ROI
              (+invert), border mask, gray channel, mirror/rotate, frame decimation,
              trial-rate limiter                 (all feed the motion detector)
  Camera      backend, FOURCC, size, fps, buffer, exposure/gain/focus/WB/brightness...
  Circuit     VS, V_BE, V_IH, RC, node tau, warm-up frames, witness rounds for Driver A
  Time code   Y, force, target error + Suggest k
  Gating      query gate for hidden/search/factor: min even-run length, every-Nth,
              min gap, probability, max per second, multi-query, reward multipliers
  Sources     hidden/search/factor/BTC options, clock jitter / wander, auto-reset
  Interface   window alpha/topmost, fonts, colors, plots on/off + decimation, history,
              overlay colour/alpha, ROI/grid drawing, alerts, confirm-close
  Automation  autostart, auto-restart, daily schedule, auto-stop, Y rotation, idle
              recalibration, autosave, analysis resets
  Capture     snapshots (on time code / motion / periodic, rotation), CSV recorder,
              stats log, session summary, log file
  Tools       profiles (built-in + yours), grant/clear bins, resets, snapshot now

Dialog: search box filters all settings, per-row reset button, modified settings are
red, 30-step Revert, live status line, Show-modified list.
Hotkeys: F5 start, F6 preview, F7 stop, F8 settings, F9 calibrate, Space start/stop.
"""
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np

import rp_circuit_camera as rp

HOME = os.path.expanduser("~")
SETTINGS_FILE = os.path.join(HOME, ".rp_circuit_settings.json")
NOCAM = -999                                   # "leave the driver default" for camera properties

DEFAULTS = dict(
    # --- Motion
    thr=18, min_px=30, k=5, blur=5, overlay=True,
    # --- Image
    blur_type="gaussian", interp="area", median_pre=False,
    brightness=0, contrast=1.0, gamma=1.0, equalize=False, clahe=False, clahe_clip=2.0,
    flicker_comp=False, flicker_strength=100, temporal_keep=0.0, quant_bits=0, sharpen=0.0,
    edge_mode="off", roi_x0=0, roi_y0=0, roi_x1=100, roi_y1=100, roi_invert=False,
    border_px=0, gray_source="gray", mirror="none", rotate="0",
    frame_decimate=1, max_trial_hz=0.0,
    # --- Camera
    cam_index=0, cam_backend="auto", cam_fourcc="default", cam_w=640, cam_h=480, cam_fps=0,
    cam_buffer=0, cam_retries=3, exposure="default", cam_exposure_val=NOCAM,
    cam_gain=NOCAM, cam_brightness=NOCAM, cam_contrast=NOCAM, cam_saturation=NOCAM,
    cam_focus=NOCAM, cam_wb_temp=NOCAM, autofocus="default", auto_wb="default",
    # --- Circuit
    vs=5.0, v_be=0.7, v_ih=3.5, rc_b_ms=4.0, tau_node_ms=4.0, warmup=8, witness_rounds=1,
    # --- Time code
    y="1111", force=False, target_err_bits=20,
    # --- Gating
    gate_min_evens=2, gate_every_n=1, gate_min_gap_ms=0, gate_prob_pct=100,
    gate_max_per_sec=0, gate_hidden=True, gate_search=True, gate_factor=True,
    multi_query=1, reward_hidden=1, reward_search=1, reward_factor=1, pending_cap=0,
    # --- Sources
    hidden_on=True, hidden_range=100, hidden_hz=17.0, hidden_jitter_pct=0.0,
    hidden_wander=False, hidden_wander_min=20.0, hidden_wander_max=2000.0,
    hidden_autoreset_min=0, search_on=True, search_bits=1000,
    fsearch_on=True, factor_bits=256, factor_hl=True, btc_period=10, btc_auto_on_start=False,
    # --- Interface
    fps_cap=30, compact=False, window_alpha=100, topmost=False, title_prefix="",
    confirm_close=False, log_font=8, log_height=14, analysis_font=8, verdict_font=11,
    draw_traj=True, draw_x=True, draw_strip=True, plot_every_n=1, x_hist_len=150,
    color_wire_on="#d9480f", color_wire_off="#999999", color_box_off="#f1efe8",
    overlay_color="#ff2828", overlay_alpha=60, show_roi=False, show_grid=False,
    preview_gray=False,
    alert_beep=True, alert_false_accept=False, alert_camera_lost=True,
    alert_cooldown_s=2.0, flash_secs=1.5, beep_count=1,
    # --- Automation
    autostart="none", autostart_delay_s=2, auto_restart=False, restart_delay_s=3, restart_max=5,
    sched_on=False, sched_start="09:00", sched_end="17:00",
    stop_codes=0, stop_secs=0, stop_frames=0, idle_warn_s=10, recal_idle_min=0,
    y_rotate=False, y_rotate_s=20, y_rotate_list="1111,1010,1000|0011,11111111",
    reset_an_on_change=True, analysis_reset_min=0, autosave_s=0,
    # --- Capture
    snap_on_code=False, snap_x=0, snap_periodic_s=0, snap_min_gap_s=2.0,
    snap_dir=os.path.join(HOME, "rp_snaps"), snap_max=200, snap_format="png", snap_full=False,
    csv_on=False, csv_path=os.path.join(HOME, "rp_stats.csv"), csv_period_s=10,
    stats_log_s=0, summary_on_stop=True, log_mode="all", log_file=False,
    log_path=os.path.join(HOME, "rp_circuit.log"),
)

BUILTIN_PROFILES = {
    "Quiet room (sensitive)": dict(thr=10, min_px=15, blur=7, k=5),
    "Busy scene (strict)": dict(thr=28, min_px=120, blur=7, k=8),
    "Flicker-proof office": dict(flicker_comp=True, temporal_keep=0.3, blur=7, median_pre=True,
                                 exposure="manual", thr=14),
    "Night / low light": dict(gamma=0.6, clahe=True, blur=9, median_pre=True, thr=12, min_px=25),
    "Edge motion only": dict(edge_mode="sobel", thr=30, min_px=40, blur=5),
    "Door watcher (centre ROI)": dict(roi_x0=25, roi_y0=10, roi_x1=75, roi_y1=100, show_roi=True,
                                      border_px=2),
    "Strict RP demo": dict(y="1111", force=False, k=10, target_err_bits=20, witness_rounds=1),
    "Boosted witness (m=3)": dict(witness_rounds=3, k=3),
    "Break one-sidedness (Thm 4)": dict(force=True, y="1000", k=5),
    "Slow analog (RC stress)": dict(rc_b_ms=12.0, v_be=0.8),
    "Fast hunt (searches)": dict(fps_cap=60, hidden_hz=500.0, hidden_range=100, search_bits=256,
                                 log_mode="minimal", multi_query=4),
    "Even-run strict (4 evens)": dict(gate_min_evens=4, gate_prob_pct=100),
    "Rate-limited sources": dict(gate_every_n=3, gate_max_per_sec=10, gate_min_gap_ms=50),
    "Soak test (unattended)": dict(auto_restart=True, csv_on=True, stats_log_s=60, autosave_s=60,
                                   log_mode="events only", summary_on_stop=True, snap_on_code=True),
    "Low CPU": dict(cam_w=320, cam_h=240, fps_cap=15, compact=True, overlay=False,
                    log_mode="minimal", draw_traj=False, plot_every_n=3, frame_decimate=2),
}

SPEC = []


def R(tab, key, label, kind, extra=None, commit=False, help_=""):
    SPEC.append((tab, key, label, kind, extra, commit, help_))


# ---- Motion
R("Motion", "thr", "Pixel diff >", "int", (4, 60, 1), False, "Per-pixel change that counts as moving.")
R("Motion", "min_px", "Min moving px", "int", (5, 800, 5), False, "Below this: no motion, no instance.")
R("Motion", "k", "k frames (SELECT window)", "int", (1, 30, 1), False, "SELECT = any Q=1 in the last k frames.")
R("Motion", "blur", "Blur kernel (0=off)", "int", (0, 15, 1), False, "Even values round up to odd.")
R("Motion", "overlay", "Show motion overlay", "bool")
# ---- Image
I = "Image"
R(I, "blur_type", "Blur type", "choice", ["gaussian", "median", "box", "bilateral"], False, "Median is best against salt-and-pepper sensor noise.")
R(I, "interp", "Downscale method", "choice", ["area", "linear", "nearest", "cubic"], False, "AREA averages pixels, which suppresses noise.")
R(I, "median_pre", "3x3 median before blur", "bool", None, False, "Kills hot pixels.")
R(I, "brightness", "Brightness offset", "int", (-100, 100, 5), False)
R(I, "contrast", "Contrast gain", "float", (0.2, 3.0, 0.1), False)
R(I, "gamma", "Gamma", "float", (0.3, 3.0, 0.1), False, "<1 brightens shadows (night scenes).")
R(I, "equalize", "Histogram equalise", "bool", None, False, "Takes priority over CLAHE.")
R(I, "clahe", "CLAHE local contrast", "bool")
R(I, "clahe_clip", "CLAHE clip limit", "float", (1.0, 8.0, 0.5))
R(I, "flicker_comp", "Flicker / exposure compensation", "bool", None, False, "Rescales each frame to a slow reference brightness.")
R(I, "flicker_strength", "Flicker comp strength %", "int", (0, 100, 10))
R(I, "temporal_keep", "Temporal smoothing (old-frame weight)", "float", (0.0, 0.95, 0.05), False, "Lowers diff amplitude, so lower 'Pixel diff' to match.")
R(I, "quant_bits", "Quantise to N bits (0=off)", "int", (0, 7, 1), False, "Posterise: ignores tiny level changes.")
R(I, "sharpen", "Sharpen amount", "float", (0.0, 3.0, 0.25))
R(I, "edge_mode", "Edge-only motion", "choice", ["off", "sobel", "laplacian"], False, "Moves only edges, ignoring flat-area noise.")
R(I, "roi_x0", "ROI left %", "int", (0, 99, 5), False, "Motion outside the ROI is ignored.")
R(I, "roi_y0", "ROI top %", "int", (0, 99, 5))
R(I, "roi_x1", "ROI right %", "int", (1, 100, 5))
R(I, "roi_y1", "ROI bottom %", "int", (1, 100, 5))
R(I, "roi_invert", "Invert ROI (ignore inside)", "bool")
R(I, "border_px", "Ignore border (px of 128x96)", "int", (0, 20, 1), False, "Edge vignetting and lens noise.")
R(I, "gray_source", "Gray source", "choice", ["gray", "red", "green", "blue", "hsv-value", "hsv-saturation"], False, "Pick the channel with the best contrast.")
R(I, "mirror", "Mirror", "choice", ["none", "horizontal", "vertical", "both"])
R(I, "rotate", "Rotate", "choice", ["0", "90", "180", "270"])
R(I, "frame_decimate", "Use every Nth camera frame", "int", (1, 10, 1), False, "Larger gaps between compared frames, so slow motion is seen.")
R(I, "max_trial_hz", "Trial-rate limit Hz (0=off)", "float", (0.0, 240.0, 1.0), False, "Caps trials per second (CPU saver).")
# ---- Camera
C = "Camera"
R(C, "cam_index", "Camera number", "int", (0, 9, 1), True, "Camera settings apply on next Start.")
R(C, "cam_backend", "Capture backend", "choice", ["auto", "any", "dshow", "msmf", "v4l2", "avfoundation"], True)
R(C, "cam_fourcc", "Pixel format", "choice", ["default", "MJPG", "YUYV", "H264"], True, "MJPG usually unlocks higher fps.")
R(C, "cam_w", "Width", "int", (160, 3840, 160), True)
R(C, "cam_h", "Height", "int", (120, 2160, 120), True)
R(C, "cam_fps", "FPS request (0=driver)", "int", (0, 240, 5), True)
R(C, "cam_buffer", "Buffer frames (0=driver)", "int", (0, 10, 1), True, "1 = lowest latency.")
R(C, "cam_retries", "Open retries", "int", (1, 10, 1), True)
R(C, "exposure", "Exposure mode", "choice", ["default", "manual", "auto"], True, "'manual' stops auto-exposure pumping.")
R(C, "cam_exposure_val", f"Exposure value ({NOCAM}=skip)", "int", (NOCAM, 9999, 1), True)
R(C, "cam_gain", f"Gain ({NOCAM}=skip)", "int", (NOCAM, 9999, 1), True)
R(C, "cam_brightness", f"Brightness ({NOCAM}=skip)", "int", (NOCAM, 9999, 1), True)
R(C, "cam_contrast", f"Contrast ({NOCAM}=skip)", "int", (NOCAM, 9999, 1), True)
R(C, "cam_saturation", f"Saturation ({NOCAM}=skip)", "int", (NOCAM, 9999, 1), True)
R(C, "cam_focus", f"Focus ({NOCAM}=skip)", "int", (NOCAM, 9999, 1), True)
R(C, "autofocus", "Autofocus", "choice", ["default", "on", "off"], True, "Focus hunting looks like motion.")
R(C, "auto_wb", "Auto white balance", "choice", ["default", "on", "off"], True)
R(C, "cam_wb_temp", f"WB temperature ({NOCAM}=skip)", "int", (NOCAM, 10000, 100), True)
# ---- Circuit
K = "Circuit"
R(K, "vs", "Supply VS (V)", "float", (2.0, 5.3, 0.1), False, "Driver B output voltage.")
R(K, "v_be", "NPN V_BE (V)", "float", (0.3, 1.0, 0.05))
R(K, "v_ih", "74HC V_IH (V)", "float", (1.0, 5.0, 0.1), False, "B is HIGH when V(C) - V_BE >= V_IH.")
R(K, "rc_b_ms", "Link RC (ms)", "float", (0.05, 60.0, 0.5), False, "Large RC: B misses within one frame (link timing misses).")
R(K, "tau_node_ms", "10 uF node tau (ms)", "float", (0.05, 100.0, 0.5), False, "Applies on next Start.")
R(K, "warmup", "Warm-up frames", "int", (0, 60, 1), False, "Ignored while auto-exposure settles.")
R(K, "witness_rounds", "Driver A witness rounds m", "int", (1, 16, 1), False, "Extra hashed witnesses per frame, p -> 1-(1-p)^m. Analysis calibration assumes m=1.")
# ---- Time code
T = "Time code"
R(T, "y", "Time code Y", "str", None, False, "Bits with | & +  e.g. 1000|0011")
R(T, "force", "'forces HIGH' as OR", "bool", None, False, "Breaks one-sidedness (Thm 4). Demo only.")
R(T, "target_err_bits", "Target error 2^-E, E =", "int", (1, 60, 1), False, "Used by Suggest k.")
# ---- Gating
G = "Gating"
R(G, "gate_min_evens", "Required sequential evens", "int", (2, 12, 1), False, "Original behaviour = 2.")
R(G, "gate_every_n", "Let every Nth armed frame through", "int", (1, 100, 1))
R(G, "gate_min_gap_ms", "Min gap between queries (ms)", "int", (0, 5000, 10))
R(G, "gate_prob_pct", "Pass probability %", "int", (0, 100, 5))
R(G, "gate_max_per_sec", "Max queries per second (0=off)", "int", (0, 500, 1))
R(G, "gate_hidden", "Gate applies to hidden number", "bool")
R(G, "gate_search", "Gate applies to bit search", "bool")
R(G, "gate_factor", "Gate applies to factoriser", "bool")
R(G, "multi_query", "Queries per armed frame (search/factor)", "int", (1, 64, 1), False, "Pools several bits per armed frame.")
R(G, "reward_hidden", "Bins per hidden-number hit", "int", (1, 20, 1))
R(G, "reward_search", "Bins per search find", "int", (1, 20, 1))
R(G, "reward_factor", "Bins per verified factorisation", "int", (1, 20, 1))
R(G, "pending_cap", "Cap on pending bins (0=off)", "int", (0, 1000, 1))
# ---- Sources
S_ = "Sources"
R(S_, "hidden_on", "Hidden number enabled", "bool")
R(S_, "hidden_range", "Hidden range 0 to", "int", (2, rp.HIDDEN_RANGE_MAX, 100), True, "Resets hidden counts.")
R(S_, "hidden_hz", "Hidden clock Hz", "float", (1.0, 2000.0, 1.0), True)
R(S_, "hidden_jitter_pct", "Clock jitter +/- %", "float", (0.0, 90.0, 5.0), False, "Retuned every 0.5 s around the base Hz.")
R(S_, "hidden_wander", "Clock wanders between min/max", "bool", None, False, "Overrides jitter.")
R(S_, "hidden_wander_min", "Wander min Hz", "float", (1.0, 2000.0, 10.0))
R(S_, "hidden_wander_max", "Wander max Hz", "float", (1.0, 2000.0, 10.0))
R(S_, "hidden_autoreset_min", "New secret every N min (0=off)", "int", (0, 1440, 1), False, "Also zeroes the luck factor.")
R(S_, "search_on", "Bit search enabled", "bool")
R(S_, "search_bits", "Secret bits", "int", (2, rp.SEARCH_BITS_MAX, 8), True, "Resets search counts.")
R(S_, "fsearch_on", "Factoriser enabled", "bool")
R(S_, "factor_bits", "Semiprime bits (Generate)", "int", (16, rp.FACTOR_BITS_MAX, 64), True)
R(S_, "factor_hl", "Typed N: high/low only", "bool")
R(S_, "btc_period", "BTC check every (s)", "int", (1, 3600, 1), True)
R(S_, "btc_auto_on_start", "BTC auto-check on launch", "bool", None, False, "Makes a Binance request at startup.")
# ---- Interface
U = "Interface"
R(U, "fps_cap", "GUI redraw cap (fps)", "int", (5, 60, 5))
R(U, "compact", "Compact layout", "bool")
R(U, "window_alpha", "Window opacity %", "int", (30, 100, 5))
R(U, "topmost", "Always on top", "bool")
R(U, "title_prefix", "Window title prefix", "str", None, True)
R(U, "confirm_close", "Confirm close while running", "bool")
R(U, "log_font", "Log font size", "int", (6, 16, 1))
R(U, "log_height", "Log height (lines)", "int", (4, 50, 1))
R(U, "analysis_font", "Analysis font size", "int", (6, 14, 1))
R(U, "verdict_font", "Verdict font size", "int", (8, 24, 1))
R(U, "draw_traj", "Draw IF/THEN plot", "bool")
R(U, "draw_x", "Draw x plot", "bool")
R(U, "draw_strip", "Draw Q strip", "bool")
R(U, "plot_every_n", "Draw plots every Nth redraw", "int", (1, 30, 1))
R(U, "x_hist_len", "x plot history (frames)", "int", (50, 1000, 50))
R(U, "color_wire_on", "Wire ON colour", "str", None, True, "Any Tk colour, e.g. #00aa55")
R(U, "color_wire_off", "Wire OFF colour", "str", None, True)
R(U, "color_box_off", "Idle box colour", "str", None, True)
R(U, "overlay_color", "Motion overlay colour", "str", None, True)
R(U, "overlay_alpha", "Motion overlay opacity %", "int", (0, 100, 10))
R(U, "show_roi", "Draw ROI rectangle on preview", "bool")
R(U, "show_grid", "Draw thirds grid on preview", "bool")
R(U, "preview_gray", "Gray preview", "bool")
R(U, "alert_beep", "Beep on time code", "bool")
R(U, "alert_false_accept", "Beep on false accept", "bool", None, False, "Hear Thm 4 break in Force mode.")
R(U, "alert_camera_lost", "Beep on unexpected stop", "bool")
R(U, "alert_cooldown_s", "Alert cooldown (s)", "float", (0.0, 60.0, 0.5))
R(U, "flash_secs", "Title flash (s)", "float", (0.0, 10.0, 0.5))
R(U, "beep_count", "Beeps per alert", "int", (1, 5, 1))
# ---- Automation
A_ = "Automation"
R(A_, "autostart", "On launch", "choice", ["none", "preview", "circuit"])
R(A_, "autostart_delay_s", "Autostart delay (s)", "int", (0, 60, 1))
R(A_, "auto_restart", "Auto-restart after unexpected stop", "bool")
R(A_, "restart_delay_s", "Restart delay (s)", "int", (1, 300, 1))
R(A_, "restart_max", "Max restarts", "int", (1, 1000, 1))
R(A_, "sched_on", "Daily schedule", "bool")
R(A_, "sched_start", "Schedule start HH:MM", "str", None, True)
R(A_, "sched_end", "Schedule end HH:MM", "str", None, True, "Overnight windows work (22:00 to 06:00).")
R(A_, "stop_codes", "Auto-stop after N time codes", "int", (0, 100000, 1), False, "0 = off")
R(A_, "stop_secs", "Auto-stop after N seconds", "int", (0, 86400, 10))
R(A_, "stop_frames", "Auto-stop after N frames", "int", (0, 10_000_000, 100))
R(A_, "idle_warn_s", "No-motion warning after s", "int", (0, 3600, 1))
R(A_, "recal_idle_min", "Recalibrate after N idle min", "int", (0, 600, 1), False, "Runs the calibrator when the scene has been still.")
R(A_, "y_rotate", "Rotate Y through list", "bool")
R(A_, "y_rotate_s", "Rotate every (s)", "int", (2, 3600, 1))
R(A_, "y_rotate_list", "Y list (comma separated)", "str")
R(A_, "reset_an_on_change", "Reset analysis when detector/Y changes", "bool", None, False, "Keeps TPR/FPR tables honest.")
R(A_, "analysis_reset_min", "Reset analysis every N min", "int", (0, 1440, 1))
R(A_, "autosave_s", "Autosave settings every N s", "int", (0, 3600, 10))
# ---- Capture
P = "Capture"
R(P, "snap_on_code", "Snapshot on each time code", "bool")
R(P, "snap_x", "Snapshot when x >= (0=off)", "int", (0, 5000, 10))
R(P, "snap_periodic_s", "Snapshot every N s during motion", "int", (0, 3600, 1))
R(P, "snap_min_gap_s", "Min gap between snapshots (s)", "float", (0.0, 600.0, 0.5))
R(P, "snap_dir", "Snapshot folder", "str", None, True)
R(P, "snap_max", "Keep newest N snapshots (0=all)", "int", (0, 100000, 10))
R(P, "snap_format", "Snapshot format", "choice", ["png", "jpg"])
R(P, "snap_full", "Full-resolution snapshot", "bool")
R(P, "csv_on", "Record stats to CSV", "bool")
R(P, "csv_path", "CSV path", "str", None, True)
R(P, "csv_period_s", "CSV row every N s", "int", (1, 3600, 1))
R(P, "stats_log_s", "Log stats line every N s (0=off)", "int", (0, 3600, 5))
R(P, "summary_on_stop", "Session summary on stop", "bool")
R(P, "log_mode", "Log detail", "choice", ["all", "events only", "minimal"])
R(P, "log_file", "Also log to file", "bool")
R(P, "log_path", "Log file path", "str", None, True)

TABS = ["Motion", "Image", "Camera", "Circuit", "Time code", "Gating", "Sources",
        "Interface", "Automation", "Capture", "Tools"]
KINDS = {s[1]: (s[3], s[4]) for s in SPEC}
COMMIT = {s[1] for s in SPEC if s[5]}
Y_PRESETS = ["1111", "1010", "11111111", "10000000", "1000|0011", "1111&1011", "1011+0001"]
RESET_AN_KEYS = {"thr", "min_px", "blur", "y", "force", "witness_rounds", "blur_type", "edge_mode",
                 "roi_x0", "roi_y0", "roi_x1", "roi_y1", "roi_invert", "gray_source", "temporal_keep"}
PACK = None
ST = {}                                      # image-pipeline state (EMA frame, flicker ref, ROI mask)


def cfg():
    return PACK.cfg if PACK else DEFAULTS


def parse_val(kind, s, extra):
    if kind == "bool":
        return bool(s)
    if kind == "int":
        return max(extra[0], min(extra[1], int(float(s))))
    if kind == "float":
        return max(extra[0], min(extra[1], float(s)))
    return str(s)


# ------------------------------------------------------------------ runtime patches
def _roi_mask(c):
    key = (c["roi_x0"], c["roi_y0"], c["roi_x1"], c["roi_y1"], c["roi_invert"], c["border_px"])
    if ST.get("mk") != key:
        x0, x1 = sorted((c["roi_x0"], c["roi_x1"]))
        y0, y1 = sorted((c["roi_y0"], c["roi_y1"]))
        X0, X1 = int(rp.SW * x0 / 100), max(int(rp.SW * x0 / 100) + 1, int(rp.SW * x1 / 100))
        Y0, Y1 = int(rp.SH * y0 / 100), max(int(rp.SH * y0 / 100) + 1, int(rp.SH * y1 / 100))
        inside = np.zeros((rp.SH, rp.SW), bool)
        inside[Y0:Y1, X0:X1] = True
        m = ~inside if c["roi_invert"] else inside
        b = c["border_px"]
        if b > 0:
            m[:b] = m[-b:] = False
            m[:, :b] = m[:, -b:] = False
        ST["mk"], ST["m"], ST["full"] = key, m, bool(m.all())
    return ST["m"], ST["full"]


def _install_core_patches():
    cv2 = rp.cv2
    interps = {}
    if cv2 is not None:
        interps = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR,
                   "nearest": cv2.INTER_NEAREST, "cubic": cv2.INTER_CUBIC}
    luts = {}

    def downscale(gray):
        c = cfg()
        small = cv2.resize(gray, (rp.SW, rp.SH), interpolation=interps[c["interp"]])
        if c["median_pre"]:
            small = cv2.medianBlur(small, 3)
        if c["brightness"] or c["contrast"] != 1.0:
            small = cv2.convertScaleAbs(small, alpha=c["contrast"], beta=c["brightness"])
        g = c["gamma"]
        if g != 1.0:
            lut = luts.get(g)
            if lut is None:
                lut = luts[g] = (255.0 * (np.arange(256) / 255.0) ** (1.0 / g)).astype(np.uint8)
            small = cv2.LUT(small, lut)
        if c["equalize"]:
            small = cv2.equalizeHist(small)
        elif c["clahe"]:
            small = cv2.createCLAHE(clipLimit=c["clahe_clip"], tileGridSize=(8, 8)).apply(small)
        if c["flicker_comp"]:
            m = float(small.mean())
            ref = ST.get("ref")
            ST["ref"] = m if ref is None else ref * 0.98 + m * 0.02
            gain = 1.0 + c["flicker_strength"] / 100.0 * (ST["ref"] / max(m, 1.0) - 1.0)
            if abs(gain - 1.0) > 0.002:
                small = np.clip(small.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        k = int(c["blur"])
        if k >= 3:
            k |= 1
            bt = c["blur_type"]
            if bt == "gaussian":
                small = cv2.GaussianBlur(small, (k, k), 0)
            elif bt == "median":
                small = cv2.medianBlur(small, min(k, 15))
            elif bt == "box":
                small = cv2.blur(small, (k, k))
            else:
                small = cv2.bilateralFilter(small, k, 50, 50)
        if c["sharpen"] > 0:
            soft = cv2.GaussianBlur(small, (0, 0), 1.0)
            small = cv2.addWeighted(small, 1 + c["sharpen"], soft, -c["sharpen"], 0)
        em = c["edge_mode"]
        if em == "sobel":
            gx = cv2.convertScaleAbs(cv2.Sobel(small, cv2.CV_16S, 1, 0))
            gy = cv2.convertScaleAbs(cv2.Sobel(small, cv2.CV_16S, 0, 1))
            small = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
        elif em == "laplacian":
            small = cv2.convertScaleAbs(cv2.Laplacian(small, cv2.CV_16S, ksize=3))
        keep = c["temporal_keep"]
        if keep > 0:
            ema = ST.get("ema")
            ST["ema"] = small.astype(np.float32) if ema is None else keep * ema + (1 - keep) * small
            small = ST["ema"].astype(np.uint8)
        qb = int(c["quant_bits"])
        if 1 <= qb <= 7:
            small = small & np.uint8((0xFF << (8 - qb)) & 0xFF)
        m, full = _roi_mask(c)
        if not full:
            small = np.where(m, small, 0).astype(np.uint8)
        return np.ascontiguousarray(small)
    rp.downscale = downscale

    # ---- camera open and read
    def src_init(self, index):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed.  Run: pip install opencv-python")
        c = cfg()
        names = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "v4l2": "CAP_V4L2",
                 "avfoundation": "CAP_AVFOUNDATION", "any": "CAP_ANY"}
        if c["cam_backend"] == "auto":
            backend = cv2.CAP_DSHOW if rp.sys.platform == "win32" else cv2.CAP_ANY
        else:
            backend = getattr(cv2, names[c["cam_backend"]], cv2.CAP_ANY)
        cap = None
        for attempt in range(int(c["cam_retries"])):
            cap = cv2.VideoCapture(index, backend)
            if c["cam_fourcc"] != "default":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c["cam_fourcc"]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["cam_w"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["cam_h"])
            if c["cam_fps"] > 0:
                cap.set(cv2.CAP_PROP_FPS, c["cam_fps"])
            if c["cam_buffer"] > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, c["cam_buffer"])
            if c["exposure"] != "default":
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if c["exposure"] == "manual" else 0.75)
            if c["autofocus"] != "default":
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if c["autofocus"] == "on" else 0)
            if c["auto_wb"] != "default":
                cap.set(cv2.CAP_PROP_AUTO_WB, 1 if c["auto_wb"] == "on" else 0)
            for key, prop in (("cam_exposure_val", cv2.CAP_PROP_EXPOSURE), ("cam_gain", cv2.CAP_PROP_GAIN),
                              ("cam_brightness", cv2.CAP_PROP_BRIGHTNESS), ("cam_contrast", cv2.CAP_PROP_CONTRAST),
                              ("cam_saturation", cv2.CAP_PROP_SATURATION), ("cam_focus", cv2.CAP_PROP_FOCUS),
                              ("cam_wb_temp", cv2.CAP_PROP_WB_TEMPERATURE)):
                if c[key] != NOCAM:
                    cap.set(prop, c[key])
            if cap.isOpened() and cap.read()[0]:
                break
            cap.release()
            cap = None
            time.sleep(0.5)
        if cap is None:
            raise RuntimeError(
                f"Cannot open camera {index} (backend {c['cam_backend']}, {c['cam_retries']} tries). Check that it is "
                f"connected, not in use by another app, and that the camera number is right.")
        self.cap = cap
        self.rgb = None
        self._tl = 0.0

    def src_read(self):
        c = cfg()
        hz = c["max_trial_hz"]
        if hz > 0:
            wait = self._tl + 1.0 / hz - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            self._tl = time.perf_counter()
        for _ in range(int(c["frame_decimate"]) - 1):
            self.cap.grab()
        ok, bgr = self.cap.read()
        if not ok:
            return None
        mi = c["mirror"]
        if mi != "none":
            bgr = cv2.flip(bgr, {"horizontal": 1, "vertical": 0, "both": -1}[mi])
        if c["rotate"] != "0":
            bgr = cv2.rotate(bgr, {"90": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180,
                                   "270": cv2.ROTATE_90_COUNTERCLOCKWISE}[c["rotate"]])
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if PACK:
            PACK.last_full = self.rgb
        gs = c["gray_source"]
        if gs == "gray":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gs in ("red", "green", "blue"):
            return np.ascontiguousarray(bgr[:, :, {"blue": 0, "green": 1, "red": 2}[gs]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return np.ascontiguousarray(hsv[:, :, 2 if gs == "hsv-value" else 1])
    rp.Source.__init__ = src_init
    rp.Source.read = src_read

    # ---- preview with colour/alpha, ROI box, grid, frame counter
    def make_preview(rgb, mask, overlay):
        c = cfg()
        if PACK:
            PACK.frame_no += 1
        h, w = rgb.shape[:2]
        pv = rgb[np.arange(rp.PH) * h // rp.PH][:, np.arange(rp.PW) * w // rp.PW].copy()
        if c["preview_gray"]:
            gy = (pv @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
            pv = np.stack([gy] * 3, axis=-1)
        if overlay and mask.any():
            big = mask[np.arange(rp.PH) * rp.SH // rp.PH][:, np.arange(rp.PW) * rp.SW // rp.PW]
            a = c["overlay_alpha"] / 100.0
            col = np.array(PACK.ov_rgb if PACK else (255, 40, 40))
            pv[big] = (pv[big] * (1 - a) + col * a).astype(np.uint8)
        if c["show_grid"]:
            for f in (1 / 3, 2 / 3):
                pv[int(rp.PH * f), :] = (200, 200, 200)
                pv[:, int(rp.PW * f)] = (200, 200, 200)
        if c["show_roi"]:
            x0, x1 = sorted((c["roi_x0"], c["roi_x1"]))
            y0, y1 = sorted((c["roi_y0"], c["roi_y1"]))
            X0, X1 = int(rp.PW * x0 / 100), min(rp.PW - 1, int(rp.PW * x1 / 100))
            Y0, Y1 = int(rp.PH * y0 / 100), min(rp.PH - 1, int(rp.PH * y1 / 100))
            yel = (255, 213, 74)
            pv[Y0, X0:X1 + 1] = yel
            pv[Y1, X0:X1 + 1] = yel
            pv[Y0:Y1 + 1, X0] = yel
            pv[Y0:Y1 + 1, X1] = yel
        return pv
    rp.make_preview = make_preview

    # ---- Driver A with m witness rounds (sim thread only; analysis/BTC keep the original)
    orig_mr = rp.miller_rabin_witness

    def mr(a, n):
        if orig_mr(a, n):
            return True
        m = cfg()["witness_rounds"]
        if m > 1 and n > 4 and isinstance(threading.current_thread(), rp.SimThread):
            for i in range(1, m):
                ai = 2 + int.from_bytes(hashlib.sha256(f"{a}|{n}|{i}".encode()).digest(), "big") % (n - 3)
                if orig_mr(ai, n):
                    return True
        return False
    rp.miller_rabin_witness = mr


# ------------------------------------------------------------------ the pack
class Pack:
    def __init__(self, app):
        self.app, self.win = app, app.window
        self.cfg = dict(DEFAULTS)
        self.profiles, self.undo = {}, []
        self.status, self.live = tk.StringVar(value=""), tk.StringVar(value="")
        self.dlg, self.vars, self.rows = None, {}, {}
        self._syncing, self._dirty, self._flush_job = False, set(), None
        self._fh = None
        self.last_S, self.last_m = {}, {}
        self.t_start = self.last_motion = time.time()
        self.last_codes = self._fa = 0
        self.t_rot = self.t_stat = self.t_csv = self.t_save = self.t_anreset = self.t_hreset = time.time()
        self.rot_i = 0
        self.cal, self.cal_started_preview = None, False
        self.base_title = self.win.title()
        self.flash_until = self._last_alert = 0.0
        self.scales = []
        self.frame_no = 0
        self.last_full = None
        self.ov_rgb = (255, 40, 40)
        self.draw_n = 0
        self._snap_req, self._snap_t, self._snap_tp = None, 0.0, 0.0
        self._streak, self._sf = 0, -2
        self._cnt = {"hidden": 0, "search": 0, "factor": 0}
        self._last = {"hidden": 0.0, "search": 0.0, "factor": 0.0}
        self._win_ts = {t: deque() for t in self._cnt}
        self.gstat = {t: dict(passed=0, blocked=0, hits=0) for t in self._cnt}
        self._was_running = False
        self.user_stop, self.last_circuit, self._auto_starting = False, True, False
        self.restarts, self._restart_job = 0, None
        self._inw_prev, self._sched_started = None, False
        self._jit_prev = False
        H = self.H = {}
        a = app
        for i, key in enumerate(("thr", "min_px", "k")):
            H[key] = (lambda v, i=i, key=key: self._scale(i, key, v))
        H.update({
            "blur": self._h_blur, "overlay": self._h_overlay, "cam_index": lambda v: a.s_cam.set(v),
            "y": self._h_y, "force": self._h_force,
            "hidden_on": lambda v: (a.v_hidden_on.set(v), a.ctl.update(hidden_on=v)),
            "hidden_range": lambda v: (a.v_hrange.set(str(v)), a._hidden_range_changed()),
            "hidden_hz": lambda v: (a.v_hhz.set(f"{v:g}"), a._hidden_hz_changed()),
            "search_on": lambda v: (a.v_search_on.set(v), a.ctl.update(search_on=v)),
            "search_bits": lambda v: (a.v_sbits.set(str(v)), a._search_bits_changed()),
            "fsearch_on": lambda v: (a.v_fs_on.set(v), a.ctl.update(fsearch_on=v)),
            "factor_bits": lambda v: a.v_fbits.set(str(v)),
            "factor_hl": lambda v: a.v_fhl.set(v),
            "btc_period": lambda v: a.btc_period_var.set(str(v)),
            "fps_cap": lambda v: setattr(rp, "FRAME_INTERVAL", max(1, int(1000 / v))),
            "compact": lambda v: (a.v_compact.set(v), a._apply_compact()),
            "log_file": lambda v: self._close_fh(), "log_path": lambda v: self._close_fh(),
            "window_alpha": lambda v: self._try(lambda: self.win.attributes("-alpha", v / 100.0)),
            "topmost": lambda v: self._try(lambda: self.win.attributes("-topmost", bool(v))),
            "title_prefix": self._h_title,
            "log_font": lambda v: a.log.config(font=("Consolas", v)),
            "log_height": lambda v: a.log.config(height=v),
            "analysis_font": lambda v: a.analysis.config(font=("Consolas", v)),
            "verdict_font": lambda v: a.verdict.config(font=("Arial", v, "bold")),
            "x_hist_len": lambda v: setattr(a, "x_hist", deque(a.x_hist, maxlen=v)),
            "color_wire_on": lambda v: (self._color(v), setattr(rp, "W_ON", v), self._repaint()),
            "color_wire_off": lambda v: (self._color(v), setattr(rp, "W_OFF", v), self._repaint()),
            "color_box_off": lambda v: (self._color(v), setattr(rp, "OFF", v), self._repaint()),
            "overlay_color": lambda v: setattr(self, "ov_rgb", self._color(v)),
            "warmup": lambda v: setattr(rp, "WARMUP", int(v)),
            "sched_start": self._h_hhmm, "sched_end": self._h_hhmm,
        })
        for key in ("vs", "v_be", "v_ih", "rc_b_ms", "tau_node_ms"):
            H[key] = self._h_circuit

    # ---------------------------------------------------------- small helpers
    def _try(self, fn):
        try:
            fn()
        except tk.TclError:
            pass

    def _color(self, v):
        try:
            return tuple(c >> 8 for c in self.win.winfo_rgb(v))
        except tk.TclError:
            raise ValueError("not a colour")

    def _repaint(self):
        pc = self.app._photo_cache
        for key in [k for k in pc if isinstance(k, tuple) and k[0] in ("wire", "fill")]:
            del pc[key]

    def _h_hhmm(self, v):
        try:
            h, m = v.split(":")
            if not (0 <= int(h) < 24 and 0 <= int(m) < 60):
                raise ValueError
        except ValueError:
            raise ValueError("use HH:MM")

    def _h_title(self, v):
        self.base_title = (v + "  " if v else "") + self._orig_title

    def _scale(self, i, key, v):
        self.app.ctl[key] = int(v)
        if i < len(self.scales):
            self.scales[i].set(v)

    def _h_blur(self, v):
        if v > 0 and v % 2 == 0:
            v += 1
        self.cfg["blur"] = 0 if v < 3 else v

    def _h_overlay(self, v):
        self.app.v_ov.set(v)
        self.app.ctl["overlay"] = v

    def _h_y(self, v):
        rp.parse_code(v)
        self.app.e_y.delete(0, "end")
        self.app.e_y.insert(0, v)
        self.app._y_changed()

    def _h_force(self, v):
        self.app.v_force.set(v)
        self.app.ctl["force"] = v
        self.app._refresh_analysis()

    def _h_circuit(self, _v):
        c = self.cfg
        rp.VS, rp.V_BE, rp.V_IH = c["vs"], c["v_be"], c["v_ih"]
        rp.RC_B, rp.TAU_NODE = c["rc_b_ms"] / 1000.0, c["tau_node_ms"] / 1000.0
        margin = c["vs"] - c["v_ih"] - c["v_be"]
        rp.LOGIC_TIME = rp.RC_B * math.log(c["vs"] / margin) if margin > 0 else float("inf")
        if margin <= 0:
            self.status.set("warning: VS - V_BE <= V_IH, so Driver B can never reach logic HIGH")

    def _find_scales(self):
        out = []

        def walk(w):
            for ch in w.winfo_children():
                if isinstance(ch, ttk.Scale):
                    out.append(ch)
                walk(ch)
        walk(self.app.root)
        return out[:3]

    def _running(self):
        return str(self.app.b_stop["state"]) == "normal"

    # ---------------------------------------------------------- install
    def install(self):
        a = self.app
        self._orig_title = self.win.title()
        self.scales = self._find_scales()
        ttk.Button(a.ctl_row2, text="Settings (F8)", command=self.open_dialog).pack(side="left", padx=10)
        orig_render, orig_show = a._render, a._show_miner
        orig_log, orig_frame = a._log, a._frame_log
        orig_start, orig_stop, orig_close = a.start, a.stop, a._close

        def render(d):
            orig_render(d)
            c = self.cfg
            self.draw_n += 1
            S = d.get("stats") or {}
            if S:
                self.last_S = S
            fa = S.get("false_acc", 0)
            if fa > self._fa and c["alert_false_accept"]:
                self.alert("false accept")
            self._fa = fa
            now = time.time()
            if d["motion"]:
                self.last_motion = now
            if self.cal and self.cal.get("collect"):
                self.cal["samples"].append(d["x"])
            reason = None
            if self._snap_req:
                reason, self._snap_req = self._snap_req, None
            elif d["motion"] and c["snap_x"] and d["x"] >= c["snap_x"]:
                reason = "motion"
            elif d["motion"] and c["snap_periodic_s"] and now - self._snap_tp >= c["snap_periodic_s"]:
                reason = "periodic"
            if reason:
                self._snap(reason, d)
        a._render = render

        def show_miner(m):
            orig_show(m)
            self.last_m = m
            n = m.get("timecodes_added", 0)
            if n > self.last_codes:
                if self.cfg["alert_beep"]:
                    self.alert("time code")
                if self.cfg["snap_on_code"]:
                    self._snap_req = "code"
            self.last_codes = n
        a._show_miner = show_miner

        def log(s):
            if self.cfg["log_mode"] == "minimal" and not any(
                    t in s for t in ("#", "ERROR", "FOUND", "GUESSED", "VERIFIED", "BTC")):
                return
            orig_log(s)
            self._write_file(s)
        a._log = log
        a._frame_log = lambda d: orig_frame(d) if self.cfg["log_mode"] == "all" else None

        def mk_draw(name, key):
            orig = getattr(a, name)

            def f():
                if self.cfg[key] and self.draw_n % self.cfg["plot_every_n"] == 0:
                    orig()
            setattr(a, name, f)
        mk_draw("_draw_traj", "draw_traj")
        mk_draw("_draw_x", "draw_x")
        mk_draw("_draw_strip", "draw_strip")

        def start(circuit=True):
            orig_start(circuit)
            self.user_stop = False
            self.last_circuit = circuit
            if not self._auto_starting:
                self.restarts = 0
            self.t_start = self.last_motion = time.time()
            self.last_S, self.last_m, self.last_codes, self._fa = {}, {}, 0, 0
            self._streak, self._sf = 0, -2
            ST.clear()
        a.start = start

        def stop():
            self.user_stop = True
            orig_stop()
        a.stop = stop
        a.b_start.config(command=lambda: a.start(True))
        a.b_prev.config(command=lambda: a.start(False))
        a.b_stop.config(command=lambda: a.stop())

        def close():
            if self.cfg["confirm_close"] and self._running() and not messagebox.askokcancel(
                    "Close", "The circuit is running. Close anyway?"):
                return
            self.save_file(silent=True)
            self._close_fh()
            orig_close()
        a._close = close
        self.win.protocol("WM_DELETE_WINDOW", close)

        self._wrap_source(a.hidden, "spot_check", "hidden", False)
        self._wrap_source(a.search, "guess", "search", True)
        self._wrap_source(a.fsearch, "guess", "factor", True)

        for key, fn in (("<F5>", lambda e: a.start(True)), ("<F6>", lambda e: a.start(False)),
                        ("<F7>", lambda e: a.stop()), ("<F8>", lambda e: self.open_dialog()),
                        ("<F9>", lambda e: self.calibrate())):
            self.win.bind_all(key, fn)
        self.win.bind_all("<space>", self._space)

        self.load_file(silent=True)
        self.apply(dict(self.cfg), undo=False, force=True)
        self.win.after(500, self._tick)
        if self.cfg["autostart"] != "none":
            self.win.after(int(self.cfg["autostart_delay_s"] * 1000), self._autostart)
        if self.cfg["btc_auto_on_start"]:
            self.win.after(1500, lambda: (a.btc_auto_var.set(True), a.btc_toggle_auto()))

    def _autostart(self):
        if not self._running():
            self._auto_starting = True
            self.app.start(self.cfg["autostart"] == "circuit")
            self._auto_starting = False

    def _space(self, e):
        if e.widget.winfo_class() in ("TEntry", "TSpinbox", "Text", "TCombobox", "Entry", "Spinbox"):
            return
        self.app.stop() if self._running() else self.app.start(True)

    # ---------------------------------------------------------- source gate (sim thread)
    def _gate(self, tag):
        c = self.cfg
        fn = self.frame_no
        if self._sf != fn:
            self._streak = self._streak + 1 if fn == self._sf + 1 else 1
            self._sf = fn
        if not c["gate_" + tag if tag != "hidden" else "gate_hidden"]:
            return True
        if self._streak + 1 < c["gate_min_evens"]:
            return False
        if c["gate_prob_pct"] < 100 and random.random() * 100 >= c["gate_prob_pct"]:
            return False
        now = time.perf_counter()
        if c["gate_min_gap_ms"] and now - self._last[tag] < c["gate_min_gap_ms"] / 1000.0:
            return False
        if c["gate_max_per_sec"]:
            q = self._win_ts[tag]
            while q and now - q[0] > 1.0:
                q.popleft()
            if len(q) >= c["gate_max_per_sec"]:
                return False
        self._cnt[tag] += 1
        if c["gate_every_n"] > 1 and self._cnt[tag] % c["gate_every_n"]:
            return False
        self._last[tag] = now
        self._win_ts[tag].append(now)
        return True

    def _wrap_source(self, obj, name, tag, multi):
        orig = getattr(obj, name)
        reward_key = {"hidden": "reward_hidden", "search": "reward_search", "factor": "reward_factor"}[tag]

        def f():
            if not self._gate(tag):
                self.gstat[tag]["blocked"] += 1
                return False
            self.gstat[tag]["passed"] += 1
            n = self.cfg["multi_query"] if multi else 1
            hit = False
            for _ in range(n):
                if orig():
                    hit = True
                    break
            if hit:
                self.gstat[tag]["hits"] += 1
                extra = self.cfg[reward_key] - 1
                if extra > 0:
                    self.app.ctl["timecode_pending"] = self.app.ctl.get("timecode_pending", 0) + extra
            return hit
        setattr(obj, name, f)

    # ---------------------------------------------------------- apply / undo
    def apply(self, new, undo=True, force=False, from_ui=False):
        changed = {k: v for k, v in new.items() if k in DEFAULTS and (force or self.cfg.get(k) != v)}
        if not changed:
            return
        if undo:
            self.undo.append(dict(self.cfg))
            del self.undo[:-30]
        mod = []
        for k, v in changed.items():
            old = self.cfg.get(k)
            self.cfg[k] = v
            try:
                h = self.H.get(k)
                if h:
                    h(v)
                if self.cfg[k] != v:
                    mod.append(k)
                if k in RESET_AN_KEYS and self.cfg["reset_an_on_change"] and not force:
                    self.app.ctl["reset_an"] = True
            except ValueError as e:
                self.cfg[k] = old
                self.status.set(f"{k}: rejected ({e})")
            except Exception as e:
                self.status.set(f"{k}: {type(e).__name__}: {e}")
        self.sync(mod if from_ui else None)

    def revert(self):
        if not self.undo:
            self.status.set("nothing to revert")
            return
        self.apply(self.undo.pop(), undo=False)
        self.status.set(f"reverted ({len(self.undo)} more steps)")

    def defaults(self):
        self.apply(dict(DEFAULTS))
        self.status.set("defaults restored (undo with Revert)")

    def modified(self):
        return [k for k in DEFAULTS if self.cfg[k] != DEFAULTS[k]]

    def show_modified(self):
        m = self.modified()
        messagebox.showinfo("Modified settings", "\n".join(f"{k} = {self.cfg[k]}   (default {DEFAULTS[k]})"
                                                           for k in m) or "Everything is at its default.",
                            parent=self.dlg)

    # ---------------------------------------------------------- persistence
    def save_file(self, silent=False, path=None):
        path = path or SETTINGS_FILE
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dict(cfg=self.cfg, profiles=self.profiles), f, indent=2)
            if not silent:
                self.status.set(f"saved {path}")
        except OSError as e:
            if not silent:
                messagebox.showerror("Settings", str(e))

    def load_file(self, silent=False, path=None):
        path = path or SETTINGS_FILE
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for k, v in data.get("cfg", {}).items():
            if k in DEFAULTS and (isinstance(v, type(DEFAULTS[k])) or
                                  (isinstance(DEFAULTS[k], float) and isinstance(v, int))) \
                    and not (isinstance(DEFAULTS[k], int) and isinstance(v, bool) != isinstance(DEFAULTS[k], bool)):
                self.cfg[k] = v
        self.profiles.update(data.get("profiles", {}))
        if not silent:
            self.apply(dict(self.cfg), force=True)
            self.status.set(f"loaded {path}")

    def export_snapshot(self):
        path = filedialog.asksaveasfilename(title="Export snapshot", defaultextension=".json",
                                            initialfile="rp_snapshot.json")
        if not path:
            return
        data = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), cfg=self.cfg, stats=self.last_S,
                    codes=self.last_m, gate=self.gstat, x_history=list(self.app.x_hist))
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            self.status.set(f"snapshot saved: {path}")
        except OSError as e:
            messagebox.showerror("Snapshot", str(e))

    # ---------------------------------------------------------- log / csv / snapshots / alerts
    def _write_file(self, s):
        if not self.cfg["log_file"]:
            return
        try:
            if self._fh is None:
                self._fh = open(self.cfg["log_path"], "a", encoding="utf-8")
            self._fh.write(time.strftime("%H:%M:%S ") + s + "\n")
            self._fh.flush()
        except OSError:
            self.cfg["log_file"], self._fh = False, None
            self.status.set("log file error: file logging turned off")

    def _close_fh(self):
        if self._fh:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None

    def _write_csv(self):
        S, m = self.last_S, self.last_m
        path = self.cfg["csv_path"]
        try:
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "frames", "motion", "in_L", "acc_in_L", "false_acc", "timecodes",
                                "hidden", "search", "factor", "pending", "k", "thr", "min_px", "y"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), S.get("frames", 0), S.get("motion", 0),
                            S.get("in_L", 0), S.get("acc_in_L", 0), S.get("false_acc", 0),
                            S.get("timecodes_added", 0), S.get("hidden_timecodes", 0),
                            S.get("search_timecodes", 0), S.get("factor_timecodes", 0),
                            m.get("timecode_pending", 0), self.cfg["k"], self.cfg["thr"],
                            self.cfg["min_px"], self.cfg["y"]])
        except OSError as e:
            self.cfg["csv_on"] = False
            self.status.set(f"CSV error, recording off: {e}")

    def _snap(self, reason, d):
        c, now = self.cfg, time.time()
        if reason != "manual" and now - self._snap_t < c["snap_min_gap_s"]:
            return
        try:
            os.makedirs(c["snap_dir"], exist_ok=True)
            img = self.last_full if (c["snap_full"] and self.last_full is not None) else d["preview"]
            ext = "jpg" if c["snap_format"] == "jpg" else "png"
            name = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(now * 1000) % 1000:03d}_x{d['x']}_{reason}.{ext}"
            rp.cv2.imwrite(os.path.join(c["snap_dir"], name), rp.cv2.cvtColor(img, rp.cv2.COLOR_RGB2BGR))
            self._snap_t = self._snap_tp = now
            if c["snap_max"] > 0:
                files = sorted(f for f in os.listdir(c["snap_dir"]) if f.lower().endswith((".png", ".jpg")))
                for f in files[:max(0, len(files) - c["snap_max"])]:
                    os.remove(os.path.join(c["snap_dir"], f))
        except Exception as e:
            self.status.set(f"snapshot failed: {e}")

    def snap_now(self):
        self._snap_req = "manual"
        if not self._running():
            self.status.set("start the camera first; a snapshot is taken on the next frame")

    def alert(self, why):
        now = time.time()
        if now - self._last_alert < self.cfg["alert_cooldown_s"]:
            return
        self._last_alert = now
        for i in range(int(self.cfg["beep_count"])):
            self.win.after(i * 150, lambda: self._try(self.win.bell))
        self.flash_until = now + self.cfg["flash_secs"]
        self.flash_why = why

    def _summary(self):
        S = self.last_S
        return (f"# session summary: {time.time() - self.t_start:.0f}s, frames {S.get('frames', 0)}, "
                f"motion {S.get('motion', 0)}, in L {S.get('in_L', 0)} (accepted {S.get('acc_in_L', 0)}), "
                f"false accepts {S.get('false_acc', 0)}, time codes {S.get('timecodes_added', 0)} "
                f"(hidden {S.get('hidden_timecodes', 0)}, search {S.get('search_timecodes', 0)}, "
                f"factor {S.get('factor_timecodes', 0)})")

    # ---------------------------------------------------------- periodic automation
    def _tick(self):
        try:
            self._automation()
        except Exception as e:
            self.status.set(f"automation error: {type(e).__name__}: {e}")
        finally:
            self.win.after(500, self._tick)

    def _in_window(self):
        try:
            mins = lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])
            s, e = mins(self.cfg["sched_start"]), mins(self.cfg["sched_end"])
        except (ValueError, IndexError):
            return None
        t = time.localtime()
        n = t.tm_hour * 60 + t.tm_min
        return (s <= n < e) if s < e else (n >= s or n < e)

    def _automation(self):
        a, c, now = self.app, self.cfg, time.time()
        running = self._running()
        if self._was_running and not running:
            if c["summary_on_stop"] and self.last_S:
                a._log(self._summary())
            if not self.user_stop:
                if c["alert_camera_lost"]:
                    self.alert("stopped unexpectedly")
                if c["auto_restart"] and self.restarts < c["restart_max"] and not self.cal:
                    self.restarts += 1
                    a._log(f"# auto-restart {self.restarts}/{c['restart_max']} in {c['restart_delay_s']} s")
                    self.win.after(int(c["restart_delay_s"] * 1000), self._do_restart)
        self._was_running = running

        if c["sched_on"]:
            inw = self._in_window()
            if inw is not None:
                if inw and not self._inw_prev and not running:
                    self._sched_started = True
                    self._auto_starting = True
                    a.start(True)
                    self._auto_starting = False
                    a._log("# schedule: started")
                elif not inw and self._inw_prev and running and self._sched_started:
                    a.stop()
                    self._sched_started = False
                    a._log("# schedule: stopped")
            self._inw_prev = inw

        title = self.base_title
        if running:
            S, m = self.last_S, self.last_m
            reason = None
            if c["stop_secs"] and now - self.t_start >= c["stop_secs"]:
                reason = f"{c['stop_secs']} s elapsed"
            elif c["stop_frames"] and S.get("frames", 0) >= c["stop_frames"]:
                reason = f"{c['stop_frames']} frames processed"
            elif c["stop_codes"] and m.get("timecodes_added", 0) >= c["stop_codes"]:
                reason = f"{c['stop_codes']} time codes earned"
            if reason and not self.cal:
                a.stop()
                a._log(f"# auto-stop: {reason}")
                self.status.set(f"auto-stopped: {reason}")
            if c["y_rotate"] and now - self.t_rot >= c["y_rotate_s"] and not self.cal:
                good = []
                for t in (s.strip() for s in c["y_rotate_list"].split(",")):
                    try:
                        if t:
                            rp.parse_code(t)
                            good.append(t)
                    except ValueError:
                        pass
                if good:
                    self.rot_i = (self.rot_i + 1) % len(good)
                    self.apply({"y": good[self.rot_i]}, undo=False)
                    a._log(f"# Y rotated to {good[self.rot_i]}")
                self.t_rot = now
            idle = now - self.last_motion
            if c["idle_warn_s"] and idle >= c["idle_warn_s"] and not self.cal:
                title += "   [NO MOTION]"
            if c["recal_idle_min"] and idle >= c["recal_idle_min"] * 60 and not self.cal:
                self.last_motion = now
                a._log("# idle: running noise-floor recalibration")
                self.calibrate()
            if c["stats_log_s"] and now - self.t_stat >= c["stats_log_s"]:
                self.t_stat = now
                a._log(self._summary().replace("session summary", "stats"))
            if c["csv_on"] and now - self.t_csv >= c["csv_period_s"]:
                self.t_csv = now
                self._write_csv()
            if c["analysis_reset_min"] and now - self.t_anreset >= c["analysis_reset_min"] * 60:
                self.t_anreset = now
                a.ctl["reset_an"] = True
            if c["pending_cap"] and a.ctl.get("timecode_pending", 0) > c["pending_cap"]:
                a.ctl["timecode_pending"] = c["pending_cap"]
        if c["autosave_s"] and now - self.t_save >= c["autosave_s"]:
            self.t_save = now
            self.save_file(silent=True)
        self._hidden_clock(now)
        if self.cal:
            title += f"   [CALIBRATING: {self.status.get()}]"
        if now < self.flash_until:
            title += f"   *** {getattr(self, 'flash_why', '').upper()} ***"
        if self.win.title() != title:
            self.win.title(title)
        if self.dlg is not None and self.dlg.winfo_exists():
            g = self.gstat
            self.live.set(
                f"even run {self._streak + 1 if self._streak else 0}   pending bins {a.ctl.get('timecode_pending', 0)}   "
                f"gate passed/blocked  hidden {g['hidden']['passed']}/{g['hidden']['blocked']}  "
                f"search {g['search']['passed']}/{g['search']['blocked']}  "
                f"factor {g['factor']['passed']}/{g['factor']['blocked']}   "
                f"modified {len(self.modified())}")

    def _do_restart(self):
        if not self._running():
            self._auto_starting = True
            self.app.start(self.last_circuit)
            self._auto_starting = False

    def _hidden_clock(self, now):
        c, h = self.cfg, self.app.hidden
        if c["hidden_wander"]:
            lo, hi = sorted((c["hidden_wander_min"], c["hidden_wander_max"]))
            h.set_hz(random.uniform(lo, hi))
            self._jit_prev = True
        elif c["hidden_jitter_pct"] > 0:
            j = c["hidden_jitter_pct"] / 100.0
            h.set_hz(c["hidden_hz"] * (1 + random.uniform(-j, j)))
            self._jit_prev = True
        elif self._jit_prev:
            h.set_hz(c["hidden_hz"])
            self._jit_prev = False
        if c["hidden_autoreset_min"] and now - self.t_hreset >= c["hidden_autoreset_min"] * 60:
            self.t_hreset = now
            self.app._hidden_new_secret()

    # ---------------------------------------------------------- calibration
    def calibrate(self):
        a = self.app
        if self.cal:
            self._cal_end("aborted", restore=True)
            return
        self.cal_started_preview = not self._running()
        if self.cal_started_preview:
            a.start(False)
        delay = 2.5 if self.cal_started_preview else 0.8
        self.cal = dict(thrs=[6, 10, 14, 18, 24, 32, 44], i=-1, phase="warm", t=time.time() + delay,
                        samples=[], res=[], collect=False)
        self.apply({"min_px": 5})
        self.status.set("keep the scene STILL ...")
        self.win.after(100, self._cal_tick)

    def _cal_tick(self):
        cal = self.cal
        if not cal:
            return
        now = time.time()
        if now >= cal["t"]:
            ph = cal["phase"]
            if ph in ("warm", "next"):
                cal["i"] += 1
                if cal["i"] >= len(cal["thrs"]):
                    self._cal_finish()
                    return
                self.apply({"thr": cal["thrs"][cal["i"]]}, undo=False)
                cal["phase"], cal["t"], cal["samples"] = "settle", now + 0.6, []
            elif ph == "settle":
                cal["phase"], cal["t"], cal["collect"] = "collect", now + 2.0, True
            elif ph == "collect":
                cal["collect"] = False
                s = sorted(cal["samples"])
                if len(s) >= 15:
                    cal["res"].append((cal["thrs"][cal["i"]], s[min(len(s) - 1, int(0.99 * len(s)))], len(s)))
                cal["phase"], cal["t"] = "next", now
            self.status.set(f"sweeping thr {cal['thrs'][max(0, cal['i'])]} ({cal['i'] + 1}/{len(cal['thrs'])})")
        self.win.after(100, self._cal_tick)

    def _cal_finish(self):
        res = self.cal["res"]
        if not res:
            self._cal_end("failed: no frames received", restore=True)
            return
        ok = [r for r in res if r[1] <= 12]
        thr, p99, _ = ok[0] if ok else res[-1]
        min_px = max(10, min(800, int(p99 * 2 + 8)))
        table = "  ".join(f"{t}:{p}" for t, p, _ in res)
        self.apply({"thr": thr, "min_px": min_px}, undo=False)
        warn = "  (scene was not still)" if p99 > 60 else ""
        self._cal_end(f"done: pixel diff {thr}, min moving px {min_px}{warn}", restore=False)
        self.app._log("# calibration p99 noise px per threshold -> " + table)

    def _cal_end(self, msg, restore):
        cal, self.cal = self.cal, None
        if restore and cal:
            self.revert()
        if self.cal_started_preview and self._running():
            self.app.stop()
        self.cal_started_preview = False
        self.status.set("calibration " + msg)
        self.app._log("# calibration " + msg)

    def suggest_k(self):
        Y, E = self.app.ctl["y"], self.cfg["target_err_bits"]
        if "1" not in Y:
            self.status.set("Y has no open bins: no k works")
            return
        for kk in range(1, 31):
            if rp.window_stats(Y, kk)[0] >= E:
                self.apply({"k": kk})
                self.status.set(f"k={kk}: worst window has >= {E} open bins, error <= 2^-{E} at p=1/2")
                return
        self.status.set(f"needs k > 30 for 2^-{E} with Y={Y}; use a denser Y or more witness rounds")

    def grant(self, n):
        self.app.ctl["timecode_pending"] = max(0, self.app.ctl.get("timecode_pending", 0) + n)
        self.status.set(f"pending bins: {self.app.ctl['timecode_pending']}")

    def reset_gate(self):
        for t in self.gstat:
            self.gstat[t] = dict(passed=0, blocked=0, hits=0)
            self._cnt[t] = 0
        self.status.set("gate statistics reset")

    # ---------------------------------------------------------- dialog
    def _scroll_tab(self, nb, name):
        outer = ttk.Frame(nb)
        nb.add(outer, text=name)
        cv = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(cv, padding=8)
        inner.grid_columnconfigure(3, weight=1)
        wid = cv.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))

        def wheel(e):
            if e.widget.winfo_class() in ("TSpinbox", "TCombobox"):
                return
            cv.yview_scroll(-3 if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else 3, "units")

        def bind(_e=None):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                cv.bind_all(ev, wheel)

        def unbind(_e=None):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                cv.unbind_all(ev)
        for w in (cv, inner):
            w.bind("<Enter>", bind)
            w.bind("<Leave>", unbind)
        return inner

    def open_dialog(self):
        if self.dlg is not None and self.dlg.winfo_exists():
            self.dlg.lift()
            return
        w = tk.Toplevel(self.win)
        w.title("Settings")
        w.geometry("1020x700")
        self.dlg = w
        top = ttk.Frame(w)
        top.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(top, text="Search settings:").pack(side="left")
        self.v_search = tk.StringVar()
        ttk.Entry(top, textvariable=self.v_search, width=34).pack(side="left", padx=6)
        ttk.Label(top, text=f"{len(SPEC)} settings   (red = changed from default)", foreground="#777").pack(side="left")
        self.v_search.trace_add("write", lambda *_: self._filter())
        nb = ttk.Notebook(w)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb
        tabs, nrow = {}, {}
        for name in TABS:
            tabs[name] = self._scroll_tab(nb, name)
        self.vars, self.rows = {}, {}
        for tab, key, label, kind, extra, commit, help_ in SPEC:
            f = tabs[tab]
            r = nrow[tab] = nrow.get(tab, -1) + 1
            lab = ttk.Label(f, text=label)
            lab.grid(row=r, column=0, sticky="w", pady=2, padx=(0, 10))
            if kind == "bool":
                var = tk.BooleanVar()
                wd = ttk.Checkbutton(f, variable=var)
            elif kind in ("int", "float"):
                var = tk.StringVar()
                wd = ttk.Spinbox(f, from_=extra[0], to=extra[1], increment=extra[2], width=12, textvariable=var)
            elif kind == "choice":
                var = tk.StringVar()
                wd = ttk.Combobox(f, values=extra, textvariable=var, width=16, state="readonly")
            else:
                var = tk.StringVar()
                wd = ttk.Entry(f, width=36, textvariable=var)
            wd.grid(row=r, column=1, sticky="w")
            rb = ttk.Button(f, text="\u21ba", width=2, command=lambda k=key: self._reset_one(k))
            rb.grid(row=r, column=2, padx=4)
            hl = ttk.Label(f, text=help_, foreground="#777", wraplength=420)
            hl.grid(row=r, column=3, sticky="w", padx=6)
            self.vars[key] = var
            self.rows[key] = dict(widgets=(lab, wd, rb, hl), lab=lab, text=f"{label} {key} {help_} {tab}".lower())
            var.trace_add("write", lambda *_a, k=key: self._changed(k))
            if kind == "choice":
                wd.bind("<<ComboboxSelected>>", lambda e, k=key: self._commit(k), add="+")
            if key in COMMIT:
                for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
                    wd.bind(ev, lambda e, k=key: self.win.after_idle(lambda: self._commit(k)), add="+")

        tf = tabs["Time code"]
        r = nrow["Time code"] + 1
        ttk.Label(tf, text="Preset").grid(row=r, column=0, sticky="w", pady=(12, 3))
        cb = ttk.Combobox(tf, values=Y_PRESETS, width=16, state="readonly")
        cb.grid(row=r, column=1, sticky="w", pady=(12, 3))
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply({"y": cb.get()}))
        ttk.Button(tf, text="Suggest k for target error", command=self.suggest_k).grid(
            row=r + 1, column=0, columnspan=2, sticky="w", pady=6)

        xf = tabs["Tools"]
        xf.grid_columnconfigure(1, weight=1)
        ttk.Label(xf, text="Profile").grid(row=0, column=0, sticky="w", pady=3)
        self.cb_prof = ttk.Combobox(xf, width=36, state="readonly")
        self.cb_prof.grid(row=0, column=1, sticky="w")
        self._refresh_profiles()
        pb = ttk.Frame(xf)
        pb.grid(row=1, column=0, columnspan=4, sticky="w", pady=6)
        for text, cmd in (("Apply profile", self._apply_profile), ("Save current as ...", self._save_profile),
                          ("Delete", self._delete_profile), ("Show modified", self.show_modified)):
            ttk.Button(pb, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(xf, text="A profile only changes the settings it lists. Every change can be undone with Revert.",
                  foreground="#777").grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Separator(xf).grid(row=3, column=0, columnspan=4, sticky="ew", pady=10)
        tb = ttk.Frame(xf)
        tb.grid(row=4, column=0, columnspan=4, sticky="w")
        tools = (("Calibrate noise floor (F9)", self.calibrate), ("Snapshot now", self.snap_now),
                 ("Grant 1 bin", lambda: self.grant(1)), ("Grant 10 bins", lambda: self.grant(10)),
                 ("Clear pending bins", lambda: self.grant(-10 ** 9)),
                 ("Reset analysis counters", lambda: self.app.ctl.update(reset_an=True)),
                 ("Reset gate stats", self.reset_gate), ("New hidden secret", self.app._hidden_new_secret))
        for i, (text, cmd) in enumerate(tools):
            ttk.Button(tb, text=text, command=cmd).grid(row=i // 3, column=i % 3, padx=3, pady=3, sticky="w")

        bar = ttk.Frame(w)
        bar.pack(fill="x", padx=6, pady=(0, 4))
        for text, cmd in (("Apply commit-fields", self._flush_all), ("Revert", self.revert),
                          ("Defaults", self.defaults), ("Save", lambda: self.save_file()),
                          ("Load...", self._load_dialog), ("Export snapshot", self.export_snapshot)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(w, textvariable=self.live, font=("Consolas", 8), foreground="#555").pack(fill="x", padx=8)
        ttk.Label(w, textvariable=self.status, foreground="#245").pack(fill="x", padx=8, pady=(0, 6))
        w.protocol("WM_DELETE_WINDOW", self._close_dlg)
        self.sync()

    def _filter(self):
        q = self.v_search.get().strip().lower()
        first = None
        for key, row in self.rows.items():
            show = (not q) or all(t in row["text"] for t in q.split())
            for wd in row["widgets"]:
                (wd.grid if show else wd.grid_remove)()
            if show and first is None:
                first = key
        if q and first:
            tab = next(s[0] for s in SPEC if s[1] == first)
            self.nb.select(TABS.index(tab))

    def _close_dlg(self):
        if self.dlg is not None:
            self.dlg.destroy()
        self.dlg = None

    def sync(self, keys=None):
        if self.dlg is None or not self.dlg.winfo_exists():
            return
        self._syncing = True
        try:
            for key in (keys if keys is not None else self.vars):
                var, v = self.vars[key], self.cfg[key]
                kind = KINDS[key][0]
                var.set(bool(v) if kind == "bool" else (f"{v:g}" if kind == "float" else str(v)))
            for key, row in self.rows.items():
                row["lab"].configure(foreground="#c0392b" if self.cfg[key] != DEFAULTS[key] else "")
        finally:
            self._syncing = False

    def _reset_one(self, key):
        self.apply({key: DEFAULTS[key]})
        self.sync([key])

    def _read(self, key):
        kind, extra = KINDS[key]
        try:
            return parse_val(kind, self.vars[key].get(), extra)
        except (ValueError, tk.TclError):
            return None

    def _changed(self, key):
        if self._syncing or key in COMMIT:
            return
        self._dirty.add(key)
        if self._flush_job:
            self.win.after_cancel(self._flush_job)
        self._flush_job = self.win.after(350, self._flush_dirty)

    def _flush_dirty(self):
        self._flush_job = None
        new = {k: self._read(k) for k in list(self._dirty)}
        self._dirty.clear()
        self.apply({k: v for k, v in new.items() if v is not None}, from_ui=True)

    def _commit(self, key):
        v = self._read(key)
        if v is not None and not self._syncing:
            self.apply({key: v}, from_ui=True)

    def _flush_all(self):
        new = {k: self._read(k) for k in COMMIT}
        self.apply({k: v for k, v in new.items() if v is not None}, from_ui=True)

    # ---------------------------------------------------------- profiles
    def _all_profiles(self):
        d = dict(BUILTIN_PROFILES)
        d.update(self.profiles)
        return d

    def _refresh_profiles(self):
        names = list(self._all_profiles())
        self.cb_prof["values"] = names
        if names and not self.cb_prof.get():
            self.cb_prof.set(names[0])

    def _apply_profile(self):
        p = self._all_profiles().get(self.cb_prof.get())
        if p:
            self.apply(dict(p))
            self.status.set(f"profile '{self.cb_prof.get()}' applied ({len(p)} settings)")

    def _save_profile(self):
        name = simpledialog.askstring("Profile", "Name for this profile:", parent=self.dlg)
        if name:
            self.profiles[name.strip()] = dict(self.cfg)
            self._refresh_profiles()
            self.cb_prof.set(name.strip())
            self.save_file(silent=True)
            self.status.set(f"profile '{name}' saved")

    def _delete_profile(self):
        n = self.cb_prof.get()
        if n in self.profiles:
            del self.profiles[n]
            self.cb_prof.set("")
            self._refresh_profiles()
            self.save_file(silent=True)
        else:
            self.status.set("built-in profiles cannot be deleted")

    def _load_dialog(self):
        p = filedialog.askopenfilename(title="Load settings", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if p:
            self.load_file(path=p)


def main():
    global PACK
    mp.freeze_support()
    _install_core_patches()
    root = tk.Tk()
    app = rp.App(root)
    PACK = Pack(app)
    PACK.install()
    root.update_idletasks()
    rp.fit_window(root, app.root.winfo_reqwidth() + 24, app.root.winfo_reqheight() + 8)
    root.mainloop()


if __name__ == "__main__":
    main()

