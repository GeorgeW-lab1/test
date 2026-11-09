"""
RP-style optoelectronic decision circuit driven by a USB camera  (single file, COMMAND PROMPT version)

    pip install numpy opencv-python
    python rp_circuit_camera_cli.py                 # starts the circuit and opens the "rp>" prompt
    python rp_circuit_camera_cli.py --help          # all start-up options

There is no window any more. Everything the GUI showed is printed in the terminal, and everything
the GUI buttons/sliders did is a typed command (type "help" at the rp> prompt). Events (hidden-number
hits, search finds, verified factorisations, PoW solves, BTC oracle checks) are printed as they
happen; "status" prints the full state of every panel; "frames on" prints one line per motion frame.

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

Driver analysis (command "analysis"): every instance frame is scored against
the truth (is x really an odd composite?). For Driver A, Driver B and the final Q
it reports TPR/FPR/PPV/NPV/accuracy and mutual information with "x in L", checks
the RP properties the bounds rest on (A never fires outside L, A's witness rate
p >= 1/2, B never misses a member of L), calibrates A against the exact witness
density of every x, and tests the independence the (1-p)^N bound assumes.

Time code Y (proved in time_code_proof_check.py, shown by "status" / "code"):
  * Gating reading, D = A.B.Y: x not in L is never accepted for ANY Y (Thm 1).
    For x in L the false-reject probability is exactly (1-p)^N, N = open bins
    in the k-frame window (Thm 2), so Y sets the error exponent, not the sidedness.
  * Y accepts arithmetic: "1000|0011" OR-adds codes (N never drops, disjoint
    codes multiply their errors), "1111&1011" ANDs them (N never rises),
    "1011+0001" adds them as binary numbers: popcount(a+b) = popcount(a) +
    popcount(b) - carries, so each carry loses one open bin (Thm 3).
  * "forces HIGH" read as OR ("force on") lets a forced bin accept any x, so the
    one-sided guarantee is lost (Thm 4). Use it to watch false accepts appear.

Concurrency: the camera read runs in a worker thread; Driver A and Driver B
process each frame concurrently as asyncio tasks; the flip-flop stage runs
once both are finished. The whole pipeline lives in a background thread with
its own asyncio loop and talks to the prompt through a queue.Queue.

Two sequential even states: a hash is mined, and the hidden number is guessed, only on a
frame whose x is even AND whose previous frame's x was also even. An odd x or a no-motion
frame in between breaks the run, so the next even frame only starts a new one.

Higher/lower search (separate from both): a secret of N bits (default 512) and a
higher/lower oracle. On every armed frame (two sequential evens) one query yields one bit
of the secret, which is POOLED; no equality check is made per query. Only when all N bits
are pooled is the candidate compared with the secret, once. A 1000-bit secret falls in
1000 pooled queries plus 1 final check instead of about 2^1000 blind guesses. A found secret earns
one time-code bin and a new secret is hidden.

Hidden-number test (separate from the bit search): a secret integer is drawn in
0..range-1. A separate clock thread cycles a candidate through that range at a frequency
that you set ("hidden hz", 1 Hz to 2 kHz, default 17 Hz). The thread never sees the
secret. Only when two sequential EVEN states have arrived does the frame loop spot-check the candidate AT
THAT MOMENT; if it equals the secret, one time-code bin is added (same mechanism as a found
secret). After EVERY spot check, hit or miss, a new secret is hidden. Expected hits = checks / range.
"hidden new" (and changing the range) draws a fresh secret and zeroes the spot checks,
hits and luck factor. The luck factor is hits / expected hits in percent (100% = average),
shown with P(>= hits by chance), the binomial tail probability.

Hash prediction -> response latch -> PoW miners (three independent async workers):
the higher/lower search is aimed at a 512-bit SHA-512 digest. Nothing in this chain waits on,
joins or is called by another stage; every worker is its own thread running its own asyncio
loop, and stages only touch each other through LATCH REGISTERS (a value plus a sequence number,
lock held for a pointer swap only, so a read or write never blocks on a slow peer):

    camera loop --pulse--> [tick box] --> ORACLE worker --write--> [message latch]
    [message latch] + [config latch] --read--> MINER lanes (N threads) --write--> [response latch]
    prompt --reads all latches on a timer, writes the config latch (header, difficulty)

  * The camera loop only appends a tick (bounded deque, drops oldest, never blocks) on every armed
    frame and later collects finished oracle hits. It never calls the search itself.
  * The oracle worker pools one bit per tick; the final check passing writes the verified 512-bit
    hash into the message latch and goes on. It does not know the miners exist.
  * Each miner lane reads whatever is in the message latch and config latch, picks its own random
    nonce start and hashes. It abandons a job only because a latch it READ has changed or holds a
    solution for that job; no lane signals another. The lane that wins publishes once.
  * With no miner running the message latch simply holds; with no oracle result the miners idle.

Response string (delimiter ":"), the latch holds this and never the bare hash:

    <header>:<padded nonce>:<padded message>:<padded sha256>

  header   custom text you set ("header TEXT", ASCII, no ":", up to 32 chars)
  nonce    256-bit field (64 hex), found by the miners
  message  the predicted 512-bit hash, 128 hex
  sha256   SHA-256 over the ASCII text "<header>:<nonce>:<message>", shown as a 512-bit zero-padded
           hex field (128 hex chars; the top 256 bits are always zero)
PoW difficulty counts leading zero bits of the 256-bit digest itself, not of the padded field.
Hashing is pure Python (hashlib holds the GIL on short inputs), so extra lanes add concurrency, not
much hash rate.

Factoriser (separate source): N is factored through the same higher/lower bit search.
The oracle holds the smallest prime factor p of N and answers only "p >= trial?". Each armed frame
pools one bit; one final check runs when the pool is full: 1 < p < N, N % p == 0 and the full prime
factorisation multiplies back to N. A time-code bin is earned ONLY after that verified solve.
"factor engine N" really factors N in a background process (trial division, Fermat, Pollard rho,
p-1, ECM, quadratic sieve; pip install gmpy2 for speed); the oracle is not ready until the engine
finishes. "factor N" (default) is the high/low-only mode. Random semiprimes ("factor gen") keep their
factors inside the oracle, which shows the mechanism only. A higher/lower oracle for a factor cannot
exist without already knowing it, so the search does not make factoring itself cheaper. Camera
motion hashes seed the engine's randomness.

Needs a real USB camera and OpenCV: motion comes only from what the camera
sees. There is no simulated source; if the camera can't be opened, it says so.
"""
import argparse
import asyncio
import hashlib
import json
import math
import multiprocessing as mp
import queue
import re
import secrets
import shlex
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:                      # reported when you start the camera
    cv2 = None

# ------------------------------------------------------------------ constants
SW, SH = 128, 96                  # processing size
PW, PH = 288, 216                 # preview size (kept for make_preview)
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
        m, c, t, r = i, b * b % p, t * b * b % p, r * b * b % p
    return r


# ------------------------------------------------------------------ hidden-number guesser
HIDDEN_RANGE_DEFAULT = 100
HIDDEN_RANGE_MAX = 1_000_000
GUESS_HZ_MIN, GUESS_HZ_MAX = 20.0, 2000.0     # the clock wanders between these
GUESS_RETUNE_S = 0.3                          # how often the clock re-reads the frequency setting
GUESS_HZ_DEFAULT = 17.0                       # starting clock frequency (adjustable)
GUESS_HZ_LIMITS = (1.0, 2000.0)               # allowed range for the frequency


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
            if new_hz != hz:                       # frequency was changed from the prompt
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
HASH_BITS = 512                     # SHA-512 digest size
SEARCH_BITS_DEFAULT = HASH_BITS
SEARCH_BITS_MAX = 4096


def short_int(n: int, keep: int = 20) -> str:
    """Hex form of a big integer, shortened to its first and last digits."""
    h = format(n, "x")
    return h if len(h) <= 2 * keep + 3 else f"{h} ({len(h) * 4} bit hex)"


def hash_secret(bits: int) -> int:
    """A fresh secret that is a SHA-512 digest (SHA-512 in counter mode when bits > 512,
    leading bits kept when bits < 512)."""
    seed = secrets.token_bytes(64)
    out, ctr = b"", 0
    while len(out) * 8 < bits:
        out += hashlib.sha512(seed + ctr.to_bytes(4, "big")).digest()
        ctr += 1
    return int.from_bytes(out, "big") >> (len(out) * 8 - bits)


# ------------------------------------------------------------------ padded response format
HEADER_DEFAULT = "RPCAM1"
HEADER_MAX = 64
NONCE_HEX = 128                      # 256-bit nonce field
DIGEST_HEX = 128                    # sha256 shown as a 512-bit padded field
DELIM = ":"
MINER_DIFF_DEFAULT = 20             # leading zero bits of the 256-bit digest
MINER_DIFF_MAX = 128
MINER_LANES_DEFAULT = 2
MINER_LANES_MAX = 16


def sanitize_header(text: str) -> str:
    """ASCII printable, no delimiter, no spaces, at most HEADER_MAX chars."""
    h = "".join(c for c in str(text) if 33 <= ord(c) < 127 and c != DELIM)[:HEADER_MAX]
    return h or HEADER_DEFAULT


def pad_hex(value: int, width: int) -> str:
    """Zero-pad an integer to a fixed number of hex characters."""
    h = format(value, "x")
    if len(h) > width:
        raise ValueError(f"value needs {len(h)} hex chars, field is {width}")
    return h.zfill(width)


def digest_text(header: str, nonce_hex: str, msg_hex: str) -> bytes:
    return hashlib.sha256((header + DELIM + nonce_hex + DELIM + msg_hex).encode("ascii")).digest()


def build_response(header: str, nonce: int, message: int, msg_bits: int = HASH_BITS) -> str:
    """<header>:<padded nonce>:<padded message>:<padded sha256>"""
    h = sanitize_header(header)
    n = pad_hex(nonce, NONCE_HEX)
    m = pad_hex(message, (msg_bits + 3) // 4)
    d = int.from_bytes(digest_text(h, n, m), "big")
    return DELIM.join((h, n, m, pad_hex(d, DIGEST_HEX)))


def digest_zero_bits(d_field: str) -> int:
    """Leading zero bits of the 256-bit digest inside the 512-bit padded field."""
    return 256 - int(d_field, 16).bit_length()


def verify_response(resp: str) -> bool:
    """Recompute the sha256 field from header, nonce and message."""
    try:
        h, n, m, d = resp.split(DELIM)
        int(n, 16), int(m, 16), int(d, 16)
    except ValueError:
        return False
    if len(n) != NONCE_HEX or len(d) != DIGEST_HEX or int(d, 16) >> 256:
        return False
    return pad_hex(int.from_bytes(digest_text(h, n, m), "big"), DIGEST_HEX) == d


# ------------------------------------------------------------------ latch registers + async workers
class Register:
    """A latch: holds the newest value and a sequence number. The lock is held only for a pointer
    swap, so a reader never waits on a slow writer and a writer never waits on a reader."""

    def __init__(self, value=None):
        self._lock = threading.Lock()
        self._v = value
        self.seq = 0

    def write(self, value) -> int:
        with self._lock:
            self._v = value
            self.seq += 1
            return self.seq

    def write_unless(self, blocked, value) -> bool:
        """Write unless blocked(current) is true (compare-and-set; first solver wins)."""
        with self._lock:
            if blocked(self._v):
                return False
            self._v = value
            self.seq += 1
            return True

    def read(self):
        with self._lock:
            return self.seq, self._v


@dataclass(frozen=True)
class MinerConfig:
    header: str = HEADER_DEFAULT
    difficulty: int = MINER_DIFF_DEFAULT


class AsyncWorker(threading.Thread):
    """A thread that runs its own asyncio loop. Subclasses implement main()."""

    def __init__(self, name: str):
        super().__init__(daemon=True, name=name)
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def stopped(self) -> bool:
        return self._halt.is_set()

    def run(self):
        asyncio.run(self.main())

    async def main(self):
        raise NotImplementedError


class OracleWorker(AsyncWorker):
    """Consumes ticks (one per armed camera frame), asks the higher/lower oracle one question per
    tick, and writes each verified solve into the message latch. It never waits for the camera
    loop or the miners; with no tick it just sleeps briefly."""

    def __init__(self, search, msg_reg: Register):
        super().__init__("oracle")
        self.search = search
        self.msg_reg = msg_reg
        self.ticks = deque(maxlen=20000)      # (frame idx, x); append/popleft are thread-safe
        self.hits = deque(maxlen=1000)        # finished solves for the camera loop to collect
        self.processed = 0

    def tick(self, idx: int, x: int):
        self.ticks.append((idx, x))           # never blocks, drops the oldest when full

    def take_hits(self):
        out = []
        while True:
            try:
                out.append(self.hits.popleft())
            except IndexError:
                return out

    async def main(self):
        while not self.stopped():
            n = 0
            while n < 256:
                try:
                    idx, x = self.ticks.popleft()
                except IndexError:
                    break
                n += 1
                self.processed += 1
                if self.search.guess():
                    sn = self.search.snapshot()
                    found = sn["last_found"]
                    if found is None:
                        continue
                    self.msg_reg.write(dict(message=found[0], bits=sn["bits"], frame=idx))
                    self.hits.append(dict(frame=idx, x=x, bits=sn["bits"], solves=sn["solves"],
                                          avg=sn["avg"], found=found))
            await asyncio.sleep(0 if n else 0.01)


class MinerLane(AsyncWorker):
    """One mining lane. Reads the message latch and config latch, hashes from its own random nonce,
    and publishes a solution to the response latch (first solver of a job wins). It stops working
    on a job only because a latch it read changed or already holds that job's solution."""

    def __init__(self, lane: int, msg_reg, cfg_reg, resp_reg):
        super().__init__(f"miner-{lane}")
        self.lane = lane
        self.msg_reg, self.cfg_reg, self.resp_reg = msg_reg, cfg_reg, resp_reg
        self.hashes = 0
        self.solved = 0
        self.hps = 0.0
        self.status = "starting"

    async def main(self):
        while not self.stopped():
            mseq, job = self.msg_reg.read()
            cseq, cfg = self.cfg_reg.read()
            if job is None:
                self.status, self.hps = "waiting for oracle", 0.0
                await asyncio.sleep(0.05)
                continue
            key = (mseq, cseq)
            _, rec = self.resp_reg.read()
            if rec is not None and rec["key"] == key:
                self.status, self.hps = "job solved, idle", 0.0
                await asyncio.sleep(0.05)
                continue
            await self._mine(key, job, cfg)
        self.status = "stopped"

    async def _mine(self, key, job, cfg: MinerConfig):
        header = sanitize_header(cfg.header)
        msg_hex = pad_hex(job["message"], (job["bits"] + 3) // 4)
        head = header + DELIM
        tail = DELIM + msg_hex
        target = 1 << (256 - cfg.difficulty)
        nonce = secrets.randbits(NONCE_HEX * 4 - 32)        # own random start, room to count up
        tried, t0 = 0, time.perf_counter()
        sha = hashlib.sha256
        self.status = f"mining frame {job['frame']}"
        while not self.stopped():
            for i in range(1024):
                d = sha((head + format(nonce, "064x") + tail).encode("ascii")).digest()
                val = int.from_bytes(d, "big")
                if val < target:
                    tried += i + 1
                    self.hashes += i + 1
                    secs = time.perf_counter() - t0
                    resp = build_response(header, nonce, job["message"], job["bits"])
                    rec = dict(key=key, response=resp, frame=job["frame"], lane=self.lane,
                               tried=tried, secs=secs, lz=256 - val.bit_length())
                    if self.resp_reg.write_unless(
                            lambda cur: cur is not None and cur["key"] == key, rec):
                        self.solved += 1
                    return
                nonce += 1
            tried += 1024
            self.hashes += 1024
            self.hps = tried / max(time.perf_counter() - t0, 1e-9)
            await asyncio.sleep(0)                           # let this lane's loop breathe
            if self.msg_reg.seq != key[0] or self.cfg_reg.seq != key[1]:
                return                                       # inputs changed (a read, not a wait)
            _, rec = self.resp_reg.read()
            if rec is not None and rec["key"] == key:
                return                                       # another lane already solved it


class MinerPool:
    """Starts and stops independent lanes. Lanes share nothing but the three latches."""

    def __init__(self, msg_reg, cfg_reg, resp_reg):
        self.regs = (msg_reg, cfg_reg, resp_reg)
        self.lanes = []
        self.solved_total = 0
        self.hashes_total = 0

    @property
    def running(self) -> bool:
        return any(l.is_alive() and not l.stopped() for l in self.lanes)

    def start(self, n: int = MINER_LANES_DEFAULT):
        if self.running:
            return
        n = max(1, min(MINER_LANES_MAX, int(n)))
        self.lanes = [MinerLane(i, *self.regs) for i in range(n)]
        for l in self.lanes:
            l.start()

    def stop(self):
        for l in self.lanes:
            l.stop()
        self.solved_total += sum(l.solved for l in self.lanes)
        self.hashes_total += sum(l.hashes for l in self.lanes)
        self.lanes = []

    def snapshot(self) -> dict:
        lanes = list(self.lanes)
        return dict(running=self.running, n=len(lanes),
                    hps=sum(l.hps for l in lanes),
                    hashes=self.hashes_total + sum(l.hashes for l in lanes),
                    solved=self.solved_total + sum(l.solved for l in lanes),
                    status=(lanes[0].status if len(lanes) == 1 else
                            "; ".join(sorted({l.status.split(" frame")[0] for l in lanes})))
                    if lanes else "stopped")


class HashPipeline:
    """Owns the latches and the workers. Starting or stopping one worker never affects another."""

    def __init__(self, search):
        self.search = search
        self.msg = Register()                       # oracle -> miners
        self.cfg = Register(MinerConfig())          # prompt -> miners
        self.resp = Register()                      # miners -> prompt
        self.oracle = OracleWorker(search, self.msg)
        self.miners = MinerPool(self.msg, self.cfg, self.resp)

    def start(self, lanes: int = MINER_LANES_DEFAULT):
        if not self.oracle.is_alive():
            self.oracle.start()
        self.miners.start(lanes)

    def shutdown(self):
        self.oracle.stop()
        self.miners.stop()

    def set_config(self, header=None, difficulty=None) -> MinerConfig:
        _, cur = self.cfg.read()
        header = cur.header if header is None else sanitize_header(header)
        diff = cur.difficulty if difficulty is None else max(1, min(MINER_DIFF_MAX, int(difficulty)))
        new = MinerConfig(header, diff)
        if new != cur:
            self.cfg.write(new)
        return new

    def response(self):
        """Held response string, or None."""
        _, rec = self.resp.read()
        return rec["response"] if rec else None

    def clear_response(self):
        """Empty the output latch without making the miners redo the job (same key)."""
        _, rec = self.resp.read()
        if rec is not None and rec["response"] is not None:
            self.resp.write(dict(rec, response=None))

    def message(self):
        _, job = self.msg.read()
        return job


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
        self._secret = hash_secret(self.bits)       # the secret is a SHA-512 digest
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
    thread froze the camera loop. Results come back through the queue."""
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
      * typed N : a background process really factors N (trial division, Fermat, Pollard rho, p-1,
                  ECM, quadratic sieve). Until it finishes the oracle is not ready, no bits are
                  pooled and no time code can be earned.
      * generated N : a random semiprime is built and its factors are kept inside the oracle. This
                  shows the mechanism; the search does not factor N, the oracle already knows p.
    A higher/lower oracle for a factor of N cannot exist without already knowing that factor, so the
    pooling is a mechanism, not a shortcut: the cost of factoring is paid by the engine."""

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
                                 "time code to earn. Type a composite N, or use 'factor gen'.")
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
        oracle = ctl.get("oracle")
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
                # The oracle runs in its own thread: the frame loop only pulses a tick (never blocks)
                # and collects whatever solves the oracle finished in the meantime.
                if oracle is not None and ctl.get("search_on", True) and even_state:
                    oracle.tick(idx, x)
                if oracle is not None:
                    for h in oracle.take_hits():
                        search_hit = True
                        ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + 1
                        S["search_timecodes"] += 1
                        S["timecodes_added"] += 1
                        out.put(("search_hit", dict(frame=h["frame"], x=h["x"], bits=h["bits"],
                                                    solves=h["solves"], avg=h["avg"],
                                                    found=h["found"])))
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


# ------------------------------------------------------------------ Bitcoin HIGH/LOW oracle
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


# ------------------------------------------------------------------ command prompt front end
FRAME_INTERVAL = 33                # ms between queue drains (~30 fps cap)
MINER_INTERVAL = 0.2               # s between miner/latch refreshes

HELP = """\
RUN CONTROL
  start                      start camera + circuit            preview   camera only, circuit idle
  stop                       stop the camera loop              quit / exit   leave the program
  status [all]               print every panel (hidden, search, factor, miners, BTC, time code)
  frames on|off              print one line per motion frame (default off)
  verdict on|off             print SELECT changes as they happen (default off)

CIRCUIT SETTINGS  ("set NAME VALUE")
  set thr N                  pixel diff threshold (4..60)       set minpx N   min moving pixels (5..800)
  set k N                    frames in the SELECT window (1..30)
  set y EXPR                 time code, e.g. 1111  1000|0011  1111&1011  1011+0001
  set force on|off           read 'forces HIGH' as OR (breaks one-sidedness, Thm 4)
  set overlay on|off         motion overlay flag (kept for the preview image)
  code                       what the time code does to the error bound (Thm 1-4)
  analysis                   print the driver analysis      reset-analysis   zero its counters

HIDDEN NUMBER
  hidden on|off | range N | new | reveal | hz F

HASH PREDICTION (higher/lower search) + RESPONSE LATCH + PoW MINERS
  search on|off | bits N | new | reveal
  miners start [threads] | miners stop
  header TEXT                response header (ASCII, no ':', up to 32 chars)
  diff N                     PoW difficulty, leading zero bits (1..40)
  latch show | verify | clear

FACTORISER
  factor on|off              enable/disable the factoriser source
  factor N                   high/low only: the oracle holds N itself (N may be 2^127-1, 0x.., 1_000_007)
  factor engine N            really factor N in a background process, then pool bits
  factor gen [bits]          random semiprime (default 256 bits)
  factor cancel | reveal | save [file.json] | status

BITCOIN HIGH/LOW ORACLE
  btc check                  one Binance check now (arms one release for the next even frame)
  btc auto [seconds]         automatic checks (default 10 s)      btc stop   stop auto-check
  btc call higher|lower      set the manual call                  btc status
"""


def onoff(word: str) -> bool:
    w = word.lower()
    if w in ("on", "1", "true", "yes", "y"):
        return True
    if w in ("off", "0", "false", "no", "n"):
        return False
    raise ValueError("use on or off")


class App:
    def __init__(self, args):
        self.args = args
        self.q = queue.Queue()
        self.cmds = queue.Queue()
        self.thread = None
        self.running = False
        self.quit = False
        self._out_lock = threading.Lock()
        self.ctl = dict(stop=False, thr=args.thr, min_px=args.min_px, k=args.k, y="1111", circuit=True,
                        overlay=True, force=False,
                        timecode_pending=0,
                        hidden_on=True, search_on=True)
        self.hidden = HiddenNumber(args.hidden_range)
        self.hidden.set_hz(args.hz)
        self.hidden.start()
        self.ctl["hidden"] = self.hidden
        self.search = BinarySearch(args.bits)
        self.ctl["search"] = self.search
        self.pipe = HashPipeline(self.search)          # oracle worker + latches + miner lanes
        self.ctl["oracle"] = self.pipe.oracle
        self.pipe.oracle.start()
        self.pipe.set_config(header=args.header, difficulty=args.difficulty)
        self.lanes = args.lanes
        if not args.no_miners:
            self.pipe.miners.start(self.lanes)
        self._seen_resp = 0
        self.pool = EntropyPool()
        self.fsearch = FactorSearch(self.pool)
        self.ctl.update(fsearch=self.fsearch, fsearch_on=True, pool=self.pool)
        self.btc = BitcoinHighLowOracle()
        self.btc_period = 10
        self.btc_auto_stop = threading.Event()
        self.btc_auto_thread = None
        self.err = False
        self.frames_on = args.frames
        self.verdict_on = False
        self._last_select = None
        self.last_stats = None
        self.last_codes = None
        self._fps = None
        self._last_codes_t = 0.0
        self._last_status_t = time.time()
        self.an_lines = ["Start the circuit and move in front of the camera."]
        self.code_info = dict(carries=0, wrapped=0, expr="1111")
        self.y_ok = True
        self._set_y(args.y, quiet=True)
        if args.force:
            self.ctl["force"] = True

    # ---------------------------------------------------------- output
    def _log(self, s):
        with self._out_lock:
            print(s, flush=True)

    def _logs(self, lines):
        with self._out_lock:
            print("\n".join(lines), flush=True)

    # ---------------------------------------------------------- text panels
    def hidden_text(self, h):
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
        return [
            f"clock: candidate {h['candidate']:>7d} of 0..{h['size'] - 1}   {h['hz']:7.1f} Hz (set)",
            f"spot checks: {h['checks']}   hits: {h['hits']}   expected ~{h['expected']:.2f}",
            luck,
            f"last check: {last}   secret: {sec}",
            f"last FOUND number: {found}"]

    def search_text(self, h):
        last = "none yet" if h["last"] is None else f"{h['last'][0]}"
        avg = "n/a" if h["avg"] is None else f"{h['avg']:.1f}"
        sec = "hidden" if h["secret"] is None else h["secret"]
        lf = h["last_found"]
        found = "none yet" if lf is None else f"{short_int(lf[0], 16)} in {lf[1]} guesses"
        return [
            f"secret: {h['bits']}-bit {'SHA-512 digest' if h['bits'] == HASH_BITS else 'hash'}   "
            f"blind guessing would need ~2^{h['bits']} tries",
            f"guesses: {h['guesses']} total, {h['round_guesses']} on this secret   "
            f"found: {h['solves']}   avg guesses/found: {avg}",
            f"pooled: {h['pooled_n']}/{h['bits']} bits   final checks: {h['final_checks']}   last: {last}",
            f"last FOUND number: {found}",
            f"current secret: {sec}"]

    def factor_text(self, h):
        st = {"idle": "idle", "generating": "building a semiprime",
              "factoring": "ENGINE RUNNING: oracle not ready",
              "ready": "READY: pooling bits on armed frames", "solved": "SOLVED and verified",
              "prime": "N is PRIME: nothing to factor, no oracle (type a composite)",
              "failed": "failed", "cancelled": "cancelled"}.get(h["state"], h["state"])
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
        return lines

    def codes_text(self, m):
        return [
            f"Time codes added: {m['timecodes_added']} (hidden-number: {m['hidden_timecodes']}, "
            f"search: {m['search_timecodes']}, factor: {m['factor_timecodes']}, "
            f"pending {m['timecode_pending']})",
            f"Armed frames (2 evens in a row): {m['armed_frames']}",
            f"{'ARMED' if m['last_armed'] else 'not armed'}"
            f"{'' if m['circuit'] else '  [circuit OFF: use start]'}"]

    def stats_text(self, S):
        rate = S["acc_in_L"] / S["in_L"] if S["in_L"] else 0.0
        return [f"frames {S['frames']}  with motion {S['motion']}",
                f"x in L (odd composite): {S['in_L']}  accepted: {S['acc_in_L']}  ({rate:.0%})",
                f"x not in L accepted (false accepts): {S['false_acc']}",
                f"time codes added: {S['timecodes_added']} (hidden {S['hidden_timecodes']}, "
                f"search {S['search_timecodes']}, factor {S['factor_timecodes']})"]

    def miner_text(self):
        ms = self.pipe.miners.snapshot()
        o = self.pipe.oracle
        sn = self.search.snapshot()
        return [f"oracle thread: {sn['pooled_n']}/{sn['bits']} bits pooled   solves {sn['solves']}   "
                f"ticks waiting {len(o.ticks)}",
                f"miners: {ms['n']} threads, {ms['status']}   {ms['hps'] / 1000:.0f} kH/s   "
                f"hashes {ms['hashes']}   solved {ms['solved']}"]

    def latch_text(self):
        v = self.pipe.response()
        job = self.pipe.message()
        if v:
            return v
        if job is not None:
            return (f"oracle message latched (frame {job['frame']}): {short_int(job['message'], 16)}  "
                    + ("(miners are hashing for a nonce)" if self.pipe.miners.running
                       else "(miners stopped: use 'miners start')"))
        return "latch empty (waiting for a verified hash prediction from the oracle)"

    def code_lines(self, S=None):
        """What the time code does to the error bound (Thm 1-4)."""
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
        return L

    def btc_text(self):
        b = self.btc
        if b.last_price is None:
            return ["BTC/USDT: --   (no check yet; use 'btc check')"]
        lp = b.last_prediction
        return [f"BTC/USDT: ${b.last_price:,.2f}   next call: {lp['guess']} (from ${lp['made_at_price']:,.2f})",
                f"checks={b.fresh_count}, factual={b.factual_count}, rejected={b.rejected_count}, "
                f"calls correct={b.correct}/{b.total}   auto-check: "
                f"{'every ' + str(self.btc_period) + ' s' if self._btc_auto_on() else 'off'}   "
                f"release armed: {bool(self.ctl.get('btc_release_armed'))}"]

    def status(self):
        L = [f"=== STATUS ({'running' if self.running else 'stopped'}"
             f"{'' if self.ctl.get('circuit', True) else ', preview only'}"
             f"{', force mode' if self.ctl.get('force') else ''}) ===",
             f"settings: pixel diff > {self.ctl['thr']}   min moving px {self.ctl['min_px']}   "
             f"k {self.ctl['k']}   Y {self.ctl['y']}"]
        if self.last_stats is not None:
            L += ["", "-- circuit --"] + self.stats_text(self.last_stats)
        if self.last_codes is not None:
            L += ["", "-- time-code sources --"] + self.codes_text(self.last_codes)
        L += ["", "-- time code Y --"] + self.code_lines(self.last_stats)
        L += ["", "-- hidden number --"] + self.hidden_text(self.hidden.snapshot())
        L += ["", "-- hash prediction (higher/lower) --"] + self.search_text(self.search.snapshot())
        L += ["", "-- PoW miners / response latch --"] + self.miner_text() + [self.latch_text()]
        L += ["", "-- factoriser --"] + self.factor_text(self.fsearch.snapshot())
        L += ["", "-- bitcoin high/low oracle --"] + self.btc_text()
        self._logs(L)

    # ---------------------------------------------------------- time code
    def _set_y(self, expr, quiet=False):
        try:
            y, info = parse_code(expr)
        except ValueError as err:
            self.y_ok = False
            self._log(f"time code invalid: {err}")
            return False
        self.y_ok = True
        self.ctl["y"] = y
        self.code_info = info
        if not quiet:
            note = ""
            if info["carries"]:
                note = f"  ({info['carries']} carries" + (", top wrapped)" if info["wrapped"] else ")")
            self._log(f"time code Y = {y}{note}")
        return True

    # ---------------------------------------------------------- run control
    def start(self, circuit=True):
        if self.thread is not None and self.thread.is_alive():
            self._log("# already running (use 'stop' first)")
            return
        self.ctl.update(stop=False, circuit=circuit, timecode_pending=0)
        self.q = queue.Queue()
        self.err = False
        self._last_select = None
        self.thread = SimThread(self.args.camera, self.ctl, self.q)
        self.thread.start()
        self.running = True
        self._log(f"# starting {'circuit' if circuit else 'camera preview (circuit idle)'} on camera {self.args.camera} ...")

    def stop(self):
        if self.running:
            self.ctl["stop"] = True
            self._log("# stopping ...")
        else:
            self._log("# not running")

    def shutdown(self):
        self.ctl["stop"] = True
        self.btc_auto_stop.set()
        self.fsearch.cancel()
        self.pipe.shutdown()
        self.hidden.stop()
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    # ---------------------------------------------------------- queue events
    def _poll(self):
        try:
            for _ in range(400):                       # bounded drain per tick
                ev = self.q.get_nowait()
                kind = ev[0]
                if kind == "frame":
                    d = ev[1]
                    self.last_stats = d["stats"]
                    fps = 1 / d["dt"]
                    self._fps = fps if self._fps is None else 0.9 * self._fps + 0.1 * fps
                    if d["circuit"]:
                        if d["motion"] and self.frames_on:
                            self._frame_log(d)
                        if self.verdict_on and d["select"] != self._last_select:
                            self._last_select = d["select"]
                            if d["select"]:
                                self._log(f"SELECT = 1  ACCEPT  (witness found within last {self.ctl['k']} frames)")
                            else:
                                self._log("SELECT = 0")
                elif kind == "codes":
                    self.last_codes = ev[1]            # keep only the newest
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
                    p, qq, n = fh["found"]
                    self._log(f"FACTORISATION VERIFIED frame {fh['frame']:5d} x={fh['x']} "
                              f"p found in {n} queries; time code added")
                    self._log(f"  p = {short_dec(p, 30)}")
                    self._log(f"  q = {short_dec(qq, 30)}")
                elif kind == "log":
                    self._log(ev[1])
                elif kind == "analysis":
                    self.an_lines = ev[1]
                elif kind == "error":
                    self.err = True
                    self._log("# ERROR: " + ev[1])
                elif kind == "info":
                    self._log("# " + ev[1])
                elif kind == "done":
                    self.running = False
                    self._log("# camera not available" if self.err else "# stopped")
        except queue.Empty:
            pass

    def _frame_log(self, d):
        kind = "odd composite" if d["in_L"] else ("even" if d["x"] % 2 == 0 else "prime/small")
        self._log(f"frame {d['idx']:5d} x={d['x']:<6d} ({kind:<13s}) r={d['r']:<6d} "
                  f"A={d['A']} B={d['B']} node={d['node']:.2f} Y={d['y_bin']} -> Q={d['Q']}")

    def _miner_tick(self):
        seq, rec = self.pipe.resp.read()
        if seq != self._seen_resp:
            self._seen_resp = seq
            if rec is not None and rec["response"] is not None:
                self._log(f"POW SOLVED frame {rec['frame']:5d} lane {rec['lane']}: {rec['lz']} zero bits, "
                          f"{rec['tried']} hashes in {rec['secs']:.1f}s; response latched")
                self._log(rec["response"])

    def _factor_tick(self):
        for s in self.fsearch.drain():
            self._log(s)

    # ---------------------------------------------------------- BTC oracle
    def _btc_auto_on(self):
        return self.btc_auto_thread is not None and self.btc_auto_thread.is_alive() \
            and not self.btc_auto_stop.is_set()

    def btc_refresh(self):
        try:
            price, result, factual, changed = self.btc.fetch_price()
            call = self.btc.last_prediction["guess"]
            self.ctl["btc_last_call"] = call
            self.ctl["btc_last_price"] = price

            checks = self.btc.last_checks
            a, b = checks["driver_a"], checks["driver_b"]
            change = "price changed" if changed else "price unchanged"

            if factual:
                self.ctl["btc_release_armed"] = True
                self._log(
                    f"BTC HIGH/LOW ORACLE CHECK: ${price:,.2f} ({change}); "
                    f"Driver A={a['driver_a']}, Driver B={b['driver_b']}; "
                    f"oracle call={call}; release armed for next even frame.")
            else:
                self.ctl["btc_release_armed"] = False
                self._log(
                    f"BTC HIGH/LOW ORACLE REJECTED: ${price:,.2f} ({change}); "
                    f"Driver A={a['driver_a']}, Driver B={b['driver_b']}; "
                    f"no release armed.")

            if result:
                outcome = "CORRECT" if result["correct"] else "WRONG"
                self._log(
                    f"BTC HIGH/LOW ORACLE RESULT: previous call {result['guess']}; "
                    f"price after check was {result['actual_direction']}; "
                    f"${result['previous']:,.2f} -> ${result['actual']:,.2f}; "
                    f"{outcome}; record {self.btc.correct}/{self.btc.total}")

            self._log(
                f"BTC HIGH/LOW ORACLE NEXT CALL: {call} from "
                f"${self.btc.last_prediction['made_at_price']:,.2f}; "
                f"totals checks={self.btc.fresh_count}, factual={self.btc.factual_count}, "
                f"rejected={self.btc.rejected_count}, released={self.btc.released_count}, "
                f"calls={self.btc.correct}/{self.btc.total}.")
        except Exception as exc:
            self.ctl["btc_release_armed"] = False
            self.btc.last_factual = False
            self._log(f"BTC HIGH/LOW ORACLE ERROR: {exc}; no release armed.")

    def btc_check_async(self):
        threading.Thread(target=self.btc_refresh, daemon=True).start()   # network call off the prompt

    def btc_start_auto(self, seconds=None):
        if seconds is not None:
            self.btc_period = max(1, int(seconds))
        if self._btc_auto_on():
            self._log(f"# BTC auto-check already on; period now {self.btc_period} s")
            return
        self.btc_auto_stop = threading.Event()
        stop = self.btc_auto_stop

        def loop():
            while not stop.is_set():
                self.btc_refresh()
                stop.wait(self.btc_period)
        self.btc_auto_thread = threading.Thread(target=loop, daemon=True)
        self.btc_auto_thread.start()
        self._log(f"BTC HIGH/LOW oracle: automatic checking enabled (every {self.btc_period} s).")

    def btc_stop_auto(self):
        self.btc_auto_stop.set()
        self.ctl["btc_release_armed"] = False
        self._log("BTC HIGH/LOW oracle: automatic checking disabled; armed release cancelled.")

    def btc_guess(self, direction):
        try:
            self.btc.guess(direction)
            call = self.btc.last_prediction["guess"]
            self.ctl["btc_last_call"] = call
            self.ctl["btc_last_price"] = self.btc.pending_price
            self._log(f"BTC HIGH/LOW ORACLE MANUAL CALL: {call} from ${self.btc.pending_price:,.2f}.")
        except Exception as exc:
            self._log(f"BTC HIGH/LOW ORACLE ERROR: {exc}")

    # ---------------------------------------------------------- commands
    def handle(self, line: str):
        line = line.strip()
        if not line:
            return
        try:
            parts = shlex.split(line)
        except ValueError as e:
            self._log(f"? {e}")
            return
        cmd, a = parts[0].lower(), parts[1:]
        fn = getattr(self, "do_" + cmd.replace("-", "_"), None)
        if fn is None:
            self._log(f"? unknown command '{cmd}' (type help)")
            return
        try:
            fn(a)
        except (ValueError, IndexError) as e:
            self._log(f"? {e}" if str(e) else "? bad arguments (type help)")

    def do_help(self, a):
        self._log(HELP)

    def do_start(self, a):
        self._set_y(self.ctl["y"], quiet=True)
        self.start(True)

    def do_preview(self, a):
        self.start(False)

    def do_stop(self, a):
        self.stop()

    def do_quit(self, a):
        self.quit = True

    do_exit = do_quit

    def do_status(self, a):
        self.status()

    def do_frames(self, a):
        self.frames_on = onoff(a[0]) if a else not self.frames_on
        self._log(f"# frame log {'on' if self.frames_on else 'off'}")

    def do_verdict(self, a):
        self.verdict_on = onoff(a[0]) if a else not self.verdict_on
        self._log(f"# verdict messages {'on' if self.verdict_on else 'off'}")

    def do_set(self, a):
        name, val = a[0].lower(), " ".join(a[1:])
        if not val:
            raise ValueError("usage: set NAME VALUE")
        if name in ("thr", "pixeldiff"):
            self.ctl["thr"] = max(4, min(60, int(val)))
            self._log(f"# pixel diff threshold = {self.ctl['thr']}")
        elif name in ("minpx", "min_px"):
            self.ctl["min_px"] = max(5, min(800, int(val)))
            self._log(f"# min moving pixels = {self.ctl['min_px']}")
        elif name == "k":
            self.ctl["k"] = max(1, min(30, int(val)))
            self._log(f"# k frames = {self.ctl['k']}")
        elif name in ("y", "code"):
            self._set_y(val)
        elif name == "force":
            self.ctl["force"] = onoff(val)
            self._log(f"# 'forces HIGH' as OR: {'ON (one-sidedness broken)' if self.ctl['force'] else 'off'}")
        elif name == "overlay":
            self.ctl["overlay"] = onoff(val)
            self._log(f"# motion overlay {'on' if self.ctl['overlay'] else 'off'}")
        else:
            raise ValueError("settable: thr, minpx, k, y, force, overlay")

    def do_code(self, a):
        self._logs(self.code_lines(self.last_stats))

    def do_analysis(self, a):
        self._logs(self.an_lines)

    def do_reset_analysis(self, a):
        self.ctl["reset_an"] = True
        self._log("# driver analysis counters will reset on the next frame")

    def do_hidden(self, a):
        sub = a[0].lower() if a else "status"
        if sub in ("on", "off"):
            self.ctl["hidden_on"] = (sub == "on")
            self._log(f"# hidden number {sub}")
        elif sub == "range":
            n = max(2, min(HIDDEN_RANGE_MAX, int(a[1])))
            if n != self.hidden.size:
                self.hidden.set_range(n)
                self._log(f"# hidden number: range 0..{n - 1}, new secret drawn, counts reset")
        elif sub == "hz":
            lo, hi = GUESS_HZ_LIMITS
            hz = max(lo, min(hi, float(a[1])))
            self.hidden.set_hz(hz)
            self._log(f"# hidden number: clock frequency set to {hz:g} Hz")
        elif sub == "new":
            self.hidden.new_secret()
            self._log("# hidden number: new secret drawn, counts and luck reset")
        elif sub == "reveal":
            self._log(f"# hidden number revealed: {self.hidden.reveal()}")
        elif sub == "status":
            self._logs(self.hidden_text(self.hidden.snapshot()))
        else:
            raise ValueError("hidden on|off|range N|hz F|new|reveal|status")

    def do_search(self, a):
        sub = a[0].lower() if a else "status"
        if sub in ("on", "off"):
            self.ctl["search_on"] = (sub == "on")
            self._log(f"# hash search {sub}")
        elif sub == "bits":
            n = max(2, min(SEARCH_BITS_MAX, int(a[1])))
            if n != self.search.bits:
                self.search.set_bits(n)
                self._log(f"# search: {n}-bit secret, new secret drawn, counts reset")
        elif sub == "new":
            self.search.new_secret()
            self._log("# search: new secret drawn, counts reset")
        elif sub == "reveal":
            self._log(f"# search secret revealed: {short_int(self.search.reveal(), 40)}")
        elif sub == "status":
            self._logs(self.search_text(self.search.snapshot()))
        else:
            raise ValueError("search on|off|bits N|new|reveal|status")

    def do_miners(self, a):
        sub = a[0].lower() if a else "status"
        if sub == "start":
            if self.pipe.miners.running:
                self._log("# miners already running")
                return
            n = int(a[1]) if len(a) > 1 else self.lanes
            self.lanes = max(1, min(MINER_LANES_MAX, n))
            self.pipe.miners.start(self.lanes)
            self._log(f"# PoW miners started: {len(self.pipe.miners.lanes)} threads")
        elif sub == "stop":
            self.pipe.miners.stop()
            self._log("# PoW miners stopped (oracle keeps running)")
        elif sub == "status":
            self._logs(self.miner_text())
        else:
            raise ValueError("miners start [threads]|stop|status")

    def do_header(self, a):
        cfg = self.pipe.set_config(header=" ".join(a) if a else None)
        self._log(f"# response header = {cfg.header}")

    def do_diff(self, a):
        cfg = self.pipe.set_config(difficulty=int(a[0]))
        self._log(f"# PoW difficulty = {cfg.difficulty} leading zero bits")

    def do_latch(self, a):
        sub = a[0].lower() if a else "show"
        if sub == "show":
            self._log(self.latch_text())
        elif sub == "verify":
            v = self.pipe.response()
            if not v:
                self._log("# latch: empty")
            elif verify_response(v):
                self._log(f"# latch: sha256 field matches header:nonce:message, "
                          f"{digest_zero_bits(v.split(DELIM)[3])} leading zero bits (of 256)")
            else:
                self._log("# latch: VERIFY FAILED")
        elif sub == "clear":
            self.pipe.clear_response()
            self._log("# latch: cleared")
        else:
            raise ValueError("latch show|verify|clear")

    def do_factor(self, a):
        if not a:
            raise ValueError("factor N | engine N | gen [bits] | on|off | cancel | reveal | save [file] | status")
        sub = a[0].lower()
        if sub in ("on", "off"):
            self.ctl["fsearch_on"] = (sub == "on")
            self._log(f"# factoriser {sub}")
        elif sub == "status":
            self._logs(self.factor_text(self.fsearch.snapshot()))
        elif sub == "cancel":
            self.fsearch.cancel()
        elif sub == "reveal":
            self.fsearch.reveal()
            v = self.fsearch.snapshot()["secret"]
            self._log(f"# factor revealed: {v}" if v else "# factor not known yet")
        elif sub == "gen":
            self.fsearch.generate(int(a[1]) if len(a) > 1 else FACTOR_BITS_DEFAULT)
        elif sub == "engine":
            self.fsearch.factor_number(parse_number(" ".join(a[1:])))
        elif sub == "save":
            path = a[1] if len(a) > 1 else "factor_numbers.json"
            data = self.fsearch.export()
            if not data["history"] and not data["current"]["n"]:
                self._log("# no numbers yet: use 'factor N' or 'factor gen'")
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except OSError as e:
                self._log(f"? {e}")
                return
            self._log(f"# saved {len(data['history'])} number record(s) to {path}")
        else:
            self.fsearch.search_number(parse_number(" ".join(a)))        # high/low only

    def do_btc(self, a):
        sub = a[0].lower() if a else "status"
        if sub == "check":
            self.btc_check_async()
        elif sub == "auto":
            self.btc_start_auto(int(a[1]) if len(a) > 1 else None)
        elif sub == "stop":
            self.btc_stop_auto()
        elif sub == "call":
            self.btc_guess(a[1].lower())
        elif sub == "status":
            self._logs(self.btc_text())
        else:
            raise ValueError("btc check|auto [s]|stop|call higher|lower|status")

    # ---------------------------------------------------------- main loop
    def _reader(self):
        """Reads typed commands in its own thread and hands them to the main loop."""
        while True:
            try:
                line = input("rp> ")
            except EOFError:
                return
            except Exception:
                return
            self.cmds.put(line)

    def run(self):
        self._log("RP circuit (command prompt).  Type 'help' for commands, 'quit' to leave.")
        if not HAVE_GMP:
            self._log("# gmpy2 not installed: factoring engine uses plain Python ints (pip install gmpy2 for speed)")
        for line in self.args.cmd or []:                 # --cmd "..." can be repeated
            self.cmds.put(line)
        if not self.args.no_start:
            self.start(circuit=not self.args.preview_only)
        if self.args.factor:
            self.cmds.put("factor " + self.args.factor)
        if self.args.btc:
            self.cmds.put(f"btc auto {self.args.btc_period}")
        if not self.args.no_prompt:
            threading.Thread(target=self._reader, daemon=True, name="stdin").start()
        t_end = time.time() + self.args.duration if self.args.duration else None
        try:
            while not self.quit:
                self._poll()
                self._miner_tick()
                self._factor_tick()
                try:
                    while True:
                        self.handle(self.cmds.get_nowait())
                        if self.quit:
                            break
                except queue.Empty:
                    pass
                now = time.time()
                if self.args.status_every and now - self._last_status_t >= self.args.status_every:
                    self._last_status_t = now
                    fps = f"{self._fps:.0f} fps" if self._fps else "no frames yet"
                    S = self.last_stats
                    if S:
                        self._log(f"[status] {fps}  frames {S['frames']}  motion {S['motion']}  x={S['last_x']}  "
                                  f"time codes {S['timecodes_added']}  false accepts {S['false_acc']}")
                if t_end and now >= t_end:
                    self._log("# duration reached")
                    break
                time.sleep(FRAME_INTERVAL / 2000)
        except KeyboardInterrupt:
            self._log("\n# interrupted")
        finally:
            if self.args.final_status:
                self.status()
                self._logs(self.an_lines)
            self.shutdown()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="RP optoelectronic decision circuit driven by a USB camera (command prompt version).")
    ap.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    ap.add_argument("--thr", type=int, default=18, help="pixel diff threshold, 4..60 (default 18)")
    ap.add_argument("--min-px", dest="min_px", type=int, default=30, help="min moving pixels, 5..800 (default 30)")
    ap.add_argument("--k", type=int, default=5, help="frames in the SELECT window, 1..30 (default 5)")
    ap.add_argument("--y", default="1111", help="time code, e.g. 1111  1000|0011  1111&1011  1011+0001")
    ap.add_argument("--force", action="store_true", help="read 'forces HIGH' as OR (breaks one-sidedness)")
    ap.add_argument("--hidden-range", type=int, default=HIDDEN_RANGE_DEFAULT, help="hidden number range 0..N-1")
    ap.add_argument("--hz", type=float, default=GUESS_HZ_DEFAULT, help="hidden number clock frequency, 1..2000 Hz")
    ap.add_argument("--bits", type=int, default=SEARCH_BITS_DEFAULT, help="higher/lower secret size in bits")
    ap.add_argument("--header", default=HEADER_DEFAULT, help="response header text")
    ap.add_argument("--difficulty", type=int, default=MINER_DIFF_DEFAULT, help="PoW leading zero bits, 1..40")
    ap.add_argument("--lanes", type=int, default=MINER_LANES_DEFAULT, help="PoW miner threads")
    ap.add_argument("--no-miners", action="store_true", help="do not start the PoW miners")
    ap.add_argument("--preview-only", action="store_true", help="camera runs, circuit idle")
    ap.add_argument("--no-start", action="store_true", help="do not start the camera; wait for 'start'")
    ap.add_argument("--factor", metavar="N", help="start with this N in the factoriser (high/low only)")
    ap.add_argument("--btc", action="store_true", help="start the BTC HIGH/LOW auto-check")
    ap.add_argument("--btc-period", type=int, default=10, help="BTC auto-check period in seconds")
    ap.add_argument("--frames", action="store_true", help="print one line per motion frame")
    ap.add_argument("--status-every", type=float, default=0.0, metavar="S",
                    help="print a one-line status every S seconds (0 = off)")
    ap.add_argument("--duration", type=float, default=0.0, metavar="S", help="quit by itself after S seconds")
    ap.add_argument("--cmd", action="append", metavar="CMD", help="run a prompt command at start (repeatable)")
    ap.add_argument("--no-prompt", action="store_true", help="no interactive prompt (events only; Ctrl-C to quit)")
    ap.add_argument("--final-status", action="store_true", help="print full status and analysis on exit")
    args = ap.parse_args(argv)
    args.thr = max(4, min(60, args.thr))
    args.min_px = max(5, min(800, args.min_px))
    args.k = max(1, min(30, args.k))
    args.hidden_range = max(2, min(HIDDEN_RANGE_MAX, args.hidden_range))
    args.hz = max(GUESS_HZ_LIMITS[0], min(GUESS_HZ_LIMITS[1], args.hz))
    args.bits = max(2, min(SEARCH_BITS_MAX, args.bits))
    args.difficulty = max(1, min(MINER_DIFF_MAX, args.difficulty))
    args.lanes = max(1, min(MINER_LANES_MAX, args.lanes))
    return args


if __name__ == "__main__":
    mp.freeze_support()
    App(parse_args()).run()
