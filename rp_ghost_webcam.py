#!/usr/bin/env python3
"""
RP-GHOST webcam oracle (software model of the paper's architecture)

Pipeline (from the spec):
  USB camera -> noise bits -> 64-bit ORACLE input
      -> EXTRANONCE2 (SHA-256 derived seed)
      -> LEFT-SIDE MINER (80-byte header, 32-bit nonce search)
      -> VERIFICATION (independent SHA-256d check)

Also emulates the RP decision circuit: Driver A (randomized witness),
Driver B (deterministic check), AND gate, and k-trial repetition.

NOTE: this is LOCAL TEST MODE only. It does not talk to any pool.
The oracle only seeds the search; it does not change the 2^k expected
hashes needed (see section 2 of the paper).

Install:  pip install opencv-python numpy
Run:      python rp_ghost_webcam.py --camera 0 --bits 20
          python rp_ghost_webcam.py --simulate        (no webcam)
          python rp_ghost_webcam.py --benchmark       (compare mining styles)
          python rp_ghost_webcam.py --benchmark --bench-bits 20 --trials 30
"""

import argparse
import hashlib
import os
import struct
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def von_neumann(bits: np.ndarray) -> np.ndarray:
    """Debias: pair (0,1)->0, (1,0)->1, discard equal pairs."""
    n = len(bits) // 2 * 2
    a, b = bits[:n:2], bits[1:n:2]
    keep = a != b
    return a[keep]


# ----------------------------------------------------------------------
# Optical acquisition (USB webcam)
# ----------------------------------------------------------------------
class WebcamOracle:
    def __init__(self, index=0, simulate=False, warmup=10):
        self.simulate = simulate or cv2 is None
        self.cap = None
        self.prev = None
        if self.simulate:
            print("[oracle] simulation mode (os.urandom frames)")
            return
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        for _ in range(warmup):          # let auto-exposure settle
            self.cap.read()
        print(f"[oracle] camera {index} opened")

    def _frame(self) -> np.ndarray:
        if self.simulate:
            return np.frombuffer(os.urandom(64 * 64), dtype=np.uint8).reshape(64, 64)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def sample(self):
        """Grab a frame; return (noise_bits, motion_energy)."""
        cur = self._frame()
        if self.prev is None:
            self.prev = cur
            cur = self._frame()
        diff = cv2.absdiff(cur, self.prev) if not self.simulate else (cur ^ self.prev)
        self.prev = cur
        # Sensor shot noise lives in the least-significant bits of the temporal difference
        bits = (diff & 1).flatten().astype(np.uint8)
        # Mix in LSB of the raw frame as well (second noise source)
        bits = bits ^ (cur & 1).flatten().astype(np.uint8)
        energy = float(np.mean(diff))
        return bits, energy

    def oracle_64(self) -> int:
        """Collect debiased camera bits until we have 64 -> 64-bit oracle input."""
        pool = np.array([], dtype=np.uint8)
        while len(pool) < 64:
            bits, _ = self.sample()
            pool = np.concatenate([pool, von_neumann(bits)])
        out = 0
        for b in pool[:64]:
            out = (out << 1) | int(b)
        return out

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------
# RP decision circuit emulation (Driver A / Driver B / AND gate / k trials)
# ----------------------------------------------------------------------
def rp_decision(oracle: WebcamOracle, k=8, clock_hz=10.0):
    """
    Driver A: randomized witness  -> parity of a noise-bit frame
    Driver B: deterministic check -> motion energy above running median
    Gate    : (A OR B) & (A AND B) == A AND B  (dual agreement)
    Latch   : 74HC74-style sample on a clock edge (period = 1/clock_hz)
    k trials: each trial is independent; accept if ANY trial latches HIGH
    """
    energies = []
    for trial in range(k):
        bits, energy = oracle.sample()
        energies.append(energy)
        A = int(bits.sum() & 1)
        B = int(energy >= np.median(energies))
        gate = (A | B) & (A & B)
        time.sleep(1.0 / clock_hz if not oracle.simulate else 0)  # clock edge
        print(f"  trial {trial+1}/{k}: A={A} B={B} Q={gate}")
        if gate:
            return True
    return False


# ----------------------------------------------------------------------
# Miner (local test job)
# ----------------------------------------------------------------------
def make_header_prefix(extranonce2: bytes) -> bytes:
    """76-byte header prefix; merkle root depends on the oracle-derived extranonce2."""
    version = struct.pack("<I", 0x20000000)
    prevhash = bytes(32)                                   # dummy previous block
    coinbase = b"RP-GHOST-local-test" + extranonce2
    merkle = sha256d(coinbase)
    ntime = struct.pack("<I", int(time.time()))
    nbits = struct.pack("<I", 0x1D00FFFF)
    return version + prevhash + merkle + ntime + nbits


def search_nonce(prefix: bytes, bits: int, max_nonce=2**32):
    target = 1 << (256 - bits)
    t0 = time.time()
    for nonce in range(max_nonce):
        h = sha256d(prefix + struct.pack("<I", nonce))
        if int.from_bytes(h[::-1], "big") < target:
            return nonce, h, nonce + 1, time.time() - t0
    return None, None, max_nonce, time.time() - t0


def verify(prefix: bytes, nonce: int, bits: int) -> bool:
    """Independent SHA-256d check."""
    header = prefix + struct.pack("<I", nonce)
    assert len(header) == 80
    h = hashlib.sha256(hashlib.sha256(header).digest()).digest()
    return int.from_bytes(h[::-1], "big") < (1 << (256 - bits))


def search_nonce_midstate(prefix: bytes, bits: int, max_nonce=2**32):
    """Same search, but hash the first 64 header bytes only once (midstate)."""
    target = 1 << (256 - bits)
    base = hashlib.sha256(prefix[:64])
    tail = prefix[64:]
    t0 = time.time()
    for nonce in range(max_nonce):
        m = base.copy()
        m.update(tail + struct.pack("<I", nonce))
        h = hashlib.sha256(m.digest()).digest()
        if int.from_bytes(h[::-1], "big") < target:
            return nonce, h, nonce + 1, time.time() - t0
    return None, None, max_nonce, time.time() - t0


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s/60:.1f} min"
    if s < 86400:
        return f"{s/3600:.1f} hours"
    if s < 86400 * 365:
        return f"{s/86400:.1f} days"
    return f"{s/(86400*365):.1f} years"


def benchmark(oracle: "WebcamOracle", bits: int, trials: int):
    """
    Compare mining styles on the same difficulty:
      1. oracle-seeded   : camera bits -> SHA-256 -> extranonce2
      2. random seed     : os.urandom
      3. sequential seed : counter
      4. midstate search : os.urandom seed, faster per-hash loop
    Seeding style should NOT change expected hashes (~2^bits).
    """
    import statistics as st

    src_name = "simulated" if oracle.simulate else "camera"
    styles = [
        (f"oracle-seeded ({src_name})", None,
         lambda i: hashlib.sha256(oracle.oracle_64().to_bytes(8, "big")).digest()[:4]),
        ("random seed (os.urandom)", None, lambda i: os.urandom(4)),
        ("sequential seed", None, lambda i: struct.pack("<I", i)),
        ("midstate search", search_nonce_midstate, lambda i: os.urandom(4)),
    ]

    print(f"\nBENCHMARK: {bits} bits, {trials} trials/style, "
          f"expected hashes {2**bits:,}")
    print(f"{'style':32s} {'mean hashes':>12s} {'mean time':>10s} {'H/s':>12s}")
    best_rate = 0.0
    for name, fn, seed_fn in styles:
        fn = fn or search_nonce
        hs, ts = [], []
        for i in range(trials):
            prefix = make_header_prefix(seed_fn(i))
            nonce, _, tried, dt = fn(prefix, bits)
            hs.append(tried)
            ts.append(dt)
        rate = sum(hs) / sum(ts) if sum(ts) else 0
        best_rate = max(best_rate, rate)
        print(f"{name:32s} {st.mean(hs):>12,.0f} {st.mean(ts):>9.2f}s {rate:>12,.0f}")

    print("\nSeeding styles differ only by sampling noise (std dev ~ mean for a")
    print(f"geometric search; error bar ~ +/-{2**bits/trials**0.5:,.0f} hashes).")

    print(f"\nProjected expected time at {best_rate:,.0f} H/s (single core):")
    print(f"{'bits':>5s} {'expected hashes':>20s} {'time':>12s}")
    for b in (20, 24, 28, 32, 40, 48):
        print(f"{b:>5d} {2**b:>20,d} {fmt_time(2**b / best_rate):>12s}")
    print("Expectation values, not upper bounds. Pool/network targets are far harder.")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RP-GHOST webcam oracle (local test)")
    ap.add_argument("--camera", type=int, default=0, help="USB camera index")
    ap.add_argument("--bits", type=int, default=20, help="TEST_BITS (2^bits expected hashes)")
    ap.add_argument("--rounds", type=int, default=1, help="number of oracle seeds to try")
    ap.add_argument("--k", type=int, default=8, help="RP decision trials")
    ap.add_argument("--simulate", action="store_true", help="no webcam, use os.urandom")
    ap.add_argument("--midstate", action="store_true", help="use faster midstate nonce search")
    ap.add_argument("--benchmark", action="store_true", help="compare mining styles and exit")
    ap.add_argument("--bench-bits", type=int, default=18, help="difficulty bits for --benchmark")
    ap.add_argument("--trials", type=int, default=20, help="trials per style for --benchmark")
    args = ap.parse_args()

    print(f"TEST_BITS={args.bits}  expected hashes ~ {2**args.bits:,}")
    oracle = WebcamOracle(args.camera, simulate=args.simulate)

    if args.benchmark:
        try:
            benchmark(oracle, args.bench_bits, args.trials)
        finally:
            oracle.close()
        return

    search = search_nonce_midstate if args.midstate else search_nonce
    try:
        for r in range(args.rounds):
            print(f"\n=== round {r+1} ===")

            print("[rp] decision circuit:")
            select = rp_decision(oracle, k=args.k)
            print(f"[rp] SELECT = {select}")

            seed = oracle.oracle_64()
            print(f"[oracle] 64-bit input : {seed:016x}")

            extranonce2 = hashlib.sha256(seed.to_bytes(8, "big")).digest()[:4]
            print(f"[seed]   extranonce2  : {extranonce2.hex()}")

            prefix = make_header_prefix(extranonce2)
            nonce, h, tried, dt = search(prefix, args.bits)
            if nonce is None:
                print("[miner] no nonce in 32-bit space; next round re-seeds")
                continue

            rate = tried / dt if dt else 0
            print(f"[miner]  nonce={nonce} (0x{nonce:08x})  tried={tried:,}  "
                  f"{dt:.2f}s  ~{rate:,.0f} H/s")
            print(f"[miner]  hash = {h[::-1].hex()}")
            print(f"[verify] independent SHA-256d check: {verify(prefix, nonce, args.bits)}")
            print("[note]   local test share only; not a valid pool share")
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
