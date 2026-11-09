
#!/usr/bin/env python3
"""
rpghost_full_dynamic.py - Complete self-contained RP-GHOST evaluation harness 
with fully dynamic metric calculations (no static placeholders).

Usage:
  python rpghost_full_dynamic.py --synthetic --frames 111 --out sim.csv
  python rpghost_full_dynamic.py --device 0 --frames 111 --out run.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

FPS = 30.0
W, H = 128, 96
N_PIX = W * H

# ------------------------------------------------------------------ Core Logic
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, int(math.isqrt(n)) + 1, 2):
        if n % d == 0:
            return False
    return True

class Prereg:
    def __init__(self, hidden_m: int, tts_k: int, n_frames: int):
        self.hidden_m = hidden_m
        self.tts_k = tts_k
        self.n_frames = n_frames
        self._seed = hashlib.sha256(f"{hidden_m}:{tts_k}:{n_frames}".encode()).digest()

    def digest(self) -> str:
        return hashlib.sha256(self._seed).hexdigest()[:16]

class OSRandomNonce:
    def __init__(self):
        self.commitment = "os-random-stream"

    def get(self, t: int) -> bytes:
        return hashlib.sha256(t.to_bytes(8, "big") + np.random.bytes(16)).digest()

class PrecommittedNonce:
    def __init__(self, nonces: list[str]):
        self.nonces = nonces
        joined = "".join(nonces).encode()
        self.commitment = hashlib.sha256(joined).hexdigest()[:16]

    def get(self, t: int) -> bytes:
        return bytes.fromhex(self.nonces[t % len(self.nonces)])

class Harness:
    def __init__(self, src, nsrc, pre: Prereg):
        self.src = src
        self.nsrc = nsrc
        self.pre = pre
        self.records = []

    def run(self, frames: int):
        for t in range(frames):
            nonce = self.nsrc.get(t)
            raw_bytes, x, r, Q, in_L, feat, guess = self.src(t)
            record = {
                "t": t,
                "x": x,
                "r": r,
                "Q": Q,
                "in_L": in_L,
                "feat": feat,
                "guess": guess,
                "nonce": nonce.hex() if isinstance(nonce, bytes) else str(nonce)
            }
            self.records.append(record)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def analyse(records: list[dict], pre: Prereg) -> dict:
    n = len(records)
    if n == 0:
        return {"error": "no records"}

    # 1. Hidden-number test (P1)
    hits = sum(1 for r in records if r["guess"] == (r["x"] & ((1 << pre.hidden_m) - 1)))
    expected_hidden = n * (1.0 / (1 << pre.hidden_m)) if pre.hidden_m > 0 else 0.0
    z_hidden = (hits - expected_hidden) / math.sqrt(max(expected_hidden, 1e-5))

    # 2. TTS test (P2) - Fully dynamic speedup calculation
    solutions = sum(1 for r in records if r["Q"] == 1)
    expected_tts = n * 0.25
    z_tts = (solutions - expected_tts) / math.sqrt(max(expected_tts * 0.75, 1e-5))
    
    if solutions > 0:
        observed_hashes_per_sol = n / solutions
        expected_hashes_per_sol = 1.0 / 0.25
        speedup_val = round(expected_hashes_per_sol / observed_hashes_per_sol, 2)
    else:
        speedup_val = 0.0

    # 3. Bit Bias test (P3) - Fully dynamic bit extraction & Bonferroni correction
    guesses_arr = np.array([r["guess"] for r in records], dtype=int)
    m_bits = pre.hidden_m if pre.hidden_m > 0 else 8
    
    worst_bit_idx = 0
    worst_bias = 0.0
    max_z = 0.0
    sd_floor_val = float(math.sqrt(0.25 / max(n, 1)))
    
    for b_idx in range(m_bits):
        bit_vals = (guesses_arr >> b_idx) & 1
        p_bit = np.mean(bit_vals) if n > 0 else 0.5
        bias_b = float(p_bit - 0.5)
        z_b = bias_b / max(sd_floor_val, 1e-6)
        if abs(z_b) > abs(max_z):
            max_z = z_b
            worst_bit_idx = b_idx
            worst_bias = bias_b

    raw_p = 2.0 * (1.0 - normal_cdf(abs(max_z))) if n > 0 else 1.0
    p_bonf = min(1.0, raw_p * m_bits)

    # 4. Correlation test (P4)
    if n > 1:
        q_arr = np.array([r["Q"] for r in records], dtype=float)
        corr_rho = float(np.corrcoef(q_arr[:-1], q_arr[1:])[0, 1]) if np.std(q_arr) > 0 else 0.0
    else:
        corr_rho = 0.0

    # 5. Witness detection test (P5)
    composite_records = [r for r in records if r["in_L"] == 1]
    n_comp = len(composite_records)
    if n_comp > 0:
        p_hat = sum(1 for r in composite_records if r["r"] > 1 and r["x"] % r["r"] == 0) / n_comp
        z_witness = (p_hat - 0.75) / math.sqrt(0.75 * 0.25 / n_comp)
    else:
        p_hat = 0.0
        z_witness = 0.0

    return {
        "hidden": {
            "n": n, 
            "hits": hits, 
            "expected": round(expected_hidden, 4), 
            "rate": round(hits / max(n, 1), 4), 
            "chance": round(1.0 / (1 << pre.hidden_m), 4) if pre.hidden_m > 0 else 0.0, 
            "z": round(z_hidden, 3)
        },
        "tts": {
            "hashes": n, 
            "solutions": solutions, 
            "expected": round(expected_tts, 4), 
            "z": round(z_tts, 3), 
            "speedup": speedup_val
        },
        "bias": {
            "n": n, 
            "worst_bit": int(worst_bit_idx), 
            "bias": round(worst_bias, 4), 
            "z": round(max_z, 3), 
            "p_bonferroni": round(p_bonf, 4), 
            "sd_floor": round(sd_floor_val, 4)
        },
        "corr": {
            "n": n, 
            "rho": round(corr_rho, 4), 
            "z": round(corr_rho * math.sqrt(n), 3)
        },
        "witness": {
            "n": n_comp, 
            "p_hat": round(p_hat, 4), 
            "z": round(z_witness, 3)
        },
        "verdict": {
            "P1 hidden-number": "null" if abs(z_hidden) < 3.0 else "signal",
            "P2 TTS": "null" if abs(z_tts) < 3.0 else "signal",
            "P3 bit bias": "null" if p_bonf >= 0.05 else "signal",
            "P4 correlation": "null" if abs(corr_rho * math.sqrt(n)) < 3.0 else "signal",
            "P5 witness": "null" if abs(z_witness) < 3.0 else "signal"
        }
    }

def est_correlation_vs_lag(records: list[dict], lags: list[int]) -> dict:
    q_arr = np.array([r["Q"] for r in records], dtype=float)
    out = {}
    for lag in lags:
        if len(q_arr) > lag and np.std(q_arr) > 0:
            rho = float(np.corrcoef(q_arr[:-lag], q_arr[lag:])[0, 1])
        else:
            rho = 0.0
        out[lag] = {"rho": round(rho, 4)}
    return out

# ------------------------------------------------------------------ Frame Sources[cite: 2]
def frame_features(binary: np.ndarray, gray: np.ndarray, m: int):
    x = int(binary.sum())
    r = 2 + int("".join(str(int(b)) for b in binary.flat[:7]), 2)
    in_L = int(x > 1 and not is_prime(x))
    A = in_L
    B = int(1 < r < x and x % r == 0)
    Y = 1
    Q = A * B * Y
    feature = float(gray.mean())
    digest = hashlib.sha256(binary.tobytes()).digest()
    guess = int.from_bytes(digest[:4], "big") & ((1 << m) - 1) if m else -1
    return x, r, Q, in_L, feature, guess

class CameraSource:
    def __init__(self, device: int = 0, threshold: str | int = "otsu", m: int = 8,
                 warmup: int = 10):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for live camera capture.")
        self.device = device
        self.threshold = threshold
        self.m = m
        self.timestamps = []
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device}")
        for i in range(warmup):
            self.cap.read()
            time.sleep(0.01)

    def _binarise(self, gray: np.ndarray) -> np.ndarray:
        if self.threshold == "otsu":
            _, b = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold == "median":
            b = (gray > np.median(gray)).astype(np.uint8)
        else:
            b = (gray > int(self.threshold)).astype(np.uint8)
        return b

    def __call__(self, t: int):
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Camera read failed at frame {t}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        binary = self._binarise(gray)
        self.timestamps.append(time.monotonic())
        x, r, Q, in_L, feat, guess = frame_features(binary, gray, self.m)
        return gray.tobytes(), x, r, Q, in_L, feat, guess

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def measured_fps(self) -> float:
        if len(self.timestamps) < 2:
            return float("nan")
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])

class SyntheticCamera:
    def __init__(self, m: int = 8, level: float = 120.0, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.level = level
        self.m = m
        self.timestamps = []

    def __call__(self, t: int):
        gray = np.clip(self.rng.poisson(self.level, size=(H, W)), 0, 255).astype(np.uint8)
        binary = (gray > np.median(gray)).astype(np.uint8)
        self.timestamps.append(time.monotonic())
        x, r, Q, in_L, feat, guess = frame_features(binary, gray, self.m)
        return gray.tobytes(), x, r, Q, in_L, feat, guess

    def close(self):
        pass

    def measured_fps(self) -> float:
        return float("nan")

# ------------------------------------------------------------------ Execution Entry[cite: 2]
def main():
    ap = argparse.ArgumentParser(description="Run the RP-GHOST protocol pipeline dynamically")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="no camera; Poisson frames")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--m", type=int, default=16, help="hidden-number bits (P1)")
    ap.add_argument("--k", type=int, default=16, help="header target bits (P2/P3)")
    ap.add_argument("--threshold", default="otsu", help="otsu | median | 0-255")
    ap.add_argument("--out", default="run.csv")
    args = ap.parse_args()

    pre = Prereg(hidden_m=args.m, tts_k=args.k, n_frames=args.frames)
    print(f"prereg sha256: {pre.digest()}")

    nsrc = OSRandomNonce()
    src = SyntheticCamera(args.m) if args.synthetic else CameraSource(
        args.device, int(args.threshold) if args.threshold.isdigit() else args.threshold, args.m)
    
    try:
        h = Harness(src, nsrc, pre)
        t0 = time.time()
        h.run(args.frames)
        print(f"{args.frames} frames processed in {time.time()-t0:.1f} s")
    finally:
        src.close()

    h.save(args.out)
    print(f"saved output to {args.out}")
    
    res = analyse(h.records, pre)
    for key in ("hidden", "tts", "bias", "corr", "witness"):
        print(key, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[key].items()})
    print("verdict:", res["verdict"])
    
    lag = est_correlation_vs_lag(h.records, [1, 2, 5, 10, 30])
    print("rho(tau):", {L: round(v["rho"], 4) for L, v in lag.items()})

if __name__ == "__main__":
    main()

