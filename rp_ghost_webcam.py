
#!/usr/bin/env python3
"""
RP-GHOST Webcam Oracle — Local Research Implementation
Author: George Wagenknecht

ARCHITECTURE
    Webcam / simulated entropy source
        |
        v
    Temporal image differences
        |
        v
    LSB extraction + Von Neumann debiasing
        |
        v
    64-bit oracle candidate
        |
        v
    SHA-256-derived extranonce2
        |
        v
    Local 80-byte header construction
        |
        v
    LEFT-SIDE MINER: sequential nonce search
        |
        v
    INDEPENDENT SHA-256d recomputation

RP DECISION CIRCUIT
    Driver A: parity of extracted noise bits
    Driver B: motion energy exceeds calibrated threshold
    Gate:     A AND B
    Latch:    simulated clock-edge sampling
    Trials:   accept if ANY trial latches HIGH

IMPORTANT
    This is a local research implementation.
    It does NOT connect to Bitcoin Stratum pools.
    It does NOT submit Bitcoin shares.
    It does NOT establish that webcam data is cryptographically secure.

    The oracle selects candidates. It does not bypass proof-of-work.
    A candidate is valid only if its independently recomputed hash
    satisfies the configured target.

INSTALL
    pip install numpy opencv-python

EXAMPLES
    python rp_ghost_webcam.py --camera 0
    python rp_ghost_webcam.py --simulate
    python rp_ghost_webcam.py --benchmark --simulate
    python rp_ghost_webcam.py --benchmark-nonce --simulate
    python rp_ghost_webcam.py --benchmark-nonce --nonce-bits 16 --bits 12
    python rp_ghost_webcam.py --midstate --simulate
"""

import argparse
import hashlib
import os
import statistics
import struct
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# CONFIGURATION / VALIDATION
# ============================================================

def validate_bits(bits):
    if not 1 <= bits <= 255:
        raise ValueError("Difficulty bits must be between 1 and 255.")


def validate_nonce_bits(bits):
    if not 1 <= bits <= 32:
        raise ValueError("Nonce width must be between 1 and 32.")


def target_from_bits(bits):
    validate_bits(bits)
    return 1 << (256 - bits)


def hash_meets_target(digest, target):
    """
    Bitcoin-style interpretation of the double-SHA-256 digest.

    The displayed digest is conventionally byte-reversed.
    """
    return int.from_bytes(digest[::-1], "big") < target


# ============================================================
# HASHING
# ============================================================

def sha256d(data):
    """Double SHA-256."""
    first = hashlib.sha256(data).digest()
    return hashlib.sha256(first).digest()


def hash_header(prefix, nonce):
    """Hash an 80-byte header."""
    if len(prefix) != 76:
        raise ValueError("Header prefix must be exactly 76 bytes.")

    if not 0 <= nonce <= 0xFFFFFFFF:
        raise ValueError("Nonce must fit into 32 bits.")

    header = prefix + struct.pack("<I", nonce)

    if len(header) != 80:
        raise ValueError("Constructed header is not 80 bytes.")

    return sha256d(header)


# ============================================================
# ENTROPY EXTRACTION
# ============================================================

def von_neumann(bits):
    """
    Von Neumann debiasing.

    01 -> 0
    10 -> 1
    00 -> discard
    11 -> discard

    This reduces first-order bias under suitable assumptions.
    It does NOT establish cryptographic entropy or independence.
    """
    n = (len(bits) // 2) * 2

    if n == 0:
        return np.array([], dtype=np.uint8)

    a = bits[:n:2]
    b = bits[1:n:2]

    keep = a != b

    return a[keep]


class WebcamOracle:

    def __init__(
        self,
        index=0,
        simulate=False,
        warmup=10,
        max_entropy_frames=10000
    ):
        self.simulate = bool(simulate or cv2 is None)
        self.cap = None
        self.prev = None
        self.max_entropy_frames = max_entropy_frames

        if self.simulate:
            print("[oracle] simulation mode enabled")
            print("[oracle] simulated frames generated using os.urandom")
            return

        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera index {index}"
            )

        for _ in range(warmup):
            self.cap.read()

        print(f"[oracle] camera {index} opened")

    def _frame(self):
        if self.simulate:
            raw = os.urandom(64 * 64)

            return np.frombuffer(
                raw,
                dtype=np.uint8
            ).reshape(64, 64)

        ok, frame = self.cap.read()

        if not ok:
            raise RuntimeError("Camera frame acquisition failed.")

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

    def sample(self):
        """
        Returns:
            noise_bits
            motion_energy

        The extracted bits are experimental sensor-derived values.
        Their entropy has not been independently established.
        """
        cur = self._frame()

        if self.prev is None:
            self.prev = cur
            cur = self._frame()

        if self.simulate:
            diff = np.bitwise_xor(cur, self.prev)
        else:
            diff = cv2.absdiff(cur, self.prev)

        self.prev = cur

        temporal_bits = (diff & 1).flatten().astype(np.uint8)

        raw_bits = (cur & 1).flatten().astype(np.uint8)

        # Mix temporal difference and raw-frame LSB measurements.
        bits = temporal_bits ^ raw_bits

        energy = float(np.mean(diff))

        return bits, energy

    def oracle_64(self):
        """
        Collect 64 debiased bits.

        A maximum frame limit prevents an endless loop if the source
        produces too few unequal bit pairs.
        """
        collected = []

        count = 0

        while sum(len(x) for x in collected) < 64:
            if count >= self.max_entropy_frames:
                raise RuntimeError(
                    "Unable to collect 64 debiased bits. "
                    "Check the entropy source."
                )

            bits, _ = self.sample()

            filtered = von_neumann(bits)

            if len(filtered):
                collected.append(filtered)

            count += 1

        pool = np.concatenate(collected)[:64]

        result = 0

        for bit in pool:
            result = (result << 1) | int(bit)

        return result

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ============================================================
# RP DECISION CIRCUIT
# ============================================================

def calibrate_driver_b(
    oracle,
    calibration_samples=20,
    mad_multiplier=3.0
):
    """
    Calibrate Driver B using a separate baseline acquisition.

    The current trial is NOT included in its own threshold calculation.

    Threshold:
        median + max(1.0, MAD * 1.4826 * multiplier)

    The MAD scaling factor makes the threshold more robust to outliers
    than a simple mean / standard deviation threshold.
    """
    energies = []

    print("\n[rp] calibrating Driver B")

    for i in range(calibration_samples):
        _, energy = oracle.sample()

        energies.append(energy)

    median_energy = float(np.median(energies))

    mad = float(
        np.median(
            np.abs(
                np.array(energies) - median_energy
            )
        )
    )

    robust_sigma = 1.4826 * mad

    threshold = median_energy + max(
        1.0,
        robust_sigma * mad_multiplier
    )

    print(
        f"[rp] baseline median : {median_energy:.4f}"
    )

    print(
        f"[rp] baseline MAD    : {mad:.4f}"
    )

    print(
        f"[rp] Driver B limit  : {threshold:.4f}"
    )

    return threshold


def rp_decision(
    oracle,
    k=8,
    clock_hz=10.0,
    calibration_samples=20,
    mad_multiplier=3.0
):
    """
    Software model of the RP decision circuit.

    Driver A:
        Parity of the acquired noise-bit frame.

    Driver B:
        Motion energy exceeds the independently calibrated threshold.

    Gate:
        A AND B.

    Latch:
        Software timing delay approximating a clock-edge sample.

    Repetition:
        Accept if ANY trial latches HIGH.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    if clock_hz <= 0:
        raise ValueError("clock_hz must be positive.")

    threshold = calibrate_driver_b(
        oracle,
        calibration_samples,
        mad_multiplier
    )

    print("\n[rp] decision trials")

    for trial in range(k):

        bits, energy = oracle.sample()

        # Driver A: parity.
        A = int(np.sum(bits, dtype=np.uint64) & 1)

        # Driver B: independent calibrated threshold.
        B = int(energy > threshold)

        # Logical agreement.
        gate = A & B

        # Simulated clock edge.
        if not oracle.simulate:
            time.sleep(1.0 / clock_hz)

        Q = gate

        print(
            f"  trial {trial + 1}/{k} "
            f"A={A} "
            f"B={B} "
            f"energy={energy:.4f} "
            f"threshold={threshold:.4f} "
            f"Q={Q}"
        )

        if Q:
            print("[rp] SELECT = HIGH")
            return True

    print("[rp] SELECT = LOW")

    return False


# ============================================================
# LOCAL HEADER CONSTRUCTION
# ============================================================

def make_header_prefix(extranonce2):
    """
    Construct a LOCAL TEST header prefix.

    This is deliberately not a real Bitcoin mining job.

    Dummy fields:
        Previous block hash: all zeros
        Coinbase: local test string
        Merkle root: locally generated
        nBits: fixed local test value

    A real Stratum implementation must use the actual pool job,
    coinbase transaction, Merkle branches and network parameters.
    """
    if not isinstance(extranonce2, bytes):
        raise TypeError("extranonce2 must be bytes.")

    version = struct.pack(
        "<I",
        0x20000000
    )

    prevhash = bytes(32)

    coinbase = (
        b"RP-GHOST-local-test"
        + extranonce2
    )

    merkle = sha256d(coinbase)

    ntime = struct.pack(
        "<I",
        int(time.time())
    )

    nbits = struct.pack(
        "<I",
        0x1D00FFFF
    )

    prefix = (
        version
        + prevhash
        + merkle
        + ntime
        + nbits
    )

    if len(prefix) != 76:
        raise ValueError(
            f"Invalid header prefix length: {len(prefix)}"
        )

    return prefix


def oracle_to_extranonce2(seed):
    """
    Convert a 64-bit oracle input into a 4-byte seed.
    """
    if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Oracle seed must be 64 bits.")

    return hashlib.sha256(
        seed.to_bytes(8, "big")
    ).digest()[:4]


# ============================================================
# SEQUENTIAL NONCE MINER
# ============================================================

def search_nonce(
    prefix,
    bits,
    max_nonce=2**32
):
    """
    Sequential nonce search.

    Returns:
        nonce
        digest
        hashes_tried
        elapsed_seconds
    """
    target = target_from_bits(bits)

    if not 1 <= max_nonce <= 2**32:
        raise ValueError("max_nonce must be between 1 and 2^32.")

    start = time.perf_counter()

    for nonce in range(max_nonce):

        digest = hash_header(
            prefix,
            nonce
        )

        if hash_meets_target(digest, target):

            elapsed = time.perf_counter() - start

            return (
                nonce,
                digest,
                nonce + 1,
                elapsed
            )

    elapsed = time.perf_counter() - start

    return (
        None,
        None,
        max_nonce,
        elapsed
    )


# ============================================================
# MIDSTATE SEARCH
# ============================================================

def search_nonce_midstate(
    prefix,
    bits,
    max_nonce=2**32
):
    """
    SHA-256 midstate optimization.

    The first 64 bytes of the header are processed once.
    Each nonce then requires processing only the remaining
    16-byte header tail, followed by the second SHA-256.

    This changes hashing efficiency, not proof-of-work probability.
    """
    target = target_from_bits(bits)

    if len(prefix) != 76:
        raise ValueError("Prefix must be 76 bytes.")

    if not 1 <= max_nonce <= 2**32:
        raise ValueError("max_nonce must be between 1 and 2^32.")

    base = hashlib.sha256(prefix[:64])

    tail = prefix[64:]

    start = time.perf_counter()

    for nonce in range(max_nonce):

        first = base.copy()

        first.update(
            tail + struct.pack("<I", nonce)
        )

        digest = hashlib.sha256(
            first.digest()
        ).digest()

        if hash_meets_target(digest, target):

            elapsed = time.perf_counter() - start

            return (
                nonce,
                digest,
                nonce + 1,
                elapsed
            )

    elapsed = time.perf_counter() - start

    return (
        None,
        None,
        max_nonce,
        elapsed
    )


# ============================================================
# INDEPENDENT VERIFICATION
# ============================================================

def verify(prefix, nonce, bits):
    """
    Independently recompute SHA-256d for the supplied candidate.

    This verifies the local header and local target.
    It does not establish pool acceptance.
    """
    if len(prefix) != 76:
        return False

    if not 0 <= nonce <= 0xFFFFFFFF:
        return False

    header = prefix + struct.pack("<I", nonce)

    if len(header) != 80:
        return False

    first_hash = hashlib.sha256(header).digest()

    second_hash = hashlib.sha256(first_hash).digest()

    target = target_from_bits(bits)

    return hash_meets_target(
        second_hash,
        target
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.3f} s"

    if seconds < 3600:
        return f"{seconds / 60:.2f} min"

    if seconds < 86400:
        return f"{seconds / 3600:.2f} hours"

    if seconds < 86400 * 365:
        return f"{seconds / 86400:.2f} days"

    return f"{seconds / (86400 * 365):.2f} years"


def print_result(
    nonce,
    digest,
    tried,
    elapsed,
    bits
):
    rate = tried / elapsed if elapsed > 0 else 0

    print("\n[miner] SEARCH RESULT")

    if nonce is None:
        print("[miner] No valid nonce found.")
        print(f"[miner] Hashes tried: {tried:,}")
        print(f"[miner] Elapsed: {elapsed:.3f}s")
        print(f"[miner] Hash rate: {rate:,.0f} H/s")
        return

    print(
        f"[miner] nonce       : {nonce} "
        f"(0x{nonce:08x})"
    )

    print(
        f"[miner] hashes tried: {tried:,}"
    )

    print(
        f"[miner] elapsed     : {elapsed:.3f}s"
    )

    print(
        f"[miner] hash rate   : {rate:,.0f} H/s"
    )

    print(
        f"[miner] digest      : {digest[::-1].hex()}"
    )

    print(
        f"[miner] target bits : {bits}"
    )


# ============================================================
# BENCHMARK 1 — SEARCH STRATEGIES
# ============================================================

def benchmark(
    oracle,
    bits,
    trials
):
    """
    Compare different candidate-seeding strategies.

    All strategies use the same local header format and target.

    The oracle does not reduce the expected number of hashes.
    The benchmark measures observed search variation and throughput.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1.")

    target = target_from_bits(bits)

    styles = [
        (
            "oracle-seeded",
            lambda i: oracle_to_extranonce2(
                oracle.oracle_64()
            ),
            search_nonce
        ),
        (
            "random-seeded",
            lambda i: os.urandom(4),
            search_nonce
        ),
        (
            "sequential-seeded",
            lambda i: struct.pack("<I", i),
            search_nonce
        ),
        (
            "midstate-search",
            lambda i: os.urandom(4),
            search_nonce_midstate
        )
    ]

    print("\n" + "=" * 76)
    print("RP-GHOST SEARCH STRATEGY BENCHMARK")
    print("=" * 76)

    print(f"Difficulty: {bits} bits")
    print(f"Expected hashes per successful search: {2**bits:,}")
    print(f"Trials per strategy: {trials}")

    print("\n{:<24s} {:>14s} {:>12s} {:>14s}".format(
        "Strategy",
        "Mean hashes",
        "Mean time",
        "Hash rate"
    ))

    best_rate = 0.0

    for name, seed_function, search_function in styles:

        hash_counts = []
        durations = []
        successes = 0

        for i in range(trials):

            seed = seed_function(i)

            prefix = make_header_prefix(seed)

            nonce, digest, tried, elapsed = search_function(
                prefix,
                bits
            )

            hash_counts.append(tried)
            durations.append(elapsed)

            if nonce is not None:
                successes += 1

                # Independently check the returned candidate.
                if not verify(prefix, nonce, bits):
                    raise RuntimeError(
                        "Independent verification failed."
                    )

        total_hashes = sum(hash_counts)
        total_time = sum(durations)

        rate = (
            total_hashes / total_time
            if total_time > 0
            else 0
        )

        best_rate = max(best_rate, rate)

        print(
            f"{name:<24s} "
            f"{statistics.mean(hash_counts):>14,.0f} "
            f"{statistics.mean(durations):>11.3f}s "
            f"{rate:>14,.0f}"
        )

        print(
            f"  successful searches: {successes}/{trials}"
        )

    print("\n[benchmark] Interpretation")

    print(
        "Different seeds change the header and therefore the "
        "individual search outcome."
    )

    print(
        "They do not change the probability of an individual "
        "hash meeting the target."
    )

    print(
        f"Expected work remains approximately {2**bits:,} hashes."
    )

    if best_rate > 0:
        print("\nProjected search times at measured best rate:")

        for b in (20, 24, 28, 32, 40, 48):

            expected = 2**b

            projected = expected / best_rate

            print(
                f"  {b:>2d} bits : "
                f"{expected:>18,d} hashes : "
                f"{fmt_time(projected)}"
            )

    print("\n[benchmark] Target value:")
    print(f"  {target:064x}")


# ============================================================
# BENCHMARK 2 — ORACLE NONCE VS SEQUENTIAL REACH
# ============================================================

def benchmark_nonce(
    oracle,
    nonce_bits,
    bits,
    trials
):
    """
    Compare two ways of evaluating the SAME oracle-selected nonce.

    TEST A — SEQUENTIAL REACH
        Hash nonce 0, 1, 2, ... until the predetermined oracle
        nonce is reached.

    TEST B — DIRECT ORACLE EVALUATION
        Hash only the predetermined oracle nonce.

    Both tests use the exact same header prefix and target.

    IMPORTANT:
        This is a time-to-reach benchmark, not a mining-success
        benchmark. The sequential test deliberately continues
        until the predetermined nonce, even if an earlier nonce
        happens to satisfy the target.
    """
    validate_nonce_bits(nonce_bits)
    validate_bits(bits)

    if trials < 1:
        raise ValueError("trials must be at least 1.")

    mask = (1 << nonce_bits) - 1

    target = target_from_bits(bits)

    sequential_times = []
    direct_times = []

    sequential_hash_counts = []

    successful_candidates = 0

    print("\n" + "=" * 82)
    print("RP-GHOST ORACLE NONCE VS SEQUENTIAL REACH")
    print("=" * 82)

    print(f"Nonce width: 2^{nonce_bits}")
    print(f"Test target: {bits} bits")
    print(f"Trials: {trials}")

    print(
        "\n"
        f"{'Trial':>5s} "
        f"{'Oracle nonce':>14s} "
        f"{'Sequential hashes':>19s} "
        f"{'Reach time':>12s} "
        f"{'Direct time':>12s} "
        f"{'Valid':>8s}"
    )

    for trial in range(trials):

        # Oracle produces one predetermined candidate.
        oracle_value = oracle.oracle_64()

        nonce = oracle_value & mask

        # Both tests use exactly the same prefix.
        extranonce2 = oracle_to_extranonce2(oracle_value)

        prefix = make_header_prefix(extranonce2)

        # ----------------------------------------------------
        # TEST A: sequentially reach the predetermined nonce
        # ----------------------------------------------------

        sequential_start = time.perf_counter()

        sequential_digest = None

        for current_nonce in range(nonce + 1):

            sequential_digest = hash_header(
                prefix,
                current_nonce
            )

        sequential_elapsed = (
            time.perf_counter() - sequential_start
        )

        sequential_count = nonce + 1

        # ----------------------------------------------------
        # TEST B: directly evaluate the oracle nonce
        # ----------------------------------------------------

        direct_start = time.perf_counter()

        direct_digest = hash_header(
            prefix,
            nonce
        )

        direct_elapsed = (
            time.perf_counter() - direct_start
        )

        # ----------------------------------------------------
        # Compare exact same candidate and digest
        # ----------------------------------------------------

        if sequential_digest != direct_digest:
            raise RuntimeError(
                "Sequential and direct hashes do not match."
            )

        valid = hash_meets_target(
            direct_digest,
            target
        )

        if valid:
            successful_candidates += 1

        sequential_times.append(sequential_elapsed)

        direct_times.append(direct_elapsed)

        sequential_hash_counts.append(sequential_count)

        print(
            f"{trial + 1:>5d} "
            f"{nonce:>14,d} "
            f"{sequential_count:>19,d} "
            f"{sequential_elapsed:>11.4f}s "
            f"{direct_elapsed * 1e6:>10.2f}us "
            f"{'YES' if valid else 'NO':>8s}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_reach_hashes = statistics.mean(
        sequential_hash_counts
    )

    mean_reach_time = statistics.mean(
        sequential_times
    )

    mean_direct_time = statistics.mean(
        direct_times
    )

    total_sequential_hashes = sum(
        sequential_hash_counts
    )

    total_sequential_time = sum(
        sequential_times
    )

    sequential_rate = (
        total_sequential_hashes / total_sequential_time
        if total_sequential_time > 0
        else 0
    )

    print("\n" + "-" * 82)

    print("[results] Sequential reach")

    print(
        f"Mean hashes to reach oracle nonce: "
        f"{mean_reach_hashes:,.0f}"
    )

    print(
        f"Mean reach time: "
        f"{mean_reach_time:.6f}s"
    )

    print(
        f"Measured sequential throughput: "
        f"{sequential_rate:,.0f} H/s"
    )

    print("\n[results] Direct oracle evaluation")

    print(
        f"Mean direct evaluation time: "
        f"{mean_direct_time * 1e6:.2f} microseconds"
    )

    print(
        f"Direct evaluations per trial: 1"
    )

    print("\n[results] Proof-of-work validity")

    print(
        f"Oracle candidates meeting target: "
        f"{successful_candidates}/{trials}"
    )

    print(
        f"Expected success probability: "
        f"approximately 1 in {2**bits:,}"
    )

    print("\n[interpretation]")

    print(
        "The direct evaluation reaches the predetermined candidate "
        "without hashing the preceding nonce values."
    )

    print(
        "It does not establish that the candidate is a valid proof "
        "of work."
    )

    print(
        "The candidate has the same validity regardless of whether "
        "it was reached sequentially or selected directly."
    )

    print(
        "Expected work to discover a valid candidate remains "
        f"approximately 2^{bits} = {2**bits:,} independent hashes."
    )

    print(
        "\nThis benchmark measures candidate reachability, "
        "not a reduction in expected proof-of-work."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RP-GHOST webcam oracle and local proof-of-work "
            "research benchmark"
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--simulate",
        action="store_true"
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=20,
        help="Local proof-of-work difficulty in bits"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=1
    )

    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of RP decision trials"
    )

    parser.add_argument(
        "--clock-hz",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=20
    )

    parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--require-rp-select",
        action="store_true",
        help="Only mine when the RP decision circuit selects HIGH"
    )

    parser.add_argument(
        "--midstate",
        action="store_true"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true"
    )

    parser.add_argument(
        "--bench-bits",
        type=int,
        default=18
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10
    )

    parser.add_argument(
        "--benchmark-nonce",
        action="store_true"
    )

    parser.add_argument(
        "--nonce-bits",
        type=int,
        default=20
    )

    args = parser.parse_args()

    validate_bits(args.bits)
    validate_bits(args.bench_bits)
    validate_nonce_bits(args.nonce_bits)

    if args.rounds < 1:
        parser.error("--rounds must be at least 1.")

    if args.trials < 1:
        parser.error("--trials must be at least 1.")

    if args.k < 1:
        parser.error("--k must be at least 1.")

    if args.calibration_samples < 1:
        parser.error("--calibration-samples must be at least 1.")

    print("=" * 72)
    print("RP-GHOST — LOCAL RESEARCH IMPLEMENTATION")
    print("=" * 72)

    print(f"TEST_BITS: {args.bits}")
    print(f"Expected hashes: {2**args.bits:,}")
    print(f"Simulation: {args.simulate}")
    print("Pool connection: DISABLED")
    print("Share submission: DISABLED")

    oracle = WebcamOracle(
        index=args.camera,
        simulate=args.simulate
    )

    try:

        # ----------------------------------------------------
        # BENCHMARK: ORACLE NONCE VS SEQUENTIAL REACH
        # ----------------------------------------------------

        if args.benchmark_nonce:

            benchmark_nonce(
                oracle,
                args.nonce_bits,
                args.bits,
                args.trials
            )

            return

        # ----------------------------------------------------
        # BENCHMARK: SEARCH STRATEGIES
        # ----------------------------------------------------

        if args.benchmark:

            benchmark(
                oracle,
                args.bench_bits,
                args.trials
            )

            return

        # ----------------------------------------------------
        # NORMAL OPERATION
        # ----------------------------------------------------

        search_function = (
            search_nonce_midstate
            if args.midstate
            else search_nonce
        )

        for round_number in range(args.rounds):

            print("\n" + "=" * 72)

            print(
                f"ROUND {round_number + 1}/{args.rounds}"
            )

            print("=" * 72)

            # ------------------------------------------------
            # RP DECISION CIRCUIT
            # ------------------------------------------------

            print("\n[rp] running decision circuit")

            selected = rp_decision(
                oracle,
                k=args.k,
                clock_hz=args.clock_hz,
                calibration_samples=args.calibration_samples,
                mad_multiplier=args.mad_multiplier
            )

            if args.require_rp_select and not selected:

                print(
                    "[system] RP decision rejected candidate generation."
                )

                continue

            # ------------------------------------------------
            # ORACLE GENERATION
            # ------------------------------------------------

            print("\n[oracle] acquiring 64-bit input")

            seed = oracle.oracle_64()

            print(
                f"[oracle] 64-bit input: {seed:016x}"
            )

            extranonce2 = oracle_to_extranonce2(seed)

            print(
                f"[oracle] extranonce2: {extranonce2.hex()}"
            )

            # ------------------------------------------------
            # HEADER CONSTRUCTION
            # ------------------------------------------------

            prefix = make_header_prefix(
                extranonce2
            )

            print(
                f"[miner] header prefix: {len(prefix)} bytes"
            )

            print(
                "[miner] local test header constructed"
            )

            # ------------------------------------------------
            # LEFT-SIDE MINER
            # ------------------------------------------------

            print("\n[miner] starting nonce search")

            nonce, digest, tried, elapsed = search_function(
                prefix,
                args.bits
            )

            print_result(
                nonce,
                digest,
                tried,
                elapsed,
                args.bits
            )

            if nonce is None:
                print(
                    "[miner] no valid nonce found in 32-bit space"
                )

                continue

            # ------------------------------------------------
            # INDEPENDENT VERIFICATION
            # ------------------------------------------------

            print("\n[verify] independently recomputing candidate")

            valid = verify(
                prefix,
                nonce,
                args.bits
            )

            print(
                f"[verify] result: {'PASS' if valid else 'FAIL'}"
            )

            if not valid:
                print(
                    "[verify] candidate rejected"
                )

                continue

            print("\n[RESULT] LOCAL TEST PROOF-OF-WORK VALID")

            print(
                f"Nonce: 0x{nonce:08x}"
            )

            print(
                f"Hash: {digest[::-1].hex()}"
            )

            print(
                f"Difficulty: {args.bits} leading-zero-bit target"
            )

            print(
                "[RESULT] Oracle generated the candidate seed."
            )

            print(
                "[RESULT] Left-side miner established target validity."
            )

            print(
                "[RESULT] Independent verification passed."
            )

            print(
                "[NOTE] This is not a Bitcoin pool share."
            )

    finally:

        oracle.close()

        print("\n[system] oracle closed")


if __name__ == "__main__":
    main()
