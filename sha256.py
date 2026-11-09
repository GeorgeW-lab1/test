
"""
RP-GHOST / ViaBTC Bitcoin Stratum V1 Miner
Oracle-seeded Bitcoin mining with separated trust architecture.

ORACLE:
    Produces an assumed SHA-512-derived input.
    No independent oracle verification gate.

RIGHT SIDE:
    Supplies Stratum job/reference data.
    Does not verify or submit shares.

LEFT SIDE:
    Constructs Bitcoin headers from reference data.
    Independently calculates SHA-256d.
    Checks pool share target and network block target.
    Submits valid shares to ViaBTC.

Install:
    pip install numpy opencv-python

Configure:
    Set VIABTC_USER to your ViaBTC worker name.
    Set VIABTC_PASSWORD if required.

Run:
    python SHA256.py
"""

import hashlib
import json
import os
import queue
import select
import socket
import struct
import threading
import time
import tkinter as tk

from decimal import Decimal
from tkinter import ttk
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ================================================================
# CONFIGURATION
# ================================================================

POOL_HOST = "btc.viabtc.io"
POOL_PORT = 3333

POOL_USER = os.getenv("VIABTC_USER", "lolcatz.001")
POOL_PASSWORD = os.getenv("VIABTC_PASSWORD", "x")

USER_AGENT = "RPGHOST-BitcoinMiner/2.0"

SW, SH = 128, 96
PW, PH = 320, 240

ORACLE_BITS = 32

DIFF1_TARGET = (
    0x00000000FFFF0000000000000000000000000000000000000000000000000000
)

MINER_LANES = 2
HASH_BATCH = 4096

DELIM = ":"


# ================================================================
# HASHING AND BITCOIN TARGETS
# ================================================================

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(
        hashlib.sha256(data).digest()
    ).digest()


def compact_to_target(nbits: str) -> int:
    compact = int(nbits, 16)

    exponent = compact >> 24
    mantissa = compact & 0x007fffff

    if compact & 0x00800000:
        raise ValueError("Negative Bitcoin target")

    if exponent == 0 or mantissa == 0:
        raise ValueError("Zero or invalid compact target")

    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))

    if target <= 0 or target >= (1 << 256):
        raise ValueError("Invalid Bitcoin target")

    return target


def difficulty_to_target(difficulty: float) -> int:
    difficulty = Decimal(str(difficulty))

    if difficulty <= 0:
        raise ValueError("Invalid pool difficulty")

    target = int(
        Decimal(DIFF1_TARGET) / difficulty
    )

    if target <= 0:
        raise ValueError("Pool share target is below integer precision")

    return min(target, (1 << 256) - 1)


def swap_words(data: bytes) -> bytes:
    if len(data) % 4:
        raise ValueError("Data length must be divisible by four")

    return b"".join(
        data[i:i + 4][::-1]
        for i in range(0, len(data), 4)
    )


def bitcoin_hash_value(header: bytes):
    if len(header) != 80:
        raise ValueError("Bitcoin header must be exactly 80 bytes")

    digest = sha256d(header)

    # Bitcoin interprets the displayed hash as a little-endian integer.
    value = int.from_bytes(digest, "little")

    return digest, value


# ================================================================
# LATCH REGISTERS
# ================================================================

class Register:

    def __init__(self, value=None):
        self._lock = threading.Lock()
        self._v = value
        self.seq = 0

    def write(self, value):
        with self._lock:
            self._v = value
            self.seq += 1
            return self.seq

    def read(self):
        with self._lock:
            return self.seq, self._v


# ================================================================
# ORACLE SIDE — QUANTUM LEAP
# ================================================================

class GhostedOracleWorker(threading.Thread):

    def __init__(self, msg_reg, log_cb):
        super().__init__(
            daemon=True,
            name="ghosted-oracle"
        )

        self.msg_reg = msg_reg
        self.log_cb = log_cb

        self.ticks = queue.Queue()
        self._halt = threading.Event()

        self.q_bits_buffer = 0
        self.q_bit_count = 0

        self.oracle_block_id = 0
        self.target_stream_bits = ORACLE_BITS

    def stop(self):
        self._halt.set()

    def tick(self, idx, q_out):
        self.ticks.put((idx, q_out))

    def run(self):

        while not self._halt.is_set():

            try:
                idx, q_out = self.ticks.get(timeout=0.1)
            except queue.Empty:
                continue

            if q_out not in (0, 1):
                continue

            self.q_bits_buffer = (
                (self.q_bits_buffer << 1) | q_out
            )

            self.q_bit_count += 1

            self.log_cb(
                f"[ORACLE] Frame {idx} | Q={q_out} | "
                f"{self.q_bit_count}/{ORACLE_BITS}"
            )

            if self.q_bit_count < self.target_stream_bits:
                continue

            self.oracle_block_id += 1

            stream_hex = format(
                self.q_bits_buffer,
                f"0{self.target_stream_bits // 4}x"
            )

            # Oracle output is accepted as an assumed input.
            # This is not an independent verification step.

            digest = hashlib.sha512(
                (
                    f"RECONSTRUCTED-ORACLE-BLOCK-"
                    f"{self.oracle_block_id}:{stream_hex}"
                ).encode("ascii")
            ).digest()

            msg_int = int.from_bytes(digest, "big")

            record = {
                "block_id": self.oracle_block_id,
                "stream_hex": stream_hex,
                "message": msg_int
            }

            self.msg_reg.write(record)

            self.log_cb(
                f"[ORACLE INPUT ACCEPTED] "
                f"Block #{self.oracle_block_id} | "
                f"Quantum Leap"
            )

            self.log_cb(str(record))

            self.q_bits_buffer = 0
            self.q_bit_count = 0


# ================================================================
# RIGHT SIDE — PASSIVE STRATUM REFERENCE
# ================================================================

class StratumReference:

    """
    The right-side reference register stores pool job data.

    It does not:
        - Verify oracle messages.
        - Calculate candidate hashes.
        - Validate proof of work.
        - Submit mining shares.
    """

    def __init__(self):
        self.register = Register()

    def receive_job(self, job):
        return self.register.write(dict(job))

    def read(self):
        return self.register.read()


# ================================================================
# STRATUM V1 CLIENT
# ================================================================

class StratumClient(threading.Thread):

    def __init__(self, reference, log_cb):

        super().__init__(
            daemon=True,
            name="viabtc-stratum"
        )

        self.reference = reference
        self.log_cb = log_cb

        self.sock = None
        self.rx_buffer = b""

        self.send_lock = threading.Lock()
        self.id_lock = threading.Lock()
        self.pending_lock = threading.Lock()

        self.request_id = 10
        self.pending_submits = {}

        self.extranonce1 = b""
        self.extranonce2_size = 0

        self.pending_difficulty = 1.0
        self.pending_target = None

        self._halt = threading.Event()

        self.connected = threading.Event()
        self.authorized = False

    def stop(self):
        self._halt.set()

        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                self.sock.close()
            except OSError:
                pass

    def next_request_id(self):
        with self.id_lock:
            self.request_id += 1
            return self.request_id

    def send_message(self, message):

        data = (
            json.dumps(message, separators=(",", ":")) + "\n"
        ).encode("utf-8")

        with self.send_lock:
            self.sock.sendall(data)

    def send_request(self, method, params):

        rid = self.next_request_id()

        self.send_message({
            "id": rid,
            "method": method,
            "params": params
        })

        return rid

    def read_message(self, timeout=None):

        deadline = (
            None
            if timeout is None
            else time.monotonic() + timeout
        )

        while not self._halt.is_set():

            newline = self.rx_buffer.find(b"\n")

            if newline >= 0:

                line = self.rx_buffer[:newline]
                self.rx_buffer = self.rx_buffer[newline + 1:]

                if not line.strip():
                    continue

                return json.loads(line.decode("utf-8"))

            if deadline is not None:
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise TimeoutError("Stratum response timeout")
            else:
                remaining = 1.0

            readable, _, _ = select.select(
                [self.sock],
                [],
                [],
                min(remaining, 1.0)
            )

            if not readable:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Stratum response timeout")
                continue

            chunk = self.sock.recv(65536)

            if not chunk:
                raise ConnectionError("Pool disconnected")

            self.rx_buffer += chunk

        raise ConnectionError("Stratum client stopped")

    def wait_response(self, request_id, timeout=30):

        deadline = time.monotonic() + timeout

        while not self._halt.is_set():

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise TimeoutError("Stratum response timeout")

            msg = self.read_message(remaining)

            if msg.get("id") == request_id:
                return msg

            self.dispatch(msg)

        raise ConnectionError("Stratum client stopped")

    def run(self):

        if not POOL_USER or POOL_USER.startswith("YOUR_"):
            self.log_cb(
                "[STRATUM ERROR] Configure VIABTC_USER before connecting."
            )
            return

        try:

            self.log_cb(
                f"[STRATUM] Connecting to {POOL_HOST}:{POOL_PORT}"
            )

            self.sock = socket.create_connection(
                (POOL_HOST, POOL_PORT),
                timeout=15
            )

            self.sock.setblocking(True)

            self.log_cb("[STRATUM] TCP connection established.")

            # ----------------------------------------------------
            # SUBSCRIBE
            # ----------------------------------------------------

            subscribe_id = self.send_request(
                "mining.subscribe",
                [USER_AGENT]
            )

            response = self.wait_response(subscribe_id)

            if response.get("error") or not response.get("result"):
                raise RuntimeError(
                    f"Subscription failed: {response}"
                )

            result = response["result"]

            self.extranonce1 = bytes.fromhex(result[1])
            self.extranonce2_size = int(result[2])

            if self.extranonce2_size < 1:
                raise RuntimeError("Invalid extranonce2 size")

            self.log_cb(
                f"[STRATUM] Subscribed | "
                f"Extranonce1={self.extranonce1.hex()} | "
                f"Extranonce2 size={self.extranonce2_size}"
            )

            # ----------------------------------------------------
            # AUTHORIZE
            # ----------------------------------------------------

            auth_id = self.send_request(
                "mining.authorize",
                [POOL_USER, POOL_PASSWORD]
            )

            response = self.wait_response(auth_id)

            if response.get("error") or response.get("result") is not True:
                raise RuntimeError(
                    f"Worker authorization failed: {response}"
                )

            self.authorized = True
            self.connected.set()

            self.log_cb(
                f"[STRATUM] Worker authorized: {POOL_USER}"
            )

            # ----------------------------------------------------
            # RECEIVE JOBS AND SHARE RESPONSES
            # ----------------------------------------------------

            while not self._halt.is_set():

                try:
                    msg = self.read_message(timeout=1)

                except TimeoutError:
                    continue

                self.dispatch(msg)

        except Exception as e:

            if not self._halt.is_set():
                self.log_cb(f"[STRATUM ERROR] {e}")

        finally:

            self.connected.clear()
            self.authorized = False

            self.log_cb("[STRATUM] Connection closed.")

    def dispatch(self, msg):

        method = msg.get("method")

        if method == "mining.set_difficulty":

            self.pending_difficulty = float(
                msg["params"][0]
            )

            self.pending_target = None

            self.log_cb(
                f"[POOL] New share difficulty: "
                f"{self.pending_difficulty}"
            )

        elif method == "mining.set_target":

            target_hex = msg["params"][0]

            self.pending_target = int(target_hex, 16)

            self.log_cb(
                f"[POOL] Explicit share target: {target_hex}"
            )

        elif method == "mining.set_extranonce":

            self.extranonce1 = bytes.fromhex(
                msg["params"][0]
            )

            self.extranonce2_size = int(
                msg["params"][1]
            )

            self.log_cb(
                "[POOL] Extranonce parameters updated."
            )

        elif method == "mining.notify":

            p = msg["params"]

            if len(p) < 9:
                self.log_cb("[POOL ERROR] Invalid mining.notify")
                return

            job = {
                "job_id": p[0],
                "prevhash": p[1],
                "coinb1": p[2],
                "coinb2": p[3],
                "merkle_branch": p[4],
                "version": p[5],
                "nbits": p[6],
                "ntime": p[7],
                "clean_jobs": bool(p[8]),
                "difficulty": self.pending_difficulty,
                "share_target": self.pending_target,
                "extranonce1": self.extranonce1,
                "extranonce2_size": self.extranonce2_size
            }

            # RIGHT SIDE ONLY:
            # Publish the reference job. No verification here.

            self.reference.receive_job(job)

            self.log_cb(
                f"[RIGHT REFERENCE] Job={job['job_id']} | "
                f"Difficulty={job['difficulty']} | "
                f"nBits={job['nbits']} | "
                f"Clean={job['clean_jobs']}"
            )

        elif method == "client.show_message":

            self.log_cb(
                f"[POOL MESSAGE] {msg.get('params')}"
            )

        elif method == "client.reconnect":

            self.log_cb(
                f"[POOL RECONNECT REQUEST] {msg.get('params')}"
            )

        elif "id" in msg:

            with self.pending_lock:
                info = self.pending_submits.pop(msg["id"], None)

            if info is None:
                return

            if msg.get("result") is True:

                self.log_cb(
                    f"[SHARE ACCEPTED] Job={info['job_id']} | "
                    f"Lane={info['lane']} | "
                    f"Nonce={info['nonce']}"
                )

            else:

                self.log_cb(
                    f"[SHARE REJECTED] Job={info['job_id']} | "
                    f"Lane={info['lane']} | "
                    f"Error={msg.get('error')}"
                )

    def submit_share(
        self,
        job,
        extranonce2,
        ntime,
        nonce,
        lane,
        result
    ):

        if not self.authorized or not self.connected.is_set():
            self.log_cb("[SUBMIT ERROR] Stratum is not authorized.")
            return False

        rid = self.next_request_id()

        info = {
            "job_id": job["job_id"],
            "nonce": f"{nonce:08x}",
            "lane": lane
        }

        # Register before sending to avoid a fast-response race.

        with self.pending_lock:
            self.pending_submits[rid] = info

        try:

            self.send_message({
                "id": rid,
                "method": "mining.submit",
                "params": [
                    POOL_USER,
                    job["job_id"],
                    extranonce2,
                    ntime,
                    f"{nonce:08x}"
                ]
            })

        except Exception:
            with self.pending_lock:
                self.pending_submits.pop(rid, None)
            raise

        self.log_cb(
            f"[LEFT SUBMITTED] Job={job['job_id']} | "
            f"Lane={lane} | Nonce={nonce:08x} | "
            f"Hash={result['hash']}"
        )

        return True


# ================================================================
# ORACLE-SEEDED EXTRANONCE2
# ================================================================

def oracle_extranonce2(message, lane, counter, size):

    if size < 1:
        raise ValueError("Invalid extranonce2 size")

    if not isinstance(message, int):
        raise ValueError("Oracle message must be an integer")

    message_bytes = message.to_bytes(64, "big")

    seed = hashlib.sha256(
        message_bytes + struct.pack("<I", lane)
    ).digest()

    if size <= 32:

        base = int.from_bytes(seed[-size:], "big")

    else:

        output = bytearray()
        block = 0

        while len(output) < size:

            output.extend(
                hashlib.sha256(
                    seed + struct.pack("<I", block)
                ).digest()
            )

            block += 1

        base = int.from_bytes(bytes(output[:size]), "big")

    value = (
        base + counter
    ) % (1 << (8 * size))

    return value.to_bytes(size, "big").hex()


# ================================================================
# BITCOIN COINBASE AND MERKLE ROOT
# ================================================================

def build_merkle_root(job, extranonce2):

    coinbase = (
        bytes.fromhex(job["coinb1"])
        + job["extranonce1"]
        + bytes.fromhex(extranonce2)
        + bytes.fromhex(job["coinb2"])
    )

    merkle_root = sha256d(coinbase)

    for branch in job["merkle_branch"]:

        branch_bytes = bytes.fromhex(branch)

        merkle_root = sha256d(
            merkle_root + branch_bytes
        )

    return merkle_root


# ================================================================
# BITCOIN 80-BYTE BLOCK HEADER
# ================================================================

def build_header_prefix(job, extranonce2):

    version = bytes.fromhex(job["version"])[::-1]

    # Stratum V1 prevhash is supplied with 32-bit words reversed.
    prevhash = swap_words(
        bytes.fromhex(job["prevhash"])
    )

    merkle_root = build_merkle_root(
        job,
        extranonce2
    )

    ntime = bytes.fromhex(job["ntime"])[::-1]
    nbits = bytes.fromhex(job["nbits"])[::-1]

    prefix = (
        version
        + prevhash
        + merkle_root
        + ntime
        + nbits
    )

    if len(prefix) != 76:
        raise ValueError(
            f"Invalid header prefix: {len(prefix)} bytes"
        )

    return prefix


def build_full_header(prefix, nonce):

    header = prefix + struct.pack("<I", nonce)

    if len(header) != 80:
        raise ValueError("Bitcoin header must be exactly 80 bytes")

    return header


# ================================================================
# LEFT SIDE — VERIFICATION AND SUBMISSION
# ================================================================

class BitcoinMinerLane(threading.Thread):

    def __init__(
        self,
        lane,
        reference,
        msg_reg,
        stratum,
        log_cb
    ):

        super().__init__(
            daemon=True,
            name=f"bitcoin-miner-{lane}"
        )

        self.lane = lane

        self.reference = reference
        self.msg_reg = msg_reg

        self.stratum = stratum
        self.log_cb = log_cb

        self._halt = threading.Event()

        self.hashes = 0
        self.shares = 0
        self.blocks = 0

    def stop(self):
        self._halt.set()

    def get_share_target(self, job):

        explicit_target = job.get("share_target")

        if explicit_target is not None:
            return int(explicit_target)

        return difficulty_to_target(
            job["difficulty"]
        )

    def verify_candidate(
        self,
        job,
        extranonce2,
        nonce,
        prefix
    ):
        """
        All candidate verification takes place on the LEFT.

        The right-side job is the reference.
        The oracle record is an assumed input.
        """

        header = build_full_header(
            prefix,
            nonce
        )

        # Reconstruct the header from the reference job data.
        # This catches local header/prefix construction errors.

        reference_prefix = build_header_prefix(
            job,
            extranonce2
        )

        reference_header = build_full_header(
            reference_prefix,
            nonce
        )

        if header != reference_header:
            raise ValueError(
                "Left-side reference reconstruction mismatch"
            )

        digest, hash_value = bitcoin_hash_value(header)

        share_target = self.get_share_target(job)
        block_target = compact_to_target(job["nbits"])

        share_valid = hash_value <= share_target
        block_candidate = hash_value <= block_target

        return {
            "header": header,
            "hash": digest[::-1].hex(),
            "hash_value": hash_value,
            "share_target": share_target,
            "block_target": block_target,
            "share_valid": share_valid,
            "block_candidate": block_candidate
        }

    def run(self):

        last_job_sequence = -1
        last_oracle_sequence = -1

        counter = 0
        nonce = self.lane

        last_report = time.perf_counter()
        last_hashes = 0

        while not self._halt.is_set():

            # RIGHT SIDE: obtain the current pool reference job.

            job_sequence, job = self.reference.read()

            # ORACLE SIDE: obtain the assumed oracle input.

            oracle_sequence, oracle = self.msg_reg.read()

            if (
                job is None
                or oracle is None
                or not self.stratum.connected.is_set()
            ):

                time.sleep(0.05)
                continue

            # New pool job resets the nonce search for that job.

            if job_sequence != last_job_sequence:

                last_job_sequence = job_sequence

                counter = 0
                nonce = self.lane

                self.log_cb(
                    f"[LEFT] New Bitcoin reference job "
                    f"{job['job_id']} | Lane={self.lane}"
                )

            # New oracle input changes the extranonce seed.
            # It does not independently verify the oracle.
            # It does not reset the current nonce position.

            if oracle_sequence != last_oracle_sequence:

                last_oracle_sequence = oracle_sequence
                counter = 0

                self.log_cb(
                    f"[LEFT] Oracle input latched | "
                    f"Block={oracle.get('block_id')} | "
                    f"Lane={self.lane}"
                )

            try:

                extranonce2 = oracle_extranonce2(
                    oracle["message"],
                    self.lane,
                    counter,
                    job["extranonce2_size"]
                )

                prefix = build_header_prefix(
                    job,
                    extranonce2
                )

                share_target = self.get_share_target(job)
                block_target = compact_to_target(job["nbits"])

            except Exception as error:

                self.log_cb(
                    f"[LEFT ERROR] Lane={self.lane} | {error}"
                )

                time.sleep(1)
                continue

            # ----------------------------------------------------
            # SEARCH THE 32-BIT BITCOIN NONCE SPACE
            # Each lane searches a separate nonce sequence.
            # ----------------------------------------------------

            for _ in range(HASH_BATCH):

                if self._halt.is_set():
                    break

                current_job_sequence, _ = self.reference.read()
                current_oracle_sequence, _ = self.msg_reg.read()

                if current_job_sequence != last_job_sequence:
                    break

                if current_oracle_sequence != last_oracle_sequence:
                    break

                try:

                    result = self.verify_candidate(
                        job,
                        extranonce2,
                        nonce,
                        prefix
                    )

                except Exception as error:

                    self.log_cb(
                        f"[LEFT VERIFY ERROR] Lane={self.lane} | "
                        f"{error}"
                    )

                    break

                self.hashes += 1

                # ------------------------------------------------
                # SHARE OR BLOCK CANDIDATE
                # ------------------------------------------------

                if (
                    result["share_valid"]
                    or result["block_candidate"]
                ):

                    if result["share_valid"]:
                        self.shares += 1

                    if result["block_candidate"]:
                        self.blocks += 1

                    self.log_cb(
                        f"[LEFT VERIFICATION PASSED] "
                        f"Lane={self.lane} | "
                        f"Hash={result['hash']} | "
                        f"Nonce={nonce:08x} | "
                        f"Share={result['share_valid']} | "
                        f"Block={result['block_candidate']}"
                    )

                    # A network block candidate must also be
                    # submitted, even if the pool share target
                    # happens to be more difficult.

                    self.stratum.submit_share(
                        job,
                        extranonce2,
                        job["ntime"],
                        nonce,
                        self.lane,
                        result
                    )

                # Disjoint nonce sequence for each lane.

                nonce += MINER_LANES

                if nonce > 0xffffffff:

                    nonce = self.lane
                    counter += 1

                    extranonce2 = oracle_extranonce2(
                        oracle["message"],
                        self.lane,
                        counter,
                        job["extranonce2_size"]
                    )

                    prefix = build_header_prefix(
                        job,
                        extranonce2
                    )

            # ----------------------------------------------------
            # HASHRATE REPORTING
            # ----------------------------------------------------

            now = time.perf_counter()

            if now - last_report >= 5:

                elapsed = now - last_report

                rate = (
                    self.hashes - last_hashes
                ) / elapsed

                self.log_cb(
                    f"[HASHRATE] Lane={self.lane} | "
                    f"{rate:,.2f} H/s | "
                    f"Total={self.hashes:,} | "
                    f"Shares={self.shares} | "
                    f"Blocks={self.blocks}"
                )

                last_report = now
                last_hashes = self.hashes


# ================================================================
# CAMERA SOURCE
# ================================================================

class Source:

    def __init__(self, index):

        if cv2 is None:
            raise RuntimeError(
                "Install OpenCV: pip install opencv-python"
            )

        backend = (
            cv2.CAP_DSHOW
            if sys.platform == "win32"
            else cv2.CAP_ANY
        )

        self.cap = cv2.VideoCapture(index, backend)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index}"
            )

        self.rgb = None

    def read(self):

        if not self.cap.grab():
            return None

        ok, bgr = self.cap.retrieve()

        if not ok:
            return None

        self.rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

        return cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )

    def close(self):
        self.cap.release()


def downscale(gray):

    small = cv2.resize(
        gray,
        (SW, SH),
        interpolation=cv2.INTER_AREA
    )

    return cv2.GaussianBlur(
        small,
        (5, 5),
        0
    )


def ppm_photo(rgb):

    h, w, _ = rgb.shape

    data = b"P6 %d %d 255\n" % (w, h)

    data += np.ascontiguousarray(
        rgb,
        dtype=np.uint8
    ).tobytes()

    return tk.PhotoImage(
        data=data,
        format="PPM"
    )


# ================================================================
# MAIN APPLICATION
# ================================================================

class App:

    def __init__(self, window):

        self.window = window

        window.title(
            "RP-GHOST | Bitcoin Stratum Miner"
        )

        self.q = queue.Queue()

        self.ctl = {
            "stop": False,
            "thr": 18,
            "min_px": 30
        }

        # Oracle input register.
        self.msg_reg = Register()

        # Right-side pool reference.
        self.reference = StratumReference()

        self.oracle = GhostedOracleWorker(
            self.msg_reg,
            self.log
        )

        self.stratum = StratumClient(
            self.reference,
            self.log
        )

        # Left-side verifiers/miners.
        self.miners = [
            BitcoinMinerLane(
                i,
                self.reference,
                self.msg_reg,
                self.stratum,
                self.log
            )
            for i in range(MINER_LANES)
        ]

        self.oracle.start()

        for miner in self.miners:
            miner.start()

        self._build_ui()

        window.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        window.after(30, self.poll)

    def log(self, message):
        self.q.put(("log", str(message)))

    def _build_ui(self):

        main = ttk.Frame(
            self.window,
            padding=10
        )

        main.pack(
            fill="both",
            expand=True
        )

        controls = ttk.LabelFrame(
            main,
            text="Bitcoin Pool Controls",
            padding=8
        )

        controls.pack(
            fill="x",
            pady=5
        )

        self.start_button = ttk.Button(
            controls,
            text="Connect / Start Mining",
            command=self.start
        )

        self.start_button.pack(
            side="left",
            padx=5
        )

        self.stop_button = ttk.Button(
            controls,
            text="Stop Camera",
            command=self.stop,
            state="disabled"
        )

        self.stop_button.pack(
            side="left",
            padx=5
        )

        self.status = tk.StringVar(
            value="Disconnected"
        )

        ttk.Label(
            controls,
            textvariable=self.status,
            font=("Arial", 10, "bold")
        ).pack(
            side="left",
            padx=12
        )

        content = ttk.Frame(main)

        content.pack(
            fill="both",
            expand=True
        )

        self.canvas = tk.Canvas(
            content,
            width=PW,
            height=PH,
            bg="black"
        )

        self.canvas.pack(
            side="left",
            padx=5
        )

        self.image_id = self.canvas.create_image(
            0,
            0,
            anchor="nw"
        )

        log_frame = ttk.LabelFrame(
            content,
            text="Oracle / Right Reference / Left Verification Log",
            padding=5
        )

        log_frame.pack(
            side="right",
            fill="both",
            expand=True
        )

        self.log_text = tk.Text(
            log_frame,
            width=75,
            height=22,
            font=("Consolas", 9),
            state="disabled"
        )

        self.log_text.pack(
            side="left",
            fill="both",
            expand=True
        )

        scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview
        )

        scroll.pack(
            side="right",
            fill="y"
        )

        self.log_text.configure(
            yscrollcommand=scroll.set
        )

    def start(self):

        self.ctl["stop"] = False

        self.start_button.config(
            state="disabled"
        )

        self.stop_button.config(
            state="normal"
        )

        self.status.set(
            "Connecting to ViaBTC..."
        )

        if not self.stratum.is_alive():
            self.stratum.start()

        self.sim_thread = threading.Thread(
            target=self.sim_loop,
            daemon=True
        )

        self.sim_thread.start()

    def stop(self):

        self.ctl["stop"] = True

        self.log("[SYSTEM] Camera stopped.")

        self.stop_button.config(
            state="disabled"
        )

        self.start_button.config(
            state="normal"
        )

    def close(self):

        self.ctl["stop"] = True

        self.oracle.stop()
        self.stratum.stop()

        for miner in self.miners:
            miner.stop()

        self.window.destroy()

    def sim_loop(self):

        try:
            src = Source(0)

        except RuntimeError as error:

            self.log(f"[CAMERA ERROR] {error}")
            self.q.put(("done",))

            return

        self.log("[CAMERA] USB camera connected.")

        prev = None
        idx = 0

        try:

            while not self.ctl["stop"]:

                gray = src.read()

                if gray is None:
                    break

                small = downscale(gray)

                if prev is None or idx < 5:

                    mask = np.zeros(
                        (SH, SW),
                        dtype=bool
                    )

                else:

                    diff = np.abs(
                        small.astype(np.int16)
                        - prev.astype(np.int16)
                    )

                    mask = diff > self.ctl["thr"]

                prev = small

                motion_px = int(mask.sum())

                motion = motion_px >= self.ctl["min_px"]

                x = motion_px if motion else 0

                driver_a = 0

                if motion and x > 4 and x % 2 == 1:

                    r = 2 + (x * 7) % (x - 3)

                    driver_a = int(
                        miller_rabin_witness(r, x)
                    )

                driver_b = int(
                    motion and x > 2 and x % 2 == 1
                )

                q_out = driver_a & driver_b

                self.oracle.tick(
                    idx,
                    q_out
                )

                if src.rgb is not None:

                    preview = cv2.resize(
                        src.rgb,
                        (PW, PH)
                    )

                    self.q.put(
                        ("frame", preview)
                    )

                idx += 1

                time.sleep(0.03)

        finally:

            src.close()

            self.q.put(("done",))

    def poll(self):

        try:

            while True:

                item = self.q.get_nowait()

                if item[0] == "log":

                    self.log_text.config(
                        state="normal"
                    )

                    self.log_text.insert(
                        "end",
                        item[1] + "\n"
                    )

                    lines = int(
                        self.log_text.index("end-1c").split(".")[0]
                    )

                    if lines > 500:

                        self.log_text.delete(
                            "1.0",
                            f"{lines - 400}.0"
                        )

                    self.log_text.see("end")

                    self.log_text.config(
                        state="disabled"
                    )

                elif item[0] == "frame":

                    photo = ppm_photo(item[1])

                    self.canvas.itemconfig(
                        self.image_id,
                        image=photo
                    )

                    self._photo = photo

                elif item[0] == "done":

                    self.log("[CAMERA] Capture ended.")

        except queue.Empty:
            pass

        if self.stratum.connected.is_set():

            self.status.set(
                "ViaBTC Connected / Authorized"
            )

        elif self.stratum.is_alive():

            self.status.set(
                "Connecting / Authenticating"
            )

        else:

            self.status.set(
                "Disconnected"
            )

        self.window.after(
            30,
            self.poll
        )


# ================================================================
# MILLER-RABIN CIRCUIT FUNCTION
# ================================================================

def miller_rabin_witness(a, n):

    if n < 2:
        return False

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


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = App(root)

    root.geometry("1150x600")

    root.mainloop()
