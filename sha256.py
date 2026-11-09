"""
Symmetric RP-Style Optoelectronic Decision Circuit & Async Hash Miner (Fixed NoneType Check)
    pip install numpy opencv-python
    python SHA256.py
"""
import asyncio
import hashlib
import queue
import secrets
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ------------------------------------------------------------------ Constants & Formatting
SW, SH = 128, 96                  # Processing resolution
PW, PH = 320, 240                 # Preview resolution
TARGET_DIFF = 47                  # Test difficulty (leading zero bits)
HEADER_DEFAULT = "RP-GHOST"
NONCE_HEX = 64                    # 256-bit nonce field (64 hex characters)
MESSAGE_HEX = 128                 # 512-bit message field (128 hex characters)
DIGEST_HEX = 128                  # 512-bit zero-padded SHA-256 field
DELIM = ":"

def pad_hex(value: int, width: int) -> str:
    h = format(value, "x")
    if len(h) > width:
        raise ValueError(f"value needs {len(h)} hex chars, field is {width}")
    return h.zfill(width)

def digest_text(header: str, nonce_hex: str, msg_hex: str) -> bytes:
    return hashlib.sha256((header + DELIM + nonce_hex + DELIM + msg_hex).encode("ascii")).digest()

def build_response(header: str, nonce: int, message: int) -> str:
    h = header
    n = pad_hex(nonce, NONCE_HEX)
    m = pad_hex(message, MESSAGE_HEX)
    d = int.from_bytes(digest_text(h, n, m), "big")
    return DELIM.join((h, n, m, pad_hex(d, DIGEST_HEX)))

def verify_response(resp: str) -> bool:
    if not resp or not isinstance(resp, str):
        return False
    try:
        h, n, m, d = resp.split(DELIM)
        int(n, 16), int(m, 16), int(d, 16)
    except ValueError:
        return False
    if len(n) != NONCE_HEX or len(m) != MESSAGE_HEX or len(d) != DIGEST_HEX or int(d, 16) >> 256:
        return False
    return pad_hex(int.from_bytes(digest_text(h, n, m), "big"), DIGEST_HEX) == d

# ------------------------------------------------------------------ Math & Miller-Rabin
def miller_rabin_witness(a: int, n: int) -> bool:
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

# ------------------------------------------------------------------ Latch Registers
class Register:
    """A latch: holds the newest value and sequence number. Pointer swaps never block[cite: 1]."""
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
        with self._lock:
            if blocked(self._v):
                return False
            self._v = value
            self.seq += 1
            return True

    def read(self):
        with self._lock:
            return self.seq, self._v

# ------------------------------------------------------------------ Left Side: Ghosted Oracle Reconstructor
class GhostedOracleWorker(threading.Thread):
    """Left Side: Asynchronously streams Q bits, reconstructs the predicted hash completely, then publishes it[cite: 1]."""
    def __init__(self, msg_reg: Register, log_cb):
        super().__init__(daemon=True, name="ghosted-oracle")
        self.msg_reg = msg_reg
        self.log_cb = log_cb
        self.ticks = queue.Queue()
        self._halt = threading.Event()
        self.q_bits_buffer = 0
        self.q_bit_count = 0
        self.oracle_block_id = 0
        self.target_stream_bits = 32

    def stop(self):
        self._halt.set()

    def tick(self, idx: int, q_out: int):
        self.ticks.put((idx, q_out))

    def run(self):
        while not self._halt.is_set():
            try:
                idx, q_out = self.ticks.get(timeout=0.1)
            except queue.Empty:
                continue

            self.q_bits_buffer = (self.q_bits_buffer << 1) | q_out
            self.q_bit_count += 1
            self.log_cb(f"[LEFT SIDE - ORACLE GHOST] Frame {idx:4d} | Q={q_out} | Reconstructing Stream: {self.q_bit_count}/{self.target_stream_bits}")

            if self.q_bit_count >= self.target_stream_bits:
                self.oracle_block_id += 1
                stream_hex = format(self.q_bits_buffer, f"0{self.target_stream_bits // 4}x")
                
                digest = hashlib.sha512(f"RECONSTRUCTED-ORACLE-BLOCK-{self.oracle_block_id}:{stream_hex}".encode("ascii")).digest()
                msg_int = int.from_bytes(digest, "big") & ((1 << (MESSAGE_HEX * 4)) - 1)

                self.log_cb(f"[LEFT SIDE - ORACLE] Fully reconstructed hash for Block #{self.oracle_block_id}: {hex(msg_int)}...")

                self.msg_reg.write(dict(message=msg_int, block_id=self.oracle_block_id))
                self.log_cb(f"[LATCH SYNC] Reconstructed Oracle Hash published to Right Miner Side[cite: 1].")

                self.q_bits_buffer = 0
                self.q_bit_count = 0

# ------------------------------------------------------------------ Right Side: Async Miner Lanes
class AsyncWorker(threading.Thread):
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

class MirroredMinerLane(AsyncWorker):
    """Right Side: Async PoW miner reading the pre-reconstructed hash from the message latch[cite: 1]."""
    def __init__(self, lane: int, msg_reg: Register, resp_reg: Register, log_cb):
        super().__init__(f"miner-lane-{lane}")
        self.lane = lane
        self.msg_reg = msg_reg
        self.resp_reg = resp_reg
        self.log_cb = log_cb
        self.solved = 0

    async def main(self):
        last_mseq = -1
        while not self.stopped():
            mseq, job = self.msg_reg.read()
            if job is None or mseq == last_mseq:
                await asyncio.sleep(0.05)
                continue

            last_mseq = mseq
            await self._mine(mseq, job)
        self.status = "stopped"

    async def _mine(self, mseq, job):
        header = HEADER_DEFAULT
        msg_hex = pad_hex(job["message"], MESSAGE_HEX)
        head = header + DELIM
        tail = DELIM + msg_hex
        target = 1 << (256 - TARGET_DIFF)
        nonce = secrets.randbits(64)
        tried, t0 = 0, time.perf_counter()
        sha = hashlib.sha256

        while not self.stopped():
            current_mseq, _ = self.msg_reg.read()
            if current_mseq != mseq:
                return

            for i in range(256):
                nonce_str = pad_hex(nonce, NONCE_HEX)
                d = sha((head + nonce_str + tail).encode("ascii")).digest()
                val = int.from_bytes(d, "big")
                if val < target:
                    tried += i + 1
                    resp = build_response(header, nonce, job["message"])
                    
                    rec = dict(key=mseq, response=resp, block_id=job["block_id"], lane=self.lane)
                    if self.resp_reg.write_unless(lambda cur: cur is not None and cur["key"] == mseq, rec):
                        self.solved += 1
                        self.log_cb(f"[RIGHT SIDE - MINER] Lane {self.lane} SOLVED Reconstructed Oracle Block #{job['block_id']}! Padded Hex Nonce: {nonce_str}")
                    return
                nonce += 1
            tried += 256
            await asyncio.sleep(0.005)

# ------------------------------------------------------------------ Camera Source
class Source:
    def __init__(self, index: int):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            raise RuntimeError(f"Cannot open camera {index}.")
        self.cap = cap
        self.rgb = None

    def read(self):
        if not self.cap.grab():
            return None
        ok, bgr = self.cap.retrieve()
        if not ok:
            return None
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()

def downscale(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (SW, SH), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0)

def ppm_photo(rgb: np.ndarray) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")

# ------------------------------------------------------------------ Main Application GUI
class App:
    def __init__(self, window):
        self.window = window
        window.title("Symmetric Architecture: Ghosted Oracle Reconstructs Hash First <---> Miners")
        
        self.q = queue.Queue()
        self.ctl = dict(stop=False, thr=18, min_px=30, circuit=True)
        self._log_buf = []
        
        self.msg_reg = Register()
        self.resp_reg = Register()

        self.ghost_oracle = GhostedOracleWorker(self.msg_reg, self.log)
        self.ghost_oracle.start()

        self.miners = [MirroredMinerLane(i, self.msg_reg, self.resp_reg, self.log) for i in range(2)]
        for m in self.miners:
            m.start()

        self._build_ui()
        window.protocol("WM_DELETE_WINDOW", self._close)
        window.after(15, self._poll)

    def log(self, text: str):
        self._log_buf.append(text)

    def _flush_log(self):
        if not self._log_buf:
            return
        lines, self._log_buf = self._log_buf[-200:], []
        self.log_text.config(state="normal")
        
        at_bottom = self.log_text.yview()[1] >= 0.999
        self.log_text.insert("end", "\n".join(lines) + "\n")
        n = int(self.log_text.index("end-1c").split(".")[0])
        if n > 400:
            self.log_text.delete("1.0", f"{n - 300}.0")
        if at_bottom:
            self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        ctrl_frame = ttk.LabelFrame(main_frame, text="Symmetric Controls & Status", padding=8)
        ctrl_frame.pack(fill="x", padx=5, pady=5)

        self.b_start = ttk.Button(ctrl_frame, text="Start Reconstructing Simulation", command=self.start)
        self.b_start.pack(side="left", padx=5)
        self.b_stop = ttk.Button(ctrl_frame, text="Stop", command=self.stop, state="disabled")
        self.b_stop.pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="Status: Idle (Ghosted Oracle Reconstructs First)")
        ttk.Label(ctrl_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(side="left", padx=15)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill="both", expand=True, pady=5)

        self.cam_canvas = tk.Canvas(content_frame, width=PW, height=PH, bg="black")
        self.cam_canvas.pack(side="left", padx=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        log_frame = ttk.LabelFrame(content_frame, text="Event Log (Ghosted Oracle Reconstructs Hash First)", padding=5)
        log_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.log_text = tk.Text(log_frame, height=15, width=60, font=("Consolas", 9), state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def start(self):
        self.ctl["stop"] = False
        self.log("[SYSTEM] Starting execution with Left Oracle reconstructing hash before Right Side mining[cite: 1].")
        self.b_start.config(state="disabled")
        self.b_stop.config(state="normal")
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()

    def stop(self):
        self.ctl["stop"] = True
        self.log("[SYSTEM] Stop requested.")

    def _close(self):
        self.stop()
        self.ghost_oracle.stop()
        for m in self.miners:
            m.stop()
        self.window.destroy()

    def _sim_loop(self):
        try:
            src = Source(0)
        except RuntimeError as e:
            self.log(f"[ERROR] {e}")
            self.q.put(("done",))
            return

        self.log(f"[CAMERA] Opened USB camera successfully.")
        prev = None
        idx = 0

        while not self.ctl["stop"]:
            gray = src.read()
            if gray is None:
                self.log("[CAMERA] Frame read failed.")
                break

            small = downscale(gray)
            if prev is None or idx < 5:
                mask = np.zeros((SH, SW), bool)
            else:
                diff = np.abs(small.astype(np.int16) - prev.astype(np.int16))
                mask = diff > self.ctl["thr"]
            prev = small

            motion_px = int(mask.sum())
            motion = motion_px >= self.ctl["min_px"]
            x = motion_px if motion else 0

            r = 0
            driver_a = 0
            if motion and x > 4 and x % 2 == 1:
                r = 2 + (x * 7) % (x - 3)
                driver_a = int(miller_rabin_witness(r, x))

            driver_b = int(motion and x > 2 and x % 2 == 1)
            q_out = driver_a & driver_b  # Circuit decision Q (0 or 1)

            self.ghost_oracle.tick(idx, q_out)

            rgb = src.rgb
            self.q.put(("frame", ppm_photo(cv2.resize(rgb, (PW, PH)))))

            idx += 1
            time.sleep(0.03)

        src.close()
        self.q.put(("done",))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "done":
                    self.b_start.config(state="normal")
                    self.b_stop.config(state="disabled")
                    self.status_var.set("Status: Stopped")
        except queue.Empty:
            pass

        self._flush_log()

        seq, rec = self.resp_reg.read()
        if rec is not None and rec.get('response') is not None:
            resp_str = rec['response']
            is_valid = verify_response(resp_str)
            valid_str = "VERIFIED VALID ✓" if is_valid else "VERIFICATION FAILED ✗"
            self.log(f"[LATCH SUCCESS] Right Miner Side solved Pre-Reconstructed Oracle Block #{rec['block_id']} (Lane {rec['lane']}) [{valid_str}]:\n{resp_str}")
            self.resp_reg.write(dict(rec, response=None))

        window_after_fn = getattr(self.window, "after", None)
        if window_after_fn:
            window_after_fn(30, self._poll)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("850x480")
    root.mainloop()
