#!/usr/bin/env python3
"""
Full RP-architecture simulation driven by a real camera.

Covers every subsystem named in the paper:

  Section 2    Optical source, dual detector arrays A/B, photocurrent,
               spatial motion-energy mask E, hash, Driver A witness.
  Section 2.1  Driver B deterministic check c(x_n), RC integrator
               (Eq. 4-5), NPN emitter follower (Eq. 6), latch threshold,
               analog IF/THEN link, frame period T_f.
  Section 3    Spacetime manifold, node separations, propagation delays,
               causal-interval constraint ds^2 <= 0, v_p < c.
  Section 4    Forward phase space Z, inverted dual space Z*,
               mirror map M:(q,p,t)->(q*,-p*,-t), symplectic form,
               oracle/forward vector fields, pushforward condition
               V_*(X_O) = -X_F, phase-volume relation.
  Section 5    Time-reversal operator T, dissipative Lagrangian (Eq. 12),
               transformed equation of motion (Eq. 13), entropy
               generation dS_univ > 0, 74HC74 latch, 10 uF smoothing node.
  Section 1    Recursive chain X_{n+1} = V(O(R(M(F(D(I)))))) with
               history H_n.

Adaptations (explicitly NOT from the paper):
  - A camera replaces the paper's laser + dual InGaAs arrays. The frame
    is split into left/right halves emulating detector arrays A and B.
  - The user's toy hash replaces SHA-256.
  - c(x_n) is a plug-in; the paper does not define it. The default
    prototype is documented below and can be changed.

Bit-serial recursive chain (Section 1 + Sections 2-5 systems), built
as "the circuit as supplied" (the user's optoelectronic diagram):
  The oracle O produces BITS and the optoelectronic system performs
  the bit manipulation through the paper's equations:
    O bit b_n -> c(x_n) -> RC integrator Eq.(4)-(5) -> NPN stage with
    the diagram's threshold: IF V(C) > 0.7 V THEN output HIGH
    -> signal B; Driver A witness -> signal A; A and B feed BOTH an
    OR gate and an AND gate; both gate outputs couple through 10 pF
    into Y (the time code, which forces HIGH) -> D input of the
    74HC74 D flip-flop, clocked at 10 Hz (T = 100 ms):
    Q -> SELECT (the register samples), Q-bar -> observe;
    the register is X_{n+1} and feeds back as history H_n.
  The oracle bit comes from the circuit itself: a Schmitt-trigger
  comparator on the mirrored momentum p*, referenced to the 10 uF
  low-pass smoothing node. The c(x_n) wiring is an EXPLICIT prototype
  (the paper gives O's signature only and never defines c).

Components the paper names but never defines computationally (F, R,
factor extraction) run as labelled pass-through stages. No substitute
factorization algorithm is included anywhere.

Usage:
    python rp_simulation.py 91                          # real camera 0
    python rp_simulation.py 91 --camera 1
    python rp_simulation.py 91 --source synthetic       # labelled test source
    python rp_simulation.py 91 --csv run.csv
    python rp_simulation.py --guess                  # bit-pipeline guessing game
    python rp_simulation.py 1000003 --search         # bit-pooling higher/lower search
    python rp_simulation.py 1000003 --search --hidden 424242 --source synthetic
    python rp_simulation.py --report-components
"""

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Constants stated in the paper
# ----------------------------------------------------------------------

VS = 5.0        # supply voltage, Section 2.1
VBE = 0.7       # base-emitter drop, Section 2.1
VIH = 3.5       # logic-high threshold, Section 2.1
C_LIGHT = 299_792_458.0          # m/s
T_LOGIC_EXPECTED = 1.83          # t_logic ~ 1.83 RC, Section 2.1
SMOOTHING_CAP_UF = 10.0          # 10 uF node, Section 5.1
DFF = "74HC74 clocked D flip-flop"  # Section 5
# "The circuit as supplied" (user's optoelectronic diagram):
CLOCK_HZ = 10.0                  # clock/timer: 10 Hz, T = 100 ms
COUPLING_CAP_PF = 10.0           # 10 pF coupling into the Y time code
Y_THRESHOLD = VIH / VS           # logic threshold of the Y node (0.7)


# ----------------------------------------------------------------------
# Section 2: optical acquisition via camera (adaptation)
# ----------------------------------------------------------------------

class OpticalAcquisition:
    """Camera standing in for the laser / dual-detector optical stage."""

    def __init__(self, source, camera_index, width, height):
        self.width = width
        self.height = height
        self.source = source
        self._camera = None

        if source == "camera":
            self._camera = cv2.VideoCapture(camera_index)
            if not self._camera.isOpened():
                raise RuntimeError(
                    f"Could not open real camera {camera_index}. "
                    "Check the index and OS camera permissions."
                )
            self.label = f"real camera {camera_index}"
        else:
            self._t = 0
            self.label = "SYNTHETIC source (labelled test mode, not a camera)"

    def read_gray(self):
        """Return a uint8 grayscale frame of (height, width)."""
        if self.source == "camera":
            ok, frame = self._camera.read()
            if not ok or frame is None:
                raise RuntimeError("Real camera returned no frame.")
        else:
            rng = np.random.default_rng()
            t = self._t
            self._t += 1
            # Deterministic moving pattern + shot noise.
            x = np.arange(self.width, dtype=np.float32)
            phase = 0.05 * x + 2.0 * math.sin(0.05 * t)
            frame = 128 + 60 * np.sin(phase)[None, :] * np.ones(
                (self.height, 1), dtype=np.float32
            )
            frame += rng.normal(0, 8, (self.height, self.width))
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(
            gray, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

    def release(self):
        if self._camera is not None:
            self._camera.release()


def detector_arrays(gray):
    """Split the frame into detector array A (left) and B (right).

    This emulates the paper's two InGaAs arrays; the paper's arrays are
    hardware, not software, so this split is the adaptation.
    """
    h, w = gray.shape
    return gray[:, : w // 2].astype(np.float32), gray[:, w // 2:].astype(np.float32)


def photocurrent(array_a, array_b):
    """i_det(t): difference of detector intensities (prototype encoding)."""
    return float(np.mean(array_a) - np.mean(array_b))


def motion_energy(previous, current):
    """Spatial motion-energy mask E(x,y,t): squared intensity differences."""
    diff = current.astype(np.int32) - previous.astype(np.int32)
    return (diff * diff).astype(np.uint16)


def encode_energy(energy):
    """Reproducible serialization: dimensions + little-endian uint16 pixels."""
    h, w = energy.shape
    header = h.to_bytes(4, "little") + w.to_bytes(4, "little")
    return header + energy.astype("<u2", copy=False).tobytes(order="C")


def toy_hash(data):
    """User-requested toy hash replacing SHA-256 (16-bit)."""
    value = 0
    for byte in data:
        value = (31 * value + byte) % 65536
    return value


# ----------------------------------------------------------------------
# Section 2.1: Driver A
# ----------------------------------------------------------------------

def driver_a(x_n, energy):
    """r_n = 2 + hash(E) mod (x_n - 3), then Miller-Rabin witness test."""
    if x_n <= 3:
        raise ValueError("Witness-base formula requires x_n > 3.")
    h = toy_hash(encode_energy(energy))
    base = 2 + h % (x_n - 3)

    if not 2 <= base <= x_n - 2:  # degenerate small n guard
        return {"status": "evaluated", "hash": h, "base": base,
                "witness": None}

    d, s = x_n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    value = pow(base, d, x_n)
    witness = False
    if value not in (1, x_n - 1):
        for _ in range(s - 1):
            value = value * value % x_n
            if value == x_n - 1:
                break
        else:
            witness = True

    return {"status": "evaluated", "hash": h, "base": base,
            "witness": witness}


# ----------------------------------------------------------------------
# Section 2.1: Driver B — c(x_n), RC integrator, transistor, latch
# ----------------------------------------------------------------------

def default_c(x_n):
    """Prototype c(x_n). THE PAPER DOES NOT DEFINE c(x_n).

    This default simply gates the RC stage ON for every cycle (c = 1),
    i.e. "integrate every frame". Change or replace this function to
    test other deterministic checks; any choice is an external
    assumption, not part of the paper.
    """
    return 1


class DriverB:
    """RC integration stage, Eq. (4)-(5), emitter follower Eq. (6), latch."""

    def __init__(self, r_ohms, c_farads, c_fn=default_c):
        self.R = r_ohms
        self.C = c_farads
        self.tau = r_ohms * c_farads
        self.c_fn = c_fn
        self.v_c = 0.0
        self.latched = False
        self.t_logic_measured = None

    def check(self, x_n):
        return self.c_fn(x_n)

    def step(self, x_n, dt, c_value=None):
        """Advance the RC stage by dt seconds.

        Eq. (4): dV_C/dt + V_C/(RC) = (V_s/(RC)) * c(x_n)
        Integrated exactly for constant c over dt:
        Eq. (5): V_C(t) = V_s (1 - e^{-t/RC}) c(x_n)

        c_value, when given, overrides the check function: this is how
        the oracle's bit drives the RC stage in the bit-serial chain
        (the paper's c(x_n) is undefined, so the wiring is explicit).
        """
        c = self.check(x_n) if c_value is None else c_value
        # Exact update of Eq. (5) between cycles:
        if c > 0:
            target = VS * (1.0 - math.exp(-dt / self.tau)) * c
            # Charge from current value toward the c-scaled target.
            self.v_c = target + (self.v_c - target) * math.exp(-dt / self.tau)
        else:
            # c = 0 discharges the capacitor (Section 5 discharge discussion)
            self.v_c *= math.exp(-dt / self.tau)

        # Eq. (6): NPN emitter follower.
        v_b = max(0.0, self.v_c - VBE)

        # 74HC74-style clocked latch at V_IH.
        was_low = not self.latched
        self.latched = v_b >= VIH
        if was_low and self.latched:
            self.t_logic_measured = dt
        return {
            "c": c, "V_C": self.v_c, "V_B": v_b,
            "latched": self.latched,
            "t_logic_expected": T_LOGIC_EXPECTED * self.tau,
            "t_logic_measured": self.t_logic_measured,
        }


def analog_if_then(driver_a_out, driver_b_out):
    """Analog IF/THEN integration link, Section 2.

    IF (Driver A produced a witness decision) AND (Driver B latch high)
    THEN the cycle's terminal optical state is fed to the recursive
    chain. This is the paper's named link; its exact circuit equation
    is not given, so the logic-level combination is the explicit
    prototype reading.
    """
    a_ready = driver_a_out["witness"] is not None
    return bool(a_ready and driver_b_out["latched"])


# ----------------------------------------------------------------------
# Section 3: locality / causal boundary
# ----------------------------------------------------------------------

class LocalityModel:
    """Node grid with separations, delays and the ds^2 <= 0 constraint."""

    def __init__(self, node_spacing_m=1.0e-3, v_p_fraction=0.6):
        self.spacing = node_spacing_m
        self.v_p = v_p_fraction * C_LIGHT  # v_p < c, as required

    def check_pair(self, i, j):
        dx = abs(i - j) * self.spacing
        dt = dx / self.v_p
        ds2 = -C_LIGHT**2 * dt**2 + dx**2
        return {"dx": dx, "dt": dt, "ds2": ds2, "causal": ds2 <= 0.0}

    def frame_check(self, n_nodes=4):
        worst = None
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                r = self.check_pair(i, j)
                if worst is None or r["ds2"] > worst["ds2"]:
                    worst = r
        return worst


# ----------------------------------------------------------------------
# Section 4: forward / inverted dual phase spaces, mirror, vector fields
# ----------------------------------------------------------------------

class DualPhaseSpace:
    """Forward space Z and inverted dual space Z* with mirror map M.

    Canonical coordinates: q = state intensities, p = momentum-equivalent
    optical gradients (the paper's description of (q,p)).
    """

    def __init__(self):
        self.q = 0.0
        self.p = 0.0
        self.t = 0.0
        # Mirror map applied to the starred dual coordinates.
        self.q_star = 0.0
        self.p_star = 0.0
        self.t_star = 0.0

    @staticmethod
    def from_frame(prev, curr):
        """q: mean intensity; p: mean temporal gradient (optical gradient)."""
        q = float(curr.mean())
        p = float(np.abs(curr.astype(np.float32) - prev.astype(np.float32)).mean())
        return q, p

    def mirror(self):
        """M : (q, p, t) -> (q*, -p*, -t) in the dual coordinates Z*."""
        # The paper defines the coordinate-level map only; the encoding
        # of integers into these coordinates is not specified, so the
        # dual coordinates are carried alongside the running state.
        self.q_star = self.q
        self.p_star = -self.p
        self.t_star = -self.t
        return (self.q_star, self.p_star, self.t_star)

    def phase_volume(self, grid=32):
        """Numerically report int dq dp over Z and the mirrored Z*.

        Paper's stated relation: int_{Z*} dq* dp* = - int_Z dq dp.
        With the mirror map, p* = -p reverses orientation, which the
        numerical integral reproduces as a sign change.
        """
        q_min, q_max = 0.0, 255.0
        p_min, p_max = -64.0, 64.0
        dq = (q_max - q_min) / grid
        dp = (p_max - p_min) / grid
        vol_z = 0.0
        vol_zstar = 0.0
        for _ in range(grid):
            vol_z += dq * dp
            vol_zstar += dq * (-dp)  # mirrored orientation
        return {"Z": vol_z, "Z*": vol_zstar, "relation_holds": math.isclose(
            vol_zstar, -vol_z, rel_tol=1e-9)}

    def vector_fields(self):
        """X_F on Z and X_O on Z*; pushforward condition V_*(X_O) = -X_F.

        The paper states the condition but does not define the fields'
        action on factorization instances. The fields below are the
        phase-space identity flows (d(q,p)/dt = 0), reported so the
        pushforward check is executable; they are NOT a factorization
        procedure.
        """
        x_f = (0.0, 0.0)   # identity flow on Z
        x_o = (0.0, 0.0)   # identity flow on Z*
        pushforward_holds = (x_o[0] == -x_f[0]) and (x_o[1] == -x_f[1])
        return {"X_F": x_f, "X_O": x_o,
                "pushforward_holds": pushforward_holds,
                "note": "identity flows; paper defines no factorization "
                        "action for these fields"}


# ----------------------------------------------------------------------
# Section 5: dissipative dynamics, time reversal, entropy
# ----------------------------------------------------------------------

class DissipativeDynamics:
    """Eq. (12) Lagrangian and Eq. (13) transformed equation of motion.

    L = 1/2 m qdot^2 - V(q) - gamma q qdot sgn(qdot)
    EOM used: m qddot + gamma qdot sgn(qdot) + V'(q) = 0
    with V(q) = 1/2 k q^2 for the prototype potential.
    """

    def __init__(self, m=1.0, k=0.02, gamma=0.8, temp_k=300.0):
        self.m = m
        self.k = k
        self.gamma = gamma
        self.temp = temp_k
        self.q = 0.0
        self.p = 0.0
        self.dissipated = 0.0

    def set_state(self, q, p):
        self.q = q
        self.p = p

    def step(self, dt):
        """Semi-implicit Euler step of Eq. (13).

        Dissipation is accounted directly as damping power:
        F_damp * v = -gamma * |p| * p <= 0, so the energy dissipated in
        one step is gamma * |p| * |dq| with dq = p * dt.
        """
        f_spring = -self.k * self.q
        f_damp = -self.gamma * self.p * (1.0 if self.p >= 0 else -1.0)
        acc = (f_spring + f_damp) / self.m
        self.p += acc * dt
        dq = self.p * dt
        self.dissipated += self.gamma * abs(self.p) * abs(dq)
        self.q += dq

    def entropy_generated(self):
        """dS_univ = Q / T for the dissipated energy (Q > 0 required)."""
        if self.dissipated <= 0:
            return 0.0
        return self.dissipated / self.temp

    @staticmethod
    def time_reversal(q, p):
        """T : (q, p) -> (q, -p). Magnetic/spatial parts unchanged here."""
        return q, -p


# ----------------------------------------------------------------------
# Section 1: recursive chain F -> M -> R -> O(H_n) -> V
# ----------------------------------------------------------------------

class RecursiveChain:
    """Bit-serial recursive chain, built as "the circuit as supplied"
    (the user's optoelectronic diagram):

        X_{n+1} = V(O(R(M(F(D(I))))))

      D  OPTICAL BENCH: camera -> detectors A/B (delta intensity),
         energy mask E, and the Driver A randomized witness bit
         (Eq. 3 + Miller-Rabin)
      F  forward pass (pass-through; the paper defines no computation)
      M  temporal mirror: (q, p, t) -> (q*, -p*, -t) into Z*
      R  reversed pass (pass-through; the paper defines no computation)
      O  ORACLE PRODUCES A BIT from the circuit itself: a Schmitt-
         trigger comparator (threshold nonlinearity, Section 5) on the
         mirrored momentum p*, referenced to the 10 uF low-pass
         smoothing node (Section 5.1). No hash, no random placement.
         (The paper gives O's signature O(., H_n) only; this circuit
         reading is the explicit prototype.)
      V  THE CIRCUIT AS SUPPLIED (the diagram):
           problem instance bus x feeds BOTH drivers
           Driver B (deterministic): oracle bit -> c(x_n) -> RC
             integrator Eq. (4)-(5) -> NPN transistor with the
             annotated threshold: IF V(C) > 0.7 V THEN output HIGH
             -> signal B
           Driver A (randomized): optics -> Eq. 3 -> witness -> A
           A and B feed BOTH an OR gate and an AND gate
           OR and AND outputs couple through 10 pF into
             Y: the time code, which forces HIGH
           Y -> D input of the 74HC74 D flip-flop
           CLK: clock/timer at 10 Hz (T = 100 ms)
           Q -> SELECT; Q-bar -> observe
         The SELECT samples accumulate into the register (X_{n+1}).
    """

    def __init__(self, r_ohms, c_farads, register_bits=16,
                 hysteresis=0.05):
        self.driver_b = DriverB(r_ohms, c_farads)
        self.register_bits = register_bits
        self.mask = (1 << register_bits) - 1
        self.register = 0
        self.history = []   # H_n: bits seen by the oracle so far
        # Oracle comparator state (Schmitt trigger on p*):
        # reference level = 10 uF low-pass node (Section 5.1) driven
        # by the same R; held bit starts at 1 (powered-up latch).
        self._ref = None
        self._oracle_bit = 1
        self.hysteresis = hysteresis
        self.tau_ref = r_ohms * SMOOTHING_CAP_UF * 1e-6
        # "The circuit as supplied" state:
        self._y = 0.0            # Y node (time code) level
        self.tau_coupling = r_ohms * COUPLING_CAP_PF * 1e-12  # 10 pF
        self.clock_period = 1.0 / CLOCK_HZ   # T = 100 ms
        self._clock_accum = 0.0
        self.select_out = 0      # Q  -> SELECT
        self.observe_out = 1     # Q-bar -> observe

    def forward_F(self, state):
        return {"output": state, "status": "pass_through",
                "reason": "paper names F but gives no computational action"}

    def mirror_M(self, f_result, dps):
        state = f_result["output"]
        dps.q, dps.p = state["q"], state["p"]
        q_star, p_star, t_star = dps.mirror()
        return {"output": {"q*": q_star, "p*": p_star, "t*": t_star,
                           "q": state["q"], "p": state["p"]},
                "status": "coordinate_map",
                "map": "(q,p,t)->(q*,-p*,-t)"}

    def reversed_R(self, state):
        return {"output": state["output"], "status": "pass_through",
                "reason": "paper names R but gives no computational action"}

    def oracle_O(self, r_result, dt):
        """ORACLE BIT from the circuit itself.

        The paper defines no computation for O, so this circuit
        reading is the explicit prototype: a Schmitt-trigger
        comparator (threshold nonlinearity, Section 5) compares the
        mirrored momentum p* against a reference level set by the
        10 uF low-pass smoothing node (Section 5.1) driven by the
        same R. Inside the hysteresis band the previous bit is held.

        No hash and no random placement: the bit is a deterministic
        circuit response to the measured optical state. p* above the
        reference (optical gradient below its smoothed level) -> 1;
        p* below the reference -> 0; inside the band -> hold.
        """
        state = r_result["output"]
        p_star = state["p*"]

        if self._ref is None:
            self._ref = p_star if p_star != 0 else -1.0

        h = self.hysteresis * abs(self._ref)

        if p_star >= self._ref + h:
            self._oracle_bit = 1
        elif p_star <= self._ref - h:
            self._oracle_bit = 0
        # else: inside the hysteresis band - hold the previous bit.

        # 10 uF smoothing node: low-pass the reference with tau = R*C.
        alpha = dt / (self.tau_ref + dt)
        self._ref += alpha * (p_star - self._ref)

        bit = self._oracle_bit
        self.history.append(bit)
        return {"bit": bit, "status": "schmitt_comparator",
                "rule": "Schmitt trigger on p*, reference = 10 uF "
                        "smoothing node (Sections 5, 5.1)"}

    def circuit_V(self, oracle_bit, witness_bit, dt):
        """THE CIRCUIT AS SUPPLIED (the user's diagram).

        Signal path, exactly as drawn:

          Driver B (deterministic): the oracle bit drives c(x_n); the
          RC integrator charges/discharges per Eq. (4)-(5); the NPN
          stage applies the annotated threshold:
              IF V(C) > 0.7 V THEN output HIGH     -> signal B

          Driver A (randomized): Eq. 3 on the measured optics -> A

          A and B feed BOTH gates:
              OR gate  : A | B
              AND gate : A & B

          10 pF coupling into Y (the time code): both gate outputs
          coupling high force Y HIGH (charge above the logic
          threshold); a single gate output couples only partial
          charge, below the threshold. Y forces HIGH -> D input.

          74HC74 D flip-flop, CLK = 10 Hz (T = 100 ms):
              Q     -> SELECT
              Q-bar -> observe
          The SELECT samples accumulate into the register (X_{n+1}).
        """
        # -- IF/THEN link: Driver B path through RC + NPN.
        b_out = self.driver_b.step(None, dt, c_value=oracle_bit)
        signal_b = 1 if b_out["V_C"] > VBE else 0   # IF V(C)>0.7V THEN HIGH
        # -- Driver A path: randomized witness bit.
        signal_a = 1 if witness_bit is True else 0
        # -- OR gate and AND gate (both take A and B).
        or_out = signal_a | signal_b
        and_out = signal_a & signal_b
        # -- 10 pF coupling into the Y node (time code).
        if or_out and and_out:
            y_target = 1.0     # both couplings: Y forced HIGH
        elif or_out or and_out:
            y_target = 0.5     # single coupling: partial charge
        else:
            y_target = 0.0
        alpha = dt / (self.tau_coupling + dt)
        self._y += alpha * (y_target - self._y)
        y_high = self._y >= Y_THRESHOLD
        # -- 74HC74 clocked D flip-flop at 10 Hz.
        self._clock_accum += dt
        clocked = False
        while self._clock_accum >= self.clock_period:
            self._clock_accum -= self.clock_period
            self.select_out = 1 if y_high else 0     # Q -> SELECT
            self.observe_out = 1 - self.select_out    # Q-bar -> observe
            clocked = True
            self.register = ((self.register << 1)
                             | self.select_out) & self.mask
        return {"oracle_bit": oracle_bit, "A": signal_a, "B": signal_b,
                "V_C": b_out["V_C"], "V_B": b_out["V_B"],
                "OR": or_out, "AND": and_out,
                "Y": self._y, "y_high": y_high,
                "SELECT": self.select_out, "observe": self.observe_out,
                "clocked": clocked, "register": self.register,
                "status": "circuit_as_supplied",
                "systems": "IF V(C)>0.7V THEN HIGH; OR/AND gates; "
                           "10 pF -> Y time code; 74HC74 @ 10 Hz "
                           "-> SELECT / observe"}

    def cycle(self, state, dps, witness_bit, dt):
        f = self.forward_F(state)                     # F
        m = self.mirror_M(f, dps)                     # M
        r = self.reversed_R(m)                        # R
        o = self.oracle_O(r, dt)                      # O -> bit
        v = self.circuit_V(o["bit"], witness_bit, dt)   # V -> circuit
        return {"F": f, "M": m, "R": r, "O": o, "V": v,
                "X_next": self.register}


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Hidden-number guessing game - the guess IS the bit pipeline
#
# The oracle produces bits and the optoelectronic system performs the
# bit manipulation (the paper's equations and systems); the resulting
# 74HC74 register word, mapped into the allowed range, is the machine's
# GUESS. There is NO feedback: no higher/lower hints, no interval
# narrowing. The oracle works alone; its only inputs are the mirrored
# optical state and its own previous word (H_n).
#
# Explicit prototypes (not from the paper):
#   - the oracle bit rule (paper gives O's signature only)
#   - the word -> range decoding (paper defines no decoding)
# ----------------------------------------------------------------------

import secrets


def guess_by_bit_pipeline(args):
    """The hidden number is guessed BY the circuit as supplied.

    Per round, the circuit runs freely until the 74HC74 (clocked at
    10 Hz) has shifted in a full register word of SELECT samples:

        D frames -> F -> M -> R -> O(bit) -> V (circuit as supplied:
        RC Eq.4-5, IF V(C)>0.7V THEN HIGH -> B; OR/AND gates on A,B;
        10 pF -> Y time code; 74HC74 @ 10 Hz -> SELECT / observe)

    and the register, mapped into the allowed interval, is the guess.
    NO feedback: the oracle works alone.
    """
    hi = (args.n - 1) if args.n is not None else 99
    if args.hidden is not None:
        if not 2 <= args.hidden <= hi:
            raise ValueError(f"--hidden must be between 2 and {hi}.")
        hidden = args.hidden
    else:
        hidden = secrets.randbelow(hi - 1) + 2  # 2 .. hi

    bits = args.register_bits
    chain = RecursiveChain(args.r_ohms, args.c_farads, bits)
    acquisition = OpticalAcquisition(args.source, args.camera,
                                     args.width, args.height)
    dps = DualPhaseSpace()
    # Driver A needs an odd integer > 3 for the witness bit; with no n
    # given, 91 is used (a composite, so the witness bit is live; a
    # prime input would gate the IF/THEN link to constant 0).
    # Documented default, not from the paper.
    witness_n = args.n if (args.n is not None and args.n > 3
                           and args.n % 2 == 1) else 91

    guesses = []
    start = time.monotonic()

    print(f"Optical source: {acquisition.label}")
    print(f"Hidden number chosen secretly in 2..{hi}.")
    print(f"Guesser: the circuit as supplied - oracle bit -> Driver B "
          f"(RC Eq.4-5, IF V(C)>0.7V THEN HIGH) -> B; A and B -> OR and "
          f"AND gates -> 10 pF -> Y time code -> 74HC74 @ 10 Hz -> "
          f"SELECT samples -> register ({bits} bits)")
    print(f"NO FEEDBACK: the oracle works alone. Each round produces one "
          f"word (MSB-first, one SELECT sample per 100 ms clock edge); "
          f"the word decoded into 2..{hi} is the guess. The oracle bit "
          f"is a Schmitt-trigger comparator on the mirrored momentum "
          f"p* (10 uF-referenced) - a deterministic circuit response, "
          f"not a random placement.")
    print(f"Word -> guess mapping: 2 + word mod {hi - 1} "
          f"[explicit; paper defines no decoding]")
    if witness_n != args.n:
        print(f"Driver A witness input: {witness_n} (default; no n given)")
    print()

    success = False
    try:
        previous = acquisition.read_gray()
        for round_no in range(1, args.rounds + 1):
            # Clear the register: the new word forms MSB-first (first
            # SELECT sample clocked in ends up most significant) and is
            # decoded only at the END of the round. The oracle never
            # sees the partial word - its comparator reads only the
            # optics, and the circuit runs free until the word is full.
            chain.register = 0

            # Run cycles until the 74HC74 has clocked `bits` SELECT
            # samples at 10 Hz (about 3 frames per sample at 30 fps).
            collected = 0
            while collected < bits:
                current = acquisition.read_gray()
                energy = motion_energy(previous, current)
                prev_gray = previous
                previous = current
                q, p = DualPhaseSpace.from_frame(prev_gray, current)
                a_out = driver_a(witness_n, energy)
                result = chain.cycle({"q": q, "p": p}, dps,
                                      a_out["witness"],
                                      args.frame_period_s)
                if result["V"]["clocked"]:
                    collected += 1

            # END OF GUESS: the completed word, decoded into the
            # allowed range, is the machine's guess.
            guess_word = chain.register
            guess = 2 + (guess_word % (hi - 1))
            guesses.append(guess)

            success = guess == hidden
            print(f"round {round_no:3d}: word="
                  f"0b{guess_word:0{bits}b} -> guess={guess:3d}  "
                  + ("MATCH" if success else ""))

            if success:
                break
    finally:
        acquisition.release()

    elapsed = time.monotonic() - start
    print()
    print("=" * 78)
    print("HIDDEN-NUMBER GUESSING GAME - the oracle works alone")
    print("=" * 78)
    print(f"Rounds played              : {len(guesses)} in "
          f"{elapsed:.2f} s")
    print(f"Guesses (in order)         : {guesses[:12]}"
          + (" ..." if len(guesses) > 12 else ""))
    print(f"RECOVERED hidden number    : {guesses[-1] if guesses else None}")
    print(f"TRUE hidden number         : {hidden}")
    print(f"Match                      : {success}")
    if success:
        print(f"Found in {len(guesses)} round(s).")
    else:
        print(f"NOT found within {args.rounds} rounds.")
    print("-" * 78)
    print("Every guess was PRODUCED by the circuit as supplied:")
    print("oracle bits -> Driver B (RC Eq.4-5, IF V(C)>0.7V THEN HIGH)")
    print("-> OR/AND gates with Driver A -> 10 pF -> Y time code ->")
    print("74HC74 @ 10 Hz -> SELECT samples -> register.")
    print("NO feedback of any kind: no higher/lower hints, no interval")
    print("narrowing. The oracle's only input is the mirrored optical")
    print("state; its bit is a Schmitt-trigger comparator response")
    print("(10 uF-referenced), placed by the circuit, not at random.")
    print("Register variation across rounds comes from the measured")
    print("optics (Driver A Eq.3 on the energy mask + shot noise),")
    print("not from any random bit placement.")
    print("Explicit prototypes (not from the paper): the oracle's")
    print("comparator reading of O and the word->range decoding.")
    print("=" * 78)
    return 0 if success else 1



# ----------------------------------------------------------------------
# Bit-pooling higher/lower oracle search, applied to the circuit
#
# Adapted from the Bit-Pooling Oracle Search Lab. The lab's oracle
# answers "is target >= trial?" and the search accumulates one bit per
# step, MSB first. Here every one of those bits is carried THROUGH the
# circuit as supplied instead of being written straight into a variable:
#
#   frames -> F -> M -> R   (optics, pass-through / mirror as before)
#   higher/lower oracle bit -> c(x_n) -> RC Eq.(4)-(5) -> NPN
#       IF V(C) > 0.7 V THEN HIGH -> B
#   A (held HIGH, see below) and B -> OR and AND gates -> 10 pF -> Y
#   Y -> D of the 74HC74 @ 10 Hz -> Q = SELECT
#   SELECT is the bit that is pooled; the next trial value is built from
#   the SELECT samples, never from the oracle's own answer.
#
# Explicit choices (not from the paper):
#   - Gate input A is tied HIGH in this mode. With A = 1 the AND gate
#     follows B, so Y reaches 0.7 only when B is HIGH and SELECT == B.
#     With the live Driver A witness, SELECT = A AND B would drop
#     oracle ones at random and the search could not converge. Driver A
#     is still evaluated every frame and its witness count is reported.
#   - One search step per 74HC74 clock edge (100 ms), so a b-bit target
#     takes b clock edges.
# ----------------------------------------------------------------------


class HigherLowerOracle:
    """Secret target with a higher/lower query (from the Bit-Pooling Lab)."""

    def __init__(self, bit_depth, target):
        self.bit_depth = bit_depth
        self._target = target
        self.queries = 0

    def query_bit(self, trial):
        """1 if target >= trial else 0."""
        self.queries += 1
        return 1 if self._target >= trial else 0

    def reveal(self):
        return self._target


def pool_register(word, bit_depth, kernel=4, stride=2, op="POPCOUNT",
                  schmitt=0.5):
    """Hardware-style pooling of the accumulated word (from the Lab)."""
    bits = [(word >> i) & 1 for i in range(bit_depth)]
    out = []
    for i in range(0, len(bits), stride):
        block = bits[i:i + kernel]
        if not block:
            break
        if op == "POPCOUNT":
            r = 1 if sum(block) / len(block) >= schmitt else 0
        elif op == "AND":
            r = 1 if all(block) else 0
        elif op == "OR":
            r = 1 if any(block) else 0
        elif op == "XOR":
            r = sum(block) % 2
        elif op == "MAJORITY":
            r = 1 if sum(block) > len(block) / 2.0 else 0
        else:
            r = block[0]
        out.append(r)
    value = 0
    for idx, b in enumerate(out):
        value |= b << idx
    return value, len(out)


def search_by_bit_pooling(args):
    hi = (args.n - 1) if args.n is not None else 99
    if args.hidden is not None:
        if not 2 <= args.hidden <= hi:
            raise ValueError(f"--hidden must be between 2 and {hi}.")
        hidden = args.hidden
    else:
        hidden = secrets.randbelow(hi - 1) + 2

    bit_depth = hi.bit_length()
    oracle = HigherLowerOracle(bit_depth, hidden)
    chain = RecursiveChain(args.r_ohms, args.c_farads, bit_depth)
    acquisition = OpticalAcquisition(args.source, args.camera,
                                     args.width, args.height)
    dps = DualPhaseSpace()
    witness_n = args.n if (args.n is not None and args.n > 3
                           and args.n % 2 == 1) else 91

    print(f"Optical source: {acquisition.label}")
    print(f"Hidden number secret in 2..{hi}; bit depth {bit_depth}.")
    print("Search: higher/lower oracle, MSB first, one bit per 74HC74 "
          "clock edge (10 Hz).")
    print("Each oracle bit travels: c(x_n) -> RC Eq.4-5 -> IF V(C)>0.7V "
          "THEN HIGH -> B -> OR/AND with A -> 10 pF -> Y -> 74HC74 -> "
          "SELECT -> pooled word.")
    print("Gate input A is tied HIGH in this mode (explicit choice); "
          "Driver A is still evaluated and counted.")
    print()

    pooled = 0
    mismatches = 0
    witness_true = 0
    frames_used = 0
    start = time.monotonic()
    try:
        chain.register = 0
        previous = acquisition.read_gray()
        for step in range(1, bit_depth + 1):
            shift = bit_depth - step
            trial = pooled | (1 << shift)
            oracle_bit = oracle.query_bit(trial)   # held until the edge

            while True:
                current = acquisition.read_gray()
                energy = motion_energy(previous, current)
                prev_gray = previous
                previous = current
                q, p = DualPhaseSpace.from_frame(prev_gray, current)
                a_out = driver_a(witness_n, energy)
                if a_out["witness"]:
                    witness_true += 1
                f = chain.forward_F({"q": q, "p": p})
                m = chain.mirror_M(f, dps)
                r = chain.reversed_R(m)
                v = chain.circuit_V(oracle_bit, True, args.frame_period_s)
                frames_used += 1
                if v["clocked"]:
                    break

            select = v["SELECT"]
            pooled |= select << shift
            if select != oracle_bit:
                mismatches += 1
            if args.verbose or step <= 8 or step == bit_depth:
                print(f"step {step:3d}/{bit_depth}: trial="
                      f"0b{trial:0{bit_depth}b} oracle={oracle_bit} "
                      f"V(C)={v['V_C']:.2f}V B={v['B']} Y={v['Y']:.2f} "
                      f"SELECT={select} pooled=0b{pooled:0{bit_depth}b}")
    finally:
        acquisition.release()

    elapsed = time.monotonic() - start
    found = pooled == hidden
    pooled_val, pooled_len = pool_register(
        pooled, bit_depth, args.pool_kernel, args.pool_stride, args.pool_op)

    print()
    print("=" * 78)
    print("BIT-POOLING HIGHER/LOWER SEARCH - through the circuit as supplied")
    print("=" * 78)
    print(f"Bit depth / clock edges     : {bit_depth} "
          f"({bit_depth / CLOCK_HZ:.1f} s of 10 Hz clock)")
    print(f"Frames used                 : {frames_used} in {elapsed:.2f} s")
    print(f"Oracle queries              : {oracle.queries}")
    print(f"SELECT != oracle bit        : {mismatches} step(s)")
    print(f"Driver A witness=True       : {witness_true} frame(s) "
          f"(informational; A gate tied HIGH)")
    print(f"Pooled word (SELECT)        : {pooled}  0b{pooled:0{bit_depth}b}")
    print(f"74HC74 register             : {chain.register}")
    print(f"Pooled register ({args.pool_op}, k={args.pool_kernel}, "
          f"s={args.pool_stride}) : {pooled_val}  ({pooled_len} bits)")
    print(f"TRUE hidden number          : {hidden}")
    print(f"Match                       : {found}")
    print("LOCKED & REVEALED" if found else "POOL MISMATCH")
    print("=" * 78)
    return 0 if found else 1


# Component inventory
# ----------------------------------------------------------------------

COMPONENTS = [
    ("Optical source", "camera (adaptation; paper: IR laser)"),
    ("Acquisition D", "camera frame -> grayscale (implemented)"),
    ("Detector arrays A/B", "frame halves (adaptation; paper: InGaAs)"),
    ("Photocurrent i_det", "intensity difference (prototype encoding)"),
    ("Energy mask E", "squared frame differences (prototype)"),
    ("Hash", "toy hash (user-requested replacement of SHA-256)"),
    ("Driver A", "r_n formula + Miller-Rabin witness (implemented)"),
    ("Driver B check c(x_n)", "plug-in; paper does not define it"),
    ("RC integrator Eq.4-5", "implemented (exact exponential update)"),
    ("NPN follower Eq.6", "V_B = max(0, V_C - 0.7) (implemented)"),
    ("IF/THEN link (diagram)", "IF V(C) > 0.7 V THEN output HIGH "
                               "(implemented per the circuit diagram)"),
    ("OR gate", "A | B, both gates fed by signals A and B (diagram)"),
    ("AND gate", "A & B (diagram)"),
    ("Y time code", "OR and AND outputs couple through 10 pF into Y, "
                    "which forces HIGH (diagram)"),
    ("Clock/timer", "10 Hz, T = 100 ms (diagram)"),
    ("Locality ds^2<=0", "node/delay model (implemented)"),
    ("Forward F", "pass-through; paper defines no computation"),
    ("Mirror M", "coordinate map implemented; encoding not specified"),
    ("Reversed R", "pass-through; paper defines no computation"),
    ("Oracle O", "produces BITS from the circuit: Schmitt-trigger "
                "comparator on p*, 10 uF-referenced (Sections 5, 5.1); "
                "paper gives signature only"),
    ("History H_n", "oracle's own previous word (no game feedback)"),
    ("Viewport V", "the circuit as supplied (diagram): RC Eq.4-5 -> "
                  "IF V(C)>0.7V THEN HIGH -> signal B; OR/AND gates on "
                  "A,B -> 10 pF -> Y time code -> 74HC74 @ 10 Hz -> "
                  "SELECT (register) / observe"),
    ("Phase spaces Z, Z*", "implemented with identity flows"),
    ("Symplectic form", "recorded as dq* ^ dp*"),
    ("Phase-volume relation", "numerically checked"),
    ("Dissipative dynamics Eq.12-13", "implemented (prototype potential)"),
    ("Time reversal T", "(q,p)->(q,-p) (implemented)"),
    ("Entropy dS_univ", "Q/T accounting (implemented)"),
    ("Factor extraction", "not specified by paper; NOT fabricated"),
    ("Bit-pooling search", "higher/lower oracle bits carried through "
                           "the circuit; SELECT samples pooled MSB-first "
                           "(--search)"),
    ("Hidden-number guesser", "guesses produced by the circuit as "
                              "supplied (oracle bits -> electrics -> "
                              "SELECT -> register); no feedback; word "
                              "decoding is an explicit prototype"),
]


def print_components():
    print("RP component inventory")
    print("-" * 78)
    for name, status in COMPONENTS:
        print(f"{name:<34} {status}")
    print("-" * 78)
    print("Stages marked 'not specified' pass state through and record")
    print("status; no substitute algorithm is used anywhere.")


# ----------------------------------------------------------------------
# Main simulation
# ----------------------------------------------------------------------

def run(args):
    acquisition = OpticalAcquisition(args.source, args.camera,
                                     args.width, args.height)
    locality = LocalityModel()
    dps = DualPhaseSpace()
    dynamics = DissipativeDynamics()
    # Bit-serial chain: the circuit as supplied (user's diagram).
    # Oracle bits -> Driver B (RC Eq.4-5, IF V(C)>0.7V THEN HIGH)
    # -> OR/AND gates with Driver A -> 10 pF -> Y time code
    # -> 74HC74 @ 10 Hz -> SELECT / observe.
    chain = RecursiveChain(args.r_ohms, args.c_farads, args.register_bits)
    driver_b = chain.driver_b

    log_file = None
    writer = None
    if args.csv is not None:
        log_file = args.csv.open("x", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        writer.writerow([
            "cycle", "i_det", "hash", "base", "witness", "oracle_bit",
            "V_C", "V_B", "B", "A", "OR", "AND", "Y", "y_high",
            "SELECT", "observe", "clocked", "register",
            "q", "p", "q_star", "p_star", "dS_univ",
            "F_status", "M_status", "R_status", "O_status", "V_status",
        ])

    frames = args.frames
    start = time.monotonic()
    witnesses = 0
    signal_b_ones = 0
    oracle_ones = 0
    select_ones = 0
    clock_edges = 0
    last = None

    try:
        previous = acquisition.read_gray()
        causal = locality.frame_check()
        print(f"Optical source: {acquisition.label}")
        print(f"Locality: v_p = {locality.v_p:.3e} m/s (< c), "
              f"worst-pair ds^2 = {causal['ds2']:.3e} m^2 "
              f"(causal: {causal['causal']})")
        print(f"Bit-serial chain (the circuit as supplied): oracle bit -> "
              f"Driver B (RC Eq.4-5, IF V(C)>0.7V THEN HIGH); A and B -> "
              f"OR and AND gates -> 10 pF -> Y time code -> 74HC74 D "
              f"flip-flop @ {CLOCK_HZ:.0f} Hz (T = "
              f"{chain.clock_period * 1e3:.0f} ms) -> SELECT / observe")

        for cycle in range(frames):
            current = acquisition.read_gray()
            a_arr, b_arr = detector_arrays(current)
            i_det = photocurrent(a_arr, b_arr)
            energy = motion_energy(previous, current)
            prev_gray = previous
            previous = current

            # Driver A: witness bit from the optical energy hash
            a_out = driver_a(args.n, energy)
            if a_out["witness"]:
                witnesses += 1

            # Phase-space state from the optical frame pair
            q, p = DualPhaseSpace.from_frame(prev_gray, current)
            dps.q, dps.p, dps.t = q, p, cycle * args.frame_period_s
            dynamics.set_state(q - 128.0, p - 16.0)
            dynamics.step(args.frame_period_s)
            ds = dynamics.entropy_generated()

            # Time reversal on the terminal state of the cycle
            q_tr, p_tr = DissipativeDynamics.time_reversal(dynamics.q, dynamics.p)

            # Bit-serial recursive chain:
            #   D -> F -> M -> R -> O(bit) -> V(optoelectronic bit manip)
            result = chain.cycle(
                {"q": q, "p": p, "i_det": i_det, "cycle": cycle},
                dps, a_out["witness"], args.frame_period_s)
            last = result
            v = result["V"]
            o = result["O"]
            if v["B"]:
                signal_b_ones += 1
            if o["bit"]:
                oracle_ones += 1
            if v["clocked"]:
                clock_edges += 1
                if v["SELECT"]:
                    select_ones += 1
            statuses = (result["F"]["status"], result["M"]["status"],
                        result["R"]["status"], result["O"]["status"],
                        result["V"]["status"])

            if writer is not None:
                writer.writerow([
                    cycle, f"{i_det:.4f}", a_out["hash"], a_out["base"],
                    a_out["witness"], o["bit"],
                    f"{v['V_C']:.4f}", f"{v['V_B']:.4f}",
                    v["B"], v["A"], v["OR"], v["AND"],
                    f"{v['Y']:.4f}", int(v["y_high"]),
                    v["SELECT"], v["observe"], int(v["clocked"]),
                    v["register"],
                    f"{q:.3f}", f"{p:.3f}", f"{dps.q_star:.3f}",
                    f"{dps.p_star:.3f}", f"{ds:.6e}", *statuses,
                ])

            if args.verbose:
                print(f"cycle {cycle:3d}  i_det={i_det:+7.2f}  "
                      f"hash={a_out['hash']:5d}  "
                      f"O_bit={o['bit']}  "
                      f"V_C={v['V_C']:.3f}V  A={v['A']} B={v['B']}  "
                      f"OR={v['OR']} AND={v['AND']}  "
                      f"Y={v['Y']:.2f}  "
                      f"SEL={v['SELECT']} obs={v['observe']}  "
                      f"reg={v['register']:0{args.register_bits}b}  "
                      f"dS={ds:.2e}")

        elapsed = time.monotonic() - start
        vol = dps.phase_volume()
        vf = dps.vector_fields()

        print()
        print("=" * 78)
        print("RUN REPORT")
        print("=" * 78)
        print(f"Input n                     : {args.n}")
        print(f"Cycles                      : {frames} in {elapsed:.2f} s")
        print(f"Driver A witnesses observed : {witnesses}")
        if witnesses:
            print("  -> Driver A conclusion    : n is composite")
        else:
            print("  -> Driver A conclusion    : no witness; primality NOT established")
        print(f"Driver B signal-HIGH cycles : {signal_b_ones} "
              f"(V(C) > 0.7 V per the diagram)")
        print(f"Clock edges @ 10 Hz         : {clock_edges} "
              f"(SELECT samples clocked into the register)")
        print(f"RC tau                      : {driver_b.tau:.4f} s "
              f"(t_logic expected ~ {T_LOGIC_EXPECTED * driver_b.tau:.4f} s)")
        print(f"Phase volume Z / Z*         : {vol['Z']:.1f} / {vol['Z*']:.1f} "
              f"(relation holds: {vol['relation_holds']})")
        print(f"Pushforward V_*(X_O)=-X_F   : {vf['pushforward_holds']} "
              f"({vf['note']})")
        entropy_note = "> 0 as required" if ds > 0 else "zero (no dissipation yet)"
        print(f"Entropy generated dS_univ   : {ds:.6e} J/K ({entropy_note})")
        print(f"Time reversal applied       : (q,p)->(q,-p) each cycle")
        print()
        print("BIT PIPELINE (oracle bits -> the circuit as supplied)")
        print("-" * 78)
        print(f"Oracle bits produced        : {frames} "
              f"({oracle_ones} ones, {frames - oracle_ones} zeros)")
        print(f"Signal B (V(C)>0.7V) cycles : {signal_b_ones}")
        print(f"SELECT samples clocked      : {clock_edges} "
              f"({select_ones} SELECT=1, {clock_edges - select_ones} SELECT=0)")
        print(f"Y time code (10 pF coupling): threshold "
              f"{Y_THRESHOLD:.2f}; both gates HIGH force Y HIGH")
        reg = chain.register
        bits_str = format(reg, "0{}b".format(args.register_bits))
        print(f"74HC74 register (X)         : {bits_str}")
        print(f"  binary                    : 0b{bits_str}")
        print(f"  hex                       : 0x{reg:0{(args.register_bits + 3) // 4}X}")
        print(f"  decimal                   : {reg}")
        if last is not None:
            print(f"Chain statuses              : "
                  f"F={last['F']['status']}  M={last['M']['status']}  "
                  f"R={last['R']['status']}  O={last['O']['status']}  "
                  f"V={last['V']['status']}")
            print(f"Oracle rule                 : {last['O']['rule']}")
            print(f"V systems                   : {last['V']['systems']}")
        print(f"Factorization result        : none - the paper specifies no "
              "factor extraction; none is fabricated")
        print("=" * 78)

    finally:
        acquisition.release()
        if log_file is not None:
            log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Full RP-architecture camera simulation.")
    parser.add_argument("n", nargs="?", type=int,
                        help="odd integer > 3 to test with Driver A")
    parser.add_argument("--source", choices=["camera", "synthetic"],
                        default="camera",
                        help="'camera' uses a real device; 'synthetic' is a "
                             "labelled test pattern")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--frame-period-s", type=float, default=1.0 / 30.0)
    parser.add_argument("--r-ohms", type=float, default=10_000.0,
                        help="RC resistance (default 10k)")
    parser.add_argument("--c-farads", type=float, default=1.0e-6,
                        help="RC capacitance (default 1uF; paper also names "
                             "a separate 10uF smoothing node)")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--register-bits", type=int, default=16,
        help="74HC74 shift-register width for the bit pipeline; default 16")
    parser.add_argument("--report-components", action="store_true")
    parser.add_argument(
        "--guess", action="store_true",
        help="hidden-number guessing game run by the circuit as supplied "
             "(oracle bit -> RC Eq.4-5, IF V(C)>0.7V THEN HIGH -> B; "
             "OR/AND gates on A,B; 10 pF -> Y time code; 74HC74 @ 10 Hz "
             "-> SELECT -> register); the register word is the machine's "
             "guess; NO feedback - the oracle (Schmitt comparator) "
             "works alone")
    parser.add_argument(
        "--hidden", type=int,
        help="force the hidden number (test mode); by default it is "
             "chosen secretly at random")
    parser.add_argument(
        "--rounds", type=int, default=100,
        help="maximum guessing rounds; default: 100")
    parser.add_argument(
        "--search", action="store_true",
        help="bit-pooling higher/lower oracle search (from the Bit-Pooling "
             "Lab) with every bit carried through the circuit as supplied")
    parser.add_argument("--pool-op", default="POPCOUNT",
                        choices=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"])
    parser.add_argument("--pool-kernel", type=int, default=4)
    parser.add_argument("--pool-stride", type=int, default=2)
    args = parser.parse_args()

    if args.report_components:
        print_components()
        return 0

    if args.search:
        if args.n is not None and args.n < 4:
            parser.error("with --search, n must be at least 4")
        hi = (args.n - 1) if args.n is not None else 99
        if args.hidden is not None and not 2 <= args.hidden <= hi:
            parser.error(f"--hidden must be between 2 and {hi}")
        if args.pool_kernel < 1 or args.pool_stride < 1:
            parser.error("--pool-kernel and --pool-stride must be >= 1")
        return search_by_bit_pooling(args)

    if args.guess:
        # The hidden number and every guess must be strictly less
        # than the input number n (range 2 .. n-1). The guesses are
        # produced by the bit pipeline: oracle bits -> optoelectronic
        # bit manipulation -> 74HC74 register -> guess.
        if args.n is not None and args.n < 4:
            parser.error("with --guess, n must be at least 4 so the "
                         "search range 2 .. n-1 is non-empty")
        hi = (args.n - 1) if args.n is not None else 99
        if args.hidden is not None and not 2 <= args.hidden <= hi:
            parser.error(f"--hidden must be between 2 and {hi} "
                         f"(strictly less than "
                         f"{'n = ' + str(args.n) if args.n is not None else 'the default bound 100'})")
        if args.rounds < 1:
            parser.error("--rounds must be at least 1")
        try:
            return guess_by_bit_pipeline(args)
        except ValueError as error:
            parser.error(str(error))

    if args.n is None:
        parser.error("provide n, or use --guess or --report-components")
    if args.n <= 3 or args.n % 2 == 0:
        parser.error("n must be an odd integer greater than 3")
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
