"""
fault_engine.py
===============
Mathematical Fault Models for Jet Engine PHM.
Based on Edward Balaban's research on Integrated Vehicle Health Management.

Implements four fault categories:
  Case 1 – Single Sensor, Single Fault  : Bias | Scale | Drift | Intermittent
  Case 2 – Single Sensor, Multi-Fault   : Compound sequential faults
  Case 3 – Multi-Sensor Faults          : Independent faults across channels
  Case 4 – System Fault                 : Physical degradation from N-CMAPSS health params

REVISION NOTES (Lab Data Version):
  • Fault magnitudes are now expressed as a fraction of the GLOBAL sensor range
    (i.e. how much the sensor moves across all speed steps and all runs).
  • This replaces within-window std which was ~0.0003 on plateau data — far too
    small to produce visible faults.
  • FaultEngine now accepts global_sensor_range (shape S,) computed from X_nominal
    in NB1 and passes it through to every fault function via sensor_idx.
  • apply_bias  : uses global_sensor_range[sensor_idx]  (fixed clip 0→1)
  • apply_drift : uses global_sensor_range[sensor_idx]  (uses ptp → now global range)
  • apply_scaling    : unchanged (multiplicative, independent of range)
  • apply_intermittent: uses global_sensor_range[sensor_idx] for spike amplitude
  • Default magnitude range [0.15, 0.45] × global_range produces offsets of
    ~0.03–0.15 which are clearly visible above the noise floor of ~0.0003.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE_HZ = 10          # 10 Hz for lab data

# Fault type labels (used by the classifier head)
FAULT_LABELS = {
    0: "Nominal",
    1: "Bias",
    2: "Scaling",
    3: "Drift",
    4: "Intermittent",
    5: "Compound",
    6: "MultiSensor",
    7: "SystemFault",
}
FAULT_LABEL_TO_INT = {v: k for k, v in FAULT_LABELS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Fault Specification Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaultSpec:
    """
    Describes a single fault applied to one sensor channel.

    Parameters
    ----------
    sensor_idx   : Column index in the sensor array.
    fault_type   : One of 'bias','scaling','drift','intermittent','compound'.
    magnitude    : Primary magnitude parameter (fraction of global sensor range).
    onset_frac   : Fraction of window length where fault begins [0, 1].
    params       : Extra parameters (e.g., second fault for compound).
    """
    sensor_idx  : int
    fault_type  : str
    magnitude   : float = 0.25
    onset_frac  : float = 0.0
    params      : Dict  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Internal scale helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_scale(
    signal: NDArray[np.float32],
    sensor_idx: Optional[int],
    global_sensor_range: Optional[NDArray[np.float32]],
    fallback_min: float = 0.05,
) -> float:
    """
    Return the scale reference for fault magnitude computation.

    Priority order:
      1. global_sensor_range[sensor_idx]  — full operating range across all runs
         (best choice for plateau data where within-window variance is near zero)
      2. fallback_min                     — safety floor so scale is never zero

    Parameters
    ----------
    signal              : The 1-D sensor signal for this window (used only as fallback).
    sensor_idx          : Index of the sensor being faulted.
    global_sensor_range : Array of shape (n_sensors,) — range of each sensor
                          across ALL nominal windows. Computed in NB1 from X_nominal.
    fallback_min        : Minimum allowed scale to prevent zero-magnitude faults.
    """
    if (global_sensor_range is not None
            and sensor_idx is not None
            and sensor_idx < len(global_sensor_range)):
        scale = float(global_sensor_range[sensor_idx])
        if scale > fallback_min:
            return scale

    # Fallback: use within-window ptp but floor it at fallback_min
    # This handles edge cases but should rarely trigger if global_sensor_range
    # is correctly computed from X_nominal.
    scale = float(np.ptp(signal))
    return max(scale, fallback_min)


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 – Single Sensor, Single Fault
# ─────────────────────────────────────────────────────────────────────────────

def apply_bias(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
    sensor_idx: Optional[int] = None,
    global_sensor_range: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Additive constant offset.
        x_f[t] = x[t] + magnitude * global_range[sensor]   for t >= onset

    magnitude is expressed as a fraction of the global sensor operating range,
    NOT the within-window std (which is ~0.0003 on plateau data).
    """
    sig   = signal.copy()
    scale = _get_scale(signal, sensor_idx, global_sensor_range)
    sig[onset:] += float(magnitude) * scale
    return np.clip(sig, 0.0, 1.0)


def apply_scaling(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
    sensor_idx: Optional[int] = None,
    global_sensor_range: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Multiplicative gain error.
        x_f[t] = (1 + magnitude) * x[t]   for t >= onset

    Scaling is multiplicative so it does not need the global range reference —
    the effect is always proportional to the signal level.
    """
    sig = signal.copy()
    sig[onset:] *= (1.0 + float(magnitude))
    return np.clip(sig, 0.0, 1.0)


def apply_drift(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
    sensor_idx: Optional[int] = None,
    global_sensor_range: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Linearly increasing offset (ramp fault).
        x_f[t] = x[t] + ramp(0 → magnitude * global_range)   for t >= onset

    Total accumulated drift by end of window = magnitude * global_range[sensor].
    Using global range as scale reference so the ramp is always clearly visible
    regardless of within-window signal variance.
    """
    sig     = signal.copy()
    n_after = len(signal) - onset
    if n_after <= 0:
        return sig
    scale = _get_scale(signal, sensor_idx, global_sensor_range)
    ramp  = np.linspace(0.0, float(magnitude) * scale, n_after).astype(np.float32)
    sig[onset:] += ramp
    return np.clip(sig, 0.0, 1.0)


def apply_intermittent(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
    pulse_prob: float = 0.35,
    rng_seed: Optional[int] = None,
    sensor_idx: Optional[int] = None,
    global_sensor_range: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Random spike pulses.
        x_f[t] = x[t] ± magnitude * global_range[sensor]   w.p. pulse_prob

    Using global range so spikes are clearly above the within-window noise floor.
    pulse_prob = 0.35 means ~35 spikes in a 100-sample window.
    """
    rng_  = np.random.default_rng(rng_seed)
    sig   = signal.copy()
    n     = len(signal) - onset
    scale = _get_scale(signal, sensor_idx, global_sensor_range)
    mask  = rng_.random(n) < pulse_prob
    signs = rng_.choice([-1.0, 1.0], size=n)
    sig[onset:][mask] += signs[mask] * float(magnitude) * scale
    return np.clip(sig, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 – Single Sensor, Multi-Fault (Compound)
# ─────────────────────────────────────────────────────────────────────────────

def apply_compound(
    signal: NDArray[np.float32],
    fault_sequence: List[Tuple[str, float, int]],
    sensor_idx: Optional[int] = None,
    global_sensor_range: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Apply a sequence of faults to the same channel.

    Parameters
    ----------
    signal         : 1-D sensor signal.
    fault_sequence : List of (fault_type, magnitude, onset) tuples.
                     Applied in order; each builds on the previous output.
    sensor_idx     : Sensor index — passed to each sub-fault for scale lookup.
    global_sensor_range : Global range array — passed to each sub-fault.
    """
    _DISPATCH = {
        "bias"         : apply_bias,
        "scaling"      : apply_scaling,
        "drift"        : apply_drift,
        "intermittent" : apply_intermittent,
    }
    sig = signal.copy()
    for fault_type, magnitude, onset in fault_sequence:
        fn = _DISPATCH.get(fault_type)
        if fn is None:
            raise ValueError(f"Unknown fault type for compound: '{fault_type}'")
        sig = fn(sig, magnitude, onset,
                 sensor_idx=sensor_idx,
                 global_sensor_range=global_sensor_range)
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 – Multi-Sensor Faults
# ─────────────────────────────────────────────────────────────────────────────

def apply_multi_sensor_faults(
    data: NDArray[np.float32],
    fault_specs: List[FaultSpec],
    global_sensor_range: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Apply independent faults to multiple sensor channels simultaneously.

    Parameters
    ----------
    data                : Array of shape (n_timesteps, n_sensors).
    fault_specs         : List of FaultSpec, one per affected sensor.
    global_sensor_range : Global range array — passed to each fault function.

    Returns
    -------
    Corrupted array of the same shape.
    """
    _DISPATCH: Dict[str, Callable] = {
        "bias"         : apply_bias,
        "scaling"      : apply_scaling,
        "drift"        : apply_drift,
        "intermittent" : apply_intermittent,
        "compound"     : None,
    }

    out    = data.copy()
    n_time = data.shape[0]

    for spec in fault_specs:
        idx   = spec.sensor_idx
        onset = int(spec.onset_frac * n_time)
        chan  = out[:, idx].copy()

        if spec.fault_type == "compound":
            seq = spec.params.get("sequence", [("bias", spec.magnitude, onset)])
            out[:, idx] = apply_compound(
                chan, seq,
                sensor_idx=idx,
                global_sensor_range=global_sensor_range,
            )
        else:
            fn = _DISPATCH.get(spec.fault_type)
            if fn is None:
                raise ValueError(f"Unknown fault type: '{spec.fault_type}'")
            out[:, idx] = fn(
                chan, spec.magnitude, onset,
                sensor_idx=idx,
                global_sensor_range=global_sensor_range,
            )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 – System Fault (Physical Degradation)
# ─────────────────────────────────────────────────────────────────────────────

def label_system_fault(
    hs: NDArray,
    rul: Optional[NDArray] = None,
    hs_nominal: int = 1,
    rul_threshold: float = 125.0,
) -> NDArray[np.int8]:
    """
    Derive binary System Fault labels from N-CMAPSS metadata.

    Two-tier labelling strategy:
      1. Primary  : hs != hs_nominal  →  System Fault
      2. Secondary: rul < rul_threshold cycles  →  System Fault
    """
    import warnings
    hs_arr = np.asarray(hs)
    labels = np.zeros(len(hs_arr), dtype=np.int8)
    labels[hs_arr != hs_nominal] = 1

    if rul is not None:
        rul_arr = np.asarray(rul)
        labels[rul_arr < rul_threshold] = 1

    if labels.sum() == 0:
        warnings.warn(
            "label_system_fault: all labels are 0. The input appears to be "
            "pre-filtered to hs==1 rows only. Pass the full multi-unit "
            "DataFrame (cruise_only=False, nominal_only=False).",
            UserWarning, stacklevel=2,
        )
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# FaultEngine – High-Level Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class FaultEngine:
    """
    Orchestrates fault injection for offline dataset augmentation.

    MAGNITUDE GUIDE (Lab Data Version):
    ─────────────────────────────────────
    Magnitudes are fractions of the GLOBAL sensor operating range, which is
    computed from X_nominal across all plateau windows in NB1:

        global_sensor_range = X_nominal.reshape(-1, N_SENSORS).max(axis=0)
                            - X_nominal.reshape(-1, N_SENSORS).min(axis=0)

    With default MAG_LOW=0.15, MAG_HIGH=0.45 and a typical sensor range of
    ~0.25, fault offsets are 0.04–0.11 — clearly visible above the noise
    floor of ~0.0003.

    Usage
    -----
    # Compute global stats in NB1:
    global_sensor_range = (X_nominal.reshape(-1, N_SENSORS).max(axis=0) -
                           X_nominal.reshape(-1, N_SENSORS).min(axis=0))

    # Pass to FaultEngine:
    engine = FaultEngine(seed=42, global_sensor_range=global_sensor_range)
    """

    # ── Magnitude bounds ──────────────────────────────────────────────────────
    MAG_LOW     = 0.15   # 15% of global sensor range  →  offset ~ 0.03–0.05
    MAG_HIGH    = 0.45   # 45% of global sensor range  →  offset ~ 0.09–0.15
    INTERM_LOW  = 0.20   # intermittent spike ~ 0.04–0.07
    INTERM_HIGH = 0.50   # intermittent spike ~ 0.10–0.18

    def __init__(
        self,
        seed: int = 42,
        global_sensor_range: Optional[NDArray[np.float32]] = None,
    ):
        self.rng                 = np.random.default_rng(seed)
        self.global_sensor_range = (
            np.asarray(global_sensor_range, dtype=np.float32)
            if global_sensor_range is not None else None
        )
        log.info(
            "FaultEngine initialized (seed=%d) | "
            "mag_range=[%.2f, %.2f] | interm_range=[%.2f, %.2f] | "
            "global_range provided=%s",
            seed,
            self.MAG_LOW, self.MAG_HIGH,
            self.INTERM_LOW, self.INTERM_HIGH,
            self.global_sensor_range is not None,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _random_sensor(self, n_sensors: int) -> int:
        return int(self.rng.integers(0, n_sensors))

    def _random_magnitude(self, fault_type: str = "bias") -> float:
        if fault_type == "intermittent":
            return float(self.rng.uniform(self.INTERM_LOW, self.INTERM_HIGH))
        return float(self.rng.uniform(self.MAG_LOW, self.MAG_HIGH))

    def _random_onset(self, n_time: int, lo: float = 0.0, hi: float = 0.25) -> int:
        return int(self.rng.uniform(lo, hi) * n_time)

    # ── Case 1 ────────────────────────────────────────────────────────────────

    def inject_single_sensor_single_fault(
        self,
        window: NDArray[np.float32],
        fault_type: Optional[str] = None,
        sensor_idx: Optional[int] = None,
        magnitude: Optional[float] = None,
    ) -> Tuple[NDArray[np.float32], int, int, str, float]:
        """
        Inject one fault type into one sensor channel.

        Returns
        -------
        faulty_window, sensor_idx, fault_label_int, fault_type_str, magnitude
        """
        n_time, n_sens = window.shape
        sensor_idx = sensor_idx if sensor_idx is not None else self._random_sensor(n_sens)
        fault_type = fault_type or str(self.rng.choice(["bias", "scaling", "drift", "intermittent"]))
        if magnitude is None:
            magnitude = self._random_magnitude(fault_type)
        onset = self._random_onset(n_time)

        out  = window.copy()
        chan = out[:, sensor_idx].copy()

        if fault_type == "bias":
            out[:, sensor_idx] = apply_bias(
                chan, magnitude, onset,
                sensor_idx=sensor_idx,
                global_sensor_range=self.global_sensor_range,
            )
        elif fault_type == "scaling":
            out[:, sensor_idx] = apply_scaling(
                chan, magnitude, onset,
                sensor_idx=sensor_idx,
                global_sensor_range=self.global_sensor_range,
            )
        elif fault_type == "drift":
            out[:, sensor_idx] = apply_drift(
                chan, magnitude, onset,
                sensor_idx=sensor_idx,
                global_sensor_range=self.global_sensor_range,
            )
        elif fault_type == "intermittent":
            out[:, sensor_idx] = apply_intermittent(
                chan, magnitude, onset,
                rng_seed=int(self.rng.integers(0, 9999)),
                sensor_idx=sensor_idx,
                global_sensor_range=self.global_sensor_range,
            )
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

        label_int = FAULT_LABEL_TO_INT[fault_type.capitalize()]
        return out, sensor_idx, label_int, fault_type, magnitude

    # ── Case 2 ────────────────────────────────────────────────────────────────

    def inject_compound_fault(
        self,
        window: NDArray[np.float32],
        sensor_idx: Optional[int] = None,
        n_faults: int = 2,
    ) -> Tuple[NDArray[np.float32], int, int, List]:
        """
        Inject a compound (sequential multi-fault) on one sensor.
        """
        n_time, n_sens = window.shape
        sensor_idx = sensor_idx if sensor_idx is not None else self._random_sensor(n_sens)
        types_     = ["bias", "scaling", "drift", "intermittent"]

        chosen = list(self.rng.choice(types_, size=n_faults, replace=False))
        seq    = [
            (t, self._random_magnitude(t), self._random_onset(n_time))
            for t in chosen
        ]

        out = window.copy()
        out[:, sensor_idx] = apply_compound(
            out[:, sensor_idx], seq,
            sensor_idx=sensor_idx,
            global_sensor_range=self.global_sensor_range,
        )

        return out, sensor_idx, FAULT_LABEL_TO_INT["Compound"], seq

    # ── Case 3 ────────────────────────────────────────────────────────────────

    def inject_multi_sensor_faults(
        self,
        window: NDArray[np.float32],
        n_affected: int = 2,
    ) -> Tuple[NDArray[np.float32], List[FaultSpec]]:
        """
        Inject independent faults across multiple sensor channels.
        """
        n_time, n_sens = window.shape
        n_affected = min(n_affected, n_sens)
        sensors_   = self.rng.choice(n_sens, size=n_affected, replace=False)
        types_     = ["bias", "scaling", "drift", "intermittent"]

        specs = []
        for s in sensors_:
            ft = str(self.rng.choice(types_))
            specs.append(FaultSpec(
                sensor_idx  = int(s),
                fault_type  = ft,
                magnitude   = self._random_magnitude(ft),
                onset_frac  = float(self.rng.uniform(0.0, 0.25)),
            ))

        out = apply_multi_sensor_faults(
            window, specs,
            global_sensor_range=self.global_sensor_range,
        )
        return out, specs

    # ── Case 4 ────────────────────────────────────────────────────────────────

    @staticmethod
    def label_system_faults(
        hs: NDArray,
        rul: Optional[NDArray] = None,
    ) -> NDArray[np.int8]:
        """Wrapper for label_system_fault utility."""
        return label_system_fault(hs, rul)

    # ── Batch injection ───────────────────────────────────────────────────────

    def inject_batch(
        self,
        X_nominal: NDArray[np.float32],
        case: int = 1,
        fault_ratio: float = 0.5,
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Inject faults into a random subset of windows.

        Parameters
        ----------
        X_nominal   : (N, T, S) nominal windows.
        case        : Fault case 1–3.
        fault_ratio : Fraction of windows to corrupt.
        """
        N        = len(X_nominal)
        X_out    = X_nominal.copy()
        y_out    = np.zeros(N, dtype=np.int64)
        n_faulty = int(N * fault_ratio)
        indices  = self.rng.choice(N, size=n_faulty, replace=False)

        for i in indices:
            win = X_out[i]
            if case == 1:
                win_f, s_idx, label, _, _ = self.inject_single_sensor_single_fault(win)
                X_out[i] = win_f
                y_out[i] = label
            elif case == 2:
                win_f, s_idx, label, _ = self.inject_compound_fault(win)
                X_out[i] = win_f
                y_out[i] = label
            elif case == 3:
                win_f, specs = self.inject_multi_sensor_faults(win)
                X_out[i] = win_f
                y_out[i] = FAULT_LABEL_TO_INT["MultiSensor"]
            else:
                raise ValueError(f"Case must be 1, 2, or 3.")

        log.info("inject_batch: case=%d | %d/%d windows injected", case, n_faulty, N)
        return X_out, y_out

    # ── Summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def summary() -> str:
        lines = [
            "FaultEngine – Supported Fault Cases  [Lab Data Version]",
            "=" * 60,
            "Case 1 (Single Sensor, Single Fault):",
            "  • Bias          – additive offset        (mag = 15–45% of global range)",
            "  • Scaling       – multiplicative gain    (mag = 15–45%)",
            "  • Drift         – linearly increasing    (mag = 15–45% of global range)",
            "  • Intermittent  – random spikes          (mag = 20–50% of global range)",
            "",
            "Case 2 (Single Sensor, Multi-Fault):",
            "  • Compound      – sequential fault composition",
            "",
            "Case 3 (Multi-Sensor Faults):",
            "  • MultiSensor   – independent faults on ≥2 channels",
            "",
            "Case 4 (System Fault):",
            "  • SystemFault   – real failure CSV data",
            "",
            "NOTE: Pass global_sensor_range=X_nominal.reshape(-1,S).ptp(axis=0)",
            "      to FaultEngine for correct magnitude scaling on plateau data.",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(FaultEngine.summary())
    print()

    rng = np.random.default_rng(0)
    # Simulate lab plateau data: 100 samples, 29 sensors, values in [0.6, 0.9]
    win = (rng.standard_normal((100, 29)).astype(np.float32) * 0.0003 + 0.75)
    win = np.clip(win, 0.0, 1.0)

    # Simulate global sensor range (~0.25 per sensor)
    global_range = np.full(29, 0.25, dtype=np.float32)

    engine = FaultEngine(seed=0, global_sensor_range=global_range)

    w1, si, lbl, ft, mag = engine.inject_single_sensor_single_fault(win, fault_type="bias")
    delta = float(np.abs(w1[:, si] - win[:, si]).mean())
    print(f"Case 1 Bias    | sensor={si} | mag={mag:.4f} | mean_delta={delta:.4f}  (target > 0.02)")

    w2, si, lbl, ft, mag = engine.inject_single_sensor_single_fault(win, fault_type="drift")
    delta = float(np.abs(w2[:, si] - win[:, si]).mean())
    print(f"Case 1 Drift   | sensor={si} | mag={mag:.4f} | mean_delta={delta:.4f}  (target > 0.02)")

    w3, si, lbl, ft, mag = engine.inject_single_sensor_single_fault(win, fault_type="intermittent")
    delta = float(np.abs(w3[:, si] - win[:, si]).mean())
    print(f"Case 1 Interm. | sensor={si} | mag={mag:.4f} | mean_delta={delta:.4f}  (target > 0.01)")

    w4, si2, lbl2, seq = engine.inject_compound_fault(win)
    print(f"Case 2 Compound | sensor={si2} | sequence={[(s[0], round(s[1],3)) for s in seq]}")

    w5, specs = engine.inject_multi_sensor_faults(win)
    print(f"Case 3 Multi    | sensors={[s.sensor_idx for s in specs]}")

    print("\n✓ fault_engine.py self-test passed.")
