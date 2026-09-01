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

REVISION NOTES (Multi-Dataset Version):
  • Fault magnitude ranges have been SIGNIFICANTLY increased for better detectability.
    Previous range: [0.05, 0.25]  →  New default range: [0.40, 0.80]
  • Bias:         magnitude = fraction of signal range (was 5–25%, now 40–80%)
  • Scaling:      gain factor (was 5–25%, now 40–80%)
  • Drift:        total accumulated ramp (was 5–25% of range, now 25–70%)
  • Intermittent: spike amplitude (was 0.05–0.25 × std, now 0.5–2.0 × std)
                  pulse probability raised from 0.15 to 0.25
  • These increases ensure faults are clearly visible after MinMaxScaling
    on real multi-dataset NCMAPSS data and improve AE/ANN training signal.
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
SAMPLING_RATE_HZ = 1          # 1 Hz

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
    magnitude    : Primary magnitude parameter (relative to signal range/std).
    onset_frac   : Fraction of window length where fault begins [0, 1].
    params       : Extra parameters (e.g., second fault for compound).
    """
    sensor_idx  : int
    fault_type  : str           # 'bias' | 'scaling' | 'drift' | 'intermittent' | 'compound'
    magnitude   : float = 0.30  # INCREASED default: relative to signal std/range
    onset_frac  : float = 0.0   # fault starts at onset_frac * window_length
    params      : Dict          = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 – Single Sensor, Single Fault
# ─────────────────────────────────────────────────────────────────────────────

def apply_bias(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
) -> NDArray[np.float32]:
    """
    Additive constant offset.
        x_f[t] = x[t] + b   for t >= onset

    Magnitude expressed as fraction of signal range.
    Increased from ~10% typical to 20–60% for clear detectability.
    """
    sig = signal.copy()
    rng = float(np.ptp(signal)) + 1e-8
    sig[onset:] += float(magnitude) * rng
    return sig


def apply_scaling(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
) -> NDArray[np.float32]:
    """
    Multiplicative gain error.
        x_f[t] = (1 + s) * x[t]   for t >= onset

    Balaban (2009): gain faults model sensor degradation of measurement amplifiers.
    Magnitude now in [0.20, 0.60] range for clear post-scaling visibility.
    """
    sig = signal.copy()
    sig[onset:] *= (1.0 + float(magnitude))
    return sig


def apply_drift(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
) -> NDArray[np.float32]:
    """
    Linearly increasing offset (ramp fault).
        x_f[t] = x[t] + slope * (t - onset)   for t >= onset

    Total accumulated drift by end of window = magnitude * signal_range.
    Magnitude now in [0.25, 0.70] for clear ramp visibility over 50-sample windows.

    NOTE: Using ptp (peak-to-peak range) as scale reference rather than std,
    so the ramp is always visually prominent regardless of signal baseline level.
    """
    sig = signal.copy()
    n   = len(signal)
    n_after = n - onset
    if n_after <= 0:
        return sig
    rng_ = float(np.ptp(signal)) + 1e-8
    # Ramp from 0 → magnitude * signal_range over the post-onset segment
    ramp = np.linspace(0.0, float(magnitude) * rng_, n_after).astype(np.float32)
    sig[onset:] += ramp
    return sig


def apply_intermittent(
    signal: NDArray[np.float32],
    magnitude: float,
    onset: int = 0,
    pulse_prob: float = 0.25,   # INCREASED from 0.15 → more frequent spikes
    rng_seed: Optional[int] = None,
) -> NDArray[np.float32]:
    """
    Random spike pulses.
        x_f[t] = x[t] ± magnitude * std(x)   with probability pulse_prob

    Models intermittent contact failure or loose wiring.
    magnitude now in [0.50, 2.0] × std for spikes clearly above noise floor.
    pulse_prob increased from 0.15 → 0.25 (25% of post-onset samples affected).
    """
    rng_  = np.random.default_rng(rng_seed)
    sig   = signal.copy()
    n     = len(signal)
    std_  = float(np.std(signal)) + 1e-8
    mask  = rng_.random(n - onset) < pulse_prob
    signs = rng_.choice([-1.0, 1.0], size=(n - onset,))
    sig[onset:][mask] += signs[mask] * float(magnitude) * std_
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 – Single Sensor, Multi-Fault (Compound)
# ─────────────────────────────────────────────────────────────────────────────

def apply_compound(
    signal: NDArray[np.float32],
    fault_sequence: List[Tuple[str, float, int]],
) -> NDArray[np.float32]:
    """
    Apply a sequence of faults to the same channel.

    Parameters
    ----------
    signal         : 1-D sensor signal.
    fault_sequence : List of (fault_type, magnitude, onset) tuples.
                     Applied in order; each builds on the previous output.

    Example
    -------
    >>> apply_compound(sig, [("scaling", 0.3, 0), ("bias", 0.25, 25)])
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
        sig = fn(sig, magnitude, onset)
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 – Multi-Sensor Faults
# ─────────────────────────────────────────────────────────────────────────────

def apply_multi_sensor_faults(
    data: NDArray[np.float32],
    fault_specs: List[FaultSpec],
) -> NDArray[np.float32]:
    """
    Apply independent faults to multiple sensor channels simultaneously.

    Parameters
    ----------
    data        : Array of shape (n_timesteps, n_sensors).
    fault_specs : List of FaultSpec, one per affected sensor.

    Returns
    -------
    Corrupted array of the same shape.
    """
    _DISPATCH: Dict[str, Callable] = {
        "bias"         : apply_bias,
        "scaling"      : apply_scaling,
        "drift"        : apply_drift,
        "intermittent" : apply_intermittent,
        "compound"     : None,  # handled separately
    }

    out = data.copy()
    n_time = data.shape[0]

    for spec in fault_specs:
        idx    = spec.sensor_idx
        onset  = int(spec.onset_frac * n_time)
        chan   = out[:, idx].copy()

        if spec.fault_type == "compound":
            seq = spec.params.get("sequence", [("bias", spec.magnitude, onset)])
            out[:, idx] = apply_compound(chan, seq)
        else:
            fn = _DISPATCH.get(spec.fault_type)
            if fn is None:
                raise ValueError(f"Unknown fault type: '{spec.fault_type}'")
            out[:, idx] = fn(chan, spec.magnitude, onset)

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

    ⚠  Requires the FULL multi-unit dataset (not a single nominal unit-cycle).
       If you pass only rows where hs==1 the output will be all zeros.
       Always call with cruise_only=False, nominal_only=False in the loader.

    Two-tier labelling strategy:
      1. Primary  : hs != hs_nominal (i.e. hs == 0)  →  System Fault
      2. Secondary: rul < rul_threshold cycles        →  System Fault

    Parameters
    ----------
    hs            : 1-D array of health-state integers (1=nominal, 0=degraded).
    rul           : 1-D array of Remaining Useful Life (cycles). Optional.
    hs_nominal    : Integer code for the nominal health state (default 1).
    rul_threshold : Cycles below which a window is flagged as system-faulted.

    Returns
    -------
    labels : int8 array  (0 = Nominal, 1 = System Fault)
    """
    import warnings
    hs_arr = np.asarray(hs)
    labels = np.zeros(len(hs_arr), dtype=np.int8)

    # Primary trigger: hs == 0 (degraded health-state class)
    labels[hs_arr != hs_nominal] = 1

    # Secondary trigger: RUL below end-of-life threshold
    if rul is not None:
        rul_arr = np.asarray(rul)
        labels[rul_arr < rul_threshold] = 1

    if labels.sum() == 0:
        warnings.warn(
            "label_system_fault: all labels are 0.  The input appears to be "
            "pre-filtered to hs==1 rows only.  Pass the full multi-unit "
            "DataFrame (cruise_only=False, nominal_only=False) so that "
            "hs==0 degraded units are included.",
            UserWarning, stacklevel=2,
        )

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# FaultEngine – High-Level Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class FaultEngine:
    """
    Orchestrates fault injection for offline dataset augmentation and
    online simulation at 1 Hz.

    MAGNITUDE GUIDE (Multi-Dataset version):
    ─────────────────────────────────────────
    Default random range is now [mag_low=0.20, mag_high=0.60] for Cases 1–3.
    For intermittent faults the magnitude is interpreted as ×std (spikes),
    so the default range [0.50, 2.0] produces clearly visible spikes above
    the noise floor even after MinMaxScaling.

    You can override per injection call:
        engine.inject_single_sensor_single_fault(win, magnitude=0.5)

    Usage
    -----
    engine = FaultEngine(seed=42)
    X_faulty, labels = engine.inject_batch(X_nominal, case=1)
    """

    # ── Magnitude bounds (class-level, easy to tune) ──────────────────────────
    MAG_LOW   = 0.40   # Minimum fault magnitude (fraction of range) for bias/scaling/drift
    MAG_HIGH  = 0.80   # Maximum fault magnitude
    INTERM_LOW  = 0.50  # Min spike amplitude (× std) for intermittent
    INTERM_HIGH = 2.00  # Max spike amplitude (× std) for intermittent

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        log.info(
            "FaultEngine initialized (seed=%d, fs=%d Hz) | "
            "mag_range=[%.2f, %.2f] | interm_range=[%.2f, %.2f]",
            seed, SAMPLING_RATE_HZ,
            self.MAG_LOW, self.MAG_HIGH,
            self.INTERM_LOW, self.INTERM_HIGH,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _random_sensor(self, n_sensors: int) -> int:
        return int(self.rng.integers(0, n_sensors))

    def _random_magnitude(self, fault_type: str = "bias") -> float:
        """Return a magnitude appropriate for the fault type."""
        if fault_type == "intermittent":
            return float(self.rng.uniform(self.INTERM_LOW, self.INTERM_HIGH))
        return float(self.rng.uniform(self.MAG_LOW, self.MAG_HIGH))

    def _random_onset(self, n_time: int, lo: float = 0.1, hi: float = 0.5) -> int:
        return int(self.rng.uniform(lo, hi) * n_time)

    # ── Case 1 ───────────────────────────────────────────────────────────────

    def inject_single_sensor_single_fault(
        self,
        window: NDArray[np.float32],
        fault_type: Optional[str] = None,
        sensor_idx: Optional[int] = None,
        magnitude: Optional[float] = None,
    ) -> Tuple[NDArray[np.float32], int, int, str, float]:
        """
        Inject one fault type into one sensor channel.

        Parameters
        ----------
        magnitude : Override the random magnitude (useful for visualization).
                    Pass None (default) to use the class-level random range.

        Returns
        -------
        faulty_window, sensor_idx, fault_label_int, fault_type_str, magnitude
        """
        n_time, n_sens = window.shape
        sensor_idx = sensor_idx if sensor_idx is not None else self._random_sensor(n_sens)
        fault_type = fault_type or str(self.rng.choice(["bias","scaling","drift","intermittent"]))
        if magnitude is None:
            magnitude = self._random_magnitude(fault_type)
        onset = self._random_onset(n_time)

        out = window.copy()
        chan = out[:, sensor_idx]

        if fault_type == "bias":
            out[:, sensor_idx] = apply_bias(chan, magnitude, onset)
        elif fault_type == "scaling":
            out[:, sensor_idx] = apply_scaling(chan, magnitude, onset)
        elif fault_type == "drift":
            out[:, sensor_idx] = apply_drift(chan, magnitude, onset)
        elif fault_type == "intermittent":
            out[:, sensor_idx] = apply_intermittent(
                chan, magnitude, onset, rng_seed=int(self.rng.integers(0, 9999))
            )
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

        return out, sensor_idx, FAULT_LABEL_TO_INT[fault_type.capitalize()], fault_type, magnitude

    # ── Case 2 ───────────────────────────────────────────────────────────────

    def inject_compound_fault(
        self,
        window: NDArray[np.float32],
        sensor_idx: Optional[int] = None,
        n_faults: int = 2,
    ) -> Tuple[NDArray[np.float32], int, int, List]:
        """
        Inject a compound (sequential multi-fault) on one sensor.
        Each fault in the sequence uses the increased magnitude range.
        """
        n_time, n_sens = window.shape
        sensor_idx = sensor_idx if sensor_idx is not None else self._random_sensor(n_sens)
        types_     = ["bias","scaling","drift","intermittent"]

        chosen = list(self.rng.choice(types_, size=n_faults, replace=False))
        seq    = [
            (t, self._random_magnitude(t), self._random_onset(n_time))
            for t in chosen
        ]

        out = window.copy()
        out[:, sensor_idx] = apply_compound(out[:, sensor_idx], seq)

        return out, sensor_idx, FAULT_LABEL_TO_INT["Compound"], seq

    # ── Case 3 ───────────────────────────────────────────────────────────────

    def inject_multi_sensor_faults(
        self,
        window: NDArray[np.float32],
        n_affected: int = 2,
    ) -> Tuple[NDArray[np.float32], List[FaultSpec]]:
        """
        Inject independent faults across multiple sensor channels.
        Each affected sensor uses an independently drawn magnitude from
        the increased range.
        """
        n_time, n_sens = window.shape
        n_affected = min(n_affected, n_sens)
        sensors_   = self.rng.choice(n_sens, size=n_affected, replace=False)
        types_     = ["bias","scaling","drift","intermittent"]

        specs = []
        for s in sensors_:
            ft = str(self.rng.choice(types_))
            specs.append(FaultSpec(
                sensor_idx  = int(s),
                fault_type  = ft,
                magnitude   = self._random_magnitude(ft),
                onset_frac  = float(self.rng.uniform(0.05, 0.4)),
            ))

        out = apply_multi_sensor_faults(window, specs)
        return out, specs

    # ── Case 4 ───────────────────────────────────────────────────────────────

    @staticmethod
    def label_system_faults(
        hs: NDArray,
        rul: Optional[NDArray] = None,
    ) -> NDArray[np.int8]:
        """Wrapper for label_system_fault utility."""
        return label_system_fault(hs, rul)

    # ── Batch injection ──────────────────────────────────────────────────────

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

        Returns
        -------
        X_out : (N, T, S) mixed nominal + faulty windows.
        y_out : (N,) fault label array (0 = nominal).
        """
        N = len(X_nominal)
        X_out = X_nominal.copy()
        y_out = np.zeros(N, dtype=np.int64)

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
                raise ValueError(f"Case must be 1, 2, or 3. Use label_system_faults() for Case 4.")

        log.info("inject_batch: case=%d | %d/%d windows injected", case, n_faulty, N)
        return X_out, y_out

    # ── Summary ──────────────────────────────────────────────────────────────

    @staticmethod
    def summary() -> str:
        lines = [
            "FaultEngine – Supported Fault Cases  [Multi-Dataset Version]",
            "=" * 60,
            "Case 1 (Single Sensor, Single Fault):",
            "  • Bias          – additive constant offset  (mag = 20–60% of range)",
            "  • Scaling       – multiplicative gain error (mag = 20–60%)",
            "  • Drift         – linearly increasing ramp  (mag = 20–60% of range)",
            "  • Intermittent  – random pulse spikes       (mag = 0.5–2.0 × std, prob=0.25)",
            "",
            "Case 2 (Single Sensor, Multi-Fault):",
            "  • Compound      – sequential fault composition",
            "",
            "Case 3 (Multi-Sensor Faults):",
            "  • MultiSensor   – independent faults on ≥2 channels",
            "",
            "Case 4 (System Fault):",
            "  • SystemFault   – N-CMAPSS physical degradation (hs, RUL)",
            "",
            "NOTE: Magnitudes significantly increased vs. original single-dataset",
            "      version to ensure clear detectability after MinMaxScaling.",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(FaultEngine.summary())
    print()

    rng = np.random.default_rng(0)
    win = rng.standard_normal((50, 14)).astype(np.float32)

    engine = FaultEngine(seed=0)

    w1, si, lbl, ft, mag = engine.inject_single_sensor_single_fault(win, fault_type="drift")
    print(f"Case 1 | sensor={si} | type={ft} | label={lbl} | mag={mag:.4f}")
    print(f"        Original T48[-1]={win[-1, si]:.4f} | Faulty T48[-1]={w1[-1, si]:.4f}")

    w2, si2, lbl2, seq = engine.inject_compound_fault(win)
    print(f"Case 2 | sensor={si2} | label={lbl2} | sequence={[(s[0],round(s[1],3)) for s in seq]}")

    w3, specs = engine.inject_multi_sensor_faults(win, n_affected=3)
    print(f"Case 3 | {len(specs)} sensors affected | mags={[round(s.magnitude,3) for s in specs]}")

    hs_  = np.array([1,1,0,0,1,0], dtype=np.float32)
    sys_ = FaultEngine.label_system_faults(hs_)
    print(f"Case 4 | system labels = {sys_}")

    X_batch = rng.standard_normal((100, 50, 14)).astype(np.float32)
    X_f, y_f = engine.inject_batch(X_batch, case=1, fault_ratio=0.4)
    print(f"Batch  | faulty windows = {(y_f > 0).sum()} / {len(y_f)}")
    print("✓ fault_engine.py smoke tests passed.")
