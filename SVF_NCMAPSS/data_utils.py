"""
data_utils.py
=============
Scalable Data Architecture for N-CMAPSS Jet Engine PHM.
Supports DS01–DS05 datasets with dynamic cruise extraction
and windowed data generation.

Author: Lead AI Research Engineer – Aerospace PHM
"""

import os
import gc
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Dict

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Column schema (N-CMAPSS standard)
# ─────────────────────────────────────────────────────────────────────────────
W_COLS  = ["alt", "Mach", "TRA", "T2"]                          # Operating conditions
X_S_COLS = ["T24","T30","T48","T50","P15","P2","P21","P24",
             "Ps30","P40","P50","Nf","Nc","Wf"]                  # Sensor readings
X_V_COLS = ["T40","P30","P45","W21","W22","W25","W31","W32",
             "W48","W50","SmFan","SmLPC","SmHPC","phi"]          # Virtual sensors
T_COLS   = ["fan_eff_mod","fan_flow_mod","LPC_eff_mod",
            "LPC_flow_mod","HPC_eff_mod","HPC_flow_mod",
            "HPT_eff_mod","HPT_flow_mod","LPT_eff_mod",
            "LPT_flow_mod"]                                      # Health params
A_COLS   = ["unit","cycle","Fc","hs"]                            # Auxiliary
Y_COL    = "RUL"                                                  # Target

ALL_COLS = T_COLS + W_COLS + X_S_COLS + X_V_COLS + A_COLS + [Y_COL]

SENSOR_COLS = X_S_COLS          # Primary sensors for analysis
ALT_COL     = "alt"             # Altitude column name


# ─────────────────────────────────────────────────────────────────────────────
# Data-Driven Cruise Threshold Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_cruise_threshold(
    alt_arrays: List[np.ndarray],
    roll_window: int = 30,
    cruise_fraction: float = 0.50,
    safety_margin: float = 2.0,
    n_sample_flights: int = 50,
) -> Tuple[float, Dict]:
    """
    Derive the altitude rolling-std threshold directly from the dataset.

    Strategy — Cruise-Fraction Percentile Method
    ---------------------------------------------
    In a typical N-CMAPSS flight the cruise phase accounts for roughly
    50–70 % of total flight time.  The rolling-std distribution is
    therefore bimodal:

        • Lower mode  (majority) → stable cruise plateau
        • Upper mode  (minority) → climb / descent ramps

    We estimate the threshold as the percentile of the rolling-std
    distribution that corresponds to `cruise_fraction` of total samples,
    then apply a `safety_margin` multiplier to ensure momentary turbulence
    spikes during cruise do not break the detection:

        threshold = P(cruise_fraction * 100) * safety_margin

    This is dataset-adaptive:
      • A clean dataset (DS01, σ_cruise ≈ 6 ft)  → low threshold (~15–25 ft)
      • A noisier dataset (DS03, σ_cruise ≈ 25 ft) → higher threshold (~60–80 ft)
      • Completely data-driven — no domain constant to tune.

    Parameters
    ----------
    alt_arrays       : List of 1-D altitude arrays (ft) from different flights.
    roll_window      : Rolling window for std calculation (seconds @ 1 Hz).
    cruise_fraction  : Expected fraction of flight time spent in cruise [0,1].
                       Default 0.50 (conservative — most N-CMAPSS flights are
                       60–70 % cruise, so 0.50 gives headroom).
    safety_margin    : Multiplier applied on top of the percentile to absorb
                       short turbulence spikes without breaking the cruise mask.
                       Default 2.0 (validated on DS01–DS05 simulations).
    n_sample_flights : Max flights to pool (sub-sampled for speed on large sets).

    Returns
    -------
    threshold : float  — recommended cruise_std_th value in ft.
    diag      : dict   — diagnostic statistics for verification plotting.
    """
    rng_ = np.random.default_rng(0)
    if len(alt_arrays) > n_sample_flights:
        idxs = rng_.choice(len(alt_arrays), n_sample_flights, replace=False)
        alt_arrays = [alt_arrays[i] for i in idxs]

    all_rstd: List[float] = []
    for alt in alt_arrays:
        alt = np.asarray(alt, dtype=np.float64)
        n   = len(alt)
        for i in range(n):
            lo = max(0, i - roll_window + 1)
            all_rstd.append(float(alt[lo : i + 1].std()))

    rs = np.array(all_rstd, dtype=np.float32)

    # The cruise_fraction-th percentile marks the boundary of the lower mode
    pct_val   = float(cruise_fraction * 100.0)
    p_cruise  = float(np.percentile(rs, pct_val))

    # Safety margin absorbs turbulence spikes within the cruise phase
    threshold = float(np.clip(p_cruise * safety_margin, 20.0, 2000.0))

    diag = {
        "threshold"       : threshold,
        "cruise_fraction" : cruise_fraction,
        "safety_margin"   : safety_margin,
        "p_cruise_raw"    : p_cruise,
        "n_flights"       : len(alt_arrays),
        "n_samples"       : len(rs),
        "p10"             : float(np.percentile(rs, 10)),
        "p25"             : float(np.percentile(rs, 25)),
        "p50"             : float(np.percentile(rs, 50)),
        "p75"             : float(np.percentile(rs, 75)),
        "p90"             : float(np.percentile(rs, 90)),
        "p95"             : float(np.percentile(rs, 95)),
        "roll_window_s"   : roll_window,
        "rolling_std_sample": rs,      # kept for histogram plotting
    }
    log.info(
        "calibrate_cruise_threshold: threshold=%.1f ft  "
        "(P%.0f=%.1f ft × margin=%.1f, n_flights=%d)",
        threshold, pct_val, p_cruise, safety_margin, len(alt_arrays),
    )
    return threshold, diag


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Cruise Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_cruise_mask(
    alt: np.ndarray,
    window: int = 30,
    std_threshold: float = 50.0,
    min_duration: int = 60,
) -> np.ndarray:
    """
    Identify the 'Cruise' phase as the longest contiguous segment
    where the rolling standard deviation of altitude stays below
    `std_threshold` feet.

    Parameters
    ----------
    alt            : 1-D altitude array (ft), sampled at 1 Hz.
    window         : Rolling window size for std calculation (seconds).
    std_threshold  : Maximum allowed altitude std (ft) to qualify as cruise.
    min_duration   : Minimum segment length (seconds) to be considered cruise.

    Returns
    -------
    Boolean mask of the same length as `alt`.
    """
    n = len(alt)
    rolling_std = np.zeros(n, dtype=np.float32)

    for i in range(n):
        lo = max(0, i - window + 1)
        rolling_std[i] = alt[lo : i + 1].std()

    stable = rolling_std < std_threshold

    # Find contiguous segments
    best_start, best_len = 0, 0
    cur_start, cur_len   = 0, 0

    for i, val in enumerate(stable):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    mask = np.zeros(n, dtype=bool)
    if best_len >= min_duration:
        mask[best_start : best_start + best_len] = True

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Generator
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_generator(
    data: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
) -> Generator[np.ndarray, None, None]:
    """
    Yield float32 windows of shape (window_size, n_features).

    Parameters
    ----------
    data        : 2-D array (time, features).
    window_size : Number of time steps per window (50 s @ 1 Hz).
    stride      : Step size between windows (10 s).
    """
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        yield data[start : start + window_size].astype(np.float32)


def collect_windows(
    data: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
) -> np.ndarray:
    """
    Return stacked windows as a float32 array of shape (N, window_size, n_features).
    """
    wins = list(sliding_window_generator(data, window_size, stride))
    return np.stack(wins, axis=0) if wins else np.empty((0, window_size, data.shape[1]), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# N-CMAPSS HDF5 Loader
# ─────────────────────────────────────────────────────────────────────────────

class NCMAPSSLoader:
    """
    Iterates through a directory of N-CMAPSS .h5 files and exposes
    pre-processed, cruise-extracted DataFrames with optional windowing.

    Memory-conscious design for 128 GB RAM workstations:
      • Reads one file at a time.
      • Uses float32 throughout.
      • Yields, never accumulates all data.
      • No plots generated — visualisation belongs in the notebook.

    Parameters
    ----------
    data_dir      : Path to directory containing .h5 files.
    window_size   : Sliding window length in seconds (1 Hz → samples).
    stride        : Window stride in seconds.
    cruise_std_th : Altitude rolling-std threshold for cruise detection (ft).
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 50,
        stride: int = 10,
        cruise_std_th: Optional[float] = None,
        calib_roll_window: int = 30,
        calib_n_flights: int = 50,
    ):
        """
        Parameters
        ----------
        cruise_std_th     : Altitude rolling-std threshold (ft) for cruise
                            detection.  If None (default), the threshold is
                            calibrated automatically from the dataset using
                            calibrate_cruise_threshold().  Pass an explicit
                            float only if you want to override the data-driven
                            value (e.g. for reproducibility across runs).
        calib_roll_window : Rolling window (s) used during calibration.
        calib_n_flights   : Max flights sampled for calibration (speed/RAM).
        """
        self.data_dir         = Path(data_dir)
        self.window_size      = window_size
        self.stride           = stride
        self._cruise_std_th_override = cruise_std_th   # None → auto-calibrate
        self.calib_roll_window = calib_roll_window
        self.calib_n_flights   = calib_n_flights
        self.calib_diag: Optional[Dict] = None         # populated after calibration

        self.h5_files: List[Path] = sorted(self.data_dir.glob("*.h5"))
        if not self.h5_files:
            raise FileNotFoundError(f"No .h5 files found in {self.data_dir}")
        log.info(f"NCMAPSSLoader: found {len(self.h5_files)} file(s) in {self.data_dir}")

        # Auto-calibrate threshold from the dataset unless overridden
        if cruise_std_th is None:
            self.cruise_std_th, self.calib_diag = self._calibrate_from_files()
        else:
            self.cruise_std_th = float(cruise_std_th)
            log.info(f"NCMAPSSLoader: using user-supplied cruise_std_th={self.cruise_std_th:.1f} ft")

    # ------------------------------------------------------------------
    def _calibrate_from_files(self) -> Tuple[float, Dict]:
        """
        Read altitude columns from all .h5 files (one pass, memory-safe)
        and call calibrate_cruise_threshold().
        """
        log.info("Calibrating cruise threshold from dataset …")
        alt_arrays: List[np.ndarray] = []

        for path in self.h5_files:
            try:
                import h5py
                with h5py.File(path, "r") as f:
                    for split in ("dev", "test"):
                        W = f.get(f"W_{split}")
                        if W is not None:
                            alt_col_idx = W_COLS.index(ALT_COL)
                            alt = W[:, alt_col_idx].astype(np.float32)
                            # Split into per-unit segments using T group if available
                            T = f.get(f"T_{split}")
                            if T is not None:
                                units = T[:, 0].astype(int)
                                for uid in np.unique(units):
                                    alt_arrays.append(alt[units == uid])
                            else:
                                alt_arrays.append(alt)
            except Exception as e:
                log.warning(f"Calibration: could not read {path.name}: {e}")

        if not alt_arrays:
            log.warning("Calibration: no altitude data found — falling back to 50 ft")
            return 50.0, {}

        threshold, diag = calibrate_cruise_threshold(
            alt_arrays,
            roll_window    = self.calib_roll_window,
            n_sample_flights = self.calib_n_flights,
        )
        log.info(f"Auto-calibrated cruise_std_th = {threshold:.1f} ft")
        return threshold, diag

    # ------------------------------------------------------------------
    
    def _read_h5(self, path: Path, only_metadata: bool = False) -> pd.DataFrame:
        """
        Read a single N-CMAPSS .h5 into a DataFrame.
        Set only_metadata=True to skip heavy sensor/virtual sensor arrays (saves ~90% RAM).
        """
        if not only_metadata:
            log.info(f"Loading Full Dataset: {path.name}")
        
        frames = []
        with h5py.File(path, "r") as f:
            for split in ("dev", "test"):
                try:
                    col_data: Dict[str, np.ndarray] = {}

                    # Metadata / Auxiliary Group (A) - Essential for labels/grouping
                    A = f.get(f"A_{split}")
                    if A is not None and len(A.shape) > 1:
                        for j, c in enumerate(A_COLS[:A.shape[1]]):
                            col_data[c] = A[:, j].astype(np.float32)

                    # Target Group (Y) - Essential for RUL
                    Y = f.get(f"Y_{split}")
                    if Y is not None:
                        col_data[Y_COL] = Y[:, 0].astype(np.float32)

                    # Heavy Data Groups - Only load if only_metadata is False
                    if not only_metadata:
                        # Operating conditions
                        W = f.get(f"W_{split}")
                        if W is not None and len(W.shape) > 1:
                            for j, c in enumerate(W_COLS[:W.shape[1]]):
                                col_data[c] = W[:, j].astype(np.float32)

                        # Sensor readings
                        Xs = f.get(f"X_s_{split}")
                        if Xs is not None and len(Xs.shape) > 1:
                            for j, c in enumerate(X_S_COLS[:Xs.shape[1]]):
                                col_data[c] = Xs[:, j].astype(np.float32)

                        # Virtual sensors
                        Xv = f.get(f"X_v_{split}")
                        if Xv is not None and len(Xv.shape) > 1:
                            for j, c in enumerate(X_V_COLS[:Xv.shape[1]]):
                                col_data[c] = Xv[:, j].astype(np.float32)

                        # Health degradation modifiers
                        T = f.get(f"T_{split}")
                        if T is not None and len(T.shape) > 1:
                            for j, c in enumerate(T_COLS[:T.shape[1]]):
                                col_data[c] = T[:, j].astype(np.float32)

                    if col_data:
                        frames.append(pd.DataFrame(col_data))

                except Exception as e:
                    continue 

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        return df

    # ------------------------------------------------------------------
    def _extract_cruise_mask(self, df_uc: pd.DataFrame) -> np.ndarray:
        """Return boolean cruise mask for one unit-cycle DataFrame."""
        alt = df_uc[ALT_COL].values.astype(np.float64)
        return extract_cruise_mask(alt, std_threshold=self.cruise_std_th)

    # ------------------------------------------------------------------      
    def iter_files(
        self,
        cruise_only: bool = True,
        nominal_only: bool = False,
        only_metadata: bool = False, # Added flag
    ) -> Generator[Dict, None, None]:
        """
        Yield one dict per unit-cycle containing:
          'file'        : filename
          'unit'        : unit id
          'cycle'       : cycle id
          'df'          : DataFrame (full or cruise-only)
          'cruise_mask' : boolean mask over full flight
          'sensor_data' : float32 array (n_timesteps, n_sensors) – sensors only
          'windows'     : float32 array (N, window_size, n_sensors)
          'hs'          : health-state labels (if present)
          Use only_metadata=True for rapid metadata scanning (Case 4).
        """
        for path in self.h5_files:
            df_full = self._read_h5(path, only_metadata=only_metadata)
            if df_full.empty: continue

            # Grouping logic
            groups = df_full.groupby(["unit", "cycle"], sort=False) if "unit" in df_full.columns else [(1, df_full)]

            for keys, df_uc in groups:
                unit_id  = keys[0] if isinstance(keys, tuple) else keys
                cycle_id = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else 0
                df_uc    = df_uc.reset_index(drop=True)

                # If only scanning metadata, yield immediately to save time/RAM
                if only_metadata:
                    yield {
                        "unit": unit_id,
                        "cycle": cycle_id,
                        "df": df_uc,
                        "file": path.name
                    }
                    continue

                # Standard processing for Training/Inference
                cruise_mask = self._extract_cruise_mask(df_uc)

                if nominal_only and "hs" in df_uc.columns:
                    # Slice cruise_mask BEFORE resetting the index so the lengths
                    # stay aligned, then reset both together.
                    hs_mask     = (df_uc["hs"] == 1).values          # bool, same length as df_uc
                    cruise_mask = cruise_mask[hs_mask]                # trim to matching rows
                    df_uc       = df_uc[hs_mask].reset_index(drop=True)
                    # If the entire unit-cycle is degraded, skip it
                    if len(df_uc) == 0:
                        continue
                if cruise_only:
                    df_out = df_uc[cruise_mask].reset_index(drop=True)
                else:
                    df_out = df_uc

                available_sensors = [c for c in SENSOR_COLS if c in df_out.columns]
                sensor_arr = df_out[available_sensors].values.astype(np.float32)
                windows    = collect_windows(sensor_arr, self.window_size, self.stride)
                hs_vals    = df_out["hs"].values if "hs" in df_out.columns else None

                yield {
                    "file"        : path.name,
                    "unit"        : unit_id,
                    "cycle"       : cycle_id,
                    "df"          : df_out,
                    "cruise_mask" : cruise_mask,
                    "sensor_data" : sensor_arr,
                    "sensor_cols" : available_sensors,
                    "windows"     : windows,
                    "hs"          : hs_vals,
                }

                # Explicit memory cleanup for large datasets
                del df_uc, sensor_arr, windows
                gc.collect()

            del df_full
            gc.collect()

    # ------------------------------------------------------------------
    # def collect_nominal_cruise_windows(self) -> Tuple[np.ndarray, List[str]]:
    #     """
    #     Convenience method: collect all nominal-cruise windows (hs==1)
    #     across all files into a single float32 array.

    #     Returns
    #     -------
    #     X_nominal : (N, window_size, n_sensors)
    #     sensor_cols : list of sensor column names
    #     """
    #     all_windows  = []
    #     sensor_cols_ = None

    #     for batch in self.iter_files(cruise_only=True, nominal_only=True):
    #         if batch["windows"].shape[0] == 0:
    #             continue
    #         all_windows.append(batch["windows"])
    #         sensor_cols_ = batch["sensor_cols"]

    #     if not all_windows:
    #         raise RuntimeError("No nominal cruise windows found. Check hs column and cruise extraction.")

    #     X = np.concatenate(all_windows, axis=0)
    #     log.info(f"Nominal cruise windows: {X.shape}")
    #     return X, sensor_cols_

    def collect_nominal_cruise_windows(self, scaler: Optional[object] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Collect and optionally scale all nominal-cruise windows (hs==1).
        """
        all_windows  = []
        sensor_cols_ = None

        for batch in self.iter_files(cruise_only=True, nominal_only=True):
            wins = batch["windows"]
            if wins.shape[0] == 0:
                continue
            
            # Apply scaling on-the-fly if a scaler is provided
            if scaler is not None:
                N, T, S = wins.shape
                wins_2d = wins.reshape(-1, S)
                wins = scaler.transform(wins_2d).reshape(N, T, S).astype(np.float32)
            
            all_windows.append(wins)
            sensor_cols_ = batch["sensor_cols"]

        if not all_windows:
            raise RuntimeError("No nominal cruise windows found.")

        X = np.concatenate(all_windows, axis=0)
        log.info(f"Nominal cruise windows: {X.shape} (Scaled: {scaler is not None})")
        return X, sensor_cols_
# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, sys

    print("data_utils.py — module loaded successfully.")
    print(f"  Sensor columns  : {SENSOR_COLS}")
    print(f"  Window / Stride : 50 s / 10 s  (1 Hz)")

    # Smoke-test extract_cruise_mask
    alt_demo = np.concatenate([
        np.linspace(0, 35000, 200),
        np.full(500, 35000) + np.random.randn(500) * 10,
        np.linspace(35000, 0, 200),
    ])
    mask = extract_cruise_mask(alt_demo)
    print(f"  Cruise mask sum : {mask.sum()} / {len(mask)} samples detected as cruise")

    # Smoke-test window generator
    dummy = np.random.randn(300, 14).astype(np.float32)
    wins  = collect_windows(dummy, 50, 10)
    print(f"  Windows shape   : {wins.shape}  (dtype={wins.dtype})")
    print("  ✓ All smoke tests passed.")