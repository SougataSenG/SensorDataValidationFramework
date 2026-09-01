"""
lab_data_utils.py
=================
Data loading and steady-state extraction for lab engine CSV data.

Replaces data_utils.py (N-CMAPSS HDF5 loader) for the lab dataset.

Key differences from N-CMAPSS:
  - Input  : CSV files (one per run), already MinMax-scaled [0,1]
  - Sensors: 29 channels named A1..A29 (or custom names if provided)
  - Freq   : 10 Hz  ->  WINDOW_SIZE=100, STRIDE=10
  - Steady : Rolling-std threshold on composite signal (no altitude column)
  - Classes: 7 (no SystemFault — all runs are nominal)
  - Scaler : Data already scaled; we just verify range, no re-fitting needed

Author: Lead AI Research Engineer
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLING_RATE = 10          # Hz
WINDOW_SIZE   = 100         # samples = 10 seconds at 10 Hz
STRIDE        = 10          # samples = 1 second overlap
N_SENSORS     = 29

# Default sensor names (A1..A29). Override with sensor_name_map if you know
# the physical names (T2, T3, NH, NL, P2, etc.)
DEFAULT_SENSOR_COLS = [f'A{i}' for i in range(1, 30)]

# 7-class label map (no SystemFault for lab data)
LAB_FAULT_LABELS: Dict[int, str] = {
    0: 'Nominal',
    1: 'Bias',
    2: 'Scaling',
    3: 'Drift',
    4: 'Intermittent',
    5: 'Compound',
    6: 'MultiSensor',
}

LAB_FAULT_LABEL_TO_INT: Dict[str, int] = {v: k for k, v in LAB_FAULT_LABELS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window helper
# ─────────────────────────────────────────────────────────────────────────────

def collect_windows(
    data       : NDArray[np.float32],
    window_size: int = WINDOW_SIZE,
    stride     : int = STRIDE,
) -> NDArray[np.float32]:
    """
    Stack sliding windows from a 2-D array.

    Parameters
    ----------
    data : (n_timesteps, n_sensors)

    Returns
    -------
    (N_windows, window_size, n_sensors)  float32
    """
    n = len(data)
    if n < window_size:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32)
    starts = range(0, n - window_size + 1, stride)
    return np.stack(
        [data[s: s + window_size].astype(np.float32) for s in starts], axis=0
    )


# ─────────────────────────────────────────────────────────────────────────────
# Steady-state (plateau) detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_plateau_mask(
    df             : pd.DataFrame,
    sensor_cols    : List[str],
    roll_window    : int   = 50,
    std_threshold  : float = 0.02,
    min_plateau_len: int   = 200,
    speed_col      : Optional[str] = None,
    method         : str   = 'pelt',
    pelt_penalty   : float = 5.0,
    edge_trim      : int   = 50,
) -> NDArray[np.bool_]:
    """
    Identify steady-state (plateau) rows in a staircase engine profile.

    method='pelt'  (recommended)
    ----------------------------
    Uses PELT change-point detection to find exact step boundaries.
    Each segment between changepoints is a candidate plateau.
    Segments shorter than min_plateau_len are discarded (transitions).
    The first/last edge_trim samples of each valid segment are trimmed
    to remove transition artefacts that bleed into the plateau body.

    Fixes the two bugs in the original rolling-std approach:
      1. One-sided lookback window created a dead zone after each transition
      2. A single global threshold failed at high-speed steps (more noise)

    method='rolling_std'  (fallback if ruptures not installed)
    -----------------------------------------------------------
    Uses a CENTERED rolling std (center=True in pandas) so the window is
    symmetric — no lookback dead zone. Still requires threshold tuning.

    Parameters
    ----------
    df              : DataFrame with sensor columns (already scaled [0,1])
    sensor_cols     : Sensor column names
    roll_window     : Window size in samples (rolling_std method only)
    std_threshold   : Stability threshold (rolling_std method only)
    min_plateau_len : Minimum contiguous plateau length (samples)
    speed_col       : Speed/RPM column (NH or NL recommended).
                      If None, uses mean across all sensors.
    method          : 'pelt' or 'rolling_std'
    pelt_penalty    : PELT penalty. Lower = more changepoints.
                      5.0 works well for MinMax-scaled engine data.
                      Increase to 20-50 if too many false changepoints.
    edge_trim       : Samples to trim at each segment edge (pelt only).
                      50 = 5 seconds at 10 Hz.
    """
    n = len(df)

    # ── Select reference signal ───────────────────────────────────────────────
    if speed_col is not None and speed_col in df.columns:
        signal = df[speed_col].values.astype(np.float64)
        signal_label = speed_col
    else:
        signal = np.mean(
            [df[c].values.astype(np.float64) for c in sensor_cols], axis=0
        )
        signal_label = 'mean_all_sensors'

    log.info("Plateau detection | method=%s | signal=%s | n=%d",
             method, signal_label, n)

    # ── Method 1: PELT ────────────────────────────────────────────────────────
    if method == 'pelt':
        try:
            import ruptures as rpt
        except ImportError:
            log.warning("ruptures not installed — falling back to rolling_std")
            method = 'rolling_std'

    if method == 'pelt':
        # Downsample by 5x for speed (10 Hz -> 2 Hz for changepoint detection)
        DS = 5
        signal_ds = signal[::DS].reshape(-1, 1)
        algo = rpt.Pelt(
            model='l2',
            min_size=max(10, min_plateau_len // DS)
        ).fit(signal_ds)
        try:
            bkps_ds = algo.predict(pen=pelt_penalty)
        except Exception as e:
            log.warning("PELT failed (%s) — falling back to rolling_std", e)
            method = 'rolling_std'

    if method == 'pelt':
        # Scale breakpoints back to original sample indices
        bkps = [min(b * DS, n) for b in bkps_ds]
        if bkps[-1] != n:
            bkps.append(n)

        # Build segments and discard short ones (transitions)
        segments, prev = [], 0
        for bp in bkps:
            if bp - prev >= min_plateau_len:
                segments.append((prev, bp))
            prev = bp

        log.info("PELT: %d valid segments | penalty=%.1f | edge_trim=%d",
                 len(segments), pelt_penalty, edge_trim)

        # Mark plateau rows, trimming edges to remove transition bleed
        mask = np.zeros(n, dtype=bool)
        for start, end in segments:
            trim_s = min(start + edge_trim, end)
            trim_e = max(end   - edge_trim, trim_s)
            if trim_e - trim_s >= min_plateau_len:
                mask[trim_s:trim_e] = True

        n_plateau = mask.sum()
        log.info("Plateau: %d / %d rows (%.1f%%)",
                 n_plateau, n, 100 * n_plateau / max(n, 1))
        return mask

    # ── Method 2: Centered rolling std (fallback) ─────────────────────────────
    # center=True: symmetric window eliminates the lookback dead zone.
    rolling_std = (
        pd.Series(signal)
        .rolling(window=roll_window, min_periods=roll_window // 2, center=True)
        .std()
        .fillna(1.0)
        .values
    )
    stable_raw = rolling_std < std_threshold

    mask = np.zeros(n, dtype=bool)
    i, cur_start, cur_len = 0, 0, 0
    while i < n:
        if stable_raw[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len >= min_plateau_len:
                mask[cur_start: cur_start + cur_len] = True
            cur_len = 0
        i += 1
    if cur_len >= min_plateau_len:
        mask[cur_start: cur_start + cur_len] = True

    n_plateau = mask.sum()
    log.info("Plateau (rolling_std centered): %d / %d rows (%.1f%%) | thr=%.4f",
             n_plateau, n, 100 * n_plateau / max(n, 1), std_threshold)
    return mask



# ─────────────────────────────────────────────────────────────────────────────
# Lab CSV Loader
# ─────────────────────────────────────────────────────────────────────────────

class LabCSVLoader:
    """
    Load lab engine CSV files and extract plateau windows for SVF training.

    Parameters
    ----------
    data_dir        : Directory containing *.csv files.
    window_size     : Samples per window (default 100 = 10 s at 10 Hz).
    stride          : Window stride in samples (default 10 = 1 s).
    sensor_cols     : List of sensor column names. If None, auto-detected
                      as all non-timestamp columns.
    sensor_name_map : Optional dict mapping A1..A29 -> physical names.
                      Used only for display/reporting, not for computation.
    speed_col       : Optional column to use as steady-state indicator.
                      If None, uses mean rolling-std across all sensors.
    std_threshold   : Rolling-std threshold for plateau detection.
    min_plateau_len : Minimum plateau length in samples.
    time_col        : Name of timestamp column. If None, auto-detected.
    """

    def __init__(
        self,
        data_dir        : str,
        window_size     : int            = WINDOW_SIZE,
        stride          : int            = STRIDE,
        sensor_cols     : Optional[List[str]] = None,
        sensor_name_map : Optional[Dict[str, str]] = None,
        speed_col       : Optional[str]  = None,
        std_threshold   : float          = 0.02,
        min_plateau_len : int            = 200,
        time_col        : Optional[str]  = None,
        method          : str            = 'pelt',
        pelt_penalty    : float          = 5.0,
        edge_trim       : int            = 50,
    ):
        self.data_dir        = Path(data_dir)
        self.window_size     = window_size
        self.stride          = stride
        self._sensor_cols    = sensor_cols
        self.sensor_name_map = sensor_name_map or {}
        self.speed_col       = speed_col
        self.std_threshold   = std_threshold
        self.min_plateau_len = min_plateau_len
        self.time_col        = time_col
        self.method          = method
        self.pelt_penalty    = pelt_penalty
        self.edge_trim       = edge_trim

        self.csv_files = sorted(self.data_dir.glob('*.csv'))
        if not self.csv_files:
            raise FileNotFoundError(f'No CSV files found in {self.data_dir}')

        log.info("LabCSVLoader: found %d file(s)", len(self.csv_files))

        # Auto-calibrate threshold from first file
        self._calibrate_threshold()

    # ── Calibration ──────────────────────────────────────────────────────────

    def _calibrate_threshold(self) -> None:
        """
        Auto-calibrate steady-state threshold from the first CSV file.

        Computes the distribution of rolling-std values and sets threshold
        at the 30th percentile — capturing the flatter portions of the
        staircase while rejecting transitions.
        """
        df = self._load_csv(self.csv_files[0])
        sensor_cols = self._get_sensor_cols(df)

        if self.speed_col and self.speed_col in df.columns:
            signal = df[self.speed_col].values.astype(np.float64)
            roll_std = pd.Series(signal).rolling(50, min_periods=25).std().dropna().values
        else:
            stds = []
            for col in sensor_cols:
                s = df[col].values.astype(np.float64)
                col_std = pd.Series(s).rolling(50, min_periods=25).std().dropna().values
                stds.append(col_std)
            roll_std = np.mean(stds, axis=0)

        # Use 60th percentile — captures plateau noise level as upper bound.
        # p30 would be too tight (below typical plateau std); p60 sits just
        # above the plateau distribution, rejecting only transition spikes.
        auto_thresh = float(np.percentile(roll_std, 60))

        # If user passed an explicit threshold, respect it; otherwise use auto
        if self.std_threshold == 0.02:   # default unchanged
            self.std_threshold = round(auto_thresh, 4)

        log.info(
            "Auto-calibrated plateau threshold: %.4f  "
            "(60th pct of rolling-std distribution)",
            self.std_threshold,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        log.debug("Loaded %s: %d rows, %d cols", path.name, len(df), len(df.columns))
        return df

    def _get_sensor_cols(self, df: pd.DataFrame) -> List[str]:
        """Return sensor columns, auto-detecting if not specified."""
        if self._sensor_cols is not None:
            return [c for c in self._sensor_cols if c in df.columns]
        # Auto-detect: all numeric columns except timestamp
        exclude = set()
        if self.time_col and self.time_col in df.columns:
            exclude.add(self.time_col)
        else:
            # Try to find timestamp column by name
            for candidate in ['timestamp', 'time', 'Time', 'Timestamp', 't']:
                if candidate in df.columns:
                    exclude.add(candidate)
                    break
        cols = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        return cols

    @property
    def sensor_cols(self) -> List[str]:
        """Sensor column names (from first file)."""
        df = self._load_csv(self.csv_files[0])
        return self._get_sensor_cols(df)

    def display_names(self, cols: Optional[List[str]] = None) -> List[str]:
        """Return display names (physical names if sensor_name_map provided)."""
        cols = cols or self.sensor_cols
        return [self.sensor_name_map.get(c, c) for c in cols]

    # ── Main iteration ────────────────────────────────────────────────────────

    def iter_files(
        self,
        plateau_only: bool = True,
    ) -> Generator[Dict, None, None]:
        """
        Yield one dict per CSV file containing plateau windows.

        Yields
        ------
        {
            'file'         : filename,
            'df'           : full DataFrame,
            'df_plateau'   : plateau-only rows,
            'plateau_mask' : boolean mask over full df,
            'sensor_cols'  : list of sensor column names,
            'windows'      : (N, window_size, n_sensors) float32,
        }
        """
        for path in self.csv_files:
            df = self._load_csv(path)
            sensor_cols = self._get_sensor_cols(df)

            plateau_mask = detect_plateau_mask(
                df, sensor_cols,
                speed_col       = self.speed_col,
                std_threshold   = self.std_threshold,
                min_plateau_len = self.min_plateau_len,
                method          = self.method,
                pelt_penalty    = self.pelt_penalty,
                edge_trim       = self.edge_trim,
            )

            if plateau_only:
                df_use = df[plateau_mask].reset_index(drop=True)
            else:
                df_use = df

            sensor_arr = df_use[sensor_cols].values.astype(np.float32)
            windows    = collect_windows(sensor_arr, self.window_size, self.stride)

            log.info(
                "%s: %d plateau rows -> %d windows",
                path.name, plateau_mask.sum(), len(windows),
            )

            yield {
                'file'         : path.name,
                'df'           : df,
                'df_plateau'   : df_use,
                'plateau_mask' : plateau_mask,
                'sensor_cols'  : sensor_cols,
                'windows'      : windows,
            }

            del df, df_use, sensor_arr, windows
            gc.collect()

    # ── Nominal window collection ─────────────────────────────────────────────

    def collect_nominal_windows(self) -> Tuple[NDArray[np.float32], List[str]]:
        """
        Collect all plateau windows across all CSV files.

        Data is already MinMax-scaled [0,1] so no scaler fitting is needed.
        We verify the range and clip any values outside [0,1] from noise.

        Returns
        -------
        X_nominal : (N_total, window_size, n_sensors) float32
        sensor_cols: list of sensor column names
        """
        chunks      = []
        sensor_cols_ = None

        for batch in self.iter_files(plateau_only=True):
            wins = batch['windows']
            if len(wins) == 0:
                log.warning("%s: no plateau windows extracted", batch['file'])
                continue
            chunks.append(wins)
            sensor_cols_ = batch['sensor_cols']

        if not chunks:
            raise RuntimeError(
                "No plateau windows extracted from any file. "
                "Try lowering std_threshold or min_plateau_len."
            )

        X = np.concatenate(chunks, axis=0)

        # Verify scaling — data should be ~[0, 1]
        vmin, vmax = X.min(), X.max()
        log.info("Nominal windows: %s  range=[%.4f, %.4f]", X.shape, vmin, vmax)
        if vmax > 1.1 or vmin < -0.1:
            log.warning(
                "Data range [%.4f, %.4f] is outside [0,1]. "
                "If data is not pre-scaled, fit a MinMaxScaler first.",
                vmin, vmax,
            )

        # Clip to [0,1] to handle any floating-point overflow
        X = np.clip(X, 0.0, 1.0)

        return X, sensor_cols_

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def plateau_summary(self) -> pd.DataFrame:
        """
        Print a summary of plateau detection across all files.
        Useful for verifying the threshold is set correctly.
        """
        rows = []
        for path in self.csv_files:
            df = self._load_csv(path)
            sensor_cols = self._get_sensor_cols(df)
            mask = detect_plateau_mask(
                df, sensor_cols,
                speed_col       = self.speed_col,
                std_threshold   = self.std_threshold,
                min_plateau_len = self.min_plateau_len,
                method          = self.method,
                pelt_penalty    = self.pelt_penalty,
                edge_trim       = self.edge_trim,
            )
            wins = collect_windows(
                df[sensor_cols][mask].values.astype(np.float32),
                self.window_size, self.stride,
            )
            rows.append({
                'file'          : path.name,
                'total_rows'    : len(df),
                'plateau_rows'  : int(mask.sum()),
                'plateau_pct'   : f"{100*mask.sum()/len(df):.1f}%",
                'n_windows'     : len(wins),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run with synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import tempfile, os

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Generate synthetic staircase CSV
    rng = np.random.default_rng(42)
    n_rows   = 20000
    n_sens   = 29
    t        = np.arange(n_rows)

    # Staircase: 5 steps of 4000 rows each
    step_levels = np.repeat([0.2, 0.4, 0.6, 0.4, 0.2], 4000)
    # Add transitions (200 rows linear ramp between steps)
    sensor_data = np.zeros((n_rows, n_sens), dtype=np.float32)
    for s in range(n_sens):
        noise      = rng.normal(0, 0.005, n_rows).astype(np.float32)
        sensor_data[:, s] = (step_levels + noise).clip(0, 1)

    df_test = pd.DataFrame(sensor_data, columns=[f'A{i+1}' for i in range(n_sens)])
    df_test.insert(0, 'timestamp', t / 10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'test_run_01.csv')
        df_test.to_csv(csv_path, index=False)

        loader = LabCSVLoader(
            data_dir=tmpdir,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )
        print(f'Auto-calibrated threshold: {loader.std_threshold}')
        print(f'Sensor cols: {loader.sensor_cols[:5]} ...')

        X_nom, scols = loader.collect_nominal_windows()
        print(f'X_nominal shape: {X_nom.shape}')
        print(f'Value range: [{X_nom.min():.4f}, {X_nom.max():.4f}]')

        summary = loader.plateau_summary()
        print(summary.to_string(index=False))

    print('\n✓ lab_data_utils.py self-test passed.')
