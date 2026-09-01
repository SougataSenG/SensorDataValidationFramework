"""
feature_engineering.py
=======================
Feature Engineering & Vector Space Clustering for Jet Engine PHM.

Pipeline:
  1. Per-window feature extraction  → Mean, Std, Skew, Kurtosis, FFT top-3
  2. Autoencoder reconstruction error as residual feature
  3. UMAP projection (PCA fallback)
  4. Nominal cluster center + severity/confidence scoring

Author: Lead AI Research Engineer – Aerospace PHM
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Statistical + FFT Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

N_FFT_PEAKS = 3     # Number of low-frequency FFT peaks to include


def extract_window_features(
    window: NDArray[np.float32],
    fft_peaks: int = N_FFT_PEAKS,
) -> NDArray[np.float32]:
    """
    Extract statistical and spectral features from a single sensor window.

    Input shape  : (T, S)  –  T time steps, S sensors
    Output shape : (S * (4 + fft_peaks),)

    Features per sensor:
      • Mean, Std Dev, Skewness, Kurtosis         (4)
      • FFT top-`fft_peaks` low-freq magnitudes   (fft_peaks)
    """
    T, S = window.shape
    feats = []

    for s in range(S):
        sig = window[:, s].astype(np.float64)

        # Statistical moments
        mean_  = float(np.mean(sig))
        std_   = float(np.std(sig) + 1e-10)
        skew_  = float(skew(sig))
        kurt_  = float(kurtosis(sig))

        # FFT – keep top `fft_peaks` low-frequency magnitudes
        fft_mag = np.abs(np.fft.rfft(sig - mean_))   # DC-removed
        # Low-freq: skip DC bin [0], take bins 1..fft_peaks
        n_bins  = len(fft_mag)
        top_k   = fft_mag[1 : min(1 + fft_peaks, n_bins)]
        # Pad if signal is very short
        if len(top_k) < fft_peaks:
            top_k = np.pad(top_k, (0, fft_peaks - len(top_k)))

        feats.extend([mean_, std_, skew_, kurt_] + top_k.tolist())

    return np.array(feats, dtype=np.float32)


def extract_batch_features(
    windows: NDArray[np.float32],
    fft_peaks: int = N_FFT_PEAKS,
) -> NDArray[np.float32]:
    """
    Extract features for a batch of windows.

    Parameters
    ----------
    windows  : (N, T, S)
    Returns  : (N, n_features)
    """
    return np.stack(
        [extract_window_features(w, fft_peaks) for w in windows],
        axis=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Autoencoder Reconstruction Error (Residual Feature)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_error(
    X_orig: NDArray[np.float32],
    X_recon: NDArray[np.float32],
    per_sensor: bool = True,
) -> NDArray[np.float32]:
    """
    Compute |X - X̂| residual features.

    Parameters
    ----------
    X_orig    : (N, T, S) original windows.
    X_recon   : (N, T, S) autoencoder reconstructions.
    per_sensor: If True, return per-sensor MAE (N, S);
                else return scalar MAE per window (N,).

    Returns
    -------
    Residuals : (N, S) or (N,).
    """
    diff = np.abs(X_orig - X_recon)                 # (N, T, S)
    if per_sensor:
        return diff.mean(axis=1).astype(np.float32) # (N, S)
    return diff.mean(axis=(1, 2)).astype(np.float32) # (N,)


def augment_features_with_residuals(
    features: NDArray[np.float32],
    residuals: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Horizontally concatenate statistical features + reconstruction residuals.
    """
    return np.concatenate([features, residuals], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sensor-Agnostic Manifold (UMAP with PCA fallback)
# ─────────────────────────────────────────────────────────────────────────────

# class ManifoldProjector:
#     """
#     Projects high-dimensional feature vectors into a low-dimensional manifold
#     for cluster-based anomaly scoring.

#     Primary  : UMAP (n_components = 2 or 3)
#     Fallback : PCA  (if umap-learn is not installed)
#     """

#     def __init__(
#         self,
#         n_components: int = 2,
#         n_neighbors: int = 15,
#         min_dist: float = 0.1,
#         random_state: int = 42,
#     ):
#         self.n_components  = n_components
#         self.n_neighbors   = n_neighbors
#         self.min_dist      = min_dist
#         self.random_state  = random_state
#         self._reducer      = None
#         self._scaler       = StandardScaler()
#         self._method       = None

#     def fit(self, X: NDArray[np.float32]) -> "ManifoldProjector":
#         """Fit scaler + UMAP (or PCA) on nominal feature vectors."""
#         Xs = self._scaler.fit_transform(X)

#         try:
#             import umap as umap_lib                             # type: ignore
#             self._reducer = umap_lib.UMAP(
#                 n_components  = self.n_components,
#                 n_neighbors   = self.n_neighbors,
#                 min_dist      = self.min_dist,
#                 random_state  = self.random_state,
#                 low_memory    = True,
#             )
#             self._reducer.fit(Xs)
#             self._method = "UMAP"
#             log.info("ManifoldProjector: fitted UMAP (n_comp=%d)", self.n_components)

#         except ImportError:
#             log.warning("umap-learn not installed → falling back to PCA")
#             self._reducer = PCA(
#                 n_components  = self.n_components,
#                 random_state  = self.random_state,
#             )
#             self._reducer.fit(Xs)
#             self._method = "PCA"
#             log.info("ManifoldProjector: fitted PCA (n_comp=%d, var=%.2f%%)",
#                      self.n_components, self._reducer.explained_variance_ratio_.sum() * 100)

#         return self

#     def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
#         """Project features to manifold space."""
#         if self._reducer is None:
#             raise RuntimeError("Call .fit() before .transform()")
#         Xs = self._scaler.transform(X)
#         return self._reducer.transform(Xs).astype(np.float32)

#     def fit_transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
#         self.fit(X)
#         return self.transform(X)

#     @property
#     def method(self) -> Optional[str]:
#         return self._method
class ManifoldProjector:
    """
    Projects high-dimensional feature vectors into a low-dimensional manifold.
    Primary  : UMAP (n_components = 2 or 3)
    Fallback : PCA  (if forced or if umap-learn is not installed)
    """

    def __init__(
        self,
        method: str = "UMAP",
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        # The 'method' parameter is now explicitly defined to prevent TypeErrors
        self.method       = method.upper() 
        self.n_components  = n_components
        self.n_neighbors   = n_neighbors
        self.min_dist      = min_dist
        self.random_state  = random_state
        self._reducer      = None
        self._scaler       = StandardScaler()

    def fit(self, X: NDArray[np.float32]) -> "ManifoldProjector":
        """Fit scaler + UMAP (or PCA) on nominal feature vectors."""
        Xs = self._scaler.fit_transform(X)

        # Force PCA if requested
        if self.method == "PCA":
            self._fit_pca(Xs)
        else:
            try:
                import umap as umap_lib
                self._reducer = umap_lib.UMAP(
                    n_components  = self.n_components,
                    n_neighbors   = self.n_neighbors,
                    min_dist      = self.min_dist,
                    random_state  = self.random_state,
                    low_memory    = True,
                )
                self._reducer.fit(Xs)
                log.info("ManifoldProjector: fitted UMAP (n_comp=%d)", self.n_components)
            except (ImportError, Exception) as e:
                log.warning(f"UMAP failed ({e}) → falling back to PCA")
                self._fit_pca(Xs)

        return self

    def _fit_pca(self, Xs):
        """Internal helper to fit PCA fallback."""
        self._reducer = PCA(n_components=self.n_components, random_state=self.random_state)
        self._reducer.fit(Xs)
        self.method = "PCA"
        log.info("ManifoldProjector: fitted PCA (n_comp=%d)", self.n_components)

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project features to manifold space."""
        if self._reducer is None:
            raise RuntimeError("Call .fit() before .transform()")
        Xs = self._scaler.transform(X)
        return self._reducer.transform(Xs).astype(np.float32)

    def fit_transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        self.fit(X)
        return self.transform(X)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Nominal Cluster Center & Severity / Confidence Scoring
# ─────────────────────────────────────────────────────────────────────────────

class NominalCluster:
    """
    Fits a Gaussian "nominal cluster" in manifold space.
    Uses Mahalanobis distance to score severity and confidence.

    Parameters
    ----------
    percentile : Percentile of nominal distances used as the
                 "healthy boundary" radius (default 95th).
    """

    def __init__(self, percentile: float = 95.0):
        self.percentile   = percentile
        self.center_      : Optional[NDArray] = None
        self.cov_inv_     : Optional[NDArray] = None
        self.radius_95_   : float = 0.0

    def fit(self, Z_nominal: NDArray[np.float32]) -> "NominalCluster":
        """
        Fit cluster center and covariance from nominal manifold embeddings.

        Parameters
        ----------
        Z_nominal : (N, n_components) nominal embeddings.
        """
        self.center_  = Z_nominal.mean(axis=0)                  # (n_comp,)
        cov           = np.cov(Z_nominal.T)                     # (n_comp, n_comp)

        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Regularise covariance
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            self.cov_inv_ = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.cov_inv_ = np.linalg.pinv(cov)

        dists = self._mahalanobis_batch(Z_nominal)
        self.radius_95_ = float(np.percentile(dists, self.percentile))
        log.info("NominalCluster fitted | center=%s | R_95=%.4f", self.center_, self.radius_95_)
        return self

    def _mahalanobis_batch(self, Z: NDArray) -> NDArray[np.float32]:
        """Compute Mahalanobis distance for each row in Z."""
        diff = Z - self.center_                                  # (N, d)
        dist = np.sqrt(
            np.einsum("ni,ij,nj->n", diff, self.cov_inv_, diff)
        ).astype(np.float32)
        return dist

    def score(
        self,
        Z: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Compute Severity (%) and Confidence for each embedding.

        Severity  = min(100, 100 * d / radius_95)
        Confidence = sigmoid(2 * (d / radius_95 - 1))

        Returns
        -------
        severity   : (N,)  percentage [0, 100]
        confidence : (N,)  probability [0, 1]
        """
        if self.center_ is None:
            raise RuntimeError("Call .fit() first")

        dist = self._mahalanobis_batch(Z)
        r    = self.radius_95_ + 1e-8

        severity   = np.clip(100.0 * dist / r, 0.0, 100.0).astype(np.float32)
        confidence = (1.0 / (1.0 + np.exp(-2.0 * (dist / r - 1.0)))).astype(np.float32)

        return severity, confidence

    def predict_anomaly(
        self,
        Z: NDArray[np.float32],
        severity_threshold: float = 50.0,
    ) -> NDArray[np.bool_]:
        """Return boolean anomaly flags (severity > threshold)."""
        severity, _ = self.score(Z)
        return severity > severity_threshold


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Feature Pipeline (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

# class FeaturePipeline:
#     """
#     End-to-end feature engineering pipeline.

#     Steps:
#       1. Extract statistical + FFT features from windows.
#       2. (Optional) Append autoencoder residuals.
#       3. Project to manifold (UMAP / PCA).
#       4. Score against nominal cluster.
#     """

#     def __init__(
#         self,
#         n_components: int = 2,
#         n_fft_peaks: int = N_FFT_PEAKS,
#         umap_neighbors: int = 15,
#         cluster_percentile: float = 95.0,
#     ):
#         self.n_fft_peaks  = n_fft_peaks
#         self.projector    = ManifoldProjector(
#             n_components = n_components,
#             n_neighbors  = umap_neighbors,
#         )
#         self.cluster      = NominalCluster(percentile=cluster_percentile)
#         self._fitted      = False

#     def fit(
#         self,
#         X_nominal: NDArray[np.float32],
#         X_recon_nominal: Optional[NDArray[np.float32]] = None,
#     ) -> "FeaturePipeline":
#         """
#         Fit the full pipeline on nominal (healthy) windows.

#         Parameters
#         ----------
#         X_nominal        : (N, T, S) nominal windows.
#         X_recon_nominal  : (N, T, S) autoencoder reconstructions (optional).
#         """
#         F = extract_batch_features(X_nominal, self.n_fft_peaks)

#         if X_recon_nominal is not None:
#             res = compute_reconstruction_error(X_nominal, X_recon_nominal)
#             F   = augment_features_with_residuals(F, res)

#         Z = self.projector.fit_transform(F)
#         self.cluster.fit(Z)
#         self._fitted = True
#         log.info("FeaturePipeline fitted | feature_dim=%d | manifold_dim=%d",
#                  F.shape[1], Z.shape[1])
#         return self

#     def transform(
#         self,
#         X: NDArray[np.float32],
#         X_recon: Optional[NDArray[np.float32]] = None,
#     ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
#         """
#         Transform new windows and return manifold embeddings + scores.

#         Returns
#         -------
#         F          : (N, n_features)   raw feature vectors
#         Z          : (N, n_components) manifold embeddings
#         severity   : (N,)              severity percentage
#         confidence : (N,)              confidence score
#         """
#         if not self._fitted:
#             raise RuntimeError("Call .fit() before .transform()")

#         F = extract_batch_features(X, self.n_fft_peaks)

#         if X_recon is not None:
#             res = compute_reconstruction_error(X, X_recon)
#             F   = augment_features_with_residuals(F, res)

#         Z          = self.projector.transform(F)
#         sev, conf  = self.cluster.score(Z)
#         return F, Z, sev, conf

#     @property
#     def n_features(self) -> int:
#         return (4 + self.n_fft_peaks)   # per-sensor; multiply by n_sensors for total

class FeaturePipeline:
    """
    High-Performance Vectorized Feature Pipeline for N-CMAPSS.
    Optimized for 128GB RAM and Multi-Dataset Global Scaling.
    """
    def __init__(self, n_components=2, n_fft_peaks=3, cluster_percentile=95):
        self.projector = ManifoldProjector(method="UMAP", n_components=n_components)
        self.cluster = NominalCluster(percentile=cluster_percentile)
        self.n_fft_peaks = n_fft_peaks

    def _extract_vectorized_features(self, windows):
        """
        Replaces the loop-based 'extract_batch_features'.
        Processes all windows and all sensors in parallel using NumPy C-kernels.
        """
        N, T, S = windows.shape
        
        # 1. Statistical Moments (Axis 1 = Time)
        means = np.mean(windows, axis=1)         # (N, S)
        stds  = np.std(windows, axis=1) + 1e-10  # (N, S)
        
        # High-speed Skew/Kurtosis calculation
        diff = windows - means[:, np.newaxis, :]
        skews = np.mean(diff**3, axis=1) / (stds**3)
        kurts = np.mean(diff**4, axis=1) / (stds**4)
        
        # 2. Vectorized Real-FFT (Frequency Domain)
        # Remove DC offset (mean) before FFT for cleaner peaks
        data_f = windows - means[:, np.newaxis, :]
        fft_vals = np.abs(np.fft.rfft(data_f, axis=1)) 
        
        # Extract the top K low-frequency peaks (ignoring the 0th DC bin)
        # Shape: (N, n_fft_peaks, S)
        top_fft = fft_vals[:, 1 : 1 + self.n_fft_peaks, :]
        
        # Reshape FFT peaks to (N, S * n_fft_peaks)
        top_fft_flat = top_fft.transpose(0, 2, 1).reshape(N, -1)
        
        # 3. Concatenate all features: [Mean, Std, Skew, Kurt] + [FFT Peaks]
        # Stats Shape: (N, S * 4)
        stats = np.stack([means, stds, skews, kurts], axis=2).reshape(N, -1)
        
        return np.concatenate([stats, top_fft_flat], axis=1).astype(np.float32)

    def fit(self, X_nominal, X_recon_nominal):
        """Fits the manifold and the healthy cluster baseline."""
        # 1. Vectorized extraction
        F_nom = self._extract_vectorized_features(X_nominal)
        
        # 2. Residual features (Delta between Raw and Autoencoder Reconstruction)
        F_res = self._extract_vectorized_features(X_nominal - X_recon_nominal)
        
        # 3. Combine and Project
        F_combined = np.concatenate([F_nom, F_res], axis=1)
        Z_nom = self.projector.fit_transform(F_combined)
        
        # 4. Define the 'Healthy' boundary
        self.cluster.fit(Z_nom)
        return self

    def transform(self, X, X_recon):
        """
        Transforms a batch of data into severity scores and 2D coordinates.
        Uses chunk-friendly logic for 7.6M rows.
        """
        # 1. Fast Batch Extraction
        F_raw = self._extract_vectorized_features(X)
        F_res = self._extract_vectorized_features(X - X_recon)
        F_all = np.concatenate([F_raw, F_res], axis=1)
        
        # 2. Manifold Projection (PCA or UMAP)
        Z = self.projector.transform(F_all)
        
        # 3. Health Scoring (Distance from Global Nominal Center)
        severity, confidence = self.cluster.score(Z)
        
        return F_all, Z, severity, confidence

# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    X_nom = rng.standard_normal((200, 50, 14)).astype(np.float32)
    X_rec = X_nom + rng.standard_normal((200, 50, 14)).astype(np.float32) * 0.05

    # Individual function tests
    feats = extract_batch_features(X_nom[:5])
    print(f"Feature shape (5 windows): {feats.shape}")

    res = compute_reconstruction_error(X_nom[:5], X_rec[:5])
    print(f"Residual shape            : {res.shape}")

    # Full pipeline
    pipe = FeaturePipeline(n_components=2)
    pipe.fit(X_nom, X_rec)

    X_test  = rng.standard_normal((20, 50, 14)).astype(np.float32)
    X_rec_t = X_test + rng.standard_normal((20, 50, 14)).astype(np.float32) * 0.3
    F, Z, sev, conf = pipe.transform(X_test, X_rec_t)
    print(f"Test features   : {F.shape}")
    print(f"Manifold embed  : {Z.shape}  method={pipe.projector.method}")
    print(f"Severity  [0:5] : {sev[:5]}")
    print(f"Confidence[0:5] : {conf[:5]}")
    print("✓ feature_engineering.py smoke tests passed.")
