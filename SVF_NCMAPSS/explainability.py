"""
explainability.py
=================
Explainability & Accommodation module for Jet Engine PHM.

  1. SHAPExplainer        – SHAP values for MultiHeadANN (GradientSHAP on flat
                            features) and MultiHeadLSTM / MultiHeadTransformer
                            (GradientSHAP on raw (N, T, S) windows).
  2. DiagnosisReport      – Human-readable structured output per window.
  3. DiagnosticOrchestrator – End-to-end pipeline that auto-routes between the
                            ANN path (needs FeaturePipeline) and the sequence
                            model path (LSTM / Transformer, raw windows only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level fault head wrapper — must be at module scope to be picklable.
# ─────────────────────────────────────────────────────────────────────────────

class _FaultHead(nn.Module):
    """Strips the multi-head tuple and returns fault logits only.
    Defined at module level (not inside a method) so it is picklable."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]   # index 0 = fault logits


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHAP Explainer  (ANN flat-feature + Sequence window variants)
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    SHAP wrapper for all three head types: MultiHeadANN, MultiHeadLSTM,
    MultiHeadTransformer.

    Strategy
    --------
    ANN    → GradientSHAP on flat feature vectors (N, F).
             `background` must be (N_bg, F) float32 ndarray.
    LSTM / Transformer → GradientSHAP on raw windows (N, T, S).
             `background` must be (N_bg, T, S) float32 ndarray.

    In both cases the wrapped model returns only fault logits (index 0 of the
    tuple), which is what SHAP differentiates through.

    Parameters
    ----------
    model        : Any of MultiHeadANN, MultiHeadLSTM, MultiHeadTransformer.
    background   : Background samples matched to the model's expected input shape.
    device       : torch.device.
    n_background : How many background samples to subsample.
    model_type   : 'ANN' | 'LSTM' | 'Transformer' (auto-detected from class name
                   if not supplied).
    """

    def __init__(
        self,
        model        : nn.Module,
        background   : np.ndarray,
        device       : torch.device = torch.device("cpu"),
        n_background : int = 100,
        model_type   : Optional[str] = None,
    ):
        self.model   = model.eval().to(device)
        self.device  = device

        # Auto-detect model type from class name if not provided
        if model_type is None:
            cname = type(model).__name__
            if "LSTM" in cname:
                model_type = "LSTM"
            elif "Transformer" in cname:
                model_type = "Transformer"
            else:
                model_type = "ANN"
        self.model_type = model_type

        # Subsample background
        idx = np.random.choice(
            len(background), min(n_background, len(background)), replace=False
        )
        bg = background[idx].astype(np.float32)
        self.background = torch.from_numpy(bg).float().to(device)
        self.input_shape = bg.shape[1:]   # (F,) for ANN or (T, S) for sequences
        self._method = "GradientSHAP"
        self._explainer = None
        self._fit()

    # ── Internal setup ───────────────────────────────────────────────────────

    def _fault_head_wrapper(self) -> nn.Module:
        """Wrap model using module-level _FaultHead (picklable)."""
        return _FaultHead(self.model).to(self.device)

    def _fit(self) -> None:
        try:
            import shap   # type: ignore
        except ImportError:
            log.warning("shap not installed – SHAPExplainer disabled")
            return

        wrapped = self._fault_head_wrapper()
        self._explainer = shap.GradientExplainer(wrapped, self.background)
        log.info(
            "SHAPExplainer [%s]: GradientSHAP initialised (bg=%d, input=%s)",
            self.model_type, len(self.background), self.input_shape,
        )

    # ── Main explain method ──────────────────────────────────────────────────

    def explain(
        self,
        X          : np.ndarray,
        top_k      : int = 10,
        class_idx  : int = 0,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute GradientSHAP values.

        Parameters
        ----------
        X         : (N, F) for ANN  or  (N, T, S) for LSTM / Transformer.
        top_k     : Top-K features / timestep-sensor pairs to report per window.
        class_idx : Fault class index to explain.

        Returns
        -------
        sv_class : (N, F) for ANN  or  (N, T, S) for sequence models.
                   Raw SHAP attributions for the requested class.
        reports  : List of dicts per window with top-k feature/timestep indices
                   and their SHAP contributions.
        """
        if self._explainer is None:
            dummy = np.zeros((len(X),) + self.input_shape)
            return dummy, []

        x_t = torch.from_numpy(X.astype(np.float32)).to(self.device)

        # cuDNN RNN (LSTM) requires train mode for backward / gradient computation.
        # We temporarily set the wrapped model to train(), run SHAP, then restore.
        # This does NOT change predictions — only enables gradient flow through RNN.
        # GradientExplainer stores the model at .explainer.model (inner _PyTorchGradient).
        # Access it safely with a fallback in case the shap version differs.
        wrapped = getattr(self._explainer, 'explainer', self._explainer)
        wrapped = getattr(wrapped, 'model', None)
        was_training = wrapped.training if wrapped is not None else False
        if wrapped is not None:
            wrapped.train()

        try:
            sv = self._explainer.shap_values(x_t)
        finally:
            if wrapped is not None and not was_training:
                wrapped.eval()

        # ── Normalise sv to (N, *input_shape) ────────────────────────────────
        # shap GradientExplainer format changed across versions:
        #   shap < 0.41  : list of arrays, one per output class
        #                  each element shape (N, *input_shape)
        #   shap >= 0.41 : single ndarray (N, *input_shape, n_classes)
        #                  class index is the LAST dimension
        # We detect which layout and always produce sv_class: (N, *input_shape).

        if isinstance(sv, (list, tuple)):
            # Old format: list[class_idx] -> (N, *input_shape)
            sv_class = np.array(sv[class_idx], dtype=np.float32)
        else:
            sv = np.array(sv, dtype=np.float32)
            # New format (shap >= 0.41): (N, *input_shape, n_classes)
            # sv.ndim == len(input_shape) + 2 in this layout.
            # For ANN:  input_shape=(F,)   -> sv is (N, F, n_classes), ndim=3
            # For LSTM: input_shape=(T, S) -> sv is (N, T, S, n_classes), ndim=4
            expected_new_ndim = len(self.input_shape) + 2
            if sv.ndim == expected_new_ndim:
                sv_class = sv[..., class_idx]   # index last dim -> (N, *input_shape)
            elif sv.ndim >= 2 and sv.shape[0] <= 32 and sv.shape[1] == len(X):
                sv_class = sv[class_idx]        # old class-first layout
            else:
                sv_class = sv                   # no class dim (single-output)

        if torch.is_tensor(sv_class):
            sv_class = sv_class.detach().cpu().numpy()
        sv_class = np.array(sv_class, dtype=np.float32)

        # Safety squeeze: remove unexpected trailing size-1 dims
        while sv_class.ndim > len(self.input_shape) + 1 and sv_class.shape[-1] == 1:
            sv_class = sv_class.squeeze(-1)

        # ── Build per-sample reports ──────────────────────────────────────────
        reports = []
        for i in range(len(X)):
            sv_i     = sv_class[i]
            abs_flat = np.abs(sv_i).ravel()
            # Use plain Python ints so indices are always scalar — prevents
            # the "only integer scalar arrays can be converted to a scalar index"
            # error that occurs when numpy arrays are used to index Python lists.
            top_idx_list = [int(j) for j in np.argsort(abs_flat)[::-1][:top_k]]
            reports.append({
                "sample_idx"   : i,
                "top_flat_idx" : top_idx_list,
                "contributions": sv_i.ravel()[top_idx_list].tolist(),
            })

        return sv_class, reports

    # ── Utility: feature / axis names ────────────────────────────────────────

    @staticmethod
    def feature_names(
        sensor_cols  : List[str],
        n_fft_peaks  : int = 3,
        has_residuals: bool = True,
    ) -> List[str]:
        """
        Ordered feature names for the ANN flat-feature path.

        Matches the FeaturePipeline output: per-sensor stats + FFT + residuals.
        """
        stat_names = ["mean", "std", "skew", "kurt"]
        fft_names  = [f"fft_{k+1}" for k in range(n_fft_peaks)]
        names = []
        for s in sensor_cols:
            for stat in stat_names + fft_names:
                names.append(f"{s}_{stat}")
        if has_residuals:
            for s in sensor_cols:
                names.append(f"{s}_residual")
        return names

    @staticmethod
    def sequence_axis_labels(sensor_cols: List[str]) -> List[str]:
        """
        Axis labels for the sequence SHAP heatmap: just the sensor column names.
        The time axis is labelled by timestep index in the plot.
        """
        return list(sensor_cols)

    # ── Sequence-specific: per-timestep importance ───────────────────────────

    @staticmethod
    def timestep_importance(sv: np.ndarray) -> np.ndarray:
        """
        Aggregate (N, T, S) SHAP values to (N, T) by summing |sv| over sensors.

        Useful for identifying *when* in a window the model detected a fault.
        """
        assert sv.ndim == 3, "Expected (N, T, S) SHAP array."
        return np.abs(sv).sum(axis=-1)   # (N, T)

    @staticmethod
    def sensor_importance(sv: np.ndarray) -> np.ndarray:
        """
        Aggregate (N, T, S) SHAP values to (N, S) by summing |sv| over timesteps.

        Useful for identifying *which sensor* drove the classification.
        """
        assert sv.ndim == 3, "Expected (N, T, S) SHAP array."
        return np.abs(sv).sum(axis=1)    # (N, S)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Diagnosis Report  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosisReport:
    """Structured output for a single diagnosed window."""

    window_idx        : int
    timestamp_s       : float
    is_anomaly        : bool
    severity_pct      : float
    confidence        : float
    fault_type        : str
    fault_type_prob   : float
    sensor_id         : str
    sensor_prob       : float
    magnitude_est     : float
    magnitude_std     : float
    top_shap_features : List[str]   = field(default_factory=list)
    shap_contributions: List[float] = field(default_factory=list)
    accommodation     : str         = "None"   # "VirtualSensor" | "None"

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    def __str__(self) -> str:
        lines = [
            f"─── Diagnosis Report [Window {self.window_idx}] ───",
            f"  Timestamp      : {self.timestamp_s:.1f} s",
            f"  Anomaly        : {'YES ⚠' if self.is_anomaly else 'no'}",
            f"  Severity       : {self.severity_pct:.1f}%",
            f"  Confidence     : {self.confidence:.2f}",
            f"  Fault Type     : {self.fault_type} (p={self.fault_type_prob:.2f})",
            f"  Sensor ID      : {self.sensor_id} (p={self.sensor_prob:.2f})",
            f"  Magnitude (est): {self.magnitude_est:.4f} ± {self.magnitude_std:.4f}",
            f"  Accommodation  : {self.accommodation}",
        ]
        if self.top_shap_features:
            lines.append("  Top SHAP features / regions:")
            for nm, cv in zip(self.top_shap_features[:5], self.shap_contributions[:5]):
                lines.append(f"    {nm:30s} {cv:+.4f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diagnostic Orchestrator  (upgraded to handle ANN, LSTM, Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticOrchestrator:
    """
    End-to-end inference pipeline.

    Automatically routes between:
      • ANN path   – uses FeaturePipeline for feature extraction and manifold
                     scoring, then passes flat features to MCDropoutWrapper.
      • Sequence path – LSTM or Transformer; uses raw (N, T, S) windows directly.
                     Anomaly detection still uses the AE; severity/confidence
                     come from the AE reconstruction error + simple threshold
                     since the manifold cluster is built on flat features.

    Parameters
    ----------
    ae_trainer     : Fitted AETrainer (always required — AE is the anomaly detector).
    mc_wrapper     : MCDropoutWrapper around whichever head model was trained.
    sensor_cols    : List of sensor column names.
    fault_labels   : Dict mapping int → fault type string.
    shap_explainer : Fitted SHAPExplainer (optional but recommended).
    feature_pipe   : Fitted FeaturePipeline (required for ANN path; ignored for
                     LSTM / Transformer).
    severity_thresh: Severity % threshold for anomaly declaration.
                     For the ANN path this comes from the manifold cluster score.
                     For the sequence path it is derived from the normalised AE
                     reconstruction error relative to the nominal distribution.
    ae_nominal_mae : Mean AE MAE on nominal data (used to normalise severity on
                     the sequence path). If None, severity is estimated as
                     raw_mae / (raw_mae + 1e-6) * 100 — provide this for
                     accuracy.
    model_type     : 'ANN' | 'LSTM' | 'Transformer'. Auto-detected if None.
    use_residual   : Whether the sequence model was trained with AE residual
                     channels appended (doubles input channels).
    """

    def __init__(
        self,
        ae_trainer,
        mc_wrapper,
        sensor_cols     : List[str],
        fault_labels    : Dict[int, str],
        shap_explainer  = None,
        feature_pipe    = None,
        severity_thresh : float = 50.0,
        ae_nominal_mae  : Optional[float] = None,
        model_type      : Optional[str] = None,
        use_residual    : bool = False,
    ):
        self.ae           = ae_trainer
        self.mc           = mc_wrapper
        self.feat_pipe    = feature_pipe
        self.shap         = shap_explainer
        self.sensor_cols  = sensor_cols
        self.fault_labels = fault_labels
        self.sev_thresh   = severity_thresh
        self.ae_nominal_mae = ae_nominal_mae
        self.use_residual = use_residual

        # Determine model type
        if model_type is None:
            cname = type(mc_wrapper.model).__name__
            if "LSTM" in cname:
                model_type = "LSTM"
            elif "Transformer" in cname:
                model_type = "Transformer"
            else:
                model_type = "ANN"
        self.model_type = model_type

        if self.model_type == "ANN" and feature_pipe is None:
            raise ValueError(
                "feature_pipe is required for model_type='ANN'. "
                "Pass the fitted FeaturePipeline from NB3."
            )

        log.info(
            "DiagnosticOrchestrator ready  model_type=%s  use_residual=%s",
            self.model_type, self.use_residual,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def diagnose_batch(
        self,
        X_windows : np.ndarray,      # (N, T, S)
        stride_s  : float = 10.0,
    ) -> Tuple[List[DiagnosisReport], np.ndarray]:
        """
        Diagnose a batch of windows end-to-end.

        Returns
        -------
        reports       : List[DiagnosisReport], one per window.
        X_accommodated: (N, T, S) with virtual sensor applied where needed.
        """
        # ── Step 1: AE reconstruction (always) ───────────────────────────────
        X_recon = self.ae.reconstruct(X_windows)

        # ── Step 2: Anomaly scoring ───────────────────────────────────────────
        severity, confidence = self._anomaly_score(X_windows, X_recon)
        anomaly_mask = severity > self.sev_thresh

        # ── Step 3: Build model input ─────────────────────────────────────────
        model_input, F_flat = self._build_model_input(X_windows, X_recon)

        # ── Step 4: MC Dropout inference ─────────────────────────────────────
        device     = next(self.mc.model.parameters()).device
        inp_t      = torch.from_numpy(model_input).float().to(device)
        fault_out  = self.mc.predict(inp_t, head="fault")
        sensor_out = self.mc.predict(inp_t, head="sensor")
        mag_out    = self.mc.predict(inp_t, head="magnitude")

        # ── Step 5: SHAP (anomalous windows only) ────────────────────────────
        shap_sv, shap_rpts = None, []
        feat_names = []
        if self.shap is not None and anomaly_mask.any():
            shap_input = model_input[anomaly_mask]
            fault_cls  = fault_out["pred_class"][anomaly_mask].cpu().numpy()
            top_cls    = int(np.bincount(fault_cls).argmax())
            shap_sv, shap_rpts = self.shap.explain(shap_input, class_idx=top_cls)
            feat_names = self._shap_feature_names()

        # ── Step 6: Virtual sensor accommodation ─────────────────────────────
        sensor_ids   = sensor_out["pred_class"].cpu().numpy()
        X_accommodated = _apply_virtual(X_windows, X_recon, anomaly_mask, sensor_ids)

        # ── Step 7: Build DiagnosisReport list ───────────────────────────────
        reports: List[DiagnosisReport] = []
        shap_ptr = 0
        for i in range(len(X_windows)):
            ft_int = int(fault_out["pred_class"][i].item())
            s_int  = int(sensor_out["pred_class"][i].item())

            top_shap, top_contr = [], []
            if anomaly_mask[i] and shap_rpts:
                r = shap_rpts[shap_ptr]; shap_ptr += 1
                top_shap  = [feat_names[fi] for fi in r["top_flat_idx"]
                             if fi < len(feat_names)]
                top_contr = r["contributions"][:len(top_shap)]

            reports.append(DiagnosisReport(
                window_idx        = i,
                timestamp_s       = i * stride_s,
                is_anomaly        = bool(anomaly_mask[i]),
                severity_pct      = float(severity[i]),
                confidence        = float(confidence[i]),
                fault_type        = self.fault_labels.get(ft_int, f"Unknown({ft_int})"),
                fault_type_prob   = float(fault_out["confidence"][i].item()),
                sensor_id         = (self.sensor_cols[s_int]
                                     if s_int < len(self.sensor_cols) else str(s_int)),
                sensor_prob       = float(sensor_out["confidence"][i].item()),
                magnitude_est     = float(mag_out["mean_mag"][i].item()),
                magnitude_std     = float(mag_out["std_mag"][i].item()),
                top_shap_features = top_shap,
                shap_contributions= top_contr,
                accommodation     = "VirtualSensor" if anomaly_mask[i] else "None",
            ))

        return reports, X_accommodated

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _anomaly_score(
        self,
        X_windows: np.ndarray,
        X_recon  : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-window (severity, confidence) scores.

        ANN path  → uses the FeaturePipeline manifold cluster score.
        Seq. path → uses normalised AE MAE as severity proxy; confidence
                    is derived from the inverse variance of per-sensor MAE.
        """
        if self.model_type == "ANN":
            # Full manifold scoring via FeaturePipeline
            _, _, severity, confidence = self.feat_pipe.transform(X_windows, X_recon)
            return severity, confidence
        else:
            # AE MAE-based severity for sequence models
            mae = np.abs(X_windows - X_recon).mean(axis=(1, 2))   # (N,)
            ref = self.ae_nominal_mae if self.ae_nominal_mae is not None else mae.mean()
            # Normalise: severity = 0% at nominal, 100% at 2× nominal MAE
            severity   = np.clip((mae / (2.0 * ref + 1e-9)) * 100.0, 0, 100)
            # Confidence proxy: how consistent is the per-sensor MAE?
            per_sensor_mae = np.abs(X_windows - X_recon).mean(axis=1)   # (N, S)
            # High variance across sensors → one sensor is very wrong → high confidence
            sensor_var  = per_sensor_mae.var(axis=1)
            confidence  = np.tanh(sensor_var / (sensor_var.mean() + 1e-9))
            confidence  = np.clip(confidence, 0.0, 1.0)
            return severity.astype(np.float32), confidence.astype(np.float32)

    def _build_model_input(
        self,
        X_windows: np.ndarray,
        X_recon  : np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns (model_input, F_flat).

        ANN       → F_flat = (N, n_feat), model_input = F_flat.
        LSTM/TF   → model_input = (N, T, S) or (N, T, 2S) with residual;
                    F_flat = None (not needed).
        """
        if self.model_type == "ANN":
            _, _, _, _ = self.feat_pipe.transform(X_windows, X_recon)
            # Re-transform to get F
            F, _, _, _ = self.feat_pipe.transform(X_windows, X_recon)
            return F.astype(np.float32), F.astype(np.float32)
        else:
            if self.use_residual:
                residual = np.abs(X_windows - X_recon).astype(np.float32)
                inp = np.concatenate([X_windows, residual], axis=-1)
            else:
                inp = X_windows.astype(np.float32)
            return inp, None

    def _shap_feature_names(self) -> List[str]:
        """
        Return human-readable axis labels for SHAP reports.

        ANN       → flat feature names (sensor_stat strings).
        LSTM/TF   → "t{timestep}__{sensor}" strings for the flat ravel order.
        """
        if self.model_type == "ANN":
            return SHAPExplainer.feature_names(self.sensor_cols)
        else:
            # SHAP values shape is (N, T, S[*2]); ravel order is C-order
            S = len(self.sensor_cols)
            # If residual channels were appended, label them too
            if self.use_residual:
                all_cols = list(self.sensor_cols) + [f"{s}_res" for s in self.sensor_cols]
            else:
                all_cols = list(self.sensor_cols)
            # Get T from the background shape stored in the explainer
            if self.shap is not None:
                T = self.shap.input_shape[0]
            else:
                T = 50   # fallback
            return [f"t{t:02d}__{col}" for t in range(T) for col in all_cols]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_virtual(
    X_raw     : np.ndarray,
    X_recon   : np.ndarray,
    mask      : np.ndarray,
    sensor_ids: np.ndarray,
) -> np.ndarray:
    """Replace faulty channel in anomalous windows with AE reconstruction."""
    out = X_raw.copy()
    for i in np.where(mask)[0]:
        s = int(sensor_ids[i])
        if s < X_raw.shape[2]:
            out[i, :, s] = X_recon[i, :, s]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sensor_cols = ["T24","T30","T48","T50","P15","P2","P21","P24",
                   "Ps30","P40","P50","Nf","Nc","Wf"]

    # ANN feature names
    names = SHAPExplainer.feature_names(sensor_cols, n_fft_peaks=3, has_residuals=True)
    print(f"ANN feature names: {len(names)}  first={names[0]}  last={names[-1]}")

    # Sequence axis labels
    seq_labels = SHAPExplainer.sequence_axis_labels(sensor_cols)
    print(f"Sequence sensor cols: {seq_labels[:4]} …")

    # Dummy sequence SHAP values (N=4, T=50, S=14)
    dummy_sv = np.random.randn(4, 50, 14).astype(np.float32)
    ts_imp  = SHAPExplainer.timestep_importance(dummy_sv)
    sen_imp = SHAPExplainer.sensor_importance(dummy_sv)
    print(f"Timestep importance : {ts_imp.shape}   min={ts_imp.min():.3f}")
    print(f"Sensor importance   : {sen_imp.shape}  min={sen_imp.min():.3f}")

    # Dummy report
    rpt = DiagnosisReport(
        window_idx=0, timestamp_s=100.0, is_anomaly=True,
        severity_pct=72.3, confidence=0.89,
        fault_type="Drift", fault_type_prob=0.78,
        sensor_id="T48", sensor_prob=0.65,
        magnitude_est=0.0821, magnitude_std=0.012,
        top_shap_features=["t12__T48","t13__T48","t14__T48"],
        shap_contributions=[0.231, -0.112, 0.089],
        accommodation="VirtualSensor",
    )
    print(rpt)
    print("✓ explainability.py smoke tests passed.")
