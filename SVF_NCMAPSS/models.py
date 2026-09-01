"""
models.py
=========
Neural-network models for Jet Engine PHM.

  1. ConvAutoencoder        – 1D-CNN Autoencoder for anomaly detection.
                              Trained on nominal cruise data (hs == 1).
  2. MultiHeadANN           – Multi-head classifier / regressor for:
                                  • Fault Type  (classification)
                                  • Sensor ID   (classification)
                                  • Magnitude   (regression)
  3. MultiHeadLSTM          – Drop-in LSTM replacement for MultiHeadANN.
                              Accepts raw windowed sensor data (N, T, S).
                              Bidirectional 2-layer LSTM, same 3 output heads.
  4. MultiHeadTransformer   – Transformer-encoder replacement.
                              Positional encoding + MHA + FFN, same 3 heads.
  5. MCDropoutWrapper       – Monte Carlo Dropout (T=100) for predictive
                              confidence. Works with all three head models.
  6. Trainer classes        – AETrainer, ANNTrainer, SequenceTrainer
                              (SequenceTrainer handles LSTM and Transformer).

Author: Lead AI Research Engineer – Aerospace PHM
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Device setup (RTX 4500 preferred)
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu")
        log.warning("CUDA unavailable – running on CPU")
    return dev


DEVICE = get_device()


# ─────────────────────────────────────────────────────────────────────────────
# Helper blocks
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
    )


def _deconv_block(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride,
                           padding=kernel // 2, output_padding=stride - 1),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared multi-head output block (reused by LSTM and Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class _MultiHead(nn.Module):
    """
    Three output heads bolted onto a shared trunk embedding.

    Input  : (batch, d_model)  – pooled sequence representation
    Output : (logits_fault, logits_sensor, magnitude)
    """

    def __init__(
        self,
        d_model       : int,
        n_fault_types : int = 8,
        n_sensors     : int = 14,
        dropout_p     : float = 0.3,
    ):
        super().__init__()
        self.head_fault_type = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, n_fault_types),
        )
        self.head_sensor_id = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, n_sensors),
        )
        self.head_magnitude = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.head_fault_type(h),
            self.head_sensor_id(h),
            self.head_magnitude(h),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. 1D-CNN Autoencoder  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for sensor data anomaly detection.

    Input  : (batch, n_sensors, window_size)  →  channels-first
    Output : (batch, n_sensors, window_size)  reconstruction

    Architecture follows the encoder–bottleneck–decoder paradigm.
    Bottleneck dimension provides latent representation for SHAP analysis.
    """

    def __init__(
        self,
        n_sensors   : int   = 14,
        window_size : int   = 50,
        base_ch     : int   = 32,
        latent_dim  : int   = 64,
        dropout_p   : float = 0.1,
    ):
        super().__init__()
        self.n_sensors   = n_sensors
        self.window_size = window_size
        self.latent_dim  = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = _conv_block(n_sensors, base_ch,      3, stride=1)
        self.enc2 = _conv_block(base_ch,   base_ch * 2,  3, stride=2)  # T/2
        self.enc3 = _conv_block(base_ch*2, base_ch * 4,  3, stride=2)  # T/4
        self.enc4 = _conv_block(base_ch*4, base_ch * 8,  3, stride=2)  # T/8

        # ── Dynamic Dimension Calculation ────────────────────────────────────
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_sensors, window_size)
            dummy_output = self.encode(dummy_input)
            self.flat_dim = dummy_output.numel()
            self.spatial_shape = dummy_output.shape[1:]

        log.info(f"AE Bottleneck: {self.flat_dim} -> {latent_dim} -> {self.flat_dim}")

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(latent_dim, self.flat_dim),
            nn.GELU(),
            nn.Unflatten(1, self.spatial_shape),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = _deconv_block(base_ch*8, base_ch*4,  3, stride=2)
        self.dec3 = _deconv_block(base_ch*4, base_ch*2,  3, stride=2)
        self.dec2 = _deconv_block(base_ch*2, base_ch,    3, stride=2)
        self.dec1 = nn.Conv1d(base_ch, n_sensors, 3, padding=1)

        self.dropout = nn.Dropout(dropout_p)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec4(z)
        z = self.dec3(z)
        z = self.dec2(z)
        z = self.dec1(z)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_hat : reconstruction
        z     : bottleneck latent
        """
        z_enc  = self.encode(x)
        z_bn   = self.bottleneck(z_enc)
        x_hat  = self.decode(z_bn)

        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode="linear",
                                  align_corners=False)
        return x_hat, z_bn

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """MAE per sample: (batch,)"""
        with torch.no_grad():
            x_hat, _ = self(x)
        return (x - x_hat).abs().mean(dim=(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Head ANN  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadANN(nn.Module):
    """
    Multi-head ANN for fault diagnosis.

    Heads:
      • head_fault_type : Cross-entropy (n_fault_types classes)
      • head_sensor_id  : Cross-entropy (n_sensors classes)
      • head_magnitude  : MSE regression (scalar magnitude)

    Input: flat feature vector (from FeaturePipeline).
    """

    def __init__(
        self,
        in_features    : int,
        n_fault_types  : int = 8,
        n_sensors      : int = 14,
        hidden_dims    : List[int] = None,
        dropout_p      : float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.dropout_p = dropout_p

        # ── Shared trunk ─────────────────────────────────────────────────────
        trunk_layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            trunk_layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout_p),
            ]
            prev = h

        self.trunk = nn.Sequential(*trunk_layers)

        # ── Classification heads ──────────────────────────────────────────────
        self.head_fault_type = nn.Linear(prev, n_fault_types)
        self.head_sensor_id  = nn.Linear(prev, n_sensors)

        # ── Regression head ───────────────────────────────────────────────────
        self.head_magnitude  = nn.Sequential(
            nn.Linear(prev, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits_fault : (batch, n_fault_types)
        logits_sensor: (batch, n_sensors)
        magnitude    : (batch, 1)
        """
        h = self.trunk(x)
        return (
            self.head_fault_type(h),
            self.head_sensor_id(h),
            self.head_magnitude(h),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-Head LSTM  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadLSTM(nn.Module):
    """
    Bidirectional multi-layer LSTM classification/regression head.

    Replaces MultiHeadANN when raw time-series windows are available.
    Takes (batch, T, S) input directly — no FeaturePipeline needed.

    Optionally accepts a per-timestep AE residual channel, appended as
    an extra feature dimension: (batch, T, S + S) = (batch, T, 2*S).

    Architecture
    ------------
    Input projection  →  Bi-LSTM (2 layers)  →  last-step pooling
                      →  _MultiHead (fault / sensor / magnitude)

    The forward hidden state of the final LSTM layer at t=T is used.
    Bidirectionality is handled by concatenating fwd+bwd: d_model = 2*hidden.

    Parameters
    ----------
    n_sensors     : number of raw sensor channels S
    hidden_size   : LSTM hidden units per direction (default 128)
    num_layers    : number of stacked LSTM layers (default 2)
    n_fault_types : number of fault classes
    dropout_p     : dropout between LSTM layers and in output heads
    use_residual  : if True, expects input (batch, T, 2*S) where the second
                    S channels are AE reconstruction residuals |X - X_hat|
    """

    def __init__(
        self,
        n_sensors     : int   = 14,
        hidden_size   : int   = 128,
        num_layers    : int   = 2,
        n_fault_types : int   = 8,
        dropout_p     : float = 0.3,
        use_residual  : bool  = False,
    ):
        super().__init__()
        self.n_sensors    = n_sensors
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.use_residual = use_residual

        in_features = n_sensors * 2 if use_residual else n_sensors
        d_model     = hidden_size * 2   # bidirectional concat

        # Input projection: linear layer to allow easier gradient flow
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size    = hidden_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout_p if num_layers > 1 else 0.0,
        )

        self.heads = _MultiHead(
            d_model       = d_model,
            n_fault_types = n_fault_types,
            n_sensors     = n_sensors,
            dropout_p     = dropout_p,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Set forget gate bias to 1 for better gradient flow
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)
        for m in self.heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x       : torch.Tensor,               # (batch, T, S) or (batch, T, 2S)
        lengths : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x       : (batch, T, S[*2]) float32 sensor windows
        lengths : optional (batch,) int64 for packed-sequence support

        Returns
        -------
        logits_fault : (batch, n_fault_types)
        logits_sensor: (batch, n_sensors)
        magnitude    : (batch, 1)
        """
        # Input projection: (batch, T, hidden_size)
        h = self.input_proj(x)

        # Optionally pack for variable-length sequences
        if lengths is not None:
            h = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        out, (h_n, _) = self.lstm(h)

        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Pool: take last timestep from each direction
        # h_n shape: (num_layers * 2, batch, hidden_size)
        # Last layer: forward = h_n[-2], backward = h_n[-1]
        fwd = h_n[-2]   # (batch, hidden_size)
        bwd = h_n[-1]   # (batch, hidden_size)
        pooled = torch.cat([fwd, bwd], dim=-1)   # (batch, 2*hidden_size)

        return self.heads(pooled)

    @property
    def dropout_p(self) -> float:
        """Expose dropout probability for MCDropoutWrapper compatibility."""
        return self.lstm.dropout


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Head Transformer  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class _SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadTransformer(nn.Module):
    """
    Transformer-encoder classification/regression head.

    Replaces MultiHeadANN when raw time-series windows are available.
    Takes (batch, T, S) input directly — no FeaturePipeline needed.

    Architecture
    ------------
    Input projection  →  Sinusoidal PE  →  TransformerEncoder (N layers)
                      →  mean-pool over T  →  _MultiHead

    The mean pool over time steps is more stable than CLS-token pooling
    for short sequences (T ≈ 50). CLS is recommended for longer sequences.

    Attention weights from the last layer are stored in `last_attn_weights`
    after each forward pass (when store_attn=True), enabling attention-based
    visualisation as a complement or replacement to SHAP.

    Parameters
    ----------
    n_sensors     : raw sensor channels S
    d_model       : internal embedding dimension (default 64)
    nhead         : number of attention heads (d_model must be divisible)
    num_layers    : number of TransformerEncoder layers (default 2)
    dim_feedforward: FFN hidden size (default 256 = 4 × d_model)
    n_fault_types : number of fault classes
    dropout_p     : dropout in attention, FFN, and output heads
    use_residual  : if True, expects (batch, T, 2*S) with AE residuals appended
    store_attn    : if True, saves attention weights for analysis
    """

    def __init__(
        self,
        n_sensors      : int   = 14,
        d_model        : int   = 64,
        nhead          : int   = 4,
        num_layers     : int   = 2,
        dim_feedforward: int   = 256,
        n_fault_types  : int   = 8,
        dropout_p      : float = 0.1,
        use_residual   : bool  = False,
        store_attn     : bool  = False,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.n_sensors   = n_sensors
        self.d_model     = d_model
        self.store_attn  = store_attn
        self.use_residual = use_residual
        self.last_attn_weights: Optional[torch.Tensor] = None

        in_features = n_sensors * 2 if use_residual else n_sensors

        # Input projection to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = _SinusoidalPositionalEncoding(
            d_model, max_len=512, dropout=dropout_p
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout_p,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            enable_nested_tensor = False,
        )

        self.heads = _MultiHead(
            d_model       = d_model,
            n_fault_types = n_fault_types,
            n_sensors     = n_sensors,
            dropout_p     = dropout_p,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x            : torch.Tensor,           # (batch, T, S[*2])
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x                   : (batch, T, S) float32 sensor windows
        src_key_padding_mask: (batch, T) bool — True positions are ignored

        Returns
        -------
        logits_fault : (batch, n_fault_types)
        logits_sensor: (batch, n_sensors)
        magnitude    : (batch, 1)
        """
        # Project and add positional encoding
        h = self.input_proj(x)               # (batch, T, d_model)
        h = self.pos_enc(h)

        # Encode
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # Capture attention from projected h (NOT raw x — shape must be d_model)
        if self.store_attn:
            self._capture_attn(h, src_key_padding_mask)

        # Mean-pool over time to get a single vector per window
        if src_key_padding_mask is not None:
            # Mask out padding before averaging
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (batch, T, 1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)            # (batch, d_model)

        return self.heads(pooled)

    @torch.no_grad()
    def _capture_attn(
        self,
        h   : torch.Tensor,               # (batch, T, d_model) — already projected
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Capture attention weights from projected h (batch, T, d_model).
        Called from forward() AFTER input_proj + pos_enc so shapes match."""
        try:
            last_layer = self.encoder.layers[-1]
            h_in = h
            # Run all but the last layer
            for layer in self.encoder.layers[:-1]:
                h_in = layer(h_in, src_key_padding_mask=mask)
            # Last layer with attn weights
            _, attn_w = last_layer.self_attn(
                h_in, h_in, h_in,
                key_padding_mask=mask,
                need_weights=True,
                average_attn_weights=True,
            )
            self.last_attn_weights = attn_w.detach().cpu()
        except Exception as e:
            log.warning("Attention capture failed: %s", e)

    @property
    def dropout_p(self) -> float:
        return self.encoder.layers[0].dropout.p


# ─────────────────────────────────────────────────────────────────────────────
# 5. Monte Carlo Dropout Wrapper  (updated to handle sequence models)
# ─────────────────────────────────────────────────────────────────────────────

class MCDropoutWrapper:
    """
    Wraps any nn.Module (MultiHeadANN, MultiHeadLSTM, MultiHeadTransformer)
    and performs T stochastic forward passes to estimate predictive mean
    and uncertainty.

    The model must have nn.Dropout / nn.LSTM dropout layers; calling
    _enable_dropout() sets them to train mode while keeping BN in eval.
    """

    def __init__(self, model: nn.Module, T: int = 100):
        self.model = model
        self.T     = T

    def _enable_dropout(self) -> None:
        """Set model to eval but keep all Dropout layers in train mode."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @torch.no_grad()
    def predict(
        self,
        x   : torch.Tensor,
        head: str = "fault",
    ) -> Dict[str, torch.Tensor]:
        """
        Run T stochastic forward passes and return statistics.

        Parameters
        ----------
        x    : Input tensor.
                 ANN   → (batch, features)
                 LSTM  → (batch, T_steps, S)
                 Transformer → (batch, T_steps, S)
        head : Which head to collect –  'fault' | 'sensor' | 'magnitude'.

        Returns dict with keys:
          mean_proba   : (batch, n_classes) mean softmax probability
          std_proba    : (batch, n_classes) epistemic uncertainty
          pred_class   : (batch,)           argmax of mean_proba
          confidence   : (batch,)           max mean_proba
        """
        self._enable_dropout()
        samples: List[torch.Tensor] = []

        for _ in range(self.T):
            out = self.model(x)
            if head == "fault":
                logits = out[0]
            elif head == "sensor":
                logits = out[1]
            elif head == "magnitude":
                samples.append(out[2].squeeze(-1))
                continue
            else:
                raise ValueError(f"Unknown head: '{head}'")
            samples.append(F.softmax(logits, dim=-1))

        stacked = torch.stack(samples, dim=0)   # (T, batch, n_classes) or (T, batch)

        if head == "magnitude":
            return {
                "mean_mag" : stacked.mean(0),
                "std_mag"  : stacked.std(0),
            }

        mean_p = stacked.mean(0)
        std_p  = stacked.std(0)
        pred   = mean_p.argmax(dim=-1)
        conf   = mean_p.max(dim=-1).values

        return {
            "mean_proba" : mean_p,
            "std_proba"  : std_p,
            "pred_class" : pred,
            "confidence" : conf,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6a. Trainer – AETrainer  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class AETrainer:
    """Training loop for ConvAutoencoder."""

    def __init__(
        self,
        model       : ConvAutoencoder,
        lr          : float = 1e-3,
        weight_decay: float = 1e-5,
        device      : torch.device = DEVICE,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.optim   = torch.optim.AdamW(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optim, T_max=50, eta_min=1e-6)

    def fit(
        self,
        X_train     : np.ndarray,
        epochs      : int = 100,
        batch_size  : int = 256,
        val_split   : float = 0.1,
        patience    : int = 10,
    ) -> List[float]:
        X = torch.from_numpy(X_train.transpose(0, 2, 1)).float()

        n_val   = max(1, int(len(X) * val_split))
        X_val   = X[:n_val].to(self.device)
        X_tr    = X[n_val:]

        loader  = DataLoader(TensorDataset(X_tr), batch_size=batch_size,
                             shuffle=True, pin_memory=True, num_workers=0)

        best_val, patience_cnt = float("inf"), 0
        val_losses = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            tr_loss = 0.0
            for (xb,) in loader:
                xb   = xb.to(self.device)
                xhat, _ = self.model(xb)
                loss = F.mse_loss(xhat, xb)
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                tr_loss += loss.item() * len(xb)

            tr_loss /= len(X_tr)
            self.sched.step()

            self.model.eval()
            with torch.no_grad():
                xhat_val, _ = self.model(X_val)
                v_loss = F.mse_loss(xhat_val, X_val).item()
            val_losses.append(v_loss)

            if epoch % 10 == 0:
                log.info(f"AE Epoch {epoch:3d}/{epochs} | train={tr_loss:.6f} | val={v_loss:.6f}")

            if v_loss < best_val - 1e-6:
                best_val    = v_loss
                patience_cnt = 0
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    log.info(f"Early stop at epoch {epoch}")
                    break

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
        return val_losses

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        xin  = torch.from_numpy(X.transpose(0, 2, 1)).float().to(self.device)
        with torch.no_grad():
            xhat, _ = self.model(xin)
        return xhat.cpu().numpy().transpose(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6b. Trainer – ANNTrainer  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class ANNTrainer:
    """Training loop for MultiHeadANN (flat feature vector input)."""

    def __init__(
        self,
        model       : MultiHeadANN,
        lr          : float = 1e-3,
        weight_decay: float = 1e-5,
        device      : torch.device = DEVICE,
        w_fault     : float = 1.0,
        w_sensor    : float = 0.5,
        w_magnitude : float = 0.3,
    ):
        self.model      = model.to(device)
        self.device     = device
        self.w_fault    = w_fault
        self.w_sensor   = w_sensor
        self.w_magnitude= w_magnitude
        self.optim      = torch.optim.AdamW(model.parameters(), lr=lr,
                                             weight_decay=weight_decay)
        self.sched      = torch.optim.lr_scheduler.ReduceLROnPlateau(
                              self.optim, patience=5, factor=0.5)

    def _loss(self, lf, ls, lm, yf, ys, ym) -> torch.Tensor:
        l_fault  = F.cross_entropy(lf, yf)
        l_sensor = F.cross_entropy(ls, ys)
        l_mag    = F.mse_loss(lm.squeeze(-1), ym.float())
        return self.w_fault * l_fault + self.w_sensor * l_sensor + self.w_magnitude * l_mag

    def fit(
        self,
        F_train    : np.ndarray,
        y_fault    : np.ndarray,
        y_sensor   : np.ndarray,
        y_mag      : np.ndarray,
        epochs     : int   = 80,
        batch_size : int   = 256,
        val_split  : float = 0.1,
    ) -> List[float]:

        F  = torch.from_numpy(F_train).float()
        yf = torch.from_numpy(y_fault).long()
        ys = torch.from_numpy(y_sensor).long()
        ym = torch.from_numpy(y_mag).float()

        n_val   = max(1, int(len(F) * val_split))
        ds_tr   = TensorDataset(F[n_val:], yf[n_val:], ys[n_val:], ym[n_val:])
        ds_val  = (F[:n_val].to(self.device), yf[:n_val].to(self.device),
                   ys[:n_val].to(self.device), ym[:n_val].to(self.device))

        loader  = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=0)
        val_losses = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            for xb, yfb, ysb, ymb in loader:
                xb, yfb, ysb, ymb = (t.to(self.device) for t in (xb, yfb, ysb, ymb))
                lf, ls, lm = self.model(xb)
                loss = self._loss(lf, ls, lm, yfb, ysb, ymb)
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

            self.model.eval()
            with torch.no_grad():
                lf_v, ls_v, lm_v = self.model(ds_val[0])
                v_loss = self._loss(lf_v, ls_v, lm_v, *ds_val[1:]).item()
            val_losses.append(v_loss)
            self.sched.step(v_loss)

            if epoch % 10 == 0:
                log.info(f"ANN Epoch {epoch:3d}/{epochs} | val_loss={v_loss:.4f}")

        return val_losses


# ─────────────────────────────────────────────────────────────────────────────
# 6c. Trainer – SequenceTrainer  (NEW — handles LSTM and Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceTrainer:
    """
    Training loop for MultiHeadLSTM and MultiHeadTransformer.

    Accepts raw windowed data (N, T, S) directly.
    Optionally concatenates AE reconstruction residuals to form (N, T, 2S).

    Parameters
    ----------
    model        : MultiHeadLSTM or MultiHeadTransformer
    lr           : learning rate (AdamW)
    weight_decay : L2 regularisation
    device       : torch.device
    w_fault      : weight for fault-type cross-entropy
    w_sensor     : weight for sensor-id cross-entropy
    w_magnitude  : weight for magnitude MSE
    """

    def __init__(
        self,
        model        : nn.Module,
        lr           : float = 2e-4,
        weight_decay : float = 1e-5,
        device       : torch.device = DEVICE,
        w_fault      : float = 1.0,
        w_sensor     : float = 0.5,
        w_magnitude  : float = 0.3,
        warmup_epochs: int   = 5,
    ):
        self.model        = model.to(device)
        self.device       = device
        self.w_fault      = w_fault
        self.w_sensor     = w_sensor
        self.w_magnitude  = w_magnitude
        self.warmup_epochs= warmup_epochs
        self.lr           = lr
        self.optim        = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Scheduler set in fit() once we know T_max (total epochs)
        self.sched = None

    def _loss(self, lf, ls, lm, yf, ys, ym) -> torch.Tensor:
        l_fault  = F.cross_entropy(lf, yf)
        l_sensor = F.cross_entropy(ls, ys)
        l_mag    = F.mse_loss(lm.squeeze(-1), ym.float())
        return self.w_fault * l_fault + self.w_sensor * l_sensor + self.w_magnitude * l_mag

    def fit(
        self,
        X_train    : np.ndarray,          # (N, T, S)
        y_fault    : np.ndarray,          # (N,) int
        y_sensor   : np.ndarray,          # (N,) int
        y_mag      : np.ndarray,          # (N,) float
        X_recon    : Optional[np.ndarray] = None,   # (N, T, S) AE reconstruction
        epochs     : int   = 120,
        batch_size : int   = 512,
        val_split  : float = 0.1,
        patience   : int   = 20,
    ) -> List[float]:
        """
        Train the sequence model.

        Parameters
        ----------
        X_train  : (N, T, S) raw windowed sensor data
        y_fault  : (N,) fault-type labels
        y_sensor : (N,) faulty-sensor labels
        y_mag    : (N,) fault magnitudes
        X_recon  : (N, T, S) AE reconstructions (optional).
                   If provided AND model.use_residual=True, the per-timestep
                   residual |X - X_recon| is appended as extra channels,
                   producing (N, T, 2*S) input.

        Returns
        -------
        val_losses : list of per-epoch validation losses
        """
        # Build input tensor — optionally with residual channels
        X_in = self._build_input(X_train, X_recon)

        X  = torch.from_numpy(X_in).float()
        yf = torch.from_numpy(y_fault).long()
        ys = torch.from_numpy(y_sensor).long()
        ym = torch.from_numpy(y_mag).float()

        n_val   = max(1, int(len(X) * val_split))
        ds_tr   = TensorDataset(X[n_val:], yf[n_val:], ys[n_val:], ym[n_val:])
        ds_val  = (
            X[:n_val].to(self.device),
            yf[:n_val].to(self.device),
            ys[:n_val].to(self.device),
            ym[:n_val].to(self.device),
        )

        loader = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=0,
        )

        # Build LambdaLR: linear warmup then cosine decay.
        # warmup_epochs=0 → pure cosine (LSTM default).
        # warmup_epochs=5 → warmup then cosine (Transformer default).
        warmup       = self.warmup_epochs
        cosine_epochs= max(1, epochs - warmup)
        base_lr      = self.lr

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = float(epoch - warmup) / float(cosine_epochs)
            cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(1e-6 / base_lr, cos_val)

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)

        best_val, patience_cnt = float("inf"), 0
        val_losses: List[float] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            tr_loss = 0.0
            for xb, yfb, ysb, ymb in loader:
                xb, yfb, ysb, ymb = (t.to(self.device) for t in (xb, yfb, ysb, ymb))
                lf, ls, lm = self.model(xb)
                loss = self._loss(lf, ls, lm, yfb, ysb, ymb)
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                tr_loss += loss.item() * len(xb)

            tr_loss /= len(X) - n_val
            self.sched.step()

            self.model.eval()
            with torch.no_grad():
                lf_v, ls_v, lm_v = self.model(ds_val[0])
                v_loss = self._loss(lf_v, ls_v, lm_v, *ds_val[1:]).item()
            val_losses.append(v_loss)

            if epoch % 10 == 0:
                log.info(
                    f"{self.model.__class__.__name__} Epoch {epoch:3d}/{epochs}"
                    f" | train={tr_loss:.4f} | val={v_loss:.4f}"
                )

            if v_loss < best_val - 1e-6:
                best_val     = v_loss
                patience_cnt = 0
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    log.info(f"Early stop at epoch {epoch}")
                    break

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
            log.info(f"Restored best checkpoint (val={best_val:.4f})")

        return val_losses

    def _build_input(
        self,
        X    : np.ndarray,
        X_rec: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Optionally append AE residuals as extra sensor channels.

        Returns (N, T, S) if use_residual=False or X_rec is None,
                (N, T, 2S) if use_residual=True and X_rec is provided.
        """
        use_res = getattr(self.model, "use_residual", False)
        if use_res and X_rec is not None:
            residual = np.abs(X - X_rec).astype(np.float32)
            return np.concatenate([X, residual], axis=-1)   # (N, T, 2S)
        return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Virtual Sensor Accommodation  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def apply_virtual_sensor(
    X_raw      : np.ndarray,
    X_recon    : np.ndarray,
    anomaly_mask : np.ndarray,
    sensor_ids   : Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Replace faulty sensor channels with their AE reconstruction.

    If `sensor_ids` is provided, only the identified faulty channel is replaced.
    Otherwise, all channels in anomalous windows are replaced.

    Returns accommodated array (N, T, S).
    """
    X_out = X_raw.copy()
    idx   = np.where(anomaly_mask)[0]

    for i in idx:
        if sensor_ids is not None:
            s = int(sensor_ids[i])
            X_out[i, :, s] = X_recon[i, :, s]
        else:
            X_out[i] = X_recon[i]

    log.info("Virtual sensor: replaced %d/%d windows", len(idx), len(X_raw))
    return X_out


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    B, T, S = 8, 50, 14

    # ── Autoencoder ──────────────────────────────────────────────────────────
    x_np = rng.standard_normal((B, T, S)).astype(np.float32)
    x_t  = torch.from_numpy(x_np.transpose(0, 2, 1)).float().to(DEVICE)
    ae   = ConvAutoencoder(n_sensors=S, window_size=T).to(DEVICE)
    xhat, z = ae(x_t)
    print(f"AE input : {x_t.shape}  |  recon: {xhat.shape}  |  latent: {z.shape}")

    # Raw sequence input for LSTM / Transformer: (B, T, S)
    seq_t = torch.from_numpy(x_np).float().to(DEVICE)

    # ── Multi-head ANN ───────────────────────────────────────────────────────
    n_feat = 98
    feat_t = torch.randn(B, n_feat).to(DEVICE)
    ann    = MultiHeadANN(in_features=n_feat, n_fault_types=8, n_sensors=S).to(DEVICE)
    lf, ls, lm = ann(feat_t)
    print(f"ANN  fault: {lf.shape}  sensor: {ls.shape}  mag: {lm.shape}")

    # ── Multi-head LSTM ───────────────────────────────────────────────────────
    lstm_model = MultiHeadLSTM(
        n_sensors=S, hidden_size=128, num_layers=2,
        n_fault_types=8, dropout_p=0.3, use_residual=False,
    ).to(DEVICE)
    lf, ls, lm = lstm_model(seq_t)
    print(f"LSTM fault: {lf.shape}  sensor: {ls.shape}  mag: {lm.shape}")
    print(f"     params: {sum(p.numel() for p in lstm_model.parameters()):,}")

    # With residual channels
    seq_res = torch.cat([seq_t, seq_t.abs()], dim=-1)   # (B, T, 2S)
    lstm_r  = MultiHeadLSTM(
        n_sensors=S, hidden_size=128, num_layers=2,
        n_fault_types=8, dropout_p=0.3, use_residual=True,
    ).to(DEVICE)
    lf, ls, lm = lstm_r(seq_res)
    print(f"LSTM+res fault: {lf.shape}")

    # ── Multi-head Transformer ────────────────────────────────────────────────
    tf_model = MultiHeadTransformer(
        n_sensors=S, d_model=64, nhead=4, num_layers=2,
        dim_feedforward=256, n_fault_types=8, dropout_p=0.1,
        use_residual=False, store_attn=True,
    ).to(DEVICE)
    lf, ls, lm = tf_model(seq_t)
    print(f"TF   fault: {lf.shape}  sensor: {ls.shape}  mag: {lm.shape}")
    print(f"     params: {sum(p.numel() for p in tf_model.parameters()):,}")
    if tf_model.last_attn_weights is not None:
        print(f"     attn shape: {tf_model.last_attn_weights.shape}")
    else:
        print("     attn shape: (captured on next eval pass with store_attn=True)")

    # ── MC Dropout ───────────────────────────────────────────────────────────
    for name, model in [("ANN", ann), ("LSTM", lstm_model), ("TF", tf_model)]:
        mc  = MCDropoutWrapper(model, T=20)
        inp = feat_t if name == "ANN" else seq_t
        out = mc.predict(inp, head="fault")
        print(f"MC-{name} conf range: {out['confidence'].min():.3f}–{out['confidence'].max():.3f}")

    print("\n✓ models.py smoke tests passed (ANN + LSTM + Transformer).")
