import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal position information into the token embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The encoding is added (not concatenated) to the embedding and is fixed
    (not learned), keeping the parameter count minimal for CPU training.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding table once and register as a buffer so it moves
        # with the model to whatever device is used (CPU/GPU).
        pe = torch.zeros(max_len, d_model)                          # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                            # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)               # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)               # odd  dims → cos

        pe = pe.unsqueeze(0)                                        # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, d_model)
        Returns x + positional encoding, same shape.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Music Transformer
# ─────────────────────────────────────────────────────────────────────────────
class MusicTransformer(nn.Module):
    """
    Decoder-only (autoregressive) Transformer for piano-roll next-step prediction.

    Architecture
    ────────────
    Input  : (batch, seq_len, input_dim=128)   binary piano roll
    ↓ Linear embedding → (batch, seq_len, d_model)
    ↓ Positional Encoding (added, not concatenated)
    ↓ TransformerEncoder with causal (look-ahead) mask
      • num_layers = 2   (lightweight for CPU)
      • nhead      = 4
      • dim_feedforward = 256
    ↓ Linear output + Sigmoid → (batch, seq_len, input_dim=128)
    Output : next-step probabilities for every position in the sequence.

    Causal Mask
    ───────────
    A square upper-triangular mask of shape (seq_len, seq_len) is generated
    each forward pass so position i can only attend to positions 0 … i.
    This is required for autoregressive training: the model must predict
    step t+1 using only steps 0 … t.
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super(MusicTransformer, self).__init__()

        self.d_model = d_model

        # ── Input projection ─────────────────────────────────────────────────
        # Maps the raw 128-dim piano-roll frame to the d_model dimension.
        self.input_projection = nn.Linear(input_dim, d_model)

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        # ── Transformer encoder stack (decoder-only = encoder + causal mask) ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # input/output shape: (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # ── Output projection ─────────────────────────────────────────────────
        self.output_layer = nn.Linear(d_model, input_dim)
        self.sigmoid = nn.Sigmoid()

    # ─────────────────────────────────────────────────────────────────────────
    # Causal mask helper
    # ─────────────────────────────────────────────────────────────────────────
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns an additive causal mask of shape (seq_len, seq_len).

        Positions that should be BLOCKED are set to -inf so that after
        softmax they contribute 0 to the attention weights.
        Positions that ARE allowed remain 0.0.

        Example for seq_len=4:
            [[ 0., -inf, -inf, -inf],
             [ 0.,  0.,  -inf, -inf],
             [ 0.,  0.,   0.,  -inf],
             [ 0.,  0.,   0.,   0.]]
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )
        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (batch, seq_len, input_dim)
        Returns: (batch, seq_len, input_dim)  — next-step probabilities.

        During training, pass x = batch[:, :-1, :] and compare the output
        against batch[:, 1:, :] (teacher-forcing / next-step prediction).
        """
        seq_len = x.size(1)
        device  = x.device

        # 1. Embed input frames into d_model space
        x = self.input_projection(x)                        # (batch, seq_len, d_model)

        # 2. Add positional encoding
        x = self.pos_encoding(x)                            # (batch, seq_len, d_model)

        # 3. Build causal mask for this sequence length
        causal_mask = self._make_causal_mask(seq_len, device)   # (seq_len, seq_len)

        # 4. Pass through Transformer encoder with causal masking
        x = self.transformer(x, mask=causal_mask, is_causal=True)  # (batch, seq_len, d_model)

        # 5. Project to output dimension and squash to [0, 1]
        out = self.sigmoid(self.output_layer(x))            # (batch, seq_len, input_dim)
        return out
