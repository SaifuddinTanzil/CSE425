import torch
import torch.nn as nn


class LSTM_VAE(nn.Module):
    """
    LSTM-based Variational Autoencoder for piano-roll sequences.

    Input shape : (batch, seq_len=256, input_dim=128)
    Latent shape: (batch, latent_dim)
    Output shape: (batch, seq_len=256, input_dim=128)  -- sigmoid activated
    """

    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128, num_layers=2):
        super(LSTM_VAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers  = num_layers
        self.latent_dim  = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        # Processes the full sequence and summarises it in the final hidden state.
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        # Two separate linear heads that project the last hidden state to the
        # mean and log-variance of the latent Gaussian.
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Projects z back to the LSTM hidden / cell state dimension, then
        # unrolls across seq_len time steps.
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    # ─────────────────────────────────────────────────────────────────────────
    # Encoder
    # ─────────────────────────────────────────────────────────────────────────
    def encode(self, x):
        """
        x : (batch, seq_len, input_dim)
        Returns mu, log_var each of shape (batch, latent_dim).
        """
        _, (hidden, _) = self.encoder_lstm(x)
        # hidden : (num_layers, batch, hidden_dim)
        #  → take only the top layer's hidden state
        h = hidden[-1]                   # (batch, hidden_dim)

        mu      = self.fc_mu(h)          # (batch, latent_dim)
        log_var = self.fc_log_var(h)     # (batch, latent_dim)
        return mu, log_var

    # ─────────────────────────────────────────────────────────────────────────
    # Reparameterization trick
    # ─────────────────────────────────────────────────────────────────────────
    def reparameterize(self, mu, log_var):
        """
        During training, sample z = mu + std * eps  where eps ~ N(0, I).
        During evaluation the model is in deterministic mode; we still use the
        trick so that callers can explicitly pass mu to get the mean prediction.
        """
        std = torch.exp(0.5 * log_var)   # σ  (batch, latent_dim)
        eps = torch.randn_like(std)       # ε ~ N(0, I)
        z   = mu + std * eps              # (batch, latent_dim)
        return z

    # ─────────────────────────────────────────────────────────────────────────
    # Decoder
    # ─────────────────────────────────────────────────────────────────────────
    def decode(self, z, seq_len=256):
        """
        z       : (batch, latent_dim)
        seq_len : number of time steps to reconstruct
        Returns reconstruction of shape (batch, seq_len, input_dim).
        """
        batch_size = z.size(0)

        # Repeat z across all time steps to form the decoder input sequence.
        # Shape: (batch, seq_len, latent_dim)
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)

        # Initialise the decoder hidden state from the latent vector so that
        # the decoder has access to z both as input and as hidden state.
        h0 = torch.tanh(self.fc_decode(z))           # (batch, hidden_dim)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        c0 = torch.zeros_like(h0)                    # zero cell state

        decoded, _ = self.decoder_lstm(z_seq, (h0, c0))  # (batch, seq_len, hidden_dim)
        reconstruction = torch.sigmoid(self.output_layer(decoded))  # (batch, seq_len, input_dim)
        return reconstruction

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        x : (batch, seq_len, input_dim)
        Returns reconstruction, mu, log_var.
        """
        seq_len          = x.size(1)
        mu, log_var      = self.encode(x)
        z                = self.reparameterize(mu, log_var)
        reconstruction   = self.decode(z, seq_len=seq_len)
        return reconstruction, mu, log_var
