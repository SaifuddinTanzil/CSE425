import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.vae import LSTM_VAE


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: beta is intentionally passed by the caller each step so that KL
# Annealing can vary it per-epoch without touching this function.
def vae_loss(reconstruction, x, mu, log_var, beta):
    """
    Combined VAE loss with KL Annealing support:
        L_VAE = L_recon + beta * D_KL

    L_recon  : MSE between reconstruction and original input.
    D_KL     : Kullback-Leibler divergence from the unit Gaussian prior.
               Closed-form: -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
    beta     : Annealed weight supplied by the training loop. Starts at 0.0
               and ramps to its target value so the model learns reconstruction
               first, preventing posterior collapse.
    """
    # Reconstruction loss — mean over every element in the batch
    recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')

    # D_KL = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_vae(
    data_path,
    epochs=10,
    batch_size=128,
    learning_rate=1e-3,
    beta=0.1,
    input_dim=128,
    hidden_dim=256,
    latent_dim=128,
    num_layers=2,
):
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading processed dataset...")
    dataset    = np.load(data_path)                         # (N, 256, 128)
    tensor_data = torch.tensor(dataset, dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(
        tensor_data, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda')
    )
    print(f"Dataset loaded: {dataset.shape[0]} sequences of shape {dataset.shape[1:]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LSTM_VAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    # KL Annealing schedule: beta ramps linearly from 0.0 → target beta over
    # the first `anneal_epochs` epochs.  This prevents posterior collapse by
    # letting the model focus on reconstruction before the KL penalty activates.
    anneal_epochs = 10   # matches total epochs so beta reaches 0.1 by the final epoch
    loss_history  = []   # combined L_VAE per epoch

    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl    = 0.0

        # KL Annealing: slowly increase beta from 0.0 to `beta` over the first
        # `anneal_epochs` epochs, then hold it constant.
        current_beta = min(beta, (epoch / anneal_epochs) * beta)

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass — VAE returns reconstruction, mu, log_var
            reconstruction, mu, log_var = model(batch)

            # 1. Reconstruction Loss (MSE)
            recon_loss = criterion(reconstruction, batch)

            # 2. KL Divergence Loss
            # D_KL = -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            # 3. Total Loss with Annealed Beta
            loss = recon_loss + (current_beta * kl_loss)

            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl    += kl_loss.item()

        n_batches = len(dataloader)
        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl    / n_batches
        loss_history.append(avg_total)

        print(
            f"Epoch [{epoch+1:>3}/{epochs}] | "
            f"Beta: {current_beta:.4f} | "
            f"Total Loss: {avg_total:.4f} | "
            f"Recon Loss: {avg_recon:.4f} | "
            f"KL Loss: {avg_kl:.4f}"
        )

    # ── Save weights ──────────────────────────────────────────────────────────
    os.makedirs('./src/models/saved_weights', exist_ok=True)
    weights_path = './src/models/saved_weights/lstm_vae.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"\nModel weights saved → {weights_path}")

    # ── Save loss history ──────────────────────────────────────────────────────
    os.makedirs('./outputs/plots', exist_ok=True)
    loss_path = './outputs/plots/vae_loss_history.npy'
    np.save(loss_path, np.array(loss_history))
    print(f"Loss history saved  → {loss_path}")

    print("\nTraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_FILE = "./data/processed/clean_midi_dataset.npy"
    if os.path.exists(DATA_FILE):
        train_vae(
            data_path=DATA_FILE,
            epochs=10,
            batch_size=128,
            learning_rate=1e-3,
            beta=0.1,
        )
    else:
        print(f"Error: Could not find {DATA_FILE}. Run preprocessing first.")
