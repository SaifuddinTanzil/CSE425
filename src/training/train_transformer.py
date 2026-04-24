import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.transformer import MusicTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_transformer(
    data_path,
    epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
):
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading processed dataset...")
    dataset     = np.load(data_path)                         # (N, 256, 128)
    tensor_data = torch.tensor(dataset, dtype=torch.float32)
    dataloader  = torch.utils.data.DataLoader(
        tensor_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"Dataset loaded: {dataset.shape[0]} sequences of shape {dataset.shape[1:]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MusicTransformer(
        input_dim=128,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    # Count and display trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    # Binary Cross-Entropy is the correct loss here: every one of the 128
    # piano-roll bins is an independent binary event (note ON / OFF).
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────────
    # Next-step prediction (teacher forcing):
    #   Input  X = batch[:, :-1, :]   (steps 0 … T-2)
    #   Target Y = batch[:, 1:,  :]   (steps 1 … T-1)
    # The model sees step t and must predict step t+1.
    # The causal mask in the Transformer ensures no future leakage.
    loss_history = []

    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)                        # (B, 256, 128)

            # Teacher-forcing split
            X = batch[:, :-1, :]                           # (B, 255, 128)
            Y = batch[:, 1:,  :]                           # (B, 255, 128)

            optimizer.zero_grad()
            predictions = model(X)                         # (B, 255, 128)

            loss = criterion(predictions, Y)
            loss.backward()

            # Gradient clipping — helps stability with Transformer training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1:>3}/{epochs}] | BCE Loss: {avg_loss:.4f}")

    # ── Save weights ──────────────────────────────────────────────────────────
    os.makedirs('./src/models/saved_weights', exist_ok=True)
    weights_path = './src/models/saved_weights/transformer.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"\nModel weights saved → {weights_path}")

    # ── Save loss history ──────────────────────────────────────────────────────
    os.makedirs('./outputs/plots', exist_ok=True)
    loss_path = './outputs/plots/transformer_loss.npy'
    np.save(loss_path, np.array(loss_history))
    print(f"Loss history saved  → {loss_path}")

    print("\nTraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_FILE = "./data/processed/clean_midi_dataset.npy"
    if os.path.exists(DATA_FILE):
        train_transformer(
            data_path=DATA_FILE,
            epochs=10,
            batch_size=64,
            learning_rate=1e-3,
        )
    else:
        print(f"Error: Could not find {DATA_FILE}. Run preprocessing first.")
