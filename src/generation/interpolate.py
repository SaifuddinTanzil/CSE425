import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.vae import LSTM_VAE
from preprocessing.midi_export import pianoroll_to_midi


def interpolate_latent(
    model_weights,
    dataset_path,
    output_dir,
    num_steps=5,
    latent_dim=128,
    seq_len=256,
    input_dim=128,
    hidden_dim=256,
    num_layers=2,
    threshold=0.5,
    fs=4,
):
    """
    Perform linear interpolation in the VAE latent space between two songs.

    Steps
    -----
    1. Load two random sequences from the dataset.
    2. Encode each sequence with the trained encoder to obtain mu_1 and mu_2
       (the mean of each posterior distribution – a deterministic representation).
    3. Linearly interpolate between mu_1 and mu_2 in `num_steps` steps
       (inclusive of endpoints so you can observe the full transition).
    4. Decode each interpolated latent vector and export as a MIDI file.

    Parameters
    ----------
    model_weights : str   Path to the saved lstm_vae.pth weights file.
    dataset_path  : str   Path to clean_midi_dataset.npy.
    output_dir    : str   Directory where interp_step_*.mid files are saved.
    num_steps     : int   Number of interpolation steps (default 5).
    latent_dim    : int   Must match value used during training.
    seq_len       : int   Time-steps in each sequence (must match dataset).
    input_dim     : int   Piano-roll pitch bins (128 MIDI notes).
    hidden_dim    : int   Must match value used during training.
    num_layers    : int   Must match value used during training.
    threshold     : float Binarisation threshold (default 0.5).
    fs            : int   Frames-per-second for MIDI export.
    """
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load trained model ────────────────────────────────────────────────────
    print(f"Loading weights from: {model_weights}")
    model = LSTM_VAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # ── Load dataset and pick two random sequences ────────────────────────────
    print(f"Loading dataset from: {dataset_path}")
    dataset = np.load(dataset_path)                         # (N, 256, 128)
    N       = dataset.shape[0]

    idx_1, idx_2 = np.random.choice(N, size=2, replace=False)
    print(f"Selected sequence indices: {idx_1} and {idx_2}")

    seq_1 = torch.tensor(dataset[idx_1], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 256, 128)
    seq_2 = torch.tensor(dataset[idx_2], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 256, 128)

    # ── Encode both sequences → get deterministic latent means ───────────────
    with torch.no_grad():
        mu_1, _ = model.encode(seq_1)   # (1, latent_dim)
        mu_2, _ = model.encode(seq_2)   # (1, latent_dim)

    print(f"Encoded both sequences. Latent dim: {mu_1.shape[-1]}")

    # ── Linear interpolation ──────────────────────────────────────────────────
    # alphas range from 0.0 (= mu_1) to 1.0 (= mu_2) in num_steps equal steps.
    # z(alpha) = (1 - alpha) * mu_1  +  alpha * mu_2
    alphas = np.linspace(0.0, 1.0, num_steps)

    print(f"Performing {num_steps}-step linear interpolation from sequence {idx_1} → {idx_2}...")

    os.makedirs(output_dir, exist_ok=True)

    for step_idx, alpha in enumerate(alphas):
        # Interpolated latent vector
        z_interp = (1.0 - alpha) * mu_1 + alpha * mu_2    # (1, latent_dim)

        with torch.no_grad():
            output = model.decode(z_interp, seq_len=seq_len)  # (1, seq_len, input_dim)

        piano_roll = output[0].cpu().numpy()               # (seq_len, input_dim)
        binarized  = (piano_roll >= threshold).astype(float)

        output_file = os.path.join(output_dir, f'interp_step_{step_idx + 1}.mid')
        pianoroll_to_midi(binarized, output_file, fs=fs)
        print(f"  Step {step_idx + 1}/{num_steps} (alpha={alpha:.2f}) → {output_file}")

    print(
        f"\nInterpolation complete. {num_steps} MIDI files saved to '{output_dir}'.\n"
        f"  interp_step_1.mid  ←  most similar to sequence {idx_1}\n"
        f"  interp_step_{num_steps}.mid  ←  most similar to sequence {idx_2}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WEIGHTS_PATH = './src/models/saved_weights/lstm_vae.pth'
    DATA_FILE    = './data/processed/clean_midi_dataset.npy'
    OUTPUT_DIR   = './outputs/generated_midis/'

    missing = [p for p in [WEIGHTS_PATH, DATA_FILE] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: Could not find '{p}'.")
        print("Please run train_vae.py before running interpolate.py.")
    else:
        interpolate_latent(
            model_weights=WEIGHTS_PATH,
            dataset_path=DATA_FILE,
            output_dir=OUTPUT_DIR,
            num_steps=5,
        )
