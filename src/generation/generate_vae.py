import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.vae import LSTM_VAE
from preprocessing.midi_export import pianoroll_to_midi


def generate_vae_samples(
    model_weights,
    output_dir,
    num_samples=8,
    latent_dim=128,
    seq_len=256,
    input_dim=128,
    hidden_dim=256,
    num_layers=2,
    threshold=0.5,
    fs=4,
):
    """
    Generate novel music by sampling random latent vectors z ~ N(0, I) and
    passing them through the trained VAE decoder.

    Parameters
    ----------
    model_weights : str   Path to the saved lstm_vae.pth weights file.
    output_dir    : str   Directory where .mid files will be written.
    num_samples   : int   Number of sequences to generate (default 8).
    latent_dim    : int   Must match the value used during training.
    seq_len       : int   Number of time steps (default 256 to match dataset).
    input_dim     : int   Piano-roll pitch bins (128 MIDI notes).
    hidden_dim    : int   Must match the value used during training.
    num_layers    : int   Must match the value used during training.
    threshold     : float Binarisation threshold applied to decoder output.
    fs            : int   Frames-per-second used by pianoroll_to_midi.
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

    # ── Sample latent vectors from the prior N(0, I) ─────────────────────────
    # This is the key difference from the autoencoder: we do NOT need any real
    # data input – we generate purely from the learned latent space.
    z = torch.randn(num_samples, latent_dim).to(device)   # (num_samples, latent_dim)

    print(f"Generating {num_samples} novel sequences from random latent vectors...")
    with torch.no_grad():
        # Pass through decoder only (no encoder, no real data)
        output = model.decode(z, seq_len=seq_len)   # (num_samples, seq_len, input_dim)

    # ── Binarise and export ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        piano_roll = output[i].cpu().numpy()              # (seq_len, input_dim)
        binarized  = (piano_roll >= threshold).astype(float)

        output_file = os.path.join(output_dir, f'vae_sample_{i + 1}.mid')
        pianoroll_to_midi(binarized, output_file, fs=fs)
        print(f"  Saved: {output_file}")

    print(f"\nGeneration complete. {num_samples} MIDI files saved to '{output_dir}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WEIGHTS_PATH = './src/models/saved_weights/lstm_vae.pth'
    OUTPUT_DIR   = './outputs/generated_midis/'

    if os.path.exists(WEIGHTS_PATH):
        generate_vae_samples(
            model_weights=WEIGHTS_PATH,
            output_dir=OUTPUT_DIR,
            num_samples=8,
        )
    else:
        print(
            f"Error: Could not find weights at '{WEIGHTS_PATH}'.\n"
            "Please run train_vae.py first."
        )
