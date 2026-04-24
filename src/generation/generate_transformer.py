import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.transformer import MusicTransformer
from preprocessing.midi_export import pianoroll_to_midi


def generate_transformer_samples(
    model_weights,
    dataset_path,
    output_dir,
    num_samples=5,
    seed_steps=10,
    target_steps=256,
    threshold=0.5,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    fs=4,
):
    """
    Autoregressively generate music sequences using the trained Transformer.

    Generation algorithm (one sample)
    ──────────────────────────────────
    1. Pick a random sequence from the dataset and take its first `seed_steps`
       time steps as the initial context. This "primes" the model with real
       musical material so the first prediction is grounded in real data rather
       than pure noise.
    2. Pass the current context through the model.
       Only the LAST predicted time step is kept (index -1).
    3. Threshold at 0.5 to binarise → append to the context sequence.
    4. Repeat steps 2-3 until the context is `target_steps` long.
    5. Export the generated portion (everything after the seed) as MIDI.

    Parameters
    ──────────
    model_weights   : str   Path to transformer.pth.
    dataset_path    : str   Path to clean_midi_dataset.npy (for seeding).
    output_dir      : str   Where to write .mid files.
    num_samples     : int   Number of distinct MIDI files to generate.
    seed_steps      : int   How many real steps to use as the initial prompt.
    target_steps    : int   Total desired sequence length (seed + generated).
    threshold       : float Binarisation threshold applied to model output.
    fs              : int   Frames-per-second for pianoroll_to_midi.
    """
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load trained model ────────────────────────────────────────────────────
    print(f"Loading weights from: {model_weights}")
    model = MusicTransformer(
        input_dim=128,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    ).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # ── Load dataset for seed sequences ───────────────────────────────────────
    print(f"Loading dataset for seed sequences: {dataset_path}")
    dataset = np.load(dataset_path)                          # (N, 256, 128)
    N = dataset.shape[0]

    os.makedirs(output_dir, exist_ok=True)

    steps_to_generate = target_steps - seed_steps

    print(f"\nGenerating {num_samples} samples "
          f"(seed={seed_steps} steps + {steps_to_generate} autoregressive steps)...")

    for sample_idx in range(num_samples):
        # ── Pick a random seed from the dataset ───────────────────────────────
        rand_idx = np.random.randint(0, N)
        seed     = dataset[rand_idx, :seed_steps, :]         # (seed_steps, 128)

        # Context window: shape (1, current_len, 128) — batch dim included
        context = torch.tensor(seed, dtype=torch.float32).unsqueeze(0).to(device)

        # ── Autoregressive loop ────────────────────────────────────────────────
        with torch.no_grad():
            for _ in range(steps_to_generate):
                # Forward pass over the entire current context
                # Output shape: (1, context_len, 128)
                output = model(context)

                # Take only the very last predicted step
                next_step_probs = output[:, -1:, :]          # (1, 1, 128)

                # Binarise at threshold
                next_step = (next_step_probs >= threshold).float()  # (1, 1, 128)

                # Append to context for the next iteration
                context = torch.cat([context, next_step], dim=1)    # (1, len+1, 128)

        # ── Extract and export ─────────────────────────────────────────────────
        # context is now (1, target_steps, 128)
        # We export the full sequence including the seed so the MIDI starts
        # coherently; remove the [:, seed_steps:, :] slice if you want only
        # the purely generated portion.
        full_sequence = context[0].cpu().numpy()             # (target_steps, 128)

        output_file = os.path.join(output_dir, f'transformer_sample_{sample_idx + 1}.mid')
        pianoroll_to_midi(full_sequence, output_file, fs=fs)
        print(f"  [{sample_idx + 1}/{num_samples}] Saved: {output_file}")

    print(f"\nGeneration complete. {num_samples} MIDI files saved to '{output_dir}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WEIGHTS_PATH = './src/models/saved_weights/transformer.pth'
    DATA_FILE    = './data/processed/clean_midi_dataset.npy'
    OUTPUT_DIR   = './outputs/generated_midis/'

    missing = [p for p in [WEIGHTS_PATH, DATA_FILE] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: Could not find '{p}'.")
        print("Please run train_transformer.py before running generate_transformer.py.")
    else:
        generate_transformer_samples(
            model_weights=WEIGHTS_PATH,
            dataset_path=DATA_FILE,
            output_dir=OUTPUT_DIR,
            num_samples=5,
            seed_steps=10,
            target_steps=256,
        )
