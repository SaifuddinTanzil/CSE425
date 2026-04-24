import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.autoencoder import LSTMAutoencoder
from preprocessing.midi_export import pianoroll_to_midi

def generate_samples(model_weights, dataset_path, output_dir, num_samples=5):
    print("Loading model and data for generation...")
    model = LSTMAutoencoder(input_dim=128, hidden_dim=64)
    model.load_state_dict(torch.load(model_weights, weights_only=True))
    model.eval()
    
    dataset = np.load(dataset_path)
    
    # Randomly select num_samples indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = dataset[indices]
    
    tensor_samples = torch.tensor(samples, dtype=torch.float32)
    
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        reconstructions = model(tensor_samples)
        
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Apply 0.5 threshold
        binarized = (reconstructions[i].numpy() >= 0.5).astype(float)
        
        output_file = os.path.join(output_dir, f'sample_{i+1}.mid')
        pianoroll_to_midi(binarized, output_file, fs=4)
        print(f"Saved {output_file}")
    
    print("Generation complete!")

if __name__ == "__main__":
    generate_samples(
        model_weights='./src/models/saved_weights/lstm_autoencoder.pth',
        dataset_path='./data/processed/clean_midi_dataset.npy',
        output_dir='./outputs/generated_midis/'
    )
