import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.autoencoder import LSTMAutoencoder

def train_autoencoder(data_path, epochs=50, batch_size=32, learning_rate=0.001):
    print("Loading processed dataset...")
    dataset = np.load(data_path)
    tensor_data = torch.tensor(dataset, dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(tensor_data, batch_size=batch_size, shuffle=True)
    
    model = LSTMAutoencoder(input_dim=128, hidden_dim=64)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] | Reconstruction Loss: {avg_loss:.4f}")
        
    os.makedirs('./src/models/saved_weights', exist_ok=True)
    torch.save(model.state_dict(), './src/models/saved_weights/lstm_autoencoder.pth')
    np.save('./outputs/plots/ae_loss_history.npy', loss_history)
    print("Training complete. Weights and loss history saved.")

if __name__ == "__main__":
    DATA_FILE = "./data/processed/clean_midi_dataset.npy" 
    if os.path.exists(DATA_FILE):
        train_autoencoder(DATA_FILE, epochs=20) 
    else:
        print(f"Error: Could not find {DATA_FILE}. Run preprocessing first.")
