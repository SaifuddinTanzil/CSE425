import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        seq_len = x.size(1)
        _, (hidden, cell) = self.encoder(x)
        z = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(z)
        reconstruction = torch.sigmoid(self.output_layer(decoded))
        return reconstruction
