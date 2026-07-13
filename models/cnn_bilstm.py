"""
cnn_bilstm.py — Mubarak et al. CNN-BiLSTM-AR Architecture
==========================================================
State-of-the-art benchmark model combining CNN, BiLSTM, and AR branches.
"""

import torch
import torch.nn as nn

class CNNBiLSTM_AR(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_quantiles: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 2. BiLSTM Temporal Layer
        # Note: after pooling, sequence length is halved (if window=168, pool=2 -> 84)
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                            num_layers=1, bidirectional=True, batch_first=True)
        
        # 3. AR Branch (Linear skip connection)
        # We use the most recent feature vector (T-1) as input to AR
        self.ar_linear = nn.Linear(input_dim, n_quantiles)
        
        # 4. Dense Fusion Head
        # BiLSTM output concats forward/backward hidden states (2 * hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, n_quantiles)
        
    def forward(self, x):
        """
        x: [Batch, Time, Features]
        """
        # Save last feature vector for AR branch
        x_last = x[:, -1, :] # [Batch, Features]
        
        # --- CNN Path ---
        # Conv1d expects [Batch, Channels, Length]
        x_cnn = x.transpose(1, 2) 
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.pool(x_cnn)
        
        # --- BiLSTM Path ---
        # LSTM expects [Batch, Length, hidden_dim]
        x_lstm = x_cnn.transpose(1, 2)
        lstm_out, _ = self.bilstm(x_lstm)
        
        # Use the last hidden state of BiLSTM
        lstm_final = lstm_out[:, -1, :]
        
        # --- Fusion ---
        out_deep = self.fc(lstm_final) # [Batch, n_quantiles]
        out_ar = self.ar_linear(x_last) # [Batch, n_quantiles]
        
        # Residual fusion: Deep correction to the AR baseline
        # (This is a robust way to implement the AR+DL hybrid)
        final_out = out_ar + out_deep
        
        return final_out

if __name__ == '__main__':
    # Shape test
    model = CNNBiLSTM_AR(input_dim=10)
    test_input = torch.randn(16, 168, 10) # Batch=16, Window=168, Features=10
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [16, 7])")
