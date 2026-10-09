"""
Neural network model implementations and utilities.

This module provides:
- RNN, LSTM, BiLSTM, and GRU classifier models with optional attention
- Transformer classifier model
- Sequence dataset for PyTorch DataLoader
- Model configuration constants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


# Model configuration constants
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

TRAIN_EPOCHS = 10
TRAIN_BATCH = 512
EVAL_BATCH = 512
LR = 1e-3

# Transformer configuration constants (shares HIDDEN_SIZE, NUM_LAYERS, DROPOUT with other models)
TRANSFORMER_NUM_HEADS = 4      # Number of attention heads (Transformer-specific)
TRANSFORMER_MAX_LEN = 36       # Maximum sequence length for positional encoding (should >= T)

# CNN configuration constants (shares HIDDEN_SIZE, DROPOUT with other models)
CNN_NUM_FILTERS = 64           # Number of filters (channels) in conv layers (CNN-specific)
CNN_KERNEL_SIZE = 3            # Convolution kernel size (CNN-specific)
CNN_NUM_CONV_LAYERS = 2        # Number of convolutional blocks (CNN-specific)
CNN_POOL_SIZE = 2              # Pooling window size (CNN-specific)
CNN_POOL_MODE = 'avg'          # Pooling mode: 'max' | 'avg' (CNN-specific)

# FNN configuration constants
FNN_FLATTEN_MODE = 'mean'      # Flatten mode: 'last' | 'flatten' | 'mean' | 'max' (FNN-specific)
# - 'last': Use only last time step [B, T, F] -> [B, F] (most recent info, recommended for prediction)
# - 'flatten': Flatten entire sequence [B, T, F] -> [B, T*F] (preserves all temporal info)
# - 'mean': Average pooling over time [B, T, F] -> [B, F] (smooth features)
# - 'max': Max pooling over time [B, T, F] -> [B, F] (extract key features)


class RNNClassifier(nn.Module):
    """
    Basic RNN classifier (SimpleRNN).
    Uses tanh activation by default.
    """
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT, use_attention: bool = False):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            nonlinearity='tanh'  # 'tanh' or 'relu'
        )
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        
        if use_attention:
            # Linear layer for attention scores
            self.attn_linear = nn.Linear(hidden_size, 1)
            # Classification layer (binary)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2)
            )
        else:
            # Standard classification head
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2)
            )

    def forward(self, x):
        out, _ = self.rnn(x)  # [batch, seq_len, hidden_size]
        
        if self.use_attention:
            # Calculate attention scores for each time step
            attn_scores = self.attn_linear(out)  # [batch, seq_len, 1]
            # Softmax over the time dimension
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
            # Weighted sum of RNN outputs
            weighted = (out * attn_weights).sum(dim=1)  # [batch, hidden_size]
            logits = self.head(weighted)
        else:
            # Standard: take the last time step
            last = out[:, -1, :]  # [batch, hidden_size]
            logits = self.head(last)
        
        return logits


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT, use_attention: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        
        if use_attention:
            # Linear layer for attention scores
            self.attn_linear = nn.Linear(hidden_size, 1)
            # Classification layer (binary)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2)
            )
        else:
            # Standard classification head
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2)
            )

    def forward(self, x):
        out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        if self.use_attention:
            # Calculate attention scores for each time step
            attn_scores = self.attn_linear(out)  # [batch, seq_len, 1]
            # Softmax over the time dimension
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
            # Weighted sum of LSTM outputs
            weighted = (out * attn_weights).sum(dim=1)  # [batch, hidden_size]
            logits = self.head(weighted)
        else:
            # Standard: take the last time step
            last = out[:, -1, :]  # [batch, hidden_size]
            logits = self.head(last)
        
        return logits


class BiLSTMClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = HIDDEN_SIZE, 
        num_layers: int = NUM_LAYERS, 
        dropout: float = DROPOUT, 
        use_attention: bool = False,
        enable_deep_head: bool = False,
        head_hidden_size: int = 128,
        head_num_layers: int = 2,
        enable_head_batch_norm: bool = False,
        sequence_aggregation: str = 'last'
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.sequence_aggregation = sequence_aggregation
        
        # Determine input dimension for classification head based on aggregation mode
        lstm_output_dim = hidden_size * 2  # BiLSTM output dimension
        if sequence_aggregation == 'concat_last_mean':
            head_input_dim = lstm_output_dim * 2  # Concatenate last and mean
        else:
            head_input_dim = lstm_output_dim
        
        if use_attention:
            # Linear layer for attention scores (BiLSTM output is hidden_size * 2)
            self.attn_linear = nn.Linear(hidden_size * 2, 1)
        
        # Build classification head
        if enable_deep_head:
            # Deep MLP head with multiple layers
            head_layers = []
            current_dim = head_input_dim
            
            for i in range(head_num_layers):
                head_layers.append(nn.Linear(current_dim, head_hidden_size))
                if enable_head_batch_norm:
                    head_layers.append(nn.BatchNorm1d(head_hidden_size))
                head_layers.append(nn.ReLU())
                head_layers.append(nn.Dropout(dropout))
                current_dim = head_hidden_size
            
            # Output layer
            head_layers.append(nn.Linear(current_dim, 2))
            self.head = nn.Sequential(*head_layers)
        else:
            # Simple single-layer head
            if enable_head_batch_norm:
                self.head = nn.Sequential(
                nn.Dropout(dropout),
                    nn.BatchNorm1d(head_input_dim),
                    nn.Linear(head_input_dim, 2),
            )
            else:
                self.head = nn.Sequential(
                nn.Dropout(dropout),
                    nn.Linear(head_input_dim, 2),
            )

    def _aggregate_sequence(self, out):
        """
        Aggregate LSTM output sequence into a single vector based on aggregation mode.
        out: [batch, seq_len, hidden_size * 2]
        returns: [batch, feature_dim] where feature_dim depends on aggregation mode
        """
        if self.use_attention:
            # Attention aggregation (overrides sequence_aggregation mode)
            attn_scores = self.attn_linear(out)  # [batch, seq_len, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
            weighted = (out * attn_weights).sum(dim=1)  # [batch, hidden_size * 2]
            return weighted
        elif self.sequence_aggregation == 'last':
            # Use only last time step
            return out[:, -1, :]  # [batch, hidden_size * 2]
        elif self.sequence_aggregation == 'mean':
            # Average pooling over time dimension
            return out.mean(dim=1)  # [batch, hidden_size * 2]
        elif self.sequence_aggregation == 'max':
            # Max pooling over time dimension
            return out.max(dim=1)[0]  # [batch, hidden_size * 2]
        elif self.sequence_aggregation == 'concat_last_mean':
            # Concatenate last time step and mean pooling
            last = out[:, -1, :]  # [batch, hidden_size * 2]
            mean = out.mean(dim=1)  # [batch, hidden_size * 2]
            return torch.cat([last, mean], dim=1)  # [batch, hidden_size * 4]
        else:
            raise ValueError(f"Unknown sequence_aggregation mode: {self.sequence_aggregation}")

    def forward(self, x):
        out, _ = self.lstm(x)  # [batch, seq_len, hidden_size * 2]
        
        # Aggregate sequence into feature vector
        features = self._aggregate_sequence(out)  # [batch, feature_dim]
        
        # Classification
        logits = self.head(features)  # [batch, 2]
        
        return logits


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT, use_attention: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        
        if use_attention:
            # Linear layer for attention scores
            self.attn_linear = nn.Linear(hidden_size, 1)
            # Classification layer (binary)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2),
            )
        else:
            # Standard classification head
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2),
            )

    def forward(self, x):
        out, _ = self.gru(x)  # [batch, seq_len, hidden_size]
        
        if self.use_attention:
            # Calculate attention scores for each time step
            attn_scores = self.attn_linear(out)  # [batch, seq_len, 1]
            # Softmax over the time dimension
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
            # Weighted sum of GRU outputs
            weighted = (out * attn_weights).sum(dim=1)  # [batch, hidden_size]
            logits = self.head(weighted)
        else:
            # Standard: take the last time step
            last = out[:, -1, :]  # [batch, hidden_size]
            logits = self.head(last)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer input.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for sequence classification (binary output).
    Shares HIDDEN_SIZE, NUM_LAYERS, and DROPOUT with other sequence models.
    """
    def __init__(
        self, 
        input_size: int, 
        num_heads: int = TRANSFORMER_NUM_HEADS, 
        num_layers: int = NUM_LAYERS, 
        hidden_dim: int = HIDDEN_SIZE, 
        dropout: float = DROPOUT, 
        max_len: int = TRANSFORMER_MAX_LEN,
        use_attention: bool = False  # For compatibility, but transformer has built-in attention
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.use_attention = use_attention  # Not used, but kept for compatibility

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.input_proj(x)                # [batch, seq_len, hidden_dim]
        x = self.pos_encoder(x)               # add positional encoding
        x = self.transformer(x)               # [batch, seq_len, hidden_dim]
        x = x[:, -1, :]                       # use the last time step's output
        x = self.dropout(x)
        return self.fc(x)                     # [batch, 2]


class FNNClassifier(nn.Module):
    """
    Feedforward Neural Network (MLP) classifier.
    For sequence inputs, flattens the sequence dimension.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = HIDDEN_SIZE, 
        num_layers: int = NUM_LAYERS, 
        dropout: float = DROPOUT,
        seq_len: int = 36,  # Sequence length T
        flatten_mode: str = 'flatten'  # 'flatten' | 'last' | 'mean' | 'max'
    ):
        super().__init__()
        self.seq_len = seq_len
        self.flatten_mode = flatten_mode
        
        # Determine input dimension based on flatten mode
        if flatten_mode == 'flatten':
            # Flatten entire sequence: [B, T, F] -> [B, T*F]
            mlp_input_size = seq_len * input_size
        elif flatten_mode in ['last', 'mean', 'max']:
            # Use single time step: [B, T, F] -> [B, F]
            mlp_input_size = input_size
        else:
            raise ValueError(f"Unknown flatten_mode: {flatten_mode}")
        
        # Build MLP layers
        layers = []
        current_size = mlp_input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, 2))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch_size = x.size(0)
        
        if self.flatten_mode == 'flatten':
            # Flatten entire sequence
            x = x.view(batch_size, -1)  # [batch, seq_len * input_size]
        elif self.flatten_mode == 'last':
            # Use last time step
            x = x[:, -1, :]  # [batch, input_size]
        elif self.flatten_mode == 'mean':
            # Average pooling over time
            x = x.mean(dim=1)  # [batch, input_size]
        elif self.flatten_mode == 'max':
            # Max pooling over time
            x = x.max(dim=1)[0]  # [batch, input_size]
        
        # Pass through MLP
        logits = self.mlp(x)  # [batch, 2]
        return logits


class CNNClassifier(nn.Module):
    """
    1D Convolutional Neural Network (CNN) classifier for sequence data.
    Uses 1D convolutions to extract local temporal patterns.
    """
    def __init__(
        self,
        input_size: int,
        num_filters: int = 64,  # Number of filters (channels) in conv layers
        kernel_size: int = 3,  # Convolution kernel size
        num_conv_layers: int = 2,  # Number of convolutional blocks
        pool_size: int = 2,  # Pooling window size
        pool_mode: str = 'max',  # 'max' | 'avg'
        hidden_size: int = HIDDEN_SIZE,  # FC layer size
        dropout: float = DROPOUT,
        seq_len: int = 36
    ):
        super().__init__()
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.seq_len = seq_len
        
        # Build convolutional blocks
        conv_layers = []
        in_channels = input_size  # Input: [B, input_size, seq_len]
        
        for i in range(num_conv_layers):
            # Conv1d: in_channels, out_channels, kernel_size
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding to preserve length
            ))
            conv_layers.append(nn.ReLU())
            
            # Pooling layer
            if pool_mode == 'max':
                conv_layers.append(nn.MaxPool1d(kernel_size=pool_size))
            elif pool_mode == 'avg':
                conv_layers.append(nn.AvgPool1d(kernel_size=pool_size))
            else:
                raise ValueError(f"Unknown pool_mode: {pool_mode}")
            
            conv_layers.append(nn.Dropout(dropout))
            in_channels = num_filters  # Next layer input channels
        
        self.conv_blocks = nn.Sequential(*conv_layers)
        
        # Calculate the output size after conv and pooling
        # After each pooling, seq_len is divided by pool_size
        # After num_conv_layers pooling operations: seq_len / (pool_size ** num_conv_layers)
        final_seq_len = seq_len // (pool_size ** num_conv_layers)
        if final_seq_len < 1:
            final_seq_len = 1
        
        # Global pooling + FC layers
        self.global_pool = nn.AdaptiveAvgPool1d(1) if pool_mode == 'avg' else nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_filters, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        # Convert to [batch, input_size, seq_len] for Conv1d
        x = x.transpose(1, 2)  # [batch, input_size, seq_len]
        
        # Apply convolutional blocks
        x = self.conv_blocks(x)  # [batch, num_filters, reduced_seq_len]
        
        # Global pooling: [batch, num_filters, reduced_seq_len] -> [batch, num_filters, 1]
        x = self.global_pool(x)  # [batch, num_filters, 1]
        
        # Flatten: [batch, num_filters, 1] -> [batch, num_filters]
        x = x.squeeze(-1)  # [batch, num_filters]
        
        # Fully connected layers
        logits = self.fc_layers(x)  # [batch, 2]
        return logits


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
