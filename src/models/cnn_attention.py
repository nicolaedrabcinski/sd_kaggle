# src/models/cnn_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAttention(nn.Module):
    """
    1D CNN with Multi-Head Attention for time series
    Best for capturing local patterns + global dependencies
    """
    def __init__(
        self,
        input_size: int,
        num_filters: int = 128,
        kernel_sizes: list = [3, 5, 7],
        num_heads: int = 8,
        num_targets: int = 424,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Multi-scale 1D Convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, num_filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])
        
        # Combine multi-scale features
        combined_size = num_filters * len(kernel_sizes)
        self.combine = nn.Sequential(
            nn.Conv1d(combined_size, num_filters, kernel_size=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU()
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.LayerNorm(num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_targets)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Multi-scale convolutions
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)  # [batch, combined_filters, seq_len]
        
        # Combine features
        x = self.combine(x)  # [batch, num_filters, seq_len]
        
        # Transpose back for attention: [batch, seq_len, num_filters]
        x = x.transpose(1, 2)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)  # [batch, num_filters]
        
        # Output
        return self.output(x)


class WaveNetStyleModel(nn.Module):
    """
    Dilated Causal Convolutions - excellent for time series
    Captures long-range dependencies efficiently
    """
    def __init__(
        self,
        input_size: int,
        residual_channels: int = 64,
        skip_channels: int = 128,
        dilation_layers: int = 8,
        num_targets: int = 424,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Input projection
        self.input_conv = nn.Conv1d(input_size, residual_channels, kernel_size=1)
        
        # Dilated causal convolutions
        self.dilated_blocks = nn.ModuleList()
        for i in range(dilation_layers):
            dilation = 2 ** i
            self.dilated_blocks.append(
                DilatedCausalBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(skip_channels, num_targets, kernel_size=1)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_conv(x)
        
        # Accumulate skip connections
        skip_connections = []
        
        for block in self.dilated_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Sum all skip connections
        skip_sum = torch.stack(skip_connections).sum(dim=0)
        
        # Output
        out = self.output_layers(skip_sum)
        
        # Take last timestep
        return out[:, :, -1]


class DilatedCausalBlock(nn.Module):
    """Single dilated causal convolution block"""
    def __init__(self, residual_channels, skip_channels, dilation, dropout):
        super().__init__()
        
        self.conv_filter = nn.Conv1d(
            residual_channels, residual_channels,
            kernel_size=2, dilation=dilation,
            padding=dilation
        )
        self.conv_gate = nn.Conv1d(
            residual_channels, residual_channels,
            kernel_size=2, dilation=dilation,
            padding=dilation
        )
        
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Gated activation
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        gated = filter_out * gate_out
        
        # Apply dropout
        gated = self.dropout(gated)
        
        # Skip connection
        skip = self.skip_conv(gated)
        
        # Residual connection
        residual = self.residual_conv(gated)
        
        # Match sizes (remove padding)
        if residual.size(2) > x.size(2):
            residual = residual[:, :, :x.size(2)]
        
        return x + residual, skip


class LightGBMStyleNN(nn.Module):
    """
    Neural network mimicking gradient boosting
    Multiple parallel branches with different depths
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_branches: int = 4,
        num_targets: int = 424,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple branches with different depths
        self.branches = nn.ModuleList([
            self._create_branch(hidden_size, depth=i+1, dropout=dropout)
            for i in range(num_branches)
        ])
        
        # Combine branches
        self.combiner = nn.Sequential(
            nn.Linear(hidden_size * num_branches, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_targets)
        )
        
    def _create_branch(self, hidden_size, depth, dropout):
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten sequence
        batch_size, seq_len, features = x.shape
        x = x.reshape(batch_size, -1)
        
        # Input embedding
        x = self.input_embed(x)
        
        # Process through branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate and combine
        combined = torch.cat(branch_outputs, dim=1)
        return self.combiner(combined)


# Factory function
def create_advanced_model(model_name: str, input_size: int, seq_len: int = 60, num_targets: int = 424):
    """Create advanced models"""
    
    models = {
        'cnn_attention': lambda: CNNAttention(
            input_size=input_size,
            num_filters=128,
            kernel_sizes=[3, 5, 7],
            num_heads=8,
            num_targets=num_targets,
            dropout=0.2
        ),
        'wavenet': lambda: WaveNetStyleModel(
            input_size=input_size,
            residual_channels=64,
            skip_channels=128,
            dilation_layers=8,
            num_targets=num_targets,
            dropout=0.2
        ),
        'lgbm_style': lambda: LightGBMStyleNN(
            input_size=input_size * seq_len,
            hidden_size=256,
            num_branches=4,
            num_targets=num_targets,
            dropout=0.2
        )
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


if __name__ == "__main__":
    # Test models
    batch_size = 32
    seq_len = 60
    input_size = 150
    num_targets = 424
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing CNNAttention...")
    model1 = CNNAttention(input_size, num_targets=num_targets)
    out1 = model1(x)
    print(f"Output shape: {out1.shape}")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    print("\nTesting WaveNetStyle...")
    model2 = WaveNetStyleModel(input_size, num_targets=num_targets)
    out2 = model2(x)
    print(f"Output shape: {out2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    print("\nTesting LightGBM Style...")
    model3 = LightGBMStyleNN(input_size * seq_len, num_targets=num_targets)
    out3 = model3(x)
    print(f"Output shape: {out3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")