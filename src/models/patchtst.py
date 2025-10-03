# src/models/patchtst.py

import torch
import torch.nn as nn
import math


class PatchTST(nn.Module):
    """
    PatchTST: Patching Time Series Transformer
    Divides time series into patches like Vision Transformer
    Much faster and better than standard Transformer
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_instance_norm: bool = True
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.use_instance_norm = use_instance_norm
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Instance normalization (important for non-stationary time series)
        if use_instance_norm:
            self.instance_norm = nn.InstanceNorm1d(input_size)
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * input_size, d_model)
        
        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_targets)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Instance normalization (per channel)
        if self.use_instance_norm:
            x = x.transpose(1, 2)  # [batch, features, seq_len]
            x = self.instance_norm(x)
            x = x.transpose(1, 2)  # [batch, seq_len, features]
        
        # Create patches
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]  # [batch, patch_len, features]
            patch = patch.reshape(batch_size, -1)  # [batch, patch_len * features]
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len * features]
        
        # Patch embedding
        x = self.patch_embedding(patches)  # [batch, num_patches, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer
        x = self.transformer(x)  # [batch, num_patches, d_model]
        
        # Output projection
        output = self.output_proj(x)
        
        return output


class PatchTSTWithChannelIndependence(nn.Module):
    """
    PatchTST with Channel Independence (CI)
    Each channel (feature) is processed independently
    Better for multivariate time series with many variables
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        
        # Number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Per-channel instance normalization
        self.instance_norm = nn.InstanceNorm1d(1)
        
        # Patch embedding (shared across channels)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
        
        # Transformer (shared across channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Channel aggregation and output
        self.channel_mix = nn.Sequential(
            nn.Linear(input_size * self.num_patches * d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_targets)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Process each channel independently
        channel_outputs = []
        
        for ch in range(features):
            # Get single channel
            ch_data = x[:, :, ch:ch+1]  # [batch, seq_len, 1]
            
            # Instance normalization
            ch_data = ch_data.transpose(1, 2)  # [batch, 1, seq_len]
            ch_data = self.instance_norm(ch_data)
            ch_data = ch_data.transpose(1, 2).squeeze(-1)  # [batch, seq_len]
            
            # Create patches
            patches = []
            for i in range(self.num_patches):
                start = i * self.stride
                end = start + self.patch_len
                patch = ch_data[:, start:end]  # [batch, patch_len]
                patches.append(patch)
            
            patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len]
            
            # Patch embedding
            ch_embedded = self.patch_embedding(patches)  # [batch, num_patches, d_model]
            
            # Add positional encoding
            ch_embedded = ch_embedded + self.pos_encoding
            
            # Transformer
            ch_out = self.transformer(ch_embedded)  # [batch, num_patches, d_model]
            
            channel_outputs.append(ch_out)
        
        # Stack and flatten all channels
        all_channels = torch.stack(channel_outputs, dim=1)  # [batch, features, num_patches, d_model]
        all_channels = all_channels.reshape(batch_size, -1)
        
        # Mix channels and output
        output = self.channel_mix(all_channels)
        
        return output


class CrossFormer(nn.Module):
    """
    CrossFormer: Cross-Dimension Dependency
    Captures dependencies across time and variables simultaneously
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        seg_len: int = 12,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seg_len = seg_len
        self.num_segs = seq_len // seg_len
        
        # Segment embedding
        self.seg_embedding = nn.Linear(seg_len * input_size, d_model)
        
        # Two-stage attention
        # Stage 1: Dimension-segment attention (DSA)
        self.dsa_blocks = nn.ModuleList([
            DSABlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Stage 2: Time-segment attention (TSA)
        self.tsa_blocks = nn.ModuleList([
            TSABlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_segs * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_targets)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Segment the time series
        x = x.reshape(batch_size, self.num_segs, self.seg_len, features)
        x = x.reshape(batch_size, self.num_segs, -1)
        
        # Segment embedding
        x = self.seg_embedding(x)  # [batch, num_segs, d_model]
        
        # Two-stage attention
        for dsa, tsa in zip(self.dsa_blocks, self.tsa_blocks):
            x = dsa(x)
            x = tsa(x)
        
        return self.output(x)


class DSABlock(nn.Module):
    """Dimension-Segment Attention"""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + self.dropout(attn_out)


class TSABlock(nn.Module):
    """Time-Segment Attention"""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # FFN
        x = x + self.ffn(self.norm(x))
        
        return x


if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_size = 150
    num_targets = 424
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing PatchTST...")
    model1 = PatchTST(input_size, seq_len, num_targets)
    out1 = model1(x)
    print(f"Output: {out1.shape}")
    print(f"Params: {sum(p.numel() for p in model1.parameters()):,}\n")
    
    print("Testing PatchTST with Channel Independence...")
    model2 = PatchTSTWithChannelIndependence(input_size, seq_len, num_targets)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}\n")
    
    print("Testing CrossFormer...")
    model3 = CrossFormer(input_size, seq_len, num_targets)
    out3 = model3(x)
    print(f"Output: {out3.shape}")
    print(f"Params: {sum(p.numel() for p in model3.parameters()):,}")