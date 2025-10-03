# src/models/ft_transformer.py

import torch
import torch.nn as nn
import math


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer
    Each feature becomes a token - excellent for tabular/financial data
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        d_token: int = 96,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        num_targets: int = 424
    ):
        super().__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_token = d_token
        
        # Feature tokenization - each feature becomes a learnable embedding
        self.feature_tokenizer = nn.Linear(1, d_token)
        
        # [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Positional encoding for time steps
        self.positional_encoding = nn.Parameter(
            torch.randn(1, seq_len * input_size + 1, d_token)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=attention_n_heads,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout
            )
            for _ in range(n_blocks)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, num_targets)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Reshape to [batch, seq_len * features, 1]
        x = x.reshape(batch_size, seq_len * features, 1)
        
        # Tokenize each feature
        x = self.feature_tokenizer(x)  # [batch, seq_len * features, d_token]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Use CLS token for prediction
        cls_output = x[:, 0]
        
        return self.head(cls_output)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float
    ):
        super().__init__()
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(d_token)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        self.dropout1 = nn.Dropout(residual_dropout)
        
        # FFN
        self.norm2 = nn.LayerNorm(d_token)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token * 4, d_token),
            nn.Dropout(ffn_dropout)
        )
        
        self.dropout2 = nn.Dropout(residual_dropout)
    
    def forward(self, x):
        # Attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        
        # FFN with residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)
        
        return x


class LinearAttention(nn.Module):
    """
    Linear Attention - O(n) complexity instead of O(n²)
    Good for long sequences
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Linear attention: compute feature map
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute attention in linear time
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        # Normalization
        z = 1 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)
        x = qkv * z.unsqueeze(-1)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class PerformerModel(nn.Module):
    """
    Performer - Fast attention approximation
    Linear complexity, good for long sequences
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        num_targets: int = 424,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # Performer layers (using linear attention)
        self.layers = nn.ModuleList([
            PerformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_targets)
        )
    
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Performer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        x = self.norm(x)
        return self.output(x)


class PerformerLayer(nn.Module):
    """Single Performer layer"""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinearAttention(d_model, n_heads, attn_drop=dropout, proj_drop=dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Attention
        x = x + self.attn(self.norm1(x))
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    import torch.nn.functional as F
    
    batch_size = 32
    seq_len = 60
    input_size = 150
    num_targets = 424
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing FT-Transformer...")
    model1 = FTTransformer(input_size, seq_len, num_targets=num_targets)
    out1 = model1(x)
    print(f"Output: {out1.shape}")
    print(f"Params: {sum(p.numel() for p in model1.parameters()):,}\n")
    
    print("Testing Performer...")
    model2 = PerformerModel(input_size, seq_len, num_targets=num_targets)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}")