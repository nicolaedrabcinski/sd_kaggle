# src/models/timesnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimesNet(nn.Module):
    """
    TimesNet: Temporal 2D-Variation Modeling
    Transforms 1D time series into 2D tensors to capture multi-periodicity
    SOTA for long-term forecasting
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        d_model: int = 64,
        d_ff: int = 128,
        num_kernels: int = 6,
        top_k: int = 3,
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_kernels = num_kernels
        self.top_k = top_k
        
        # Input embedding
        self.embed = nn.Linear(input_size, d_model)
        
        # TimesBlocks for multi-periodicity
        self.blocks = nn.ModuleList([
            TimesBlock(
                seq_len=seq_len,
                d_model=d_model,
                d_ff=d_ff,
                num_kernels=num_kernels,
                top_k=top_k,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, num_targets)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        
        # Embed
        x = self.embed(x)  # [batch, seq_len, d_model]
        
        # Process through TimesBlocks
        for block in self.blocks:
            x = block(x)
        
        # Take last timestep and project
        output = self.projection(x[:, -1, :])
        
        return output


class TimesBlock(nn.Module):
    """
    Single TimesBlock: FFT -> 2D Conv -> iFFT
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
        top_k: int,
        dropout: float
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.top_k = top_k
        
        # Parameter for period selection
        self.conv_kernels = nn.Parameter(
            torch.randn(1, num_kernels, 1)
        )
        
        # 2D Convolution for each period
        self.conv2d = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_ff,
                    kernel_size=(1, 3),
                    padding=(0, 1)
                ),
                nn.BatchNorm2d(d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(
                    in_channels=d_ff,
                    out_channels=d_model,
                    kernel_size=(1, 3),
                    padding=(0, 1)
                )
            )
            for _ in range(top_k)
        ])
        
        # Residual and normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size = x.shape[0]
        
        # FFT to find dominant periods
        x_freq = torch.fft.rfft(x, dim=1)
        freq_amplitudes = torch.abs(x_freq).mean(dim=-1)  # [batch, freq_bins]
        
        # Find top-k periods
        _, top_indices = torch.topk(freq_amplitudes, self.top_k, dim=1)
        
        # Convert frequency indices to periods
        periods = []
        for k in range(self.top_k):
            period_idx = top_indices[:, k]
            # Avoid zero period
            period_idx = torch.clamp(period_idx, min=1)
            period = (self.seq_len // period_idx).int()
            periods.append(period)
        
        # Process each period
        res = []
        
        for i, conv in enumerate(self.conv2d):
            period = periods[i]
            
            # Reshape to 2D: [batch, d_model, period, seq_len/period]
            reshaped = []
            for b in range(batch_size):
                p = period[b].item()
                if p == 0 or self.seq_len % p != 0:
                    p = self.seq_len  # Fallback
                
                # Pad if necessary
                pad_len = (p - self.seq_len % p) % p
                if pad_len > 0:
                    x_padded = F.pad(x[b:b+1], (0, 0, 0, pad_len))
                else:
                    x_padded = x[b:b+1]
                
                # Reshape
                x_2d = x_padded.reshape(1, -1, p, self.d_model)
                x_2d = x_2d.permute(0, 3, 2, 1)  # [1, d_model, period, length]
                reshaped.append(x_2d)
            
            # Apply 2D convolution
            x_2d = torch.cat(reshaped, dim=0)
            x_conv = conv(x_2d)
            
            # Reshape back
            x_conv = x_conv.permute(0, 3, 2, 1)  # [batch, length, period, d_model]
            x_conv = x_conv.reshape(batch_size, -1, self.d_model)
            
            # Trim to original length
            x_conv = x_conv[:, :self.seq_len, :]
            
            res.append(x_conv)
        
        # Aggregate multiple periods
        res = torch.stack(res, dim=-1).mean(dim=-1)
        
        # Residual connection
        x = x + self.dropout(res)
        x = self.norm(x)
        
        return x


class AutoFormer(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation
    Better than vanilla Transformer for long-term forecasting
    Uses auto-correlation instead of self-attention
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        moving_avg: int = 25,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input embedding
        self.embed = nn.Linear(input_size, d_model)
        
        # Series decomposition
        self.decomp = SeriesDecomposition(moving_avg)
        
        # Encoder with auto-correlation
        self.encoder_layers = nn.ModuleList([
            AutoCorrelationLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                moving_avg=moving_avg,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_targets)
        )
    
    def forward(self, x):
        # Embed
        x = self.embed(x)  # [batch, seq_len, d_model]
        
        # Decompose and process
        for layer in self.encoder_layers:
            x = layer(x, self.decomp)
        
        # Global pooling and projection
        x = x.mean(dim=1)
        output = self.projection(x)
        
        return output


class AutoCorrelationLayer(nn.Module):
    """Auto-Correlation layer with series decomposition"""
    def __init__(self, d_model, n_heads, d_ff, moving_avg, dropout):
        super().__init__()
        
        self.auto_correlation = AutoCorrelation(d_model, n_heads, dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, decomp):
        # Auto-correlation
        x_corr = self.auto_correlation(x)
        x, _ = self.decomp1(x + x_corr)
        
        # FFN
        x_ffn = self.ffn(x)
        x, _ = self.decomp2(x + x_ffn)
        
        return x


class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism"""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Project Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Auto-correlation in frequency domain (simplified)
        Q_fft = torch.fft.rfft(Q, dim=1)
        K_fft = torch.fft.rfft(K, dim=1)
        
        # Multiply in frequency domain
        corr = Q_fft * torch.conj(K_fft)
        
        # Back to time domain
        corr = torch.fft.irfft(corr, n=seq_len, dim=1)
        
        # Aggregate with V
        corr = torch.softmax(corr, dim=1)
        out = torch.einsum('bshd,bshd->bshd', corr, V)
        
        # Reshape and project
        out = out.reshape(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        return self.dropout(out)


class SeriesDecomposition(nn.Module):
    """
    Series decomposition into trend and seasonal components
    Moving average for trend extraction
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False
        )
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Transpose for AvgPool1d
        x_t = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # Extract trend (moving average)
        trend = self.avg_pool(x_t)
        trend = trend.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # Seasonal = original - trend
        seasonal = x - trend
        
        return seasonal, trend


class FEDformer(nn.Module):
    """
    FEDformer: Frequency Enhanced Decomposed Transformer
    Works in frequency domain for efficiency
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        moving_avg: int = 25,
        dropout: float = 0.1,
        modes: int = 32
    ):
        super().__init__()
        
        self.modes = modes
        
        # Input embedding
        self.embed = nn.Linear(input_size, d_model)
        
        # Decomposition
        self.decomp = SeriesDecomposition(moving_avg)
        
        # Frequency Enhanced Blocks
        self.feb_layers = nn.ModuleList([
            FEBlock(d_model, modes, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_targets)
        )
    
    def forward(self, x):
        # Embed
        x = self.embed(x)
        
        # Process through FE blocks
        for feb in self.feb_layers:
            x = feb(x, self.decomp)
        
        # Output
        x = x.mean(dim=1)
        return self.projection(x)


class FEBlock(nn.Module):
    """Frequency Enhanced Block"""
    def __init__(self, d_model, modes, d_ff, dropout):
        super().__init__()
        
        self.freq_attention = FrequencyAttention(d_model, modes)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, decomp):
        # Frequency attention
        x_freq = self.freq_attention(x)
        x, _ = decomp(self.norm1(x + x_freq))
        
        # FFN
        x_ffn = self.ffn(x)
        x, _ = decomp(self.norm2(x + x_ffn))
        
        return x


class FrequencyAttention(nn.Module):
    """Attention in frequency domain"""
    def __init__(self, d_model, modes):
        super().__init__()
        
        self.modes = modes
        self.weights = nn.Parameter(
            torch.randn(modes, d_model, d_model, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        # FFT
        x_fft = torch.fft.rfft(x, dim=1)
        
        # Frequency mixing (keep only first 'modes' frequencies)
        out_fft = torch.zeros_like(x_fft)
        
        for i in range(min(self.modes, x_fft.shape[1])):
            out_fft[:, i, :] = torch.matmul(x_fft[:, i, :], self.weights[i])
        
        # Inverse FFT
        x_out = torch.fft.irfft(out_fft, n=x.shape[1], dim=1)
        
        return x_out


if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_size = 150
    num_targets = 424
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing TimesNet...")
    model1 = TimesNet(input_size, seq_len, num_targets)
    out1 = model1(x)
    print(f"Output: {out1.shape}")
    print(f"Params: {sum(p.numel() for p in model1.parameters()):,}\n")
    
    print("Testing Autoformer...")
    model2 = AutoFormer(input_size, seq_len, num_targets)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}\n")
    
    print("Testing FEDformer...")
    model3 = FEDformer(input_size, seq_len, num_targets)
    out3 = model3(x)
    print(f"Output: {out3.shape}")
    print(f"Params: {sum(p.numel() for p in model3.parameters()):,}")