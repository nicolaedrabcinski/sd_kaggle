# src/models/dlinear.py

import torch
import torch.nn as nn


class DLinear(nn.Module):
    """
    DLinear: Decomposition + Linear
    Surprisingly simple but VERY effective baseline
    Often beats complex transformers!
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        moving_avg: int = 25,
        individual: bool = False
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.individual = individual
        
        # Decomposition
        self.decomposition = MovingAvgDecomposition(moving_avg)
        
        if individual:
            # Separate linear layer for each variable
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, num_targets)
                for _ in range(input_size)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, num_targets)
                for _ in range(input_size)
            ])
        else:
            # Shared linear layers
            self.linear_seasonal = nn.Linear(seq_len, num_targets)
            self.linear_trend = nn.Linear(seq_len, num_targets)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Decompose into seasonal and trend
        seasonal, trend = self.decomposition(x)
        
        if self.individual:
            # Process each feature separately
            seasonal_out = []
            trend_out = []
            
            for i in range(features):
                s = self.linear_seasonal[i](seasonal[:, :, i])
                t = self.linear_trend[i](trend[:, :, i])
                seasonal_out.append(s)
                trend_out.append(t)
            
            seasonal_out = torch.stack(seasonal_out, dim=-1).mean(dim=-1)
            trend_out = torch.stack(trend_out, dim=-1).mean(dim=-1)
        else:
            # Shared processing
            seasonal = seasonal.permute(0, 2, 1)  # [batch, features, seq_len]
            trend = trend.permute(0, 2, 1)
            
            seasonal_out = self.linear_seasonal(seasonal).mean(dim=1)  # [batch, num_targets]
            trend_out = self.linear_trend(trend).mean(dim=1)
        
        # Combine
        output = seasonal_out + trend_out
        
        return output


class NLinear(nn.Module):
    """
    NLinear: Normalized Linear
    Handles distribution shift better than DLinear
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        individual: bool = False
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.individual = individual
        
        if individual:
            self.linear = nn.ModuleList([
                nn.Linear(seq_len, num_targets)
                for _ in range(input_size)
            ])
        else:
            self.linear = nn.Linear(seq_len, num_targets)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Normalization
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        
        if self.individual:
            # Process each feature separately
            outputs = []
            for i in range(features):
                out = self.linear[i](x[:, :, i])
                outputs.append(out)
            output = torch.stack(outputs, dim=-1).mean(dim=-1)
        else:
            x = x.permute(0, 2, 1)  # [batch, features, seq_len]
            output = self.linear(x).mean(dim=1)
        
        # Denormalization
        output = output + seq_last.squeeze(1).mean(dim=-1, keepdim=True)
        
        return output


class MovingAvgDecomposition(nn.Module):
    """Moving average decomposition"""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Pad to maintain length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Moving average (trend)
        x_padded = x_padded.permute(0, 2, 1)  # [batch, features, seq_len]
        trend = self.avg(x_padded)
        trend = trend.permute(0, 2, 1)  # [batch, seq_len, features]
        
        # Seasonal = original - trend
        seasonal = x - trend
        
        return seasonal, trend


class RLinear(nn.Module):
    """
    RLinear: Reversible Instance Normalization + Linear
    Best for non-stationary time series
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        individual: bool = False
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.individual = individual
        
        # RevIN (learnable affine transformation)
        self.affine_weight = nn.Parameter(torch.ones(input_size))
        self.affine_bias = nn.Parameter(torch.zeros(input_size))
        
        if individual:
            self.linear = nn.ModuleList([
                nn.Linear(seq_len, num_targets)
                for _ in range(input_size)
            ])
        else:
            self.linear = nn.Linear(seq_len, num_targets)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # RevIN: Normalize
        means = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = (x - means) / stdev
        x_norm = x_norm * self.affine_weight + self.affine_bias
        
        # Linear projection
        if self.individual:
            outputs = []
            for i in range(features):
                out = self.linear[i](x_norm[:, :, i])
                outputs.append(out)
            output = torch.stack(outputs, dim=-1).mean(dim=-1)
        else:
            x_norm = x_norm.permute(0, 2, 1)
            output = self.linear(x_norm).mean(dim=1)
        
        # RevIN: Denormalize
        output = output * stdev.mean(dim=-1).squeeze(1) + means.mean(dim=-1).squeeze(1)
        
        return output


class FITS(nn.Module):
    """
    FITS: Frequency Interpolation Time Series
    Ultra-simple but effective: FFT -> Interpolate -> iFFT
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        cut_freq: int = None
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_targets = num_targets
        
        # Frequency cutoff
        if cut_freq is None:
            self.cut_freq = seq_len // 2
        else:
            self.cut_freq = cut_freq
        
        # Complex-valued weights for frequency domain
        self.freq_weight = nn.Parameter(
            torch.randn(input_size, self.cut_freq, 2)  # real and imaginary
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=1)
        
        # Keep only low frequencies
        x_fft_cut = x_fft[:, :self.cut_freq, :]
        
        # Apply learned weights in frequency domain
        weight = torch.complex(self.freq_weight[:, :, 0], self.freq_weight[:, :, 1])
        x_fft_weighted = x_fft_cut * weight.unsqueeze(0)
        
        # Interpolate to target length
        x_fft_interp = torch.nn.functional.interpolate(
            x_fft_weighted.abs().permute(0, 2, 1),
            size=self.num_targets // 2 + 1,
            mode='linear'
        ).permute(0, 2, 1)
        
        # Preserve phase
        phase = torch.angle(x_fft_weighted)
        phase_interp = torch.nn.functional.interpolate(
            phase.permute(0, 2, 1),
            size=self.num_targets // 2 + 1,
            mode='linear'
        ).permute(0, 2, 1)
        
        # Reconstruct complex tensor
        x_fft_final = x_fft_interp * torch.exp(1j * phase_interp)
        
        # Inverse FFT
        output = torch.fft.irfft(x_fft_final, n=self.num_targets, dim=1)
        
        # Average across features
        output = output.mean(dim=-1)
        
        return output


if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_size = 150
    num_targets = 424
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing DLinear...")
    model1 = DLinear(input_size, seq_len, num_targets)
    out1 = model1(x)
    print(f"Output: {out1.shape}")
    print(f"Params: {sum(p.numel() for p in model1.parameters()):,}\n")
    
    print("Testing NLinear...")
    model2 = NLinear(input_size, seq_len, num_targets)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}\n")
    
    print("Testing RLinear...")
    model3 = RLinear(input_size, seq_len, num_targets)
    out3 = model3(x)
    print(f"Output: {out3.shape}")
    print(f"Params: {sum(p.numel() for p in model3.parameters()):,}\n")
    
    print("Testing FITS...")
    model4 = FITS(input_size, seq_len, num_targets)
    out4 = model4(x)
    print(f"Output: {out4.shape}")
    print(f"Params: {sum(p.numel() for p in model4.parameters()):,}")