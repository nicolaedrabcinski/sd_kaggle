# src/models/nhits.py

import torch
import torch.nn as nn
import numpy as np


class NHiTS(nn.Module):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series
    SOTA model for time series forecasting
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        num_stacks: int = 3,
        num_blocks: int = 3,
        num_layers: int = 2,
        layer_size: int = 512,
        pooling_sizes: list = None,
        dropout: float = 0.1,
        max_pool_size: int = 16
    ):
        super().__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_targets = num_targets
        
        if pooling_sizes is None:
            # Default hierarchical pooling
            pooling_sizes = [max_pool_size, max_pool_size // 2, 1]
        
        self.pooling_sizes = pooling_sizes[:num_stacks]
        
        # Input projection
        self.input_proj = nn.Linear(input_size * seq_len, layer_size)
        
        # Multiple stacks for hierarchical decomposition
        self.stacks = nn.ModuleList()
        
        for pool_size in self.pooling_sizes:
            stack = NHiTSStack(
                input_size=layer_size,
                num_blocks=num_blocks,
                num_layers=num_layers,
                layer_size=layer_size,
                horizon=num_targets,
                pool_size=pool_size,
                dropout=dropout
            )
            self.stacks.append(stack)
    
    def forward(self, x):
        # Flatten input
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Project input
        x = self.input_proj(x)
        
        # Hierarchical forecasting
        forecast = 0
        
        for stack in self.stacks:
            stack_forecast, backcast = stack(x)
            forecast = forecast + stack_forecast
            x = x - backcast
        
        return forecast


class NHiTSStack(nn.Module):
    """Single stack in N-HiTS"""
    def __init__(
        self,
        input_size: int,
        num_blocks: int,
        num_layers: int,
        layer_size: int,
        horizon: int,
        pool_size: int,
        dropout: float
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_size=input_size,
                num_layers=num_layers,
                layer_size=layer_size,
                horizon=horizon,
                pool_size=pool_size,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        forecast = 0
        backcast = 0
        
        for block in self.blocks:
            block_forecast, block_backcast = block(x)
            forecast = forecast + block_forecast
            backcast = backcast + block_backcast
            x = x - block_backcast
        
        return forecast, backcast


class NHiTSBlock(nn.Module):
    """Single block in N-HiTS"""
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        layer_size: int,
        horizon: int,
        pool_size: int,
        dropout: float
    ):
        super().__init__()
        
        self.pool_size = pool_size
        self.horizon = horizon
        
        # MLP for feature extraction
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, layer_size))
            else:
                layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Backcast and forecast heads
        self.backcast_head = nn.Linear(layer_size, input_size)
        
        # Forecast with interpolation
        if pool_size > 1:
            self.forecast_head = nn.Linear(layer_size, horizon // pool_size)
        else:
            self.forecast_head = nn.Linear(layer_size, horizon)
    
    def forward(self, x):
        # Feature extraction
        h = self.mlp(x)
        
        # Backcast
        backcast = self.backcast_head(h)
        
        # Forecast with hierarchical interpolation
        if self.pool_size > 1:
            forecast_pooled = self.forecast_head(h)
            # Upsample using interpolation
            forecast = torch.nn.functional.interpolate(
                forecast_pooled.unsqueeze(1),
                size=self.horizon,
                mode='linear',
                align_corners=True
            ).squeeze(1)
        else:
            forecast = self.forecast_head(h)
        
        return forecast, backcast


class NBeatsInterpretable(nn.Module):
    """
    N-BEATS with interpretable basis functions
    Decomposes into trend + seasonality
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        num_stacks: int = 2,
        num_blocks_per_stack: int = 3,
        layer_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        flat_input = input_size * seq_len
        
        # Trend stack
        self.trend_stack = nn.ModuleList([
            NBeatsBlock(
                flat_input, layer_size, num_targets,
                basis_function='trend', dropout=dropout
            )
            for _ in range(num_blocks_per_stack)
        ])
        
        # Seasonality stack
        self.seasonality_stack = nn.ModuleList([
            NBeatsBlock(
                flat_input, layer_size, num_targets,
                basis_function='seasonality', dropout=dropout
            )
            for _ in range(num_blocks_per_stack)
        ])
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Trend component
        trend_forecast = 0
        residual = x
        
        for block in self.trend_stack:
            forecast, backcast = block(residual)
            trend_forecast = trend_forecast + forecast
            residual = residual - backcast
        
        # Seasonality component
        seasonal_forecast = 0
        
        for block in self.seasonality_stack:
            forecast, backcast = block(residual)
            seasonal_forecast = seasonal_forecast + forecast
            residual = residual - backcast
        
        # Combine
        return trend_forecast + seasonal_forecast


class NBeatsBlock(nn.Module):
    """N-BEATS block with basis functions"""
    def __init__(
        self,
        input_size: int,
        layer_size: int,
        horizon: int,
        basis_function: str = 'generic',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.basis_function = basis_function
        self.horizon = horizon
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if basis_function == 'trend':
            # Polynomial basis
            self.forecast_head = nn.Linear(layer_size, 3)  # degree 2 polynomial
            self.backcast_head = nn.Linear(layer_size, input_size)
        elif basis_function == 'seasonality':
            # Fourier basis
            self.forecast_head = nn.Linear(layer_size, horizon)
            self.backcast_head = nn.Linear(layer_size, input_size)
        else:
            # Generic
            self.forecast_head = nn.Linear(layer_size, horizon)
            self.backcast_head = nn.Linear(layer_size, input_size)
    
    def forward(self, x):
        h = self.mlp(x)
        
        if self.basis_function == 'trend':
            # Polynomial trend
            theta_f = self.forecast_head(h)
            t = torch.arange(self.horizon, dtype=torch.float32, device=x.device) / self.horizon
            t = t.unsqueeze(0).expand(x.shape[0], -1)
            forecast = theta_f[:, 0:1] + theta_f[:, 1:2] * t + theta_f[:, 2:3] * (t ** 2)
        else:
            forecast = self.forecast_head(h)
        
        backcast = self.backcast_head(h)
        
        return forecast, backcast


if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_size = 150
    num_targets = 424
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing N-HiTS...")
    model1 = NHiTS(input_size, seq_len, num_targets)
    out1 = model1(x)
    print(f"Output: {out1.shape}")
    print(f"Params: {sum(p.numel() for p in model1.parameters()):,}\n")
    
    print("Testing N-BEATS Interpretable...")
    model2 = NBeatsInterpretable(input_size, seq_len, num_targets)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}")