# src/training/losses.py

import torch
import torch.nn as nn


class DirectionalLoss(nn.Module):
    """Penalize wrong direction - balanced scaling"""
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha  # 0.3 = 30% directional, 70% MSE
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE component (magnitude)
        mse_loss = self.mse(pred, target)
        
        # Directional component (0-1 scale)
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        direction_error = (pred_sign != target_sign).float().mean()
        
        # Balanced combination
        return (1 - self.alpha) * mse_loss + self.alpha * direction_error


class SharpeRatioLoss(nn.Module):
    """Maximize Sharpe ratio - NOT RECOMMENDED for training"""
    def forward(self, pred, target):
        # This is problematic - Sharpe ratio unstable for gradients
        returns = pred.mean(dim=1)
        volatility = pred.std(dim=1) + 1e-6
        sharpe = -returns / volatility
        return sharpe.mean()


class HybridLoss(nn.Module):
    """MSE + Directional + Magnitude-weighted directional"""
    def __init__(self, alpha=0.2, beta=0.1):
        super().__init__()
        self.alpha = alpha  # directional weight
        self.beta = beta    # magnitude-weighted directional
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE
        mse_loss = self.mse(pred, target)
        
        # Simple directional
        direction_error = (torch.sign(pred) != torch.sign(target)).float()
        simple_dir_loss = direction_error.mean()
        
        # Magnitude-weighted directional (penalize mistakes on big moves)
        magnitude = torch.abs(target)
        weighted_dir_loss = (direction_error * magnitude).mean()
        
        return (1 - self.alpha - self.beta) * mse_loss + \
               self.alpha * simple_dir_loss + \
               self.beta * weighted_dir_loss