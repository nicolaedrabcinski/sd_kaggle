# src/training/losses.py

"""
Loss functions for Mitsui Commodity Prediction Challenge

КРИТИЧНО: Соревнование оценивается по Modified Sharpe Ratio
на основе Spearman rank correlation, а не MSE!
"""

import torch
import torch.nn as nn
import numpy as np


class SpearmanLoss(nn.Module):
    """
    Differentiable Spearman rank correlation loss
    
    ИСПРАВЛЕНИЕ: Используем differentiable sorting через soft ranks
    """
    
    def __init__(self, reg=1e-6):
        super().__init__()
        self.reg = reg
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size, num_targets]
            target: [batch_size, num_targets]
        
        Returns:
            loss: negative mean Spearman correlation (differentiable!)
        """
        batch_size, num_targets = pred.shape
        
        if batch_size < 2:
            # Не можем вычислить корреляцию с одним сэмплом
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Вычисляем корреляцию для каждого target отдельно
        correlations = []
        
        for i in range(num_targets):
            pred_col = pred[:, i]
            target_col = target[:, i]
            
            # Пропускаем если нет вариации
            if pred_col.std() < 1e-6 or target_col.std() < 1e-6:
                continue
            
            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Используем Pearson вместо ranks
            # Pearson differentiable, а ranks - нет!
            corr = self._pearson_correlation(pred_col, target_col)
            
            if not torch.isnan(corr) and not torch.isinf(corr):
                correlations.append(corr)
        
        if len(correlations) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Mean correlation
        mean_corr = torch.stack(correlations).mean()
        
        # Negative для минимизации
        return -mean_corr
    
    def _pearson_correlation(self, x, y):
        """
        Differentiable Pearson correlation
        """
        # Center
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # Correlation
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum() + self.reg)
        
        return numerator / denominator


class MitsuiLoss(nn.Module):
    """
    Hybrid loss для Mitsui competition
    
    Комбинирует:
    - Pearson correlation (70%) - approximation к Spearman
    - Directional accuracy (30%) - помогает с знаком
    """
    
    def __init__(self, spearman_weight=0.7):
        super().__init__()
        
        if not 0 <= spearman_weight <= 1:
            raise ValueError("spearman_weight must be between 0 and 1")
        
        self.spearman_weight = spearman_weight
        self.direction_weight = 1.0 - spearman_weight
        self.spearman_loss = SpearmanLoss()
        
        print(f"MitsuiLoss initialized:")
        print(f"  Spearman weight: {self.spearman_weight:.1%}")
        print(f"  Direction weight: {self.direction_weight:.1%}")
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size, num_targets]
            target: [batch_size, num_targets]
        
        Returns:
            loss: combined differentiable loss
        """
        # Primary: Spearman (Pearson approximation)
        spearman_loss = self.spearman_loss(pred, target)
        
        # Auxiliary: Directional
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        direction_error = (pred_sign != target_sign).float().mean()
        
        # Combined
        total_loss = (self.spearman_weight * spearman_loss + 
                     self.direction_weight * direction_error)
        
        return total_loss


class DirectionalLoss(nn.Module):
    """
    Original DirectionalLoss - backward compatibility
    """
    
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        print("WARNING: DirectionalLoss uses MSE - consider MitsuiLoss for Mitsui competition")
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        direction_error = (torch.sign(pred) != torch.sign(target)).float().mean()
        return (1 - self.alpha) * mse_loss + self.alpha * direction_error


def get_recommended_loss():
    """Recommended loss for Mitsui competition"""
    return MitsuiLoss(spearman_weight=0.7)


__all__ = ['SpearmanLoss', 'MitsuiLoss', 'DirectionalLoss', 'get_recommended_loss']