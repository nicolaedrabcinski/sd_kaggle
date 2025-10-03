# src/training/losses.py

class DirectionalLoss(nn.Module):
    """Penalize wrong direction more than magnitude"""
    def __init__(self, direction_weight=2.0):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE component
        mse_loss = self.mse(pred, target)
        
        # Directional component
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        direction_loss = (pred_sign != target_sign).float().mean()
        
        return mse_loss + self.direction_weight * direction_loss


class SharpeRatioLoss(nn.Module):
    """Maximize Sharpe ratio instead of minimizing MSE"""
    def forward(self, pred, target):
        returns = pred.mean(dim=1)
        volatility = pred.std(dim=1) + 1e-6
        sharpe = -returns / volatility  # Negative because we minimize
        return sharpe.mean()