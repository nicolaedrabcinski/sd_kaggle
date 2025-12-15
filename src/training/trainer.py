# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr

# Import custom losses
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.losses import MitsuiLoss, SpearmanLoss, DirectionalLoss


class Trainer:
    """
    Trainer for Mitsui Commodity Prediction Challenge
    
    КРИТИЧНО: Использует MitsuiLoss (Spearman correlation)
    вместо MSE для соответствия метрике соревнования
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        save_dir: str = 'models/checkpoints',
        loss_type: str = 'mitsui',  # 'mitsui', 'spearman', or 'directional'
        spearman_weight: float = 0.7
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function - ALIGNED WITH COMPETITION METRIC
        print(f"\nLoss configuration:")
        if loss_type == 'mitsui':
            self.criterion = MitsuiLoss(spearman_weight=spearman_weight)
            print(f"✓ Using MitsuiLoss (Spearman {spearman_weight:.0%} + Direction {1-spearman_weight:.0%})")
        elif loss_type == 'spearman':
            self.criterion = SpearmanLoss()
            print(f"✓ Using pure SpearmanLoss")
        else:
            # Fallback (не рекомендуется для Mitsui!)
            self.criterion = DirectionalLoss(alpha=0.3)
            print(f"⚠️  Using DirectionalLoss (NOT recommended for Mitsui!)")
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
            # verbose=False  # Changed to False to reduce output
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_spearman = float('-inf')
        self.train_losses = []
        self.val_losses = []
        
        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int, num_epochs: int) -> float:
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        
        # Progress bar for batches
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch}/{num_epochs} [Train]',
            leave=False,
            ncols=120
        )
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.criterion(predictions, y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\n⚠️  WARNING: NaN loss at batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Gradient clipping (важно для stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{avg_loss:.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch: int, num_epochs: int) -> Dict[str, float]:
        """
        Validate with progress bar
        
        КРИТИЧНО: Теперь вычисляет Spearman correlation!
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Progress bar for validation
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch}/{num_epochs} [Val]',
            leave=False,
            ncols=120
        )
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.model(x)
                loss = self.criterion(predictions, y)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'val_loss': f'{avg_loss:.6f}'})
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # ====================================================================
        # METRICS
        # ====================================================================
        
        # 1. MSE/RMSE/MAE (traditional)
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))
        
        # 2. R² (может быть отрицательным для efficient markets)
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
        
        # 3. Directional accuracy (sign prediction)
        dir_acc = np.mean(np.sign(all_predictions) == np.sign(all_targets))
        
        # 4. ⭐ SPEARMAN CORRELATION - ГЛАВНАЯ МЕТРИКА ⭐
        spearman_correlations = []
        
        for i in range(all_predictions.shape[1]):
            # Skip if no variance
            if all_targets[:, i].std() < 1e-6 or all_predictions[:, i].std() < 1e-6:
                continue
            
            try:
                corr, _ = spearmanr(all_predictions[:, i], all_targets[:, i])
                if not np.isnan(corr):
                    spearman_correlations.append(corr)
            except:
                continue
        
        mean_spearman = np.mean(spearman_correlations) if spearman_correlations else 0.0
        std_spearman = np.std(spearman_correlations) if len(spearman_correlations) > 1 else 0.0
        
        # Modified Sharpe Ratio (Mitsui metric)
        if std_spearman > 1e-6:
            modified_sharpe = mean_spearman / std_spearman
        else:
            modified_sharpe = 0.0
        
        return {
            'loss': total_loss / len(self.val_loader),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': dir_acc,
            'spearman_mean': mean_spearman,
            'spearman_std': std_spearman,
            'modified_sharpe': modified_sharpe
        }
    
    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ):
        """Train model with progress tracking"""
        print("=" * 80)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Save best: {save_best}")
        print("=" * 80)
        
        patience_counter = 0
        
        # Main progress bar for epochs
        epoch_pbar = tqdm(range(1, num_epochs + 1), desc='Training Progress', ncols=120)
        
        for epoch in epoch_pbar:
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch, num_epochs)
            
            # Validate
            val_metrics = self.validate(epoch, num_epochs)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            old_lr = current_lr
            self.scheduler.step(val_metrics['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            
            epoch_time = time.time() - start_time
            
            # Update main progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_metrics["loss"]:.6f}',
                'spearman': f'{val_metrics["spearman_mean"]:.4f}',
                'dir_acc': f'{val_metrics["directional_accuracy"]:.4f}',
                'lr': f'{current_lr:.2e}',
                'time': f'{epoch_time:.1f}s'
            })
            
            # Print detailed summary every 10 epochs or if improvement
            if epoch % 10 == 0 or val_metrics['spearman_mean'] > self.best_val_spearman:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch}/{num_epochs} Summary:")
                print(f"  Train Loss:      {train_loss:.6f}")
                print(f"  Val Loss:        {val_metrics['loss']:.6f}")
                print(f"  Val RMSE:        {val_metrics['rmse']:.6f}")
                print(f"  Val R²:          {val_metrics['r2']:.4f}")
                print(f"  Dir Acc:         {val_metrics['directional_accuracy']:.4f} ({val_metrics['directional_accuracy']*100:.1f}%)")
                print(f"  ⭐ Spearman:     {val_metrics['spearman_mean']:.6f} ± {val_metrics['spearman_std']:.6f}")
                print(f"  ⭐ Mod. Sharpe:  {val_metrics['modified_sharpe']:.4f}")
                print(f"  LR:              {current_lr:.6f}")
                print(f"  Time:            {epoch_time:.2f}s")
                
                if new_lr != old_lr:
                    print(f"  LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Check for improvement (по Spearman, не по loss!)
            improved = False
            
            if val_metrics['spearman_mean'] > self.best_val_spearman:
                self.best_val_spearman = val_metrics['spearman_mean']
                improved = True
                improvement_type = "Spearman"
            
            if improved:
                patience_counter = 0
                
                if save_best:
                    save_path = self.save_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'best_spearman': self.best_val_spearman
                    }, save_path)
                    print(f"  ✓ New best model saved! (improved {improvement_type})")
                print("=" * 80)
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    print(f"  No improvement for {patience_counter}/{early_stopping_patience} epochs")
                    print("=" * 80)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                print(f"Best Spearman correlation: {self.best_val_spearman:.6f}")
                break
        
        epoch_pbar.close()
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best Spearman correlation: {self.best_val_spearman:.6f}")
        print("=" * 80)
    
    def load_best_model(self):
        """Load best model checkpoint"""
        checkpoint_path = self.save_dir / 'best_model.pth'
        
        if not checkpoint_path.exists():
            print(f"⚠️  No checkpoint found at {checkpoint_path}")
            return {}
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")
        print(f"  Best Spearman: {checkpoint.get('best_spearman', 'N/A')}")
        
        return checkpoint.get('val_metrics', {})