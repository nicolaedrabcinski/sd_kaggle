# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import time
import numpy as np
from pathlib import Path

class Trainer:
    """Trainer for commodity prediction models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        save_dir: str = 'models/checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss (Huber - robust to outliers)
        self.criterion = nn.HuberLoss(delta=1.0)
        
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
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.criterion(predictions, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.6f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.model(x)
                loss = self.criterion(predictions, y)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))
        
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ):
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            old_lr = current_lr
            self.scheduler.step(val_metrics['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"  Val MAE: {val_metrics['mae']:.6f}")
            print(f"  Val R²: {val_metrics['r2']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                if save_best:
                    save_path = self.save_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics
                    }, save_path)
                    print(f"  ✓ Saved best model")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epochs")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def load_best_model(self):
        checkpoint_path = self.save_dir / 'best_model.pth'
        # Fix for PyTorch 2.6: use weights_only=False
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
        return checkpoint['val_metrics']
