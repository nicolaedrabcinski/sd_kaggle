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


class Trainer:
    """Trainer for commodity prediction models with progress tracking"""
    
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
            
            loss.backward()
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
        """Validate with progress bar"""
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
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'val_loss': f'{avg_loss:.6f}'})
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))
        
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
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
        """Train model with progress tracking"""
        print("=" * 80)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {early_stopping_patience}")
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
                'val_r2': f'{val_metrics["r2"]:.4f}',
                'lr': f'{current_lr:.2e}',
                'time': f'{epoch_time:.1f}s'
            })
            
            # Print detailed summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_metrics['loss']:.6f}")
            print(f"  Val RMSE:   {val_metrics['rmse']:.6f}")
            print(f"  Val MAE:    {val_metrics['mae']:.6f}")
            print(f"  Val R²:     {val_metrics['r2']:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Time:       {epoch_time:.2f}s")
            
            if new_lr != old_lr:
                print(f"  📉 LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Check for improvement
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
                    print(f"  ✓ New best model saved!")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")
            
            print("=" * 80)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                break
        
        epoch_pbar.close()
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("=" * 80)
    
    def load_best_model(self):
        """Load best model checkpoint"""
        checkpoint_path = self.save_dir / 'best_model.pth'
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
        return checkpoint['val_metrics']