# train.py

import torch
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.lstm_attention import LSTMAttention
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer


def main():
    """Main training script for LSTM Attention model"""
    
    # Configuration
    CONFIG = {
        # Model architecture
        'input_size': None,
        'hidden_size': 128,  # Reduced from 256 to prevent NaN
        'num_layers': 2,     # Reduced from 3
        'num_targets': 424,
        'dropout': 0.3,      # Increased dropout
        'bidirectional': True,
        
        # Training hyperparameters
        'batch_size': 32,    # Reduced batch size
        'learning_rate': 5e-4,  # Lower learning rate
        'weight_decay': 1e-4,   # Higher weight decay
        'num_epochs': 100,
        'early_stopping_patience': 15,
        
        # Data configuration
        'lookback': 60,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4
    }
    
    print("=" * 60)
    print("MITSUI Commodity Prediction - LSTM Attention Training")
    print("=" * 60)
    print(f"\nDevice: {CONFIG['device']}")
    
    # Step 1: Create dataloaders
    print("\n[1/4] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        batch_size=CONFIG['batch_size'],
        lookback=CONFIG['lookback'],
        num_workers=CONFIG['num_workers']
    )
    
    # Detect input size and validate data
    sample_x, sample_y = next(iter(train_loader))
    
    print(f"\nValidating data batch...")
    print(f"Batch X shape: {sample_x.shape}")
    print(f"Batch Y shape: {sample_y.shape}")
    print(f"X contains NaN: {torch.isnan(sample_x).any().item()}")
    print(f"Y contains NaN: {torch.isnan(sample_y).any().item()}")
    print(f"X contains Inf: {torch.isinf(sample_x).any().item()}")
    print(f"Y contains Inf: {torch.isinf(sample_y).any().item()}")
    print(f"X range: [{sample_x.min():.4f}, {sample_x.max():.4f}]")
    print(f"Y range: [{sample_y.min():.4f}, {sample_y.max():.4f}]")
    
    if torch.isnan(sample_x).any() or torch.isnan(sample_y).any():
        print("\nERROR: Data contains NaN! Please check data preprocessing.")
        return
    
    CONFIG['input_size'] = sample_x.shape[-1]
    CONFIG['num_targets'] = sample_y.shape[-1]
    
    print(f"\nData configuration:")
    print(f"  Input features: {CONFIG['input_size']}")
    print(f"  Output targets: {CONFIG['num_targets']}")
    print(f"  Lookback window: {CONFIG['lookback']} days")
    print(f"  Batch size: {CONFIG['batch_size']}")
    
    # Step 2: Create model
    print("\n[2/4] Creating LSTM Attention model...")
    model = LSTMAttention(
        input_size=CONFIG['input_size'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        num_targets=CONFIG['num_targets'],
        dropout=CONFIG['dropout'],
        bidirectional=CONFIG['bidirectional']
    )
    
    # Initialize weights carefully
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                torch.nn.init.xavier_normal_(param, gain=0.5)  # Smaller gain
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel architecture:")
    print(f"  Hidden size: {CONFIG['hidden_size']}")
    print(f"  Number of layers: {CONFIG['num_layers']}")
    print(f"  Bidirectional: {CONFIG['bidirectional']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        test_out = model(sample_x)
        print(f"Output shape: {test_out.shape}")
        print(f"Output contains NaN: {torch.isnan(test_out).any().item()}")
        print(f"Output range: [{test_out.min():.4f}, {test_out.max():.4f}]")
    
    if torch.isnan(test_out).any():
        print("\nERROR: Model produces NaN! Check model initialization.")
        return
    
    # Step 3: Create trainer and train
    print("\n[3/4] Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        save_dir='models/checkpoints'
    )
    
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience'],
        save_best=True
    )
    
    # Step 4: Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    print("=" * 60)
    
    # Check if best model was saved
    best_model_path = Path('models/checkpoints/best_model.pth')
    if not best_model_path.exists():
        print("No best model saved (all epochs had NaN). Training failed.")
        return
    
    # Load best model
    val_metrics = trainer.load_best_model()
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            predictions = model(x)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\nTest Set Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.4f}")
    
    # Save final results
    results = {
        'config': {k: v for k, v in CONFIG.items() if k != 'device'},
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'test_r2': float(r2),
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_metrics': val_metrics
    }
    
    import json
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Results saved to outputs/training_results.json")
    print(f"Best model saved to models/checkpoints/best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
