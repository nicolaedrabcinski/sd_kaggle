#!/usr/bin/env python3
# optimize_hyperparams.py - Optuna hyperparameter optimization

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import sys
from pathlib import Path
import numpy as np
import json
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer

# Import all models
from src.models.cnn_attention import CNNAttention, WaveNetStyleModel
from src.models.ft_transformer import FTTransformer, PerformerModel
from src.models.nhits import NHiTS, NBeatsInterpretable
from src.models.patchtst import PatchTST, PatchTSTWithChannelIndependence
from src.models.timesnet import TimesNet, AutoFormer, FEDformer
from src.models.dlinear import DLinear, NLinear, RLinear


def suggest_hyperparameters(trial, model_name):
    """Suggest hyperparameters for each model"""
    
    if model_name == 'dlinear':
        return {
            'moving_avg': trial.suggest_int('moving_avg', 15, 35, step=5),
            'individual': trial.suggest_categorical('individual', [True, False])
        }
    
    elif model_name == 'nlinear':
        return {
            'individual': trial.suggest_categorical('individual', [True, False])
        }
    
    elif model_name == 'rlinear':
        return {
            'individual': trial.suggest_categorical('individual', [True, False])
        }
    
    elif model_name == 'patchtst':
        return {
            'patch_len': trial.suggest_int('patch_len', 8, 16, step=4),
            'stride': trial.suggest_int('stride', 8, 16, step=4),
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3)
        }
    
    elif model_name == 'patchtst_ci':
        return {
            'patch_len': trial.suggest_int('patch_len', 8, 16, step=4),
            'stride': trial.suggest_int('stride', 8, 16, step=4),
            'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
            'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3)
        }
    
    elif model_name == 'cnn_attention':
        return {
            'num_filters': trial.suggest_categorical('num_filters', [64, 128, 256]),
            'kernel_sizes': [3, 5, 7],  # Fixed
            'num_heads': trial.suggest_categorical('num_heads', [4, 8]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3)
        }
    
    elif model_name == 'wavenet':
        return {
            'residual_channels': trial.suggest_categorical('residual_channels', [32, 64, 128]),
            'skip_channels': trial.suggest_categorical('skip_channels', [64, 128, 256]),
            'dilation_layers': trial.suggest_int('dilation_layers', 6, 10),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3)
        }
    
    elif model_name == 'nhits':
        return {
            'num_stacks': trial.suggest_int('num_stacks', 2, 4),
            'num_blocks': trial.suggest_int('num_blocks', 2, 4),
            'layer_size': trial.suggest_categorical('layer_size', [256, 512, 1024]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.2)
        }
    
    elif model_name == 'nbeats':
        return {
            'num_stacks': trial.suggest_int('num_stacks', 2, 3),
            'num_blocks_per_stack': trial.suggest_int('num_blocks_per_stack', 2, 4),
            'layer_size': trial.suggest_categorical('layer_size', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.2)
        }
    
    elif model_name == 'timesnet':
        return {
            'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
            'd_ff': trial.suggest_categorical('d_ff', [64, 128, 256]),
            'num_blocks': trial.suggest_int('num_blocks', 1, 3),
            'top_k': trial.suggest_int('top_k', 2, 5),
            'dropout': trial.suggest_float('dropout', 0.05, 0.2)
        }
    
    elif model_name == 'autoformer':
        return {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'moving_avg': trial.suggest_int('moving_avg', 15, 35, step=5),
            'dropout': trial.suggest_float('dropout', 0.05, 0.2)
        }
    
    elif model_name == 'ft_transformer':
        return {
            'd_token': trial.suggest_categorical('d_token', [64, 96, 128]),
            'n_blocks': trial.suggest_int('n_blocks', 2, 4),
            'attention_n_heads': trial.suggest_categorical('attention_n_heads', [4, 8]),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.3),
            'ffn_dropout': trial.suggest_float('ffn_dropout', 0.05, 0.2)
        }
    
    elif model_name == 'performer':
        return {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.2)
        }
    
    else:
        return {}


def get_model_class(model_name):
    """Get model class by name"""
    models = {
        'dlinear': DLinear,
        'nlinear': NLinear,
        'rlinear': RLinear,
        'patchtst': PatchTST,
        'patchtst_ci': PatchTSTWithChannelIndependence,
        'cnn_attention': CNNAttention,
        'wavenet': WaveNetStyleModel,
        'nhits': NHiTS,
        'nbeats': NBeatsInterpretable,
        'timesnet': TimesNet,
        'autoformer': AutoFormer,
        'fedformer': FEDformer,
        'ft_transformer': FTTransformer,
        'performer': PerformerModel
    }
    return models.get(model_name)


def objective(trial, model_name, data_config, n_epochs=50):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    model_params = suggest_hyperparameters(trial, model_name)
    
    # Suggest training hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        batch_size=batch_size,
        lookback=data_config['lookback'],
        num_workers=data_config['num_workers'],
        use_enhanced=data_config['use_enhanced']
    )
    
    # Get data dimensions
    sample_x, sample_y = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    num_targets = sample_y.shape[-1]
    seq_len = sample_x.shape[1]
    
    # Create model
    model_class = get_model_class(model_name)
    
    model_params['input_size'] = input_size
    model_params['num_targets'] = num_targets
    
    # Add seq_len for models that need it
    needs_seq_len = ['patchtst', 'patchtst_ci', 'nhits', 'nbeats', 
                     'timesnet', 'autoformer', 'fedformer', 
                     'ft_transformer', 'performer', 
                     'dlinear', 'nlinear', 'rlinear']
    
    if model_name in needs_seq_len:
        model_params['seq_len'] = seq_len
    
    try:
        model = model_class(**model_params)
    except Exception as e:
        print(f"Error creating model: {e}")
        raise optuna.TrialPruned()
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_dir=f'models/optuna/{model_name}/trial_{trial.number}'
    )
    
    # Train with pruning callback
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(1, n_epochs + 1):
        # Train one epoch
        train_loss = trainer.train_epoch(epoch, n_epochs)
        
        # Validate
        val_metrics = trainer.validate(epoch, n_epochs)
        val_loss = val_metrics['loss']
        val_r2 = val_metrics['r2']
        
        # Report to Optuna
        trial.report(val_loss, epoch)
        
        # Pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Return best validation metric
    return best_val_loss


def optimize_model(model_name, n_trials=100, n_epochs=50, use_enhanced=True):
    """Run Optuna optimization for a model"""
    
    print("="*80)
    print(f"OPTUNA OPTIMIZATION: {model_name.upper()}")
    print("="*80)
    print(f"Trials: {n_trials}")
    print(f"Epochs per trial: {n_epochs}")
    print(f"Enhanced features: {use_enhanced}")
    print("="*80)
    
    # Data configuration
    data_config = {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'lookback': 60,
        'num_workers': 4,
        'use_enhanced': use_enhanced
    }
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, model_name, data_config, n_epochs),
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'n_trials': n_trials,
        'best_trial': study.best_trial.number,
        'best_value': float(study.best_value),
        'best_params': study.best_params,
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    Path('outputs/optuna').mkdir(parents=True, exist_ok=True)
    output_file = f'outputs/optuna/{model_name}_optimization.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)
    
    return study.best_params


def train_with_best_params(model_name, best_params, use_enhanced=True):
    """Train final model with best parameters"""
    
    print(f"\n\nTraining final {model_name} model with best parameters...")
    
    # Extract training params
    batch_size = best_params.pop('batch_size', 64)
    learning_rate = best_params.pop('learning_rate', 1e-3)
    weight_decay = best_params.pop('weight_decay', 1e-5)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=batch_size,
        lookback=60,
        num_workers=4,
        use_enhanced=use_enhanced
    )
    
    # Get dimensions
    sample_x, sample_y = next(iter(train_loader))
    
    # Create model with best params
    model_class = get_model_class(model_name)
    
    model_params = best_params.copy()
    model_params['input_size'] = sample_x.shape[-1]
    model_params['num_targets'] = sample_y.shape[-1]
    
    if model_name in ['patchtst', 'patchtst_ci', 'nhits', 'nbeats', 
                      'timesnet', 'autoformer', 'ft_transformer', 'performer',
                      'dlinear', 'nlinear', 'rlinear']:
        model_params['seq_len'] = sample_x.shape[1]
    
    model = model_class(**model_params)
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_dir=f'models/checkpoints/{model_name}_optimized'
    )
    
    trainer.train(num_epochs=100, early_stopping_patience=15)
    
    # Evaluate
    val_metrics = trainer.load_best_model()
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2 = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
    
    print(f"\nFinal model performance:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²: {r2:.4f}")
    
    return rmse, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization')
    
    parser.add_argument('--model', type=str, required=True, help='Model to optimize')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per trial')
    parser.add_argument('--enhanced', action='store_true', default=True, help='Use enhanced features')
    parser.add_argument('--no-enhanced', dest='enhanced', action='store_false')
    parser.add_argument('--train-best', action='store_true', help='Train final model with best params')
    
    args = parser.parse_args()
    
    # Run optimization
    best_params = optimize_model(
        args.model,
        n_trials=args.trials,
        n_epochs=args.epochs,
        use_enhanced=args.enhanced
    )
    
    # Optionally train final model
    if args.train_best:
        train_with_best_params(args.model, best_params, args.enhanced)