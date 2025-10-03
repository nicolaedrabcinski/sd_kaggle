#!/usr/bin/env python3
# train_all.py - Universal training script for all models

import torch
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
from src.models.tabnet import TabNetModel, ResidualMLP, XGBoostStyleNN
from src.models.ft_transformer import FTTransformer, PerformerModel
from src.models.nhits import NHiTS, NBeatsInterpretable
from src.models.patchtst import PatchTST, PatchTSTWithChannelIndependence, CrossFormer
from src.models.timesnet import TimesNet, AutoFormer, FEDformer
from src.models.dlinear import DLinear, NLinear, RLinear, FITS


# Model registry with optimal hyperparameters
MODEL_REGISTRY = {
    # === Simple & Fast Baselines ===
    'dlinear': {
        'class': DLinear,
        'params': {'moving_avg': 25, 'individual': True},
        'batch_size': 128,
        'lr': 1e-3,
        'category': 'baseline',
        'description': 'Decomposition + Linear - Simple but VERY effective baseline'
    },
    'nlinear': {
        'class': NLinear,
        'params': {'individual': True},
        'batch_size': 128,
        'lr': 1e-3,
        'category': 'baseline',
        'description': 'Normalized Linear - Handles distribution shift'
    },
    'rlinear': {
        'class': RLinear,
        'params': {'individual': True},
        'batch_size': 128,
        'lr': 1e-3,
        'category': 'baseline',
        'description': 'RevIN + Linear - Best for non-stationary series'
    },
    'fits': {
        'class': FITS,
        'params': {},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'baseline',
        'description': 'Frequency Interpolation - Ultra simple FFT-based'
    },
    
    # === Convolutional Models ===
    'cnn_attention': {
        'class': CNNAttention,
        'params': {'num_filters': 128, 'kernel_sizes': [3, 5, 7], 'num_heads': 8, 'dropout': 0.2},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'cnn',
        'description': 'Multi-scale CNN + Attention - Fast and effective'
    },
    'wavenet': {
        'class': WaveNetStyleModel,
        'params': {'residual_channels': 64, 'skip_channels': 128, 'dilation_layers': 8, 'dropout': 0.2},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'cnn',
        'description': 'Dilated causal convolutions - Long-range dependencies'
    },
    
    # === Transformer-based ===
    'patchtst': {
        'class': PatchTST,
        'params': {'patch_len': 12, 'stride': 12, 'd_model': 128, 'n_heads': 8, 'n_layers': 3, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'transformer',
        'description': 'PatchTST - Best Transformer for time series (RECOMMENDED)'
    },
    'patchtst_ci': {
        'class': PatchTSTWithChannelIndependence,
        'params': {'patch_len': 12, 'stride': 12, 'd_model': 64, 'n_heads': 4, 'n_layers': 2, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'transformer',
        'description': 'PatchTST with Channel Independence - For many variables'
    },
    'ft_transformer': {
        'class': FTTransformer,
        'params': {'d_token': 96, 'n_blocks': 3, 'attention_n_heads': 8, 'dropout': 0.2},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'transformer',
        'description': 'Feature Tokenizer - Each feature as token'
    },
    'performer': {
        'class': PerformerModel,
        'params': {'d_model': 128, 'n_layers': 3, 'n_heads': 8, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'transformer',
        'description': 'Performer - Linear attention, efficient'
    },
    'crossformer': {
        'class': CrossFormer,
        'params': {'seg_len': 12, 'd_model': 128, 'n_heads': 8, 'n_layers': 2, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'transformer',
        'description': 'CrossFormer - Cross-dimension dependencies'
    },
    
    # === Advanced Time Series Models ===
    'nhits': {
        'class': NHiTS,
        'params': {'num_stacks': 3, 'num_blocks': 3, 'layer_size': 512, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'advanced',
        'description': 'N-HiTS - Hierarchical interpolation, SOTA forecasting'
    },
    'nbeats': {
        'class': NBeatsInterpretable,
        'params': {'num_stacks': 2, 'num_blocks_per_stack': 3, 'layer_size': 256, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'advanced',
        'description': 'N-BEATS - Interpretable basis functions (trend+seasonal)'
    },
    'timesnet': {
        'class': TimesNet,
        'params': {'d_model': 64, 'd_ff': 128, 'num_blocks': 2, 'top_k': 3, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'advanced',
        'description': 'TimesNet - 2D variation modeling, multi-periodicity'
    },
    'autoformer': {
        'class': AutoFormer,
        'params': {'d_model': 128, 'n_heads': 8, 'n_layers': 2, 'moving_avg': 25, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'advanced',
        'description': 'Autoformer - Auto-correlation + decomposition'
    },
    'fedformer': {
        'class': FEDformer,
        'params': {'d_model': 128, 'n_heads': 8, 'n_layers': 2, 'moving_avg': 25, 'modes': 32, 'dropout': 0.1},
        'batch_size': 64,
        'lr': 5e-4,
        'category': 'advanced',
        'description': 'FEDformer - Frequency enhanced, efficient'
    },
    
    # === Tabular-focused ===
    'tabnet': {
        'class': TabNetModel,
        'params': {'n_d': 64, 'n_a': 64, 'n_steps': 3, 'gamma': 1.3, 'dropout': 0.2},
        'batch_size': 128,
        'lr': 1e-3,
        'category': 'tabular',
        'description': 'TabNet - Attentive interpretable, best for tabular'
    },
    'residual_mlp': {
        'class': ResidualMLP,
        'params': {'hidden_sizes': [512, 384, 256, 128], 'dropout': 0.3},
        'batch_size': 128,
        'lr': 1e-3,
        'category': 'tabular',
        'description': 'Deep MLP with residuals - Simple baseline'
    },
    'xgboost_nn': {
        'class': XGBoostStyleNN,
        'params': {'num_trees': 10, 'tree_depth': 3, 'dropout': 0.1},
        'batch_size': 128,
        'lr': 1e-3,
        'category': 'tabular',
        'description': 'Neural network mimicking XGBoost'
    }
}


def list_models(category=None):
    """List available models"""
    print("\n" + "="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    
    if category:
        models = {k: v for k, v in MODEL_REGISTRY.items() if v['category'] == category}
        print(f"\nCategory: {category.upper()}")
    else:
        models = MODEL_REGISTRY
        
        # Group by category
        categories = {}
        for name, config in models.items():
            cat = config['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, config))
        
        for cat in ['baseline', 'cnn', 'transformer', 'advanced', 'tabular']:
            if cat in categories:
                print(f"\n{cat.upper()}:")
                for name, config in categories[cat]:
                    print(f"  {name:20} - {config['description']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("  1. Start with: dlinear (fastest baseline)")
    print("  2. Then try: patchtst (best transformer)")
    print("  3. For SOTA: nhits")
    print("="*80)


def train_model(model_name, use_enhanced=True, epochs=100, patience=15):
    """Train a specific model"""
    
    if model_name not in MODEL_REGISTRY:
        print(f"\nERROR: Unknown model '{model_name}'")
        list_models()
        return None
    
    config = MODEL_REGISTRY[model_name]
    
    print("\n" + "="*80)
    print(f"Training: {model_name.upper()}")
    print(f"Category: {config['category']}")
    print(f"Description: {config['description']}")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'model_name': model_name,
        'input_size': None,
        'num_targets': 424,
        'lookback': 60,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'use_enhanced': use_enhanced,
        'batch_size': config['batch_size'],
        'learning_rate': config['lr'],
        'weight_decay': 1e-5,
        'num_epochs': epochs,
        'early_stopping_patience': patience,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4
    }
    
    print(f"\nConfiguration:")
    print(f"  Device: {CONFIG['device']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Max epochs: {CONFIG['num_epochs']}")
    print(f"  Enhanced features: {use_enhanced}")
    
    # Load data
    print("\n[1/5] Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        batch_size=CONFIG['batch_size'],
        lookback=CONFIG['lookback'],
        num_workers=CONFIG['num_workers'],
        use_enhanced=use_enhanced
    )
    
    # Get dimensions
    sample_x, sample_y = next(iter(train_loader))
    CONFIG['input_size'] = sample_x.shape[-1]
    CONFIG['num_targets'] = sample_y.shape[-1]
    
    print(f"  Input: {sample_x.shape}")
    print(f"  Output: {sample_y.shape}")
    print(f"  Target stats: mean={sample_y.mean():.6f}, std={sample_y.std():.6f}")
    
    # Create model
    print(f"\n[2/5] Creating {model_name} model...")
    
    model_params = config['params'].copy()
    model_params['input_size'] = CONFIG['input_size']
    model_params['num_targets'] = CONFIG['num_targets']
    
    # Add seq_len for models that need it
    needs_seq_len = ['tabnet', 'ft_transformer', 'performer', 'residual_mlp', 
                     'xgboost_nn', 'nhits', 'nbeats', 'timesnet', 'autoformer', 
                     'fedformer', 'dlinear', 'nlinear', 'rlinear', 'fits',
                     'patchtst', 'patchtst_ci', 'crossformer']
    
    if model_name in needs_seq_len:
        model_params['seq_len'] = CONFIG['lookback']
    
    try:
        model = config['class'](**model_params)
    except Exception as e:
        print(f"ERROR creating model: {e}")
        return None
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward
    print("\n  Testing forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            test_out = model(sample_x)
            assert test_out.shape == sample_y.shape, f"Shape mismatch: {test_out.shape} vs {sample_y.shape}"
            assert not torch.isnan(test_out).any(), "Model outputs NaN!"
            print(f"  Forward pass OK: {test_out.shape}")
        except Exception as e:
            print(f"  ERROR in forward pass: {e}")
            return None
    
    # Baseline
    print("\n[3/5] Calculating baseline...")
    all_val_targets = []
    for _, y in val_loader:
        all_val_targets.append(y.numpy())
    all_val_targets = np.concatenate(all_val_targets)
    
    baseline_pred = np.mean(all_val_targets, axis=0, keepdims=True)
    baseline_pred = np.repeat(baseline_pred, len(all_val_targets), axis=0)
    baseline_rmse = np.sqrt(np.mean((all_val_targets - baseline_pred) ** 2))
    print(f"  Baseline RMSE: {baseline_rmse:.6f}")
    print(f"  Model must beat this!")
    
    # Train
    print(f"\n[4/5] Training...")
    print("="*80)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        save_dir=f'models/checkpoints/{model_name}'
    )
    
    start_time = datetime.now()
    
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience']
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate
    print("\n[5/5] Final evaluation...")
    print("="*80)
    
    val_metrics = trainer.load_best_model()
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
    
    dir_acc = np.mean(np.sign(all_preds) == np.sign(all_targets))
    
    test_baseline = np.sqrt(np.mean((all_targets - np.mean(all_targets, axis=0)) ** 2))
    improvement = (1 - rmse/test_baseline) * 100 if rmse < test_baseline else (rmse/test_baseline - 1) * -100
    
    # Results
    print(f"\nFINAL RESULTS - {model_name.upper()}")
    print("="*80)
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Dir Acc: {dir_acc:.4f} ({dir_acc*100:.1f}%)")
    print(f"\n  Baseline RMSE: {test_baseline:.6f}")
    print(f"  Improvement: {improvement:+.2f}%")
    print(f"\n  Training time: {training_time/60:.1f} minutes")
    print(f"  Parameters: {total_params:,}")
    print("="*80)
    
    # Save results
    results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'config': {k: v for k, v in CONFIG.items() if k not in ['device']},
        'test_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'directional_accuracy': float(dir_acc)
        },
        'baseline_rmse': float(test_baseline),
        'improvement_pct': float(improvement),
        'training_time_seconds': float(training_time),
        'num_parameters': int(total_params),
        'best_val_metrics': {k: float(v) for k, v in val_metrics.items()}
    }
    
    Path('outputs').mkdir(exist_ok=True)
    output_file = f'outputs/{model_name}_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def compare_models(models_to_compare=None):
    """Compare results from multiple models"""
    
    results_dir = Path('outputs')
    
    if not results_dir.exists():
        print("No results found in outputs/")
        return
    
    # Load all results
    all_results = []
    
    for result_file in results_dir.glob('*_results.json'):
        with open(result_file) as f:
            data = json.load(f)
            all_results.append(data)
    
    if not all_results:
        print("No model results found!")
        return
    
    # Filter if specific models requested
    if models_to_compare:
        all_results = [r for r in all_results if r['model_name'] in models_to_compare]
    
    # Sort by R²
    all_results.sort(key=lambda x: x['test_metrics']['r2'], reverse=True)
    
    # Print comparison
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<20} {'RMSE':<12} {'R²':<10} {'Dir Acc':<10} {'Improve':<10} {'Params':<12} {'Time(m)':<10}")
    print("-"*100)
    
    for result in all_results:
        name = result['model_name']
        rmse = result['test_metrics']['rmse']
        r2 = result['test_metrics']['r2']
        dir_acc = result['test_metrics']['directional_accuracy']
        improve = result['improvement_pct']
        params = result['num_parameters']
        time_m = result['training_time_seconds'] / 60
        
        print(f"{name:<20} {rmse:<12.6f} {r2:<10.4f} {dir_acc:<10.4f} {improve:>+8.2f}% {params:>10,}  {time_m:>8.1f}")
    
    print("="*100)
    
    # Best models
    best_r2 = max(all_results, key=lambda x: x['test_metrics']['r2'])
    best_rmse = min(all_results, key=lambda x: x['test_metrics']['rmse'])
    fastest = min(all_results, key=lambda x: x['training_time_seconds'])
    
    print("\nBEST MODELS:")
    print(f"  Highest R²: {best_r2['model_name']} (R²={best_r2['test_metrics']['r2']:.4f})")
    print(f"  Lowest RMSE: {best_rmse['model_name']} (RMSE={best_rmse['test_metrics']['rmse']:.6f})")
    print(f"  Fastest: {fastest['model_name']} ({fastest['training_time_seconds']/60:.1f} minutes)")
    print("="*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train time series models')
    
    parser.add_argument('--model', type=str, help='Model name to train')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--category', type=str, help='List models in category')
    parser.add_argument('--compare', action='store_true', help='Compare all trained models')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced features')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--all', action='store_true', help='Train all models sequentially')
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.category:
        list_models(args.category)
    elif args.compare:
        compare_models()
    elif args.all:
        # Train all models
        print("Training ALL models sequentially...")
        for model_name in MODEL_REGISTRY.keys():
            print(f"\n\n{'='*80}")
            print(f"Starting: {model_name}")
            print(f"{'='*80}\n")
            train_model(model_name, args.enhanced, args.epochs, args.patience)
        
        # Compare at the end
        print("\n\nFinal comparison:")
        compare_models()
    elif args.model:
        train_model(args.model, args.enhanced, args.epochs, args.patience)
    else:
        print("Usage:")
        print("  python train_all.py --list                    # List all models")
        print("  python train_all.py --model dlinear           # Train DLinear")
        print("  python train_all.py --model patchtst --enhanced  # Train with enhanced features")
        print("  python train_all.py --all                     # Train all models")
        print("  python train_all.py --compare                 # Compare results")