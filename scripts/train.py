#!/usr/bin/env python3
"""
train.py - Universal training script for Mitsui Commodity Prediction Challenge

КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ:
- Использует MitsuiLoss (Spearman correlation) вместо MSE
- Загружает данные из train_features_v2.csv (с lagged targets)
- Вычисляет Spearman correlation в метриках
- Правильное временное разбиение данных
"""

import torch
import sys
from pathlib import Path
import numpy as np
import json
import argparse
from datetime import datetime
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.append(str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.models.registry import MODEL_REGISTRY


# Model registry imported from src.models.registry
# All model configurations are now centralized in src/models/registry.py


def list_models(category=None):
    """List available models"""
    print("\n" + "="*80)
    print("AVAILABLE MODELS - Mitsui Commodity Challenge")
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
        
        for cat in ['baseline', 'tabular', 'cnn', 'transformer', 'advanced']:
            if cat in categories:
                print(f"\n{cat.upper()}:")
                for name, config in categories[cat]:
                    print(f"  {name:20} - {config['description']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR MITSUI:")
    print("  1. Start with: dlinear (fastest, often best for financial data)")
    print("  2. Try tabular: residual_mlp, xgboost_nn (good for commodity features)")
    print("  3. Advanced: nhits (SOTA forecasting)")
    print("="*80)
    print("\nNOTE: All models now use MitsuiLoss (Spearman correlation)")
    print("="*80)


def train_model(
    model_name, 
    epochs=100, 
    patience=15, 
    lookback=60,
    loss_type='mitsui',
    spearman_weight=0.7
):
    """
    Train a specific model for Mitsui competition
    
    Args:
        model_name: Name of model from MODEL_REGISTRY
        epochs: Maximum epochs
        patience: Early stopping patience
        lookback: Temporal lookback window
        loss_type: 'mitsui', 'spearman', or 'directional'
        spearman_weight: Weight for Spearman component (if using mitsui)
    """
    
    if model_name not in MODEL_REGISTRY:
        print(f"\nERROR: Unknown model '{model_name}'")
        list_models()
        return None
    
    config = MODEL_REGISTRY[model_name]
    
    print("\n" + "="*80)
    print(f"TRAINING: {model_name.upper()}")
    print("="*80)
    print(f"Category:    {config['category']}")
    print(f"Description: {config['description']}")
    print(f"Loss:        {loss_type} (Spearman weight: {spearman_weight:.0%})")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'model_name': model_name,
        'input_size': None,  # Will be determined from data
        'num_targets': 424,
        'lookback': lookback,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'batch_size': config['batch_size'],
        'learning_rate': config['lr'],
        'weight_decay': 1e-5,
        'num_epochs': epochs,
        'early_stopping_patience': patience,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'loss_type': loss_type,
        'spearman_weight': spearman_weight
    }
    
    print(f"\nConfiguration:")
    print(f"  Device:           {CONFIG['device']}")
    print(f"  Batch size:       {CONFIG['batch_size']}")
    print(f"  Learning rate:    {CONFIG['learning_rate']}")
    print(f"  Max epochs:       {CONFIG['num_epochs']}")
    print(f"  Early stopping:   {CONFIG['early_stopping_patience']} epochs")
    print(f"  Lookback:         {CONFIG['lookback']} steps")
    print(f"  Train/Val/Test:   {CONFIG['train_ratio']:.0%}/{CONFIG['val_ratio']:.0%}/{1-CONFIG['train_ratio']-CONFIG['val_ratio']:.0%}")
    
    # ====================================================================
    # [1/6] LOAD DATA
    # ====================================================================
    print("\n" + "="*80)
    print("[1/6] Loading data...")
    print("="*80)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path='data/processed/train_features_v2.csv',  # НОВЫЕ ДАННЫЕ!
            train_ratio=CONFIG['train_ratio'],
            val_ratio=CONFIG['val_ratio'],
            batch_size=CONFIG['batch_size'],
            lookback=CONFIG['lookback'],
            num_workers=CONFIG['num_workers']
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n⚠️  Did you run feature engineering?")
        print("Run: python scripts/create_proper_features.py")
        return None
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Get dimensions from data
    try:
        sample_x, sample_y = next(iter(train_loader))
        CONFIG['input_size'] = sample_x.shape[-1]
        CONFIG['num_targets'] = sample_y.shape[-1]
        
        print(f"\n✓ Data loaded successfully:")
        print(f"  Input shape:  {sample_x.shape} (batch, lookback, features)")
        print(f"  Output shape: {sample_y.shape} (batch, targets)")
        print(f"  Features:     {CONFIG['input_size']}")
        print(f"  Targets:      {CONFIG['num_targets']}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")
        
        # Target statistics
        print(f"\n  Target statistics:")
        print(f"    Mean: {sample_y.mean():.6f}")
        print(f"    Std:  {sample_y.std():.6f}")
        print(f"    Min:  {sample_y.min():.6f}")
        print(f"    Max:  {sample_y.max():.6f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: Could not load sample batch: {e}")
        return None
    
    # ====================================================================
    # [2/6] CREATE MODEL
    # ====================================================================
    print("\n" + "="*80)
    print(f"[2/6] Creating {model_name} model...")
    print("="*80)
    
    model_params = config['params'].copy()
    model_params['input_size'] = CONFIG['input_size']
    model_params['num_targets'] = CONFIG['num_targets']
    
    # Add seq_len for models that need it
    needs_seq_len = [
        'tabnet', 'ft_transformer', 'performer', 'residual_mlp', 
        'xgboost_nn', 'nhits', 'nbeats', 'timesnet', 'autoformer', 
        'fedformer', 'dlinear', 'nlinear', 'rlinear', 'fits',
        'patchtst', 'patchtst_ci', 'crossformer'
    ]
    
    if model_name in needs_seq_len:
        model_params['seq_len'] = CONFIG['lookback']
    
    try:
        model = config['class'](**model_params)
        model = model.to(CONFIG['device'])
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully:")
        print(f"  Total parameters:      {total_params:,}")
        print(f"  Trainable parameters:  {trainable_params:,}")
        print(f"  Model size:            {total_params * 4 / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ ERROR creating model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test forward pass
    print("\n  Testing forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            sample_x = sample_x.to(CONFIG['device'])
            sample_y = sample_y.to(CONFIG['device'])
            
            test_out = model(sample_x)
            
            # Validate output
            assert test_out.shape == sample_y.shape, \
                f"Shape mismatch: {test_out.shape} vs {sample_y.shape}"
            assert not torch.isnan(test_out).any(), \
                "Model outputs NaN!"
            assert not torch.isinf(test_out).any(), \
                "Model outputs Inf!"
            
            print(f"  ✓ Forward pass OK: {test_out.shape}")
            print(f"    Output range: [{test_out.min():.6f}, {test_out.max():.6f}]")
            
        except Exception as e:
            print(f"\n  ❌ ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ====================================================================
    # [3/6] CALCULATE BASELINE
    # ====================================================================
    print("\n" + "="*80)
    print("[3/6] Calculating baseline metrics...")
    print("="*80)
    
    # Collect all validation targets
    all_val_targets = []
    for _, y in val_loader:
        all_val_targets.append(y.numpy())
    all_val_targets = np.concatenate(all_val_targets)
    
    # Naive baseline: predict mean
    baseline_pred = np.mean(all_val_targets, axis=0, keepdims=True)
    baseline_pred = np.repeat(baseline_pred, len(all_val_targets), axis=0)
    
    baseline_rmse = np.sqrt(np.mean((all_val_targets - baseline_pred) ** 2))
    
    # Baseline Spearman (should be ~0)
    baseline_spearman = []
    for i in range(all_val_targets.shape[1]):
        if all_val_targets[:, i].std() > 1e-6:
            try:
                corr, _ = spearmanr(baseline_pred[:, i], all_val_targets[:, i])
                if not np.isnan(corr):
                    baseline_spearman.append(corr)
            except:
                pass
    
    baseline_spearman_mean = np.mean(baseline_spearman) if baseline_spearman else 0.0
    
    print(f"Baseline (predict mean):")
    print(f"  RMSE:              {baseline_rmse:.6f}")
    print(f"  Spearman:          {baseline_spearman_mean:.6f}")
    print(f"\n⚠️  Model must beat these baselines!")
    
    # ====================================================================
    # [4/6] TRAIN MODEL
    # ====================================================================
    print("\n" + "="*80)
    print("[4/6] Training model...")
    print("="*80)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        save_dir=f'models/checkpoints/{model_name}',
        loss_type=CONFIG['loss_type'],
        spearman_weight=CONFIG['spearman_weight']
    )
    
    start_time = datetime.now()
    
    try:
        trainer.train(
            num_epochs=CONFIG['num_epochs'],
            early_stopping_patience=CONFIG['early_stopping_patience']
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Loading best model so far...")
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # ====================================================================
    # [5/6] EVALUATE ON TEST SET
    # ====================================================================
    print("\n" + "="*80)
    print("[5/6] Final evaluation on test set...")
    print("="*80)
    
    # Load best model
    val_metrics = trainer.load_best_model()
    
    # Evaluate on test
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\nGenerating predictions on test set...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(CONFIG['device'])
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    print(f"✓ Generated {len(all_preds)} predictions")
    
    # Calculate metrics
    print("\nCalculating test metrics...")
    
    # 1. MSE/RMSE/MAE
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # 2. R² (может быть отрицательным для efficient markets!)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
    
    # 3. Directional accuracy
    dir_acc = np.mean(np.sign(all_preds) == np.sign(all_targets))
    
    # 4. ⭐ SPEARMAN CORRELATION ⭐
    spearman_correlations = []
    for i in range(all_preds.shape[1]):
        if all_targets[:, i].std() > 1e-6 and all_preds[:, i].std() > 1e-6:
            try:
                corr, _ = spearmanr(all_preds[:, i], all_targets[:, i])
                if not np.isnan(corr):
                    spearman_correlations.append(corr)
            except:
                continue
    
    mean_spearman = np.mean(spearman_correlations) if spearman_correlations else 0.0
    std_spearman = np.std(spearman_correlations) if len(spearman_correlations) > 1 else 0.0
    
    # Modified Sharpe Ratio (Mitsui metric!)
    if std_spearman > 1e-6:
        modified_sharpe = mean_spearman / std_spearman
    else:
        modified_sharpe = 0.0
    
    # Baseline comparison
    test_baseline_rmse = np.sqrt(np.mean((all_targets - np.mean(all_targets, axis=0)) ** 2))
    improvement_pct = ((test_baseline_rmse - rmse) / test_baseline_rmse) * 100
    
    # ====================================================================
    # [6/6] REPORT RESULTS
    # ====================================================================
    print("\n" + "="*80)
    print(f"[6/6] FINAL RESULTS - {model_name.upper()}")
    print("="*80)
    
    print(f"\nTest Metrics:")
    print(f"  RMSE:              {rmse:.6f}")
    print(f"  MAE:               {mae:.6f}")
    print(f"  R²:                {r2:.4f}")
    print(f"  Directional Acc:   {dir_acc:.4f} ({dir_acc*100:.1f}%)")
    
    print(f"\n⭐ Competition Metrics:")
    print(f"  Spearman (mean):   {mean_spearman:.6f} ± {std_spearman:.6f}")
    print(f"  Modified Sharpe:   {modified_sharpe:.4f}")
    print(f"  (higher is better)")
    
    print(f"\nBaseline Comparison:")
    print(f"  Baseline RMSE:     {test_baseline_rmse:.6f}")
    print(f"  Improvement:       {improvement_pct:+.2f}%")
    
    print(f"\nTraining Info:")
    print(f"  Training time:     {training_time/60:.1f} minutes")
    print(f"  Parameters:        {total_params:,}")
    print(f"  Best val epoch:    {val_metrics.get('epoch', 'N/A')}")
    
    # Interpretation
    print(f"\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    
    if mean_spearman > 0.01:
        print("✓ GOOD: Positive Spearman correlation - model has predictive power!")
    elif mean_spearman > 0:
        print("⚠️  WEAK: Small positive Spearman - model barely beats random")
    else:
        print("❌ BAD: Negative Spearman - model worse than random!")
    
    if r2 < 0:
        print("⚠️  NOTE: Negative R² is NORMAL for efficient markets")
        print("   (commodity prices near random walk)")
    
    if dir_acc > 0.52:
        print(f"✓ GOOD: {dir_acc*100:.1f}% directional accuracy (above 50%)")
    else:
        print(f"⚠️  WEAK: {dir_acc*100:.1f}% directional accuracy (near random)")
    
    print("="*80)
    
    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    # results = {
    #     'model_name': model_name,
    #     'timestamp': datetime.now().isoformat(),
    #     'config': {k: v for k, v in CONFIG.items() if k != 'device'},
    #     'test_metrics': {
    #         'rmse': float(rmse),
    #         'mae': float(mae),
    #         'r2': float(r2),
    #         'directional_accuracy': float(dir_acc),
    #         'spearman_mean': float(mean_spearman),
    #         'spearman_std': float(std_spearman),
    #         'modified_sharpe': float(modified_sharpe)
    #     },
    #     'baseline': {
    #         'rmse': float(test_baseline_rmse),
    #         'spearman': float(baseline_spearman_mean)
    #     },
    #     'improvement_pct': float(improvement_pct),
    #     'training_time_seconds': float(training_time),
    #     'num_parameters': int(total_params),
    #     'best_val_metrics': val_metrics
    # }
    
    def to_python_type(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_python_type(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'config': to_python_type({k: v for k, v in CONFIG.items() if k != 'device'}),
        'test_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'directional_accuracy': float(dir_acc),
            'spearman_mean': float(mean_spearman),
            'spearman_std': float(std_spearman),
            'modified_sharpe': float(modified_sharpe)
        },
        'baseline': {
            'rmse': float(test_baseline_rmse),
            'spearman': float(baseline_spearman_mean)
        },
        'improvement_pct': float(improvement_pct),
        'training_time_seconds': float(training_time),
        'num_parameters': int(total_params),
        'best_val_metrics': to_python_type(val_metrics)  # ← ИСПРАВЛЕНО
    }

    # Save to file
    Path('outputs').mkdir(exist_ok=True)
    output_file = f'outputs/{model_name}_v2_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*80)
    
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
        try:
            with open(result_file) as f:
                data = json.load(f)
                if 'test_metrics' in data and 'model_name' in data:
                    all_results.append(data)
        except Exception as e:
            print(f"Error loading {result_file.name}: {e}")
            continue
    
    if not all_results:
        print("No valid model results found!")
        return
    
    if models_to_compare:
        all_results = [r for r in all_results if r['model_name'] in models_to_compare]
    
    if not all_results:
        print("No matching models found!")
        return
    
    # Sort by Spearman correlation (главная метрика!)
    all_results.sort(
        key=lambda x: x['test_metrics'].get('spearman_mean', -999), 
        reverse=True
    )
    
    # Print comparison table
    print("\n" + "="*120)
    print("MODEL COMPARISON - Sorted by Spearman Correlation")
    print("="*120)
    print(f"{'Model':<20} {'Spearman':<12} {'Mod.Sharpe':<12} {'Dir.Acc':<10} {'RMSE':<12} {'R²':<10} {'Params':<12} {'Time(m)':<10}")
    print("-"*120)
    
    for result in all_results:
        name = result['model_name']
        spearman = result['test_metrics'].get('spearman_mean', 0)
        sharpe = result['test_metrics'].get('modified_sharpe', 0)
        dir_acc = result['test_metrics'].get('directional_accuracy', 0)
        rmse = result['test_metrics'].get('rmse', 0)
        r2 = result['test_metrics'].get('r2', 0)
        params = result.get('num_parameters', 0)
        time_m = result.get('training_time_seconds', 0) / 60
        
        print(f"{name:<20} {spearman:>11.6f} {sharpe:>11.4f} {dir_acc:>9.4f} {rmse:>11.6f} {r2:>9.4f} {params:>11,} {time_m:>9.1f}")
    
    print("="*120)
    
    # Best models
    best_spearman = max(all_results, key=lambda x: x['test_metrics'].get('spearman_mean', -999))
    best_sharpe = max(all_results, key=lambda x: x['test_metrics'].get('modified_sharpe', -999))
    best_dir = max(all_results, key=lambda x: x['test_metrics'].get('directional_accuracy', 0))
    fastest = min(all_results, key=lambda x: x.get('training_time_seconds', float('inf')))
    
    print("\nBEST MODELS:")
    print(f"  ⭐ Best Spearman:       {best_spearman['model_name']:<20} ({best_spearman['test_metrics'].get('spearman_mean', 0):.6f})")
    print(f"  ⭐ Best Mod. Sharpe:    {best_sharpe['model_name']:<20} ({best_sharpe['test_metrics'].get('modified_sharpe', 0):.4f})")
    print(f"  📊 Best Dir. Accuracy:  {best_dir['model_name']:<20} ({best_dir['test_metrics'].get('directional_accuracy', 0):.4f})")
    print(f"  ⚡ Fastest:             {fastest['model_name']:<20} ({fastest.get('training_time_seconds', 0)/60:.1f} min)")
    
    print("\n" + "="*120)
    print("RECOMMENDATION: Submit the model with highest Spearman correlation to Kaggle!")
    print("="*120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train models for Mitsui Commodity Prediction Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --list                    # List all available models
  python train.py --model dlinear           # Train DLinear baseline
  python train.py --model residual_mlp      # Train MLP
  python train.py --compare                 # Compare all results
  python train.py --all --epochs 50         # Train all models (quick)
  
For Mitsui competition:
  1. python scripts/create_proper_features.py  # First create features
  2. python train.py --model dlinear           # Train your models
  3. python train.py --compare                 # Compare results
        """
    )
    
    parser.add_argument('--model', type=str, 
                       help='Model name to train (see --list)')
    parser.add_argument('--list', action='store_true', 
                       help='List available models')
    parser.add_argument('--category', type=str, 
                       help='List models in specific category')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare all trained models')
    parser.add_argument('--all', action='store_true', 
                       help='Train all models sequentially')
    
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Maximum epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=15, 
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--lookback', type=int, default=60, 
                       help='Lookback window size (default: 60)')
    
    parser.add_argument('--loss', type=str, default='mitsui',
                       choices=['mitsui', 'spearman', 'directional'],
                       help='Loss function type (default: mitsui)')
    parser.add_argument('--spearman-weight', type=float, default=0.7,
                       help='Spearman weight for mitsui loss (default: 0.7)')
    
    args = parser.parse_args()
    
    # Execute commands
    if args.list:
        list_models()
    elif args.category:
        list_models(args.category)
    elif args.compare:
        compare_models()
    elif args.all:
        print("\n" + "="*80)
        print("TRAINING ALL MODELS FOR MITSUI COMPETITION")
        print("="*80)
        print(f"Epochs: {args.epochs}")
        print(f"Patience: {args.patience}")
        print(f"Loss: {args.loss}")
        print("="*80)
        
        results = []
        for model_name in MODEL_REGISTRY.keys():
            print(f"\n\n{'='*80}")
            print(f"Starting: {model_name}")
            print(f"{'='*80}\n")
            
            result = train_model(
                model_name, 
                epochs=args.epochs,
                patience=args.patience,
                lookback=args.lookback,
                loss_type=args.loss,
                spearman_weight=args.spearman_weight
            )
            
            if result:
                results.append(result)
        
        print(f"\n\n{'='*80}")
        print(f"TRAINING COMPLETED: {len(results)}/{len(MODEL_REGISTRY)} models")
        print(f"{'='*80}\n")
        
        # Final comparison
        compare_models()
        
    elif args.model:
        train_model(
            args.model,
            epochs=args.epochs,
            patience=args.patience,
            lookback=args.lookback,
            loss_type=args.loss,
            spearman_weight=args.spearman_weight
        )
    else:
        parser.print_help()