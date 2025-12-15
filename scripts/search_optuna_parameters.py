#!/usr/bin/env python3
"""
Hyperparameter Search Script
=============================
Performs automatic hyperparameter optimization for all models in MODEL_REGISTRY.
Supports internal (model params) and external (batch_size, lr) hyperparameters.
Results are saved to results/hyperparams_results.json
"""

import os
import json
import argparse
import itertools
import numpy as np
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === Import model registry ===
from src.models.registry import MODEL_REGISTRY  # adjust this path if needed


# === Synthetic dataset ===
def generate_synthetic_data(n_samples=512, input_dim=16, output_dim=1):
    """Generates a simple nonlinear synthetic regression dataset."""
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = (np.sin(X).sum(axis=1, keepdims=True) + 0.1 * np.random.randn(n_samples, 1)).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)


# === Training & evaluation ===
def train_and_eval(model_class, params, lr, batch_size, input_size, device='cpu', epochs=10):
    """Trains the model on synthetic data and returns validation loss."""
    X, y = generate_synthetic_data(input_dim=input_size)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Try to create model with input_size if supported
    try:
        model = model_class(input_size=input_size, **params).to(device)
    except TypeError as e:
        if "unexpected keyword argument 'input_size'" in str(e):
            model = model_class(**params).to(device)
        elif "missing 1 required positional argument" in str(e) and "input_size" in str(e):
            # Some models require input_size but it's not in params
            model = model_class(input_size=input_size, **params).to(device)
        else:
            raise RuntimeError(f"Model init failed for {model_class.__name__}: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            try:
                pred = model(xb)
            except Exception as e:
                raise RuntimeError(f"Forward pass failed for {model_class.__name__}: {e}")
            if pred.shape != yb.shape:
                # Handle mismatched output shapes
                try:
                    pred = pred.view_as(yb)
                except Exception:
                    raise RuntimeError(f"Output shape mismatch: got {pred.shape}, expected {yb.shape}")
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X.to(device))
        if pred.shape != y.shape:
            try:
                pred = pred.view_as(y)
            except Exception:
                raise RuntimeError(f"Validation output mismatch: got {pred.shape}, expected {y.shape}")
        val_loss = float(loss_fn(pred, y.to(device)))
    return val_loss


# === Parameter grid expansion ===
def expand_grid(param_grid):
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


# === Main hyperparameter search ===
def hyperparam_search(category=None, max_combinations=10, results_path="results/hyperparams_results.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results = {}

    print(f"\n🔍 Starting hyperparameter search on device: {device}\n")

    models_to_run = {
        name: cfg for name, cfg in MODEL_REGISTRY.items()
        if category is None or cfg['category'] == category
    }

    input_size = 16  # synthetic input dimension

    for name, cfg in tqdm(models_to_run.items(), desc="Models"):
        base_params = deepcopy(cfg.get('params', {}))
        lr = cfg.get('lr', 1e-3)
        batch_size = cfg.get('batch_size', 32)

        # External + internal search space
        search_space = {
            'lr': [lr, lr * 0.5, lr * 2],
            'batch_size': [batch_size, max(8, batch_size // 2), batch_size * 2],
        }

        # Extend with internal params
        for p, v in base_params.items():
            if isinstance(v, (int, float)):
                search_space[p] = list(set([v, max(1, int(v * 0.5)), int(v * 1.5)]))
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], int):
                search_space[p] = [
                    v,
                    [max(1, int(x * 0.5)) for x in v],
                    [int(x * 1.5) for x in v],
                ]

        combinations = list(itertools.islice(expand_grid(search_space), max_combinations))
        best_score = float('inf')
        best_params = None

        for combo in combinations:
            params = deepcopy(base_params)
            for k, v in combo.items():
                if k in ['lr', 'batch_size']:
                    continue
                params[k] = v

            # Clean incompatible arguments
            if "FTTransformer" in name and "dropout" in params:
                params.pop("dropout")

            try:
                score = train_and_eval(
                    cfg['class'],
                    params,
                    combo['lr'],
                    combo['batch_size'],
                    input_size,
                    device
                )
            except Exception as e:
                print(f"⚠️  {name} failed with params {combo}: {e}")
                continue

            if score < best_score:
                best_score = score
                best_params = combo

        results[name] = {
            'category': cfg.get('category', 'unknown'),
            'description': cfg.get('description', ''),
            'best_params': best_params,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat(),
        }

        print(f"\n✅ {name}: Best Score = {best_score:.4f}")
        if best_params:
            print(f"   → Params: {json.dumps(best_params, indent=4)}")

    # Save all results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n📁 Results saved to: {results_path}\n")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Hyperparameter Search for Model Registry")
    parser.add_argument("--category", type=str, default=None, help="Filter models by category (e.g. transformer, cnn, baseline)")
    parser.add_argument("--max_combinations", type=int, default=10, help="Limit combinations per model")
    parser.add_argument("--results", type=str, default="results/hyperparams_results.json", help="Path to save JSON results")
    args = parser.parse_args()

    hyperparam_search(
        category=args.category,
        max_combinations=args.max_combinations,
        results_path=args.results
    )
