# visualize_optuna.py

import joblib
import optuna
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_study():
    """Visualize Optuna optimization results"""
    
    # Load study
    study_path = Path('outputs/optuna_study.pkl')
    if not study_path.exists():
        print("Error: optuna_study.pkl not found. Run optimize_hyperparameters.py first.")
        return
    
    study = joblib.load(study_path)
    
    print("Creating visualizations...")
    
    # Create output directory
    viz_dir = Path('outputs/optuna_viz')
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Optimization history
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(viz_dir / 'optimization_history.png', dpi=150)
    print("✓ Saved: optimization_history.png")
    plt.close()
    
    # 2. Parameter importances
    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(viz_dir / 'param_importances.png', dpi=150)
        print("✓ Saved: param_importances.png")
        plt.close()
    except:
        print("⚠ Skipped param importances (need more trials)")
    
    # 3. Parallel coordinate plot
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig(viz_dir / 'parallel_coordinate.png', dpi=150)
    print("✓ Saved: parallel_coordinate.png")
    plt.close()
    
    # 4. Slice plot
    fig = optuna.visualization.matplotlib.plot_slice(study)
    plt.tight_layout()
    plt.savefig(viz_dir / 'slice_plot.png', dpi=150)
    print("✓ Saved: slice_plot.png")
    plt.close()
    
    print(f"\nAll visualizations saved to {viz_dir}/")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation loss: {study.best_value:.6f}")


if __name__ == "__main__":
    visualize_study()