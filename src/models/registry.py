# src/models/registry.py

from src.models.cnn_attention import CNNAttention, WaveNetStyleModel
from src.models.tabnet import TabNetModel, ResidualMLP, XGBoostStyleNN
from src.models.ft_transformer import FTTransformer, PerformerModel
from src.models.nhits import NHiTS, NBeatsInterpretable
from src.models.patchtst import PatchTST, PatchTSTWithChannelIndependence, CrossFormer
from src.models.timesnet import TimesNet, AutoFormer, FEDformer
from src.models.dlinear import DLinear, NLinear, RLinear, FITS


MODEL_REGISTRY = {
    # === Simple & Fast Baselines ===
    'dlinear': {
        'class': DLinear,
        'params': {'moving_avg': 25, 'individual': True},
        'batch_size': 32,
        'lr': 1e-3,
        'category': 'baseline',
        'description': 'Decomposition + Linear - Simple but VERY effective baseline'
    },
    'nlinear': {
        'class': NLinear,
        'params': {'individual': True},
        'batch_size': 32,
        'lr': 1e-3,
        'category': 'baseline',
        'description': 'Normalized Linear - Handles distribution shift'
    },
    'rlinear': {
        'class': RLinear,
        'params': {'individual': True},
        'batch_size': 32,
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
