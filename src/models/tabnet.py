# src/models/tabnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabNetEncoder(nn.Module):
    """
    TabNet - Attentive Interpretable Tabular Learning
    Best for financial data with mixed feature types
    """
    def __init__(
        self,
        input_size: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        super().__init__()
        
        self.input_size = input_size
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        
        # Initial batch normalization
        self.input_bn = nn.BatchNorm1d(input_size, momentum=momentum)
        
        # Shared and independent layers
        self.initial_splitter = FeatureTransformer(
            input_size, n_d + n_a, n_shared, n_independent, 
            virtual_batch_size, momentum
        )
        
        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        
        for step in range(n_steps):
            transformer = FeatureTransformer(
                input_size, n_d + n_a, n_shared, n_independent,
                virtual_batch_size, momentum
            )
            attention = AttentiveTransformer(
                n_a, input_size, virtual_batch_size, momentum
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)
    
    def forward(self, x):
        # Flatten sequence if needed
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x = x.reshape(batch_size, -1)
        
        x = self.input_bn(x)
        
        # Prior scale for attention
        prior = torch.ones(x.shape).to(x.device)
        M_loss = 0
        att_weights = []
        
        # Initial split
        x_a = self.initial_splitter(x)[:, self.n_d:]
        
        for step in range(self.n_steps):
            # Feature selection with attention
            M = self.att_transformers[step](prior, x_a)
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            
            # Update prior
            prior = torch.mul(self.gamma - M, prior)
            
            # Masked features
            masked_x = torch.mul(M, x)
            
            # Feature processing
            x_split = self.feat_transformers[step](masked_x)
            x_d = x_split[:, :self.n_d]
            x_a = x_split[:, self.n_d:]
            
            att_weights.append(M)
            
            if step == 0:
                output = x_d
            else:
                output = output + x_d
        
        return output, M_loss, att_weights


class FeatureTransformer(nn.Module):
    """Feature transformation block"""
    def __init__(self, input_size, output_size, n_shared, n_independent, 
                 virtual_batch_size, momentum):
        super().__init__()
        
        # Shared layers
        self.shared = nn.ModuleList()
        for i in range(n_shared):
            if i == 0:
                self.shared.append(nn.Linear(input_size, output_size))
            else:
                self.shared.append(nn.Linear(output_size, output_size))
            self.shared.append(nn.BatchNorm1d(output_size, momentum=momentum))
        
        # Independent layers
        self.independent = nn.ModuleList()
        for i in range(n_independent):
            if i == 0 and n_shared == 0:
                self.independent.append(nn.Linear(input_size, output_size))
            else:
                self.independent.append(nn.Linear(output_size, output_size))
            self.independent.append(nn.BatchNorm1d(output_size, momentum=momentum))
    
    def forward(self, x):
        # Shared layers
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:  # BatchNorm
                x = layer(x)
                x = F.gelu(x)
        
        # Independent layers
        for layer in self.independent:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:  # BatchNorm
                x = layer(x)
                x = F.gelu(x)
        
        return x


class AttentiveTransformer(nn.Module):
    """Attention mechanism for feature selection"""
    def __init__(self, input_size, output_size, virtual_batch_size, momentum):
        super().__init__()
        
        self.fc = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=momentum)
    
    def forward(self, prior, x):
        x = self.fc(x)
        x = self.bn(x)
        x = torch.mul(prior, x)
        return F.softmax(x, dim=-1)


class TabNetModel(nn.Module):
    """Complete TabNet model for time series prediction"""
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_targets: int = 424,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Flatten input for TabNet
        self.encoder = TabNetEncoder(
            input_size=input_size * seq_len,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(n_d, n_d // 2),
            nn.BatchNorm1d(n_d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_d // 2, num_targets)
        )
        
        self.n_steps = n_steps
    
    def forward(self, x, return_attention=False):
        # Flatten sequence
        batch_size, seq_len, features = x.shape
        x = x.reshape(batch_size, -1)
        
        # Encode with attention
        encoded, M_loss, att_weights = self.encoder(x)
        
        # Output
        output = self.output(encoded)
        
        if return_attention:
            return output, M_loss, att_weights
        return output


class XGBoostStyleNN(nn.Module):
    """
    Neural network mimicking XGBoost's boosting approach
    Multiple weak learners combined
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_trees: int = 10,
        tree_depth: int = 3,
        num_targets: int = 424,
        dropout: float = 0.1
    ):
        super().__init__()
        
        flat_input = input_size * seq_len
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(flat_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple "trees" (weak learners)
        self.trees = nn.ModuleList([
            self._create_tree(256, tree_depth, dropout)
            for _ in range(num_trees)
        ])
        
        # Tree outputs to predictions
        self.tree_outputs = nn.ModuleList([
            nn.Linear(256, num_targets)
            for _ in range(num_trees)
        ])
        
        # Learning rate for each tree (like XGBoost)
        self.tree_weights = nn.Parameter(torch.ones(num_trees) / num_trees)
    
    def _create_tree(self, hidden_size, depth, dropout):
        """Create a single tree (deep network)"""
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through each tree
        tree_predictions = []
        for tree, output_layer in zip(self.trees, self.tree_outputs):
            tree_out = tree(x)
            pred = output_layer(tree_out)
            tree_predictions.append(pred)
        
        # Weighted sum (boosting)
        tree_predictions = torch.stack(tree_predictions, dim=0)
        weights = F.softmax(self.tree_weights, dim=0).view(-1, 1, 1)
        
        output = (tree_predictions * weights).sum(dim=0)
        
        return output


class ResidualMLP(nn.Module):
    """
    Simple but effective: Deep MLP with residual connections
    Often beats complex models on tabular data
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        hidden_sizes: list = [512, 384, 256, 128],
        num_targets: int = 424,
        dropout: float = 0.3
    ):
        super().__init__()
        
        flat_input = input_size * seq_len
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(flat_input, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.blocks.append(
                ResidualBlock(
                    hidden_sizes[i], 
                    hidden_sizes[i + 1], 
                    dropout
                )
            )
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], num_targets)
    
    def forward(self, x):
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Input
        x = self.input_layer(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block with projection"""
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
            nn.BatchNorm1d(output_size)
        )
        
        # Projection for residual
        self.projection = None
        if input_size != output_size:
            self.projection = nn.Linear(input_size, output_size)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.layers(x)
        
        if self.projection is not None:
            residual = self.projection(residual)
        
        out = out + residual
        out = self.activation(out)
        
        return out


if __name__ == "__main__":
    # Test models
    batch_size = 32
    seq_len = 60
    input_size = 150
    num_targets = 424
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing TabNet...")
    model1 = TabNetModel(input_size, seq_len, num_targets)
    out1 = model1(x)
    print(f"Output: {out1.shape}")
    print(f"Params: {sum(p.numel() for p in model1.parameters()):,}")
    
    print("\nTesting XGBoostStyleNN...")
    model2 = XGBoostStyleNN(input_size, seq_len, num_targets=num_targets)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}")
    
    print("\nTesting ResidualMLP...")
    model3 = ResidualMLP(input_size, seq_len, num_targets=num_targets)
    out3 = model3(x)
    print(f"Output: {out3.shape}")
    print(f"Params: {sum(p.numel() for p in model3.parameters()):,}")