# src/models/lstm_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AttentionLayer(nn.Module):
    """
    Bahdanau-style attention mechanism
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention weights
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(
        self, 
        hidden: torch.Tensor,  # (batch, hidden_size)
        encoder_outputs: torch.Tensor  # (batch, seq_len, hidden_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: Current decoder hidden state
            encoder_outputs: All encoder hidden states
            
        Returns:
            context: Weighted sum of encoder outputs (batch, hidden_size)
            attention_weights: Attention weights (batch, seq_len)
        """
        # Compute attention scores
        # (batch, seq_len, hidden_size)
        hidden_expanded = hidden.unsqueeze(1).expand_as(encoder_outputs)
        
        # Energy calculation
        energy = torch.tanh(
            self.W_a(encoder_outputs) + self.U_a(hidden_expanded)
        )  # (batch, seq_len, hidden_size)
        
        # Attention scores
        scores = self.v_a(energy).squeeze(-1)  # (batch, seq_len)
        
        # Attention weights (softmax)
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Context vector (weighted sum)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            encoder_outputs  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights


class LSTMAttention(nn.Module):
    """
    LSTM with Attention for time series forecasting
    
    Architecture:
    1. Bidirectional LSTM encoder
    2. Attention mechanism
    3. Fully connected decoder
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_targets: int = 424,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.bidirectional = bidirectional
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = AttentionLayer(lstm_output_size)
        
        # Decoder (fully connected)
        self.decoder = nn.Sequential(
            nn.Linear(lstm_output_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_targets)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: (batch_size, num_targets)
            attention_weights: Optional (batch_size, seq_len)
        """
        batch_size = x.size(0)
        
        # Encode with LSTM
        encoder_outputs, (hidden, cell) = self.encoder(x)
        # encoder_outputs: (batch, seq_len, hidden_size * 2 if bidirectional)
        # hidden: (num_layers * 2, batch, hidden_size) if bidirectional
        
        # Get last hidden state for attention query
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden_last = torch.cat([
                hidden[-2, :, :],  # Forward last layer
                hidden[-1, :, :]   # Backward last layer
            ], dim=1)  # (batch, hidden_size * 2)
        else:
            hidden_last = hidden[-1, :, :]  # (batch, hidden_size)
        
        # Apply attention
        context, attention_weights = self.attention(hidden_last, encoder_outputs)
        
        # Decode to predictions
        predictions = self.decoder(context)  # (batch, num_targets)
        
        if return_attention:
            return predictions, attention_weights
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            attention_weights: (batch_size, seq_len)
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights


# Example usage and testing
if __name__ == "__main__":
    # Model configuration
    INPUT_SIZE = 150  # Number of features
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    NUM_TARGETS = 424
    SEQ_LEN = 60
    BATCH_SIZE = 32
    
    # Create model
    model = LSTMAttention(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_targets=NUM_TARGETS,
        dropout=0.2,
        bidirectional=True
    )
    
    # Print model info
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test with attention weights
    output_with_attn, attn_weights = model(dummy_input, return_attention=True)
    print(f"Attention weights shape: {attn_weights.shape}")
    
    print("\n✓ Model test passed!")