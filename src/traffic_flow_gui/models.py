"""
Model definitions for v3 global-capacity inference (GUI only).
Contains FeedforwardNN_GlobalCapacity for FF model inference.
"""

import torch
import torch.nn as nn


class FeedforwardNN_GlobalCapacity(nn.Module):
    """Feedforward model trained on local + global capacity context.
    
    This is the only model type used in the inference-only GUI.
    It supports single-task (flow) or multi-task (flow+speed) prediction
    based on the output_dim parameter.
    """

    def __init__(self, input_size, hidden_sizes=(512, 256), dropout=0.2, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_sizes[1], output_dim)

    def forward(self, x):
        h = self.dropout1(self.relu1(self.fc1(x)))
        h = self.dropout2(self.relu2(self.fc2(h)))
        out = self.fc3(h)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
