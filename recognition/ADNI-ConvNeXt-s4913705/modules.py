# =========================================================
# modules.py — ConvNeXt Model Setup (v3.3)
# =========================================================
"""
Author: Darshan Shaji (s4913705)
Course: COMP3710 - Pattern Analysis
University of Queensland, 2025
"""

import torch
import torch.nn as nn
import timm
import numpy as np, random

def set_seed(seed: int = 4913705):
    """Ensure deterministic results for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🧩 Random seed fixed at: {seed}")

class ConvNeXtADNI(nn.Module):
    """ConvNeXt-Tiny model fine-tuned for AD vs NC classification."""
    def __init__(self, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.model = timm.create_model("convnext_tiny", pretrained=True, drop_rate=drop_rate)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(device="cuda", lr=3e-5, weight_decay=1e-4):
    """Return model, criterion, optimizer, and scheduler."""
    set_seed()
    model = ConvNeXtADNI().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    print(f"✅ Model initialized on {device} | Dropout: 0.3 | LR: {lr}")
    return model, criterion, optimizer, scheduler
