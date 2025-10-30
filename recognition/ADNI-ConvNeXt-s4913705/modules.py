# =========================================================
# modules.py — ConvNeXt Model Setup (v3.4)
# =========================================================
"""
Author: Darshan Shaji (s4913705)
Course: COMP3710 - Pattern Analysis
University of Queensland, 2025
"""

import torch
import torch.nn as nn
import timm
import numpy as np, random, time

def set_seed(seed: int = 4913705):
    """Ensure deterministic results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🧩 Seed fixed: {seed}")

class ConvNeXtADNI(nn.Module):
    """Fine-tuned ConvNeXt-Tiny for binary classification."""
    def __init__(self, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.model = timm.create_model("convnext_tiny", pretrained=True, drop_rate=drop_rate)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Total parameters: {total:,} | Trainable: {trainable:,}")

def get_model(device="cuda", lr=3e-5, weight_decay=1e-4):
    """Return initialized model, optimizer, scheduler."""
    set_seed()
    model = ConvNeXtADNI().to(device)
    count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    print(f"✅ Model ready on {device} | LR={lr}")
    return model, criterion, optimizer, scheduler
