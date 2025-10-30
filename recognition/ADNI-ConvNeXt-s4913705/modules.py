import torch
import torch.nn as nn
import timm

class ConvNeXtADClassifier(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2):
        super().__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=True, drop_rate=drop_rate)
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
