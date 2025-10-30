# =========================================================
# predict.py — Evaluation (v3.4)
# =========================================================
"""
Author: Darshan Shaji (s4913705)
Course: COMP3710 - Pattern Analysis
University of Queensland, 2025
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from dataset import get_dataloaders
from modules import ConvNeXtADNI

def evaluate_model(weights_path, train_dir, test_dir, device="cuda"):
    """Evaluate ConvNeXt ADNI model and show confusion matrix + metrics."""
    _, test_loader, classes = get_dataloaders(train_dir, test_dir)
    model = ConvNeXtADNI(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="🔍 Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    print("\n✅ Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='plasma', values_format='d')
    plt.title("Confusion Matrix — ADNI ConvNeXt (v3.4)")
    plt.show()

if __name__ == "__main__":
    evaluate_model(
        weights_path="/content/drive/MyDrive/best_convnext_adni_v3_4.pth",
        train_dir="/content/drive/MyDrive/ADNI/AD_NC/train",
        test_dir="/content/drive/MyDrive/ADNI/AD_NC/test"
    )
