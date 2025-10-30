# =========================================================
# predict.py — Evaluation (v3.3)
# =========================================================
"""
Author: Darshan Shaji (s4913705)
Course: COMP3710 - Pattern Analysis
University of Queensland, 2025
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import get_dataloaders
from modules import ConvNeXtADNI

def evaluate_model(weights_path, train_dir, test_dir, device="cuda"):
    """Evaluate trained model on ADNI dataset with per-class accuracy."""
    _, test_loader, classes = get_dataloaders(train_dir, test_dir)
    model = ConvNeXtADNI(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    per_class = torch.zeros(len(classes), 2)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="🔍 Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for label, pred in zip(labels, preds):
                per_class[label][1] += 1
                if label == pred:
                    per_class[label][0] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    overall_acc = 100 * correct / total
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.2f}%")
    print("📊 Per-Class Accuracy:")
    for i, cls in enumerate(classes):
        if per_class[i][1] > 0:
            acc = 100 * per_class[i][0] / per_class[i][1]
            print(f"   • {cls:<10}: {acc:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Purples', values_format='d')
    plt.title("Confusion Matrix — ADNI ConvNeXt (v3.3)")
    plt.show()

if __name__ == "__main__":
    evaluate_model(
        weights_path="/content/drive/MyDrive/best_convnext_adni_v3_3.pth",
        train_dir="/content/drive/MyDrive/ADNI/AD_NC/train",
        test_dir="/content/drive/MyDrive/ADNI/AD_NC/test"
    )
