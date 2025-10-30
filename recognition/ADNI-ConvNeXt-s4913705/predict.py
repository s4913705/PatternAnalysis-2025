# =========================================================
# predict.py — Evaluation Script (Final v3.1)
# =========================================================
import torch
from tqdm import tqdm
from dataset import get_dataloaders
from modules import ConvNeXtADNI
import matplotlib.pyplot as plt

def evaluate_model(weights_path, train_dir, test_dir, device="cuda"):
    """Evaluate trained model and print per-class accuracy with example images."""
    _, test_loader, classes = get_dataloaders(train_dir, test_dir)
    model = ConvNeXtADNI(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    per_class = torch.zeros(len(classes), 2)

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="🔍 Evaluating Model"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                per_class[label][1] += 1
                if label == pred:
                    per_class[label][0] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"\n✅ Overall Test Accuracy: {100*correct/total:.2f}%")
    print("📊 Class-wise Accuracy:")
    for i, cls in enumerate(classes):
        if per_class[i][1] > 0:
            acc = 100 * per_class[i][0] / per_class[i][1]
            print(f"   • {cls:<10}: {acc:.2f}%")

    # --- Optional: visualize 4 sample predictions ---
    batch = next(iter(test_loader))
    imgs, labels = batch
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        axes[i].imshow(imgs[i].permute(1, 2, 0))
        axes[i].set_title(f"True: {classes[labels[i]]}")
        axes[i].axis('off')
    plt.suptitle("Example Test Samples (Unnormalized)")
    plt.show()

if __name__ == "__main__":
    evaluate_model(
        weights_path="/content/drive/MyDrive/best_convnext_adni_v3_1.pth",
        train_dir="/content/drive/MyDrive/ADNI/AD_NC/train",
        test_dir="/content/drive/MyDrive/ADNI/AD_NC/test"
    )
