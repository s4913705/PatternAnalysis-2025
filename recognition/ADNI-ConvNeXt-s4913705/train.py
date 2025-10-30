# =========================================================
# train.py — Training Loop (v3.4)
# =========================================================
"""
Author: Darshan Shaji (s4913705)
Course: COMP3710 - Pattern Analysis
University of Queensland, 2025
"""

import time, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from dataset import get_dataloaders
from modules import get_model

def train_model(train_dir, test_dir, save_path, device="cuda", epochs=25, patience=4):
    """Train ConvNeXt ADNI model with detailed logs."""
    start_time = time.time()
    train_loader, test_loader, classes = get_dataloaders(train_dir, test_dir)
    model, criterion, optimizer, scheduler = get_model(device=device)
    print(f"🚀 Training {epochs} epochs | Start time: {datetime.now().strftime('%H:%M:%S')}")

    best_val_acc, counter = 0, 0
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, epoch_loss = 0, 0, 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_acc_hist.append(train_acc)

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total
        val_acc_hist.append(val_acc)

        print(f"📈 Epoch {epoch:02d}: Train={train_acc:.2f}% | Val={val_acc:.2f}% | LR={optimizer.param_groups[0]['lr']:.1e}")
        scheduler.step()

        # --- Early stopping ---
        if val_acc > best_val_acc:
            best_val_acc, counter = val_acc, 0
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model improved at epoch {epoch} → saved.")
        else:
            counter += 1
            if counter >= patience:
                print("⚠️ Early stopping.")
                torch.save(model.state_dict(), save_path.replace(".pth", "_final.pth"))
                print(f"💾 Final snapshot saved at {datetime.now().strftime('%H:%M:%S')}")
                break

    # --- Plot Accuracy Curves ---
    plt.figure(figsize=(6,4))
    plt.plot(train_acc_hist, label="Train Accuracy")
    plt.plot(val_acc_hist, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ConvNeXt ADNI Training (v3.4)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    duration = (time.time() - start_time) / 60
    print(f"🏁 Best Validation Accuracy: {best_val_acc:.2f}% | Total Training Time: {duration:.1f} min")

if __name__ == "__main__":
    train_model(
        train_dir="/content/drive/MyDrive/ADNI/AD_NC/train",
        test_dir="/content/drive/MyDrive/ADNI/AD_NC/test",
        save_path="/content/drive/MyDrive/best_convnext_adni_v3_4.pth"
    )
