# =========================================================
# train.py — Training Loop (v3.2)
# =========================================================
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from modules import get_model

def train_model(train_dir, test_dir, save_path, device="cuda", epochs=25, patience=4):
    """Train ConvNeXt ADNI model with accuracy tracking and early stopping."""
    train_loader, test_loader, classes = get_dataloaders(train_dir, test_dir)
    model, criterion, optimizer, scheduler = get_model(device=device)

    best_val_acc, counter = 0, 0
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, epoch_loss = 0, 0, 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} — Training"):
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

        print(f"📈 Epoch {epoch:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Loss: {epoch_loss/len(train_loader):.4f}")
        scheduler.step()

        # --- Early stopping ---
        if val_acc > best_val_acc:
            best_val_acc, counter = val_acc, 0
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model improved — saved at {save_path}")
        else:
            counter += 1
            if counter >= patience:
                print("⛔ Early stopping triggered — no improvement.")
                break

    # --- Plot accuracy curves ---
    plt.figure(figsize=(6,4))
    plt.plot(train_acc_hist, label="Train Accuracy")
    plt.plot(val_acc_hist, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ConvNeXt ADNI Training Curve (v3.2)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"🏁 Final Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model(
        train_dir="/content/drive/MyDrive/ADNI/AD_NC/train",
        test_dir="/content/drive/MyDrive/ADNI/AD_NC/test",
        save_path="/content/drive/MyDrive/best_convnext_adni_v3_2.pth"
    )
