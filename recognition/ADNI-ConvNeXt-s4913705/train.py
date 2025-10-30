import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import ConvNeXtADClassifier
from dataset import get_dataloaders

# ------------------------------
# Setup
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "/content/drive/MyDrive/ADNI/AD_NC/train"
test_dir  = "/content/drive/MyDrive/ADNI/AD_NC/test"

train_loader, test_loader, classes = get_dataloaders(train_dir, test_dir)
model = ConvNeXtADClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 25
patience = 3
best_val_acc = 0
counter = 0
train_acc_hist, val_acc_hist = [], []

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = 100 * correct / total
    train_acc_hist.append(train_acc)

    # Validation
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

    print(f"Epoch [{epoch}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    scheduler.step()

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "/content/drive/MyDrive/best_convnext_adni_optimized.pth")
    else:
        counter += 1
        if counter >= patience:
            print("⚠️ Early stopping triggered - accuracy plateaued.")
            break

# ------------------------------
# Plot Accuracy
# ------------------------------
plt.figure(figsize=(6,4))
plt.plot(train_acc_hist, label='Train Accuracy')
plt.plot(val_acc_hist, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ConvNeXt ADNI Training")
plt.legend()
plt.show()

print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
