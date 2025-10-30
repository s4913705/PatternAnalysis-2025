# =========================================================
# dataset.py — ADNI Alzheimer's Dataset Loader (v3.2)
# =========================================================
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(train_dir, test_dir, batch_size=32, num_workers=2):
    """
    Loads ADNI Alzheimer's dataset and returns dataloaders with appropriate
    augmentations, normalization, and validation pipeline.
    """
    # --- Training augmentations ---
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomRotation(12),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Validation/test preprocessing ---
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Directory verification ---
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("❌ Dataset paths not found. Verify train/test folders.")

    # --- Dataset loading ---
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir, transform=test_tf)

    # --- DataLoader setup ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"📦 Loaded {len(train_ds)} training and {len(test_ds)} testing samples.")
    print(f"✅ Classes: {train_ds.classes}")
    return train_loader, test_loader, train_ds.classes
