# =========================================================
# dataset.py — ADNI Dataset Loader (Final v3.1)
# =========================================================
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(train_dir, test_dir, batch_size=32, num_workers=2):
    """
    Loads ADNI Alzheimer's dataset with augmentations and normalization.
    Now includes subtle Gaussian blur for stronger generalization.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("Dataset path missing — verify ADNI directories!")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"📂 ADNI Datasets Ready → Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {train_ds.classes}")
    return train_loader, test_loader, train_ds.classes
