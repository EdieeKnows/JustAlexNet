# main.py
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ---- 1. Set Paths ----
dataset_root = Path("/app/data/ImageNet")   # <-- modify

# ---- 2. Define Transforms ----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- 3. Load Full Dataset ----
full_dataset = ImageFolder(root=dataset_root, transform=train_transform)

# ---- 4. Split ----
train_ratio = 0.8
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Replace validation transform
val_dataset.dataset.transform = val_transform

# ---- 5. DataLoaders ----
batch_size = 32
num_workers = 16

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)

# ---- 6. Quick check ----
if __name__ == "__main__":
    print("\n=== Dataset Statistics ===")
    print(f"Total images: {len(full_dataset)}")
    print("Classes (index → name):")
    for idx, cls in enumerate(full_dataset.classes):
        print(f"  {idx:2d} → {cls}")