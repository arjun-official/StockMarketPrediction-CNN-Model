
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model.image_dataset import ChartImageDataset
from model.image_model import ChartClassifier


BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.15
PATIENCE = 4


def _transforms(train: bool):
    t = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(t)


def train_classifier(labels_csv: str,
                     epochs: int = EPOCHS,
                     batch_size: int = BATCH_SIZE,
                     lr: float = LR,
                     weight_decay: float = WEIGHT_DECAY,
                     val_fraction: float = VAL_FRACTION,
                     patience: int = PATIENCE,
                     out_path: str = "chart_cnn.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = ChartImageDataset(labels_csv, transform=_transforms(train=True))
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ChartClassifier(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        va_loss /= len(val_loader.dataset)
        acc = correct / max(1, total)

        scheduler.step(va_loss)
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | acc {acc:.3f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        if va_loss + 1e-8 < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")
    return model.cpu()


def predict_class(model_path: str, image_path: str) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChartClassifier(num_classes=3)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    x = _transforms(train=False)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).cpu()
    pred = int(logits.argmax(dim=1).item())
    # map 0,1,2 -> -1,0,1 for short, neutral, long
    mapping = {-1:0, 0:1, 1:2}  # placeholder not used
    # We'll define order as [-1, 0, 1] = indices [0,1,2]
    return [-1, 0, 1][pred]


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))
