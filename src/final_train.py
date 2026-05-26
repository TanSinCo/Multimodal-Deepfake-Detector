
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from tqdm import tqdm

from multimodal_cache_dataset import (
    MultimodalCacheDataset
)

from models.fusion_model import FusionModel


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# DATASET
# ==========================================

dataset = MultimodalCacheDataset(
    "/kaggle/working/project/multimodal_cache"
)

print("Dataset size:", len(dataset))


# ==========================================
# SPLIT
# ==========================================

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(

    dataset,

    [train_size, val_size]
)




def collate_fn(batch):

    batch = [
        b for b in batch
        if b is not None
    ]

    if len(batch) == 0:
        return None

    return {

        key: torch.stack([
            item[key]
            for item in batch
        ])

        for key in batch[0]
    }


# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(

    train_dataset,

    batch_size=4,

    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(

    val_dataset,

    batch_size=4,

    shuffle=False,
    collate_fn=collate_fn
)


# ==========================================
# MODEL
# ==========================================

model = FusionModel().to(device)


# ==========================================
# LOSS
# ==========================================

criterion = nn.BCEWithLogitsLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-4
)


# ==========================================
# CHECKPOINTS
# ==========================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)


best_acc = 0.0

train_losses = []

val_losses = []

val_accuracies = []



# ==========================================
# TRAINING
# ==========================================

EPOCHS = 3

for epoch in range(EPOCHS):

    print(f"\n===== EPOCH {epoch+1} =====")

    # ======================================
    # TRAIN
    # ======================================

    model.train()

    train_loss = 0

    train_preds = []

    train_labels = []

    for batch in tqdm(train_loader):

        frames = batch["frames"].to(device)

        rppg = batch["rppg"].to(device)

        blink = batch["blink"].to(device)

        motion = batch["motion"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(

            frames,
            rppg,
            blink,
            motion

        ).squeeze(1)

        labels = labels.squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        train_preds.extend(
            preds.cpu().numpy()
        )

        train_labels.extend(
            labels.cpu().numpy()
        )


    train_acc = accuracy_score(
        train_labels,
        train_preds
    )


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_loss = 0

    val_preds = []

    val_labels = []

    with torch.no_grad():

        for batch in tqdm(val_loader):

            frames = batch["frames"].to(device)

            rppg = batch["rppg"].to(device)

            blink = batch["blink"].to(device)

            motion = batch["motion"].to(device)

            labels = batch["label"].to(device)

            outputs = model(

                frames,
                rppg,
                blink,
                motion

            ).squeeze(1)

            labels = labels.squeeze(1)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).float()

            val_preds.extend(
                preds.cpu().numpy()
            )

            val_labels.extend(
                labels.cpu().numpy()
            )


    val_acc = accuracy_score(
        val_labels,
        val_preds
    )

    val_f1 = f1_score(
        val_labels,
        val_preds
    )

    val_precision = precision_score(
        val_labels,
        val_preds
    )

    val_recall = recall_score(
        val_labels,
        val_preds
    )


    # ======================================
    # METRICS
    # ======================================

    
    train_losses.append(train_loss)

    val_losses.append(val_loss)

    val_accuracies.append(val_acc)

    print("\nTRAIN LOSS:", train_loss)


    print("VAL LOSS:", val_loss)

    print("VAL ACC:", val_acc)

    print("VAL F1:", val_f1)

    print("VAL PRECISION:", val_precision)

    print("VAL RECALL:", val_recall)


    # ======================================
    # SAVE BEST
    # ======================================

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(

            model.state_dict(),

            "checkpoints/best_model.pth"
        )

        print("\nBEST MODEL SAVED")


# ==========================================
# TRAINING CURVES
# ==========================================

import matplotlib.pyplot as plt


# LOSS CURVE
plt.figure()

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.savefig("training_loss_curve.png")

print("Saved training_loss_curve.png")


# ACCURACY CURVE
plt.figure()

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Validation Accuracy Curve")

plt.legend()

plt.savefig("validation_accuracy_curve.png")

print("Saved validation_accuracy_curve.png")
