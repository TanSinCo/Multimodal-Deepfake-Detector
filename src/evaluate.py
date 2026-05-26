
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    f1_score
)

from torch.utils.data import DataLoader

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


# ==========================================
# DATASET
# ==========================================

dataset = MultimodalCacheDataset(
    "/kaggle/working/project/multimodal_cache"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)


# ==========================================
# MODEL
# ==========================================

model = FusionModel().to(device)

model.load_state_dict(

    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()


# ==========================================
# EVALUATION
# ==========================================

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for batch in loader:

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

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.squeeze(1).cpu().numpy()
        )


# ==========================================
# METRICS
# ==========================================

acc = accuracy_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

print("Accuracy:", acc)

print("F1 Score:", f1)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

print("Saved confusion_matrix.png")


# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, _ = roc_curve(
    all_labels,
    all_probs
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

print("Saved roc_curve.png")
