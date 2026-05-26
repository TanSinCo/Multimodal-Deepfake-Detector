
import torch
import sys

sys.path.append("/kaggle/working/project/src")

from dataset import DeepfakeDataset
from models.fusion_model import FusionModel


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# LOAD MODEL
# ==========================================

model = FusionModel().to(device)

model.load_state_dict(

    torch.load(
        "/kaggle/working/final_project/checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded")


# ==========================================
# LOAD SAMPLE VIDEO
# ==========================================

dataset = DeepfakeDataset(
    "/kaggle/working/project/data/train_kaggle.csv"
)

sample = dataset[0]

frames = sample["frames"].unsqueeze(0).to(device)

rppg = sample["rppg"].unsqueeze(0).to(device)

blink = sample["blink"].unsqueeze(0).to(device)

motion = sample["motion"].unsqueeze(0).to(device)

label = sample["label"].item()


# ==========================================
# INFERENCE
# ==========================================

with torch.no_grad():

    output = model(

        frames,
        rppg,
        blink,
        motion
    )

    probability = torch.sigmoid(
        output
    ).item()


# ==========================================
# PREDICTION
# ==========================================

prediction = 1 if probability > 0.5 else 0

label_name = "FAKE" if prediction == 1 else "REAL"

true_name = "FAKE" if label == 1 else "REAL"


# ==========================================
# RESULTS
# ==========================================

print("\n======================")

print("TRUE LABEL:", true_name)

print("PREDICTION:", label_name)

print(f"FAKE PROBABILITY: {probability:.4f}")

print("======================")
