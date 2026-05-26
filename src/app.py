
import gradio as gr
import torch
import numpy as np
import sys

sys.path.append("/kaggle/working/final_project/src")

from preprocess.video import extract_frames_and_landmarks
from preprocess.roi import extract_all_rois
from preprocess.rppg import extract_rppg
from preprocess.filtering import filter_rppg
from preprocess.window import create_windows
from preprocess.blink import extract_blink_signal
from preprocess.motion import extract_motion_signal

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
# PREDICTION FUNCTION
# ==========================================

def predict_video(video_path):

    try:

        frames_15, landmarks_15 = extract_frames_and_landmarks(

            video_path,

            target_fps=15,
            max_frames=120
        )

        frames_30, landmarks_30 = extract_frames_and_landmarks(

            video_path,

            target_fps=30,
            max_frames=120
        )

        rois = extract_all_rois(
            frames_30,
            landmarks_30
        )

        rppg = extract_rppg(rois)

        filtered_rppg = filter_rppg(rppg)

        windows = create_windows(

            filtered_rppg,

            window_size=40,
            stride=20
        )

        rppg_tensor = torch.tensor(

            windows[0],

            dtype=torch.float32
        )

        blink_signal = extract_blink_signal(
            landmarks_15
        )

        motion_signal = extract_motion_signal(
            landmarks_15
        )

        blink_signal = blink_signal[:40]
        motion_signal = motion_signal[:40]

        if len(blink_signal) < 40:

            blink_signal = np.pad(

                blink_signal,

                (0, 40 - len(blink_signal))
            )

        if len(motion_signal) < 40:

            motion_signal = np.pad(

                motion_signal,

                (0, 40 - len(motion_signal))
            )

        indices = np.linspace(

            0,

            len(frames_15) - 1,

            40,

            dtype=int
        )

        selected_frames = frames_15[indices]

        frames_tensor = torch.tensor(

            selected_frames,

            dtype=torch.float32

        ).permute(0, 3, 1, 2)

        frames_tensor = frames_tensor.unsqueeze(0).to(device)

        rppg_tensor = rppg_tensor.unsqueeze(0).to(device)

        blink_tensor = torch.tensor(

            blink_signal,

            dtype=torch.float32

        ).unsqueeze(0).to(device)

        motion_tensor = torch.tensor(

            motion_signal,

            dtype=torch.float32

        ).unsqueeze(0).to(device)


        with torch.no_grad():

            output = model(

                frames_tensor,
                rppg_tensor,
                blink_tensor,
                motion_tensor
            )

            probability = torch.sigmoid(
                output
            ).item()


        prediction = (

            "FAKE"

            if probability > 0.5

            else "REAL"
        )

        confidence = (

            probability

            if prediction == "FAKE"

            else 1 - probability
        )

        confidence_bar = "█" * int(confidence * 20)

        result = f"""

================================

VIDEO ANALYSIS RESULT

================================

Prediction:
{prediction}

Confidence:
{confidence*100:.2f}%

Fake Probability:
{probability*100:.2f}%

Confidence Meter:
{confidence_bar}

================================
"""

        return result

    except Exception as e:

        return f"Error: {str(e)}"


# ==========================================
# GRADIO UI
# ==========================================

app = gr.Interface(

    fn=predict_video,

    inputs=gr.Video(
        label="Upload Video"
    ),

    outputs=gr.Textbox(

        label="Detection Result",

        lines=15
    ),

    title="Multimodal Deepfake Detector",

    description="""
Upload an MP4 video and the AI system will analyze:

• Facial features
• Temporal inconsistencies
• rPPG physiological signals
• Blink patterns
• Facial motion

to determine whether the video is REAL or FAKE.
""",

    theme=gr.themes.Soft(),

    allow_flagging="never"
)


app.launch(debug=True)
