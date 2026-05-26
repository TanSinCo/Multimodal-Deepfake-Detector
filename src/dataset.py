
import pandas as pd
import torch
import numpy as np

from preprocess.video import extract_frames_and_landmarks
from preprocess.roi import extract_all_rois
from preprocess.rppg import extract_rppg
from preprocess.filtering import filter_rppg
from preprocess.window import create_windows
from preprocess.blink import extract_blink_signal
from preprocess.motion import extract_motion_signal


class DeepfakeDataset:

    def __init__(self, csv_path):

        self.data = pd.read_csv(csv_path)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        video_path = row["video_path"]
        label = row["label"]

        frames_15, landmarks_15 = extract_frames_and_landmarks(
            video_path,
            target_fps=15
        )

        frames_30, landmarks_30 = extract_frames_and_landmarks(
            video_path,
            target_fps=30
        )

        if frames_15 is None or frames_30 is None:
            return None

        rois = extract_all_rois(
            frames_30,
            landmarks_30
        )

        rppg = extract_rppg(rois)

        filtered_rppg = filter_rppg(rppg)

        rppg_windows = create_windows(
            filtered_rppg,
            window_size=40,
            stride=20
        )

        if len(rppg_windows) == 0:
            return None

        blink_signal = extract_blink_signal(
            landmarks_15
        )

        motion_signal = extract_motion_signal(
            landmarks_15
        )

        TARGET_SIGNAL_LENGTH = 40

        if len(blink_signal) < TARGET_SIGNAL_LENGTH:

            pad = TARGET_SIGNAL_LENGTH - len(blink_signal)

            blink_signal = np.pad(
                blink_signal,
                (0, pad)
            )

        else:

            blink_signal = blink_signal[
                :TARGET_SIGNAL_LENGTH
            ]

        if len(motion_signal) < TARGET_SIGNAL_LENGTH:

            pad = TARGET_SIGNAL_LENGTH - len(motion_signal)

            motion_signal = np.pad(
                motion_signal,
                (0, pad)
            )

        else:

            motion_signal = motion_signal[
                :TARGET_SIGNAL_LENGTH
            ]

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

        rppg_tensor = torch.tensor(
            rppg_windows[0],
            dtype=torch.float32
        )

        blink_tensor = torch.tensor(
            blink_signal,
            dtype=torch.float32
        )

        motion_tensor = torch.tensor(
            motion_signal,
            dtype=torch.float32
        )

        label_tensor = torch.tensor(
            [label],
            dtype=torch.float32
        )

        return {
            "frames": frames_tensor,
            "rppg": rppg_tensor,
            "blink": blink_tensor,
            "motion": motion_tensor,
            "label": label_tensor
        }
