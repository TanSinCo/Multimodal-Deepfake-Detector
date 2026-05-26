
import os
import torch

from torch.utils.data import Dataset


class MultimodalCacheDataset(Dataset):

    def __init__(self, cache_dir):

        self.cache_dir = cache_dir

        self.files = [

            f for f in os.listdir(cache_dir)

            if f.endswith(".pt")
        ]

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = os.path.join(
            self.cache_dir,
            self.files[idx]
        )

        try:

            sample = torch.load(path)

            return {

                "frames": sample["frames"].float(),

                "rppg": sample["rppg"].float(),

                "blink": sample["blink"].float(),

                "motion": sample["motion"].float(),

                "label": sample["label"].float()
            }

        except Exception as e:

            print(f"\nCORRUPTED FILE: {path}")
            print(e)

            return None
