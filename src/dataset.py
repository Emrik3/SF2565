from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

from utils import lprint

# this import logic is based of the following git repo
# source: https://github.com/jfhealthcare/Chexpert


CSV_PATH = "data/CheXpert-v1.0-small/train.csv"
CSV_PATH_test = "data/CheXpert-v1.0-small/test.csv"
afflictions = [
    "Pleural Effusion",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pneumonia",
]


class CheXpertDataset(Dataset):
    """This simpler datset class excludes uncertain labels"""

    def __init__(self, label_path):
        data = pd.read_csv(label_path)
        print(list(data.columns))
        lprint(f"Afflictaions included: {afflictions}")
        data = data[
            (data["No Finding"] == 1.0) | (data[afflictions] == 1.0).any(axis=1)
        ]
        data = data[["Path", "No Finding"] + afflictions]
        data = data.head(2000)
        self.data = data
        self._transform = ResNet50_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = "data/" + self.data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")

        labels = (
            self.data.iloc[idx, 5:].fillna(0).values.astype("float")
        )  # skip metadata columns
        if self._transform:
            image = self._transform(image)

        return image, torch.tensor(labels, dtype=torch.float)


def test():
    dataset = CheXpertDataset(CSV_PATH)
    df = dataset.data
    print(df.head())
    print((df["No Finding"] == df[afflictions].any(axis=1)).sum())

    # image_array = dataset[4][0].permute(1, 2, 0).numpy()
    # plt.imshow(image_array)
    # plt.show()


if __name__ == "__main__":
    test()
