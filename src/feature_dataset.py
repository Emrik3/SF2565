"""
Dataset class for loading pre-extracted ResNet50 features from train.pt
"""

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted features.

    Args:
        data_path: Path to the .pt file containing features and labels
        transform: Optional transform to apply to features (default: None)
    """

    def __init__(self, data_path, transform=None):
        self.transform = transform

        # Load the pre-extracted features
        print(f"Loading features from {data_path}...")
        data = torch.load(data_path)

        self.features = data["features"]
        self.labels = data["labels"]

        print(f"Loaded {len(self.features)} samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Label shape: {self.labels.shape}")

        self.labels = self.labels[:, 0:1]

        # For 50/50 split, uncomment this code below
        """
        print(self.labels[0:10])
        # Balance the dataset
        print(f"  Original dataset size: {len(self.labels)}")

        # Find indices for each class
        positive_indices = torch.where(self.labels[:, 0] == 1)[0]  # Not sick
        negative_indices = torch.where(self.labels[:, 0] == 0)[0]  # Sick

        print(
            f"  Original - Positive: {len(positive_indices)}, Negative: {len(negative_indices)}"
        )

        # Balance by taking equal number of samples from each class
        min_class_size = min(len(positive_indices), len(negative_indices))

        # Randomly sample equal number from each class
        positive_indices = positive_indices[
            torch.randperm(len(positive_indices))[
                :min_class_size
            ]  # TODO: Change by what is run, if you want 50/50 or not and such
        ]
        negative_indices = negative_indices[
            torch.randperm(len(negative_indices))[:min_class_size]
        ]

        # Combine and shuffle indices
        balanced_indices = torch.cat([positive_indices, negative_indices])
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

        # Update features and labels with balanced subset
        self.features = self.features[balanced_indices]
        self.labels = self.labels[balanced_indices]

        print(f"  Balanced dataset size: {len(self.labels)}")
        print(f"  Final label shape: {self.labels.shape}")
        """
        print(
            f"  Label distribution - Negative: {(self.labels == 1).sum().item()}, Positive: {(self.labels == 0).sum().item()}"
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label

    def get_feature_dim(self):
        """Returns the dimensionality of the features"""
        return self.features.shape[1]

    def get_num_classes(self):
        """Returns the number of output classes"""
        return self.labels.shape[1] if self.labels.dim() > 1 else 1


def test_dataset():
    """Test the dataset loading"""
    import os

    # Adjust path as needed
    data_path = "data/xray_features_frontal_only.pt"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return

    dataset = FeatureDataset(data_path)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    print(f"Number of classes: {dataset.get_num_classes()}")

    # Get a sample
    feature, label = dataset[0]
    print(f"\nSample feature shape: {feature.shape}")
    print(f"Sample label shape: {label.shape}")
    print(f"Sample label value: {label}")

    # Check label distribution
    all_labels = dataset.labels.numpy()
    print(f"\nLabel distribution:")
    print(f"  Positive samples: {(all_labels == 1).sum()}")
    print(f"  Negative samples: {(all_labels == 0).sum()}")


if __name__ == "__main__":
    test_dataset()
