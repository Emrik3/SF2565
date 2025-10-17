from tracemalloc import start
from torch.utils.data import DataLoader
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from dataset import CheXpertDataset
from tqdm import tqdm


def transform_dataset(in_path, out_path):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(6)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    dataset = CheXpertDataset(in_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    print(len(dataset))
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            feats = feature_extractor(images)
            # feats = torch.flatten(feats, start_dim=1)
            feats = nn.AdaptiveAvgPool2d(1)(feats)
            feats = feats.squeeze(-1).squeeze(-1)  # or use global avg pool
            features_list.append(feats.cpu())  # move back to CPU if saving
            labels_list.append(labels)

    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    torch.save({"features": all_features, "labels": all_labels}, out_path)
    print(f"Feature extraction complete. Saved to '{out_path}'")
    return {"features": all_features, "labels": all_labels}
