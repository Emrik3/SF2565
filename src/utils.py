import torch
import numpy as np
import matplotlib.pyplot as plt


DEBUG = True
LOG = True


def dprint(*args, **kwargs):
    if DEBUG:
        print("DEBUG:", *args, **kwargs)


def lprint(*args, **kwargs):
    if LOG:
        print("LOG:", *args, **kwargs)


def load_dataset(
    path: str,
    size: int | None = None,
    seed: int | None = None,
    balanced: bool = False,
    label: int = 3,
    replace: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads a cetrain size of the dataset if specified, all else"""

    data = torch.load(path)
    X = data["features"].numpy()
    y = data["labels"].int().numpy()

    is_healthy = y[:, 0] == 1
    has_affliction = y[:, 1:].sum(axis=1) > 0
    overlap = is_healthy & has_affliction
    n_overlap = np.sum(overlap)

    y = has_affliction.astype(int)

    mask = ~overlap & (is_healthy | has_affliction)
    X = X[mask]
    y = y[mask]
    lprint(f"Dropped {n_overlap} non-exclusive samples out of {len(y)} total.")

    # defines probabilites of a certain index based on the distribution
    # this makes classes that are less common more probable
    # to appear in the dataset
    if balanced:
        p = 1 / np.bincount(y)[y]
        p = p / p.sum()
    else:
        p = None

    # This allows for a subset of the dataset to be loaded
    if size:
        rng = np.random.default_rng(seed=seed)
        total_size = len(y)
        random_indices = rng.choice(total_size, size=size, replace=replace, p=p)
        return_y = y[random_indices]
        lprint(f"Loaded Dataset: {path.split("/")[-1]}")
        lprint(
            f"Size: {size}. Class split (true): {round(np.sum(return_y)/size, 2)*100}%"
        )
        print(np.bincount(return_y))

        return X[random_indices], return_y
    lprint(f"Loaded Dataset: {path.split("/")[-1]}")
    lprint(f"Size: {len(y)}. Class split (true): {round(np.sum(y)/len(y), 2)*100}%")
    return X, y


def load_transform(
    path: str, transforms: list, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Loads a data set just like load_dataset but also transforms the data
    using specified transformations"""
    X, y = load_dataset(path, **kwargs)
    for transform in transforms:
        X = transform.fit_transform(X, y)
    return X, y


def load_dataset_partitions(
    path: str,
    n_partitions: int,
    seed: int | None = None,
    balanced: bool = False,
    label: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Loads a dataset and splits it into n disjoint partitions.

    Returns a list of (X_part, y_part) tuples.
    """

    data = torch.load(path)
    X = data["features"].numpy()
    y = data["labels"].numpy()

    mask = (y[:, 0] == 1) | (y[:, label] == 1)
    X = X[mask]
    y = y[mask]
    y = (y[:, label] == 1).astype(int)

    total_size = len(y)
    rng = np.random.default_rng(seed=seed)

    if balanced:
        class_ratio = np.mean(y)
        p = (1 - class_ratio) * (y == 1) + class_ratio * (y == 0)
        p = p / p.sum()
        indices = rng.choice(total_size, size=total_size, replace=False, p=p)
    else:
        indices = rng.permutation(total_size)

    partitions = np.array_split(indices, n_partitions)

    result = []
    for part in partitions:
        result.append((X[part], y[part]))

    lprint(f"Loaded Partitioned Dataset: {path.split("/")[-1]}")
    lprint(f"Size: {len(y)}. Class split (true): {round(np.sum(y)/len(y), 2)*100}%")
    return result


def test():
    data = torch.load("data/xray_features_frontal_only.pt")
    X = data["features"].numpy()
    y = data["labels"]
    # plt.imshow(np.cov(y.numpy().T))
    # plt.show()
    X, y = load_dataset("data/xray_features_frontal_only.pt", size=400, balanced=True)
    # print(np.sum(y))


if __name__ == "__main__":

    test()
