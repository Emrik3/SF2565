from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

from kernel import RNG_SEED
from model_evaluation import evaluate_model
from utils import load_dataset, load_dataset_partitions, load_transform, dprint, lprint


from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.svm import SVC, LinearSVC

RNG_SEED = 42


def RBF_gs():
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[],
        size=14000,
        balanced=True,
        seed=RNG_SEED,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = pipeline = Pipeline(
        [("RBF", RBFSampler(n_components=8000)), ("svm", LinearSVC())]
    )
    values_C = 2.0 ** np.array([*range(-5, 4)])
    gammas = [8e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    param_grid = {
        "RBF__gamma": gammas,
        "svm__C": values_C,
    }

    grid = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=-1, verbose=3, scoring="accuracy"
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid.best_score_)

    best_model = Pipeline(
        [
            (
                "RBF",
                RBFSampler(gamma=best_params["RBF__gamma"], n_components=1000),
            ),
            ("svm", LinearSVC(C=best_params["svm__C"])),
        ]
    )
    print(evaluate_model(best_model, X_train, y_train, X_test, y_test))


def Nystroem_gs():

    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[],
        size=14000,
        balanced=True,
        seed=RNG_SEED,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = pipeline = Pipeline(
        [("nystroem", Nystroem(kernel="rbf", n_components=1000)), ("svm", LinearSVC())]
    )
    values_C = 2.0 ** np.array([*range(-5, 4)])
    gammas = [8e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    param_grid = {
        "nystroem__gamma": gammas,
        "svm__C": values_C,
    }

    grid = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=-1, verbose=3, scoring="accuracy"
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid.best_score_)

    best_model = Pipeline(
        [
            (
                "nystroem",
                Nystroem(gamma=best_params["nystroem__gamma"], n_components=1000),
            ),
            ("svm", LinearSVC(C=best_params["svm__C"])),
        ]
    )
    print(evaluate_model(best_model, X_train, y_train, X_test, y_test))


def plot_time_accuracy():
    """
    Compare Nyström and RBFSampler kernel approximations by plotting
    accuracy and fit time vs. n_components on the same graph.
    """
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[],
        size=14000,
        balanced=True,
        seed=RNG_SEED,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED
    )

    n_comp_list = [100, 500, 1000, 2000, 3000, 5000]
    results = {
        "nystroem": {"acc": [], "time": []},
        "rbf_sampler": {"acc": [], "time": []},
    }

    for method_name, feature_map in [
        ("nystroem", Nystroem(kernel="rbf", gamma=0.001)),
        ("rbf_sampler", RBFSampler(gamma=0.001)),
    ]:
        pipeline = Pipeline(
            [("feature_map", feature_map), ("svm", LinearSVC(max_iter=5000))]
        )

        print(f"\n=== {method_name.upper()} ===")
        for n_comp in n_comp_list:
            pipeline.set_params(feature_map__n_components=n_comp)

            start = time.time()
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - start

            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            results[method_name]["acc"].append(acc)
            results[method_name]["time"].append(elapsed)

            print(
                f"n_components={n_comp:4d} | Accuracy={acc:.4f} | Time={elapsed:.2f}s"
            )

    # Plot accuracy and time vs. n_components
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("n_components")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(
        n_comp_list,
        results["nystroem"]["acc"],
        color="tab:blue",
        marker="o",
        label="Nyström Accuracy",
    )
    ax1.plot(
        n_comp_list,
        results["rbf_sampler"]["acc"],
        color="tab:cyan",
        marker="^",
        label="RBF Sampler Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Fit Time (s)", color="tab:red")
    ax2.plot(
        n_comp_list,
        results["nystroem"]["time"],
        color="tab:red",
        marker="s",
        linestyle="--",
        label="Nyström Time",
    )
    ax2.plot(
        n_comp_list,
        results["rbf_sampler"]["time"],
        color="tab:orange",
        marker="x",
        linestyle="--",
        label="RBF Sampler Time",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title("Nyström vs RBF Sampler: Accuracy & Fit Time vs n_components")
    fig.tight_layout()
    plt.show()


def main():
    RBF_gs()


if __name__ == "__main__":
    main()
