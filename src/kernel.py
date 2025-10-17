from __future__ import annotations
from typing import Protocol, Sequence
from numbers import Real
import tracemalloc
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import log_loss
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split

from model_evaluation import evaluate_model
from utils import load_dataset, load_dataset_partitions, load_transform, dprint, lprint

RNG_SEED = 42  # Set to None for random


class ScikitModel(Protocol):
    """Scikit classifer interface"""

    def fit(self, X, y) -> ScikitModel: ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...
    def predict_proba(self, X): ...


TrainTestTuple = tuple[np.ndarray, np.ndarray]


def compare_models() -> None:
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[StandardScaler()],
        seed=RNG_SEED,
        balanced=True,
        size=1000,
    )

    values_C = 2.0 ** np.array([*range(-10, 10)])
    svc_params = [
        {"kernel": ["rbf"], "C": values_C, "gamma": ["scale"]},
        {"kernel": ["poly"], "C": values_C},
    ]
    svc = SVC(probability=True)
    svc_cv = GridSearchCV(
        svc,
        svc_params,
        scoring="neg_log_loss",
        verbose=3,
        n_jobs=-1,
        cv=3,
    )
    svc_cv.fit(X, y)
    svc_best_params = svc_cv.best_params_
    dprint(f"Best params svc: {svc_best_params}")
    dprint(f"Best CV score svc (log-loss): {svc_cv.best_score_}")


def plot_train_test_err(
    train: np.ndarray, test: np.ndarray, x: Sequence, title: str, xlabel: str, log=True
):
    """Plots the difference between the train and test error."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("err")
    if log:
        plt.loglog(x, train, "-o", label="train err")
        plt.loglog(x, test, "-o", label="test err")
    else:
        plt.plot(x, train, "-o", label="train err")
        plt.plot(x, test, "-o", label="test err")

    plt.legend()
    plt.grid()
    plt.show()


# ---------------------- train test size
# --- This section is for measuring how train and test error is affected by
# --- the number of samples.


def train_test_size(sizes: Sequence[int], model: ScikitModel) -> TrainTestTuple:
    """Fits a clasifier on different sizes of the data. train and test error
    using log cross-entropy loss"""
    k = len(sizes)
    train_err = np.zeros((1, k)).ravel()
    test_err = np.zeros((1, k)).ravel()
    for idx, size in enumerate(sizes):
        X, y = load_dataset(
            "data/xray_features_frontal_only.pt", size=size, seed=RNG_SEED
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        lprint(f"run {idx+1}/{k}. Current size: {size}. Model: {type(model)}")
        model.fit(X_train, y_train)
        train_err[idx] = log_loss(y_train, model.predict_proba(X_train))
        test_err[idx] = log_loss(y_test, model.predict_proba(X_test))
    return train_err, test_err


def run_train_test_err_size() -> None:
    """Runs the code that generate the train test plot for different sizes"""
    sizes = (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)

    with open("kernel_output/best_params_kernel", "rb") as f:
        best_params = pickle.load(f)
    svm_rbf = SVC(probability=True, **best_params)

    train_err, test_err = train_test_size(sizes=sizes, model=svm_rbf)
    plot_train_test_err(
        train_err,
        test_err,
        sizes,
        xlabel="Sample sizes",
        title="cross-entropy loss at differnet sample sizes",
    )
    dprint(f"train err: {train_err}, test err: {test_err}")


# ---------------- regularization train test error
# --- This section measures how the train and test error is affected by
# --- the strength of regularization.


def regularization_train_test(
    model: ScikitModel, param_list: list[Real], dataset_size: int = 400
) -> TrainTestTuple:
    """Fits the classifier multiple times with varying value for C. Train and
    test err by log cross cross-entropy"""

    k = len(param_list)
    train_err = np.zeros((1, k)).ravel()
    test_err = np.zeros((1, k)).ravel()

    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        [],
        size=dataset_size,
        seed=RNG_SEED,
        balanced=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    gam0 = 0.001
    n_gam = 8
    gammas = np.logspace(np.log10(gam0) - 2, np.log10(gam0) + 2, n_gam)
    for gamma in gammas:

        lprint(f"Running regularization fit with gamma={gamma:.1e}")
        for idx, param in enumerate(param_list):
            lprint(
                f"Running regularization fit with gamma={gamma:.1e} and C={param:.1e}"
            )

            model.set_params(C=param, gamma=gamma)
            model.fit(X_train, y_train)
            # print(evaluate_model(model, X_train, y_train, X_test, y_test))
            train_err[idx] = log_loss(y_train, model.predict_proba(X_train))
            test_err[idx] = log_loss(y_test, model.predict_proba(X_test))
        plt.loglog(1 / np.array(param_list), test_err, "-o", label=f"Gamma:{gamma:.1e}")
        # plt.loglog(
        #     1 / np.array(param_list), train_err,"-o", label=f"Gamma: {gamma:.1e}"
        # )
    plt.legend()
    plt.xlabel("lambda")
    plt.ylabel("error")
    plt.title("Test error for different models")
    plt.grid(which="both")
    plt.show()
    dprint(f"All model fits complete: train err: {train_err}\n\ttest err: {test_err}")
    return train_err, test_err


def run_regularization() -> None:
    c_params = np.power(2.0, np.array([*range(-1, 11)]))
    with open("kernel_output/best_params_kernel", "rb") as f:
        best_params = pickle.load(f)
    dprint(best_params)
    svm_rbf = SVC(probability=True, **best_params)
    train_err, test_err = regularization_train_test(model=svm_rbf, param_list=c_params)
    plot_train_test_err(
        train_err,
        test_err,
        c_params,
        xlabel="C",
        title="cross-entropy loss for different regularization strengths",
        log=True,
    )


# -------------------------- Performance Evaluation
# --- The memory allocation benchmark didn't work since the library tracemalloc
# --- doesn't measure any memory allocated outside python i.e. by C subrutines
# --- that numpy runs on. Time benchmark works as expected though.


def test_mem_time(sizes: Sequence[int], model: ScikitModel) -> None:
    """measures memory usage (MB) and execution time when fitting a model
    for different sample sizes. Saves resutlts to disk."""
    # The memory benchmark didn't work since it only measures the memory
    # allocated by python and not any C/C++ subrutine which numpy runs on:(.
    k = len(sizes)
    sizes = np.array(sizes).ravel()

    used_mem_list = np.zeros((1, k)).ravel()
    times = np.zeros((1, k)).ravel()

    tracemalloc.start()

    for idx, size in enumerate(sizes):
        tracemalloc.clear_traces()
        X, y = load_dataset(
            "data/xray_features_frontal_only.pt", size=size, seed=RNG_SEED
        )
        lprint(f"run {idx+1}/{k}. Current size: {size}.")
        start = tracemalloc.take_snapshot()
        t0 = time.time()
        model.fit(X, y)
        t = time.time() - t0
        end = tracemalloc.take_snapshot()

        stats = end.compare_to(start, "lineno")
        mem_diff = sum([stat.size_diff for stat in stats])
        used_mem_list[idx] = round(mem_diff / (1024**2), 3)
        times[idx] = t

    tracemalloc.stop()
    np.save("kernel_output/times_test_output_SVC", np.stack((sizes, times)))
    np.save("kernel_output/mem_test_output_SVC", np.stack((sizes, used_mem_list)))


def plot_times():
    """Loads and plots the execution time results with n^2 scaling reference"""
    with open("kernel_output/times_test_output_SVC.npy", "rb") as f:
        data = np.load(f)
    x, y = data[0, :], data[1, :]

    plt.figure(figsize=(7, 5))
    plt.loglog(x, y, marker="o", linestyle="-", label="Execution Time")

    ref = (y[0] / (x[0] ** 2)) * (x**2)
    plt.loglog(x, ref, linestyle="--", label=r"$\mathcal{O}(n^2)$ ref")

    plt.title("Execution Times vs Sample Size")
    plt.xlabel("Sample Size", fontsize=12)
    plt.ylabel("Elapsed Time (s)", fontsize=12)
    plt.grid(True, which="both", linewidth=0.7, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_used_mem():
    """Loads and plots the used memory results with n^2 scaling reference"""
    # This produces a O(n) plot since the memory benchmark doesn't work
    # a expected, see test_mem_time funciton comment.
    with open("kernel_output/mem_test_output_SVC.npy", "rb") as f:
        data = np.load(f)
    x, y = data[0, :], data[1, :]

    plt.figure(figsize=(7, 5))
    plt.loglog(x, y, marker="s", linestyle="-", color="tab:green", label="Peak Memory")

    ref = (y[0] / (x[0])) * (x)
    plt.loglog(x, ref, linestyle="--", color="tab:red", label=r"$\mathcal{O}(n^3)$ ref")

    plt.title("Peak Memory Allocation vs Size", fontsize=14, fontweight="bold")
    plt.xlabel("Sample Size", fontsize=12)
    plt.ylabel("Peak Memory (MB)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_time_mem_test() -> None:
    sizes = list(map(lambda x: 2**x, range(6, 13)))
    model = SVC(probability=True, kernel="rbf", gamma=0.004, C=1000)
    test_mem_time(sizes, model)


# ------------------------- Nyström approximations ---------------------


def train_on_nystom(model: LinearSVC, X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    values_C = 2.0 ** np.array([*range(-5, 4)])
    svc_params = [{"C": values_C}]
    svc_cv = GridSearchCV(
        model,
        svc_params,
        scoring="roc_auc",
        verbose=3,
        n_jobs=-1,
        cv=3,
    )
    svc_cv.fit(X_train, y_train)
    lprint("Evaluating model")
    best_model = LinearSVC(**svc_cv.best_params_)
    print(best_model.get_params())
    print(evaluate_model(best_model, X_train, y_train, X_test, y_test))


def run_nystrom():
    pca = PCA(n_components=0.60)
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[],
        size=14000,
        balanced=True,
    )
    feature_map_ns = Nystroem(gamma=0.004, n_components=2000)
    lprint("transforming features...")
    transfomed_X = feature_map_ns.fit_transform(X, y)
    model = LinearSVC(C=100000, fit_intercept=True)

    train_on_nystom(model, transfomed_X, y)


# ------------------ Fourier Features ------------------------


def train_on_rff(model: LinearSVC, X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    values_C = 2.0 ** np.array([*range(-5, 4)])
    gammas = [8e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    svc_params = [{"C": values_C}, {"gamma": gammas}]
    svc_cv = GridSearchCV(
        model,
        svc_params,
        scoring="roc_auc",
        verbose=3,
        n_jobs=-1,
        cv=3,
    )
    svc_cv.fit(X_train, y_train)
    lprint("Evaluating model")
    best_model = LinearSVC(**svc_cv.best_params_)
    print(best_model.get_params())
    print(evaluate_model(best_model, X_train, y_train, X_test, y_test))


def run_RFF():
    pca = PCA(n_components=0.60)
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[],
        size=30000,
        balanced=True,
    )
    feature_map_ns = RBFSampler(gamma=0.04, n_components=2000)
    lprint("transforming features...")
    transfomed_X = feature_map_ns.fit_transform(X, y)
    model = LinearSVC(class_weight="balanced", fit_intercept=True)

    train_on_nystom(model, transfomed_X, y)


# -------------------------- Divide and Conquer -------------------
def best_alignment(X, y) -> dict:
    """Determines the best kernel and kernel parameter through best alignment conditions"""
    gam0 = 0.1
    y_pm = (2 * (y > 0) - 1).astype(float)
    n_gam = 40
    gammas = np.logspace(np.log10(gam0) - 3, np.log10(gam0) + 3, n_gam)
    y_outer = np.outer(y_pm, y_pm)
    y_outer_norm = np.linalg.norm(y_outer, "fro")
    best_alignment = -1.0e10
    best_params = {}

    degrees = range(1, 4)

    lprint("Testing rbf kernel")
    aligns = []
    for g in gammas:
        K = rbf_kernel(X, gamma=g)
        num = np.sum(K * y_outer)
        denom = np.linalg.norm(K, "fro") * y_outer_norm
        aligns.append(num / denom)
    best_idx = np.argmax(aligns)
    if aligns[best_idx] > best_alignment:
        lprint(
            f"best gamma: {round(gammas[best_idx], 3)}, best_alignment: {round(aligns[best_idx], 3)}"
        )
        best_alignment = aligns[best_idx]
        best_params = {"kernel": "rbf", "gamma": gammas[best_idx]}

    lprint("testing polynomial kernels")
    for degree in degrees:
        aligns = []
        for g in gammas:
            K = polynomial_kernel(X, degree=degree, gamma=g)
            num = np.sum(K * np.outer(y_pm, y_pm))
            denom = np.linalg.norm(K, "fro") * y_outer_norm
            aligns.append(num / denom)
        best_idx = np.argmax(aligns)
        if aligns[best_idx] > best_alignment:
            lprint(
                f"best gamma: {round(gammas[best_idx], 3)}, best_alignment: {round(aligns[best_idx], 3)}"
            )
            best_alignment = aligns[best_idx]
            best_params = {
                "kernel": "poly",
                "degree": degree,
                "gamma": gammas[best_idx],
            }

    return {"best_params": best_params, "best_alignment": best_alignment}


def train_auto_cv(X, y):
    """Determines the best regularization parameter through CV"""
    values_C = 2.0 ** np.array([*range(-6, 10)])
    svc_params = [{"C": values_C}]
    ker_gamma_parameter = best_alignment(X, y)["best_params"]
    svc = SVC(**ker_gamma_parameter, class_weight="balanced")
    svc_cv = GridSearchCV(
        svc,
        svc_params,
        scoring="roc_auc",
        verbose=3,
        n_jobs=1,
        cv=3,
    )
    svc_cv.fit(X, y)
    best_model = SVC(
        **ker_gamma_parameter,
        **svc_cv.best_params_,
        probability=True,
        class_weight="balanced",
    )
    print(best_model.get_params())
    return best_model.fit(X, y)


def main():
    # run_regularization()
    run_train_test_err_size()
    # SVC_cv()kernel
    # compare_models()
    # run_time_mem_test()
    # plot_times()

    # run_nystrom()
    # run_RFF()

if __name__ == "__main__":
    main()
