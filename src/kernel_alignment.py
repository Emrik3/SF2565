from ctypes import alignment
from tqdm import tqdm
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import (
    cosine_similarity,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from utils import dprint, load_transform, lprint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


RNG_SEED = 42


def pairwise_dist():
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[StandardScaler()],
        seed=RNG_SEED,
        balanced=True,
        size=500,
    )
    distances = pairwise_distances(X)
    print(np.std(distances))
    plt.hist(distances)
    plt.show()

    print(distances)


def nystom_alignment():
    
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[StandardScaler()],
        size=14000,
        balanced=True
    )
    gam0 = 0.01
    y_pm = (2 * (y > 0) - 1).astype(float)
    n_gam = 40
    gammas = np.logspace(np.log10(gam0) - 3, np.log10(gam0) + 3, n_gam)
    plt.figure(figsize=(12, 5))

    lprint("fitting Nystroem...")

    aligns = []
    for g in tqdm(gammas):
        ns = Nystroem(gamma=g, n_components=1000).fit(X)
        Phi = ns.transform(X)  # shape (N, d)

        # numerator = y^T (Phi Phi^T) y
        v = Phi.T @ y_pm  # shape (d,)
        num = np.dot(v, v)

        # denom part 1: ||K||_F = ||Phi^T Phi||_F
        G = Phi.T @ Phi  # shape (d,d)
        froK = np.linalg.norm(G, "fro")

        # denom part 2: ||yy^T||_F = ||y||^2
        froY = np.linalg.norm(y_pm) ** 2

        aligns.append(num / (froK * froY))
    aligns = np.array(aligns)
    plt.semilogx(gammas, aligns, marker="o", label="nyström rbf kernel")
    plt.xlabel("gamma")
    plt.ylabel("alignment")
    plt.title("Alignment vs gamma (RBF)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def kernel_alignment():
    alpha = 0.3

    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[],
        balanced=True,
        size=400,
        seed=RNG_SEED
        
    )
    gam0 = 0.01
    y_pm = (2 * (y > 0) - 1).astype(float)
    n_gam = 40
    gammas = np.logspace(np.log10(gam0) - 3, np.log10(gam0) + 3, n_gam)
    plt.figure(figsize=(12, 5))

    lprint("fitting rbf...")
    aligns = []
    for g in gammas:
        K = rbf_kernel(X, gamma=g)
        num = np.sum(K * np.outer(y_pm, y_pm))
        denom = np.linalg.norm(K, "fro") * np.linalg.norm(np.outer(y_pm, y_pm), "fro")
        aligns.append(num / denom)
    aligns = np.array(aligns)
    plt.semilogx(gammas, aligns, marker="o", label="rbf kernel")

    # lprint("fitting linear")
    # aligns = []
    # for g in gammas:
    #     K = linear_kernel(X)
    #     num = np.sum(K * np.outer(y_pm, y_pm))
    #     denom = np.linalg.norm(K, "fro") * np.linalg.norm(np.outer(y_pm, y_pm), "fro")
    #     aligns.append(num / denom)
    # aligns = np.array(aligns)
    # plt.semilogx(gammas, aligns, marker="_", label="linear kernel")
    #
    # lprint("fitting sigmoids...")
    # coefs = range(-1,2)
    # for coeff in coefs:
    #     aligns = []
    #     for g in gammas:
    #         K = sigmoid_kernel(X, coef0=coeff, gamma=g)
    #         num = np.sum(K * np.outer(y_pm, y_pm))
    #         denom = np.linalg.norm(K, "fro") * np.linalg.norm(
    #             np.outer(y_pm, y_pm), "fro"
    #         )
    #         aligns.append(num / denom)
    #     aligns = np.array(aligns)
    #     plt.semilogx(gammas, aligns, marker="x", label=f"sigmoid kernel, c0={coeff}")
    #
    lprint("fitting polynomials...")
    degrees = range(1, 7)
    for degree in degrees:
        aligns = []
        for g in gammas:
            K =polynomial_kernel(X, degree=degree, gamma=g)
            num = np.sum(K * np.outer(y_pm, y_pm))
            denom = np.linalg.norm(K, "fro") * np.linalg.norm(
                np.outer(y_pm, y_pm), "fro"
            )
            aligns.append(num / denom)
        aligns = np.array(aligns)
        plt.semilogx(gammas, aligns, marker="o", label=f"poly deg{degree}")
    #
    plt.xlabel("gamma")
    plt.ylabel("alignment")
    plt.title("Alignment vs gamma (RBF)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():

    # npz_path = "data/pneumoniamnist.npz"
    # data = np.load(npz_path)
    # image_key = "train_images"
    # labels_key = "train_labels"
    # X = data[image_key]  # shape: [N, H, W] or [N, H, W, C]
    # n,h,w = X.shape
    # X = np.reshape(X, (n, h*w))
    # y = data[labels_key]
    # X = StandardScaler().fit_transform(X,y)
    # print(X)

    # X, y = load_transform(
    #     "data/xray_features_frontal_only.pt",
    #     transforms=[StandardScaler()],
    #     balanced=True,
    #     size=1000,
    # )
    # pairwise_dist()
    kernel_alignment()
    # kernel_diagnostics(X, y)
    # deeper_diagnostics(X, y)
    # nystom_alignment()


if __name__ == "__main__":
    main()
