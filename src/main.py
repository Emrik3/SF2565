from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from model_evaluation import evaluate_model
from utils import load_dataset, load_transform


from transform_data import transform_dataset

RAND_STATE = 42

CSV_PATH = "data/CheXpert-v1.0-small/train.csv"

CSV_TEST_PATH = "data/CheXpert-v1.0-small/valid.csv"


def LDA_plot(X, y):

    # y = y.argmax(axis=1)
    lda = LDA(n_components=1)
    reduced = lda.fit_transform(X, y)

    scatter = plt.scatter(reduced[:, 0], y, c=y)
    plt.legend(*scatter.legend_elements(), title="classes")
    plt.show()


def t_sne():
    # data = np.load("data/pneumoniamnist.npz")
    # X = data['train_images'].reshape((-1, 28**2))
    # y = data['train_labels'].ravel()
    # print(y.shape)
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[StandardScaler()],
        size=4000,
        balanced=False,
    )

    tsne = TSNE(
        n_components=2, perplexity=30, max_iter=2000, learning_rate="auto", verbose=1
    )
    tsne_results = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.title("t-SNE Visualization of MobileNetV3 X-Ray Features")

    # Create a DataFrame for easy plotting
    tsne_df = pd.DataFrame(
        {"TSNE-1": tsne_results[:, 0], "TSNE-2": tsne_results[:, 1], "Label": y}
    )

    # Get unique labels to create the legend
    unique_labels = np.unique(y)

    # Scatter plot
    for label in unique_labels:
        subset = tsne_df[tsne_df["Label"] == label]
        plt.scatter(
            subset["TSNE-1"],
            subset["TSNE-2"],
            label=f"Label {int(label)}",
            alpha=0.6,
            s=20,
        )  # 's' controls marker size

    plt.legend(title="Class (Example)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def random_f():
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[StandardScaler()],
        size=600,
        balanced=True,
        replace=True,
    )
    X_test, y_test = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[StandardScaler()],
        size=600,
        balanced=True,
        replace=True,
    )

    model = RandomForestClassifier(max_depth=2)
    model.fit(X, y)
    print(evaluate_model(model, X, y, X_test, y_test))


def linear_svm():
    pca = PCA(n_components=0.95)
    X, y = load_transform(
        "data/xray_features_frontal_only.pt",
        transforms=[StandardScaler(), pca],
        size=14000,
        balanced=True,
        replace=False,
    )
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model = LinearSVC()
    model.fit(X, y)
    print(evaluate_model(model, X_train, y_train, X_test, y_test))


def main():
    # X, y = load_transform(
    #     "data/xray_features_frontal_only.pt",
    #     transforms=[StandardScaler()],
    #     size=2000,
    #     balanced=True,
    # )
    # plt.plot(np.sum(X[y==1],axis=1))
    # plt.show()
    # plt.imshow(np.reshape(X[5, :], (32, 32)))
    # plt.show()

    # X_test, y_test = load_dataset("data/xray_features_test_frontal_only.pt")
    # SVM_classification(X, y)
    # LDA_plot(X, y)
    transform_dataset(CSV_PATH, "data/xray_features_frontal_only.pt")
    # t_sne()
    # linear_svm()
    # random_f()


if __name__ == "__main__":
    main()
