from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "C:/amiti_hamalka/iml/IML.HUJI/datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "C:/amiti_hamalka/iml/IML.HUJI/datasets/linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=lambda p, Xi, yi: losses.append(perceptron.loss(X, y)))
        perceptron.fit(X, y)

        # Plot figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(losses) + 1)), y=losses))
        fig.update_layout(title=f"{n} dataset loss as function of iteration number",
                          xaxis_title="Iteration number", yaxis_title="loss")
        # fig.show()
        output_path = "C:/amiti_hamalka/iml"
        fig.write_html(output_path + "/" + str(n) + ".html")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy",
              "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        gaussian = GaussianNaiveBayes()
        lda.fit(X, y)
        gaussian.fit(X, y)
        lda_pred = lda.predict(X)
        gaussian_pred = gaussian.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        model_names = ["Linear Discriminant Analysis. Accuracy: " + str(accuracy(y, lda_pred)),
                       "Gaussian Naive Bayes. Accuracy: " + str(accuracy(y, gaussian_pred))]

        fig = make_subplots(rows=1, cols=2, subplot_titles=model_names)
        fig.update_layout(title="classifier evaluation for dataset " + str(f.split('.npy')[0]))
        fig.append_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
            marker=dict(color=lda_pred, symbol=class_symbols[y])), row=1, col=1)
        fig.append_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                    marker=dict(color="Black", symbol="x")), row=1, col=1)
        for k in range(lda.classes_.shape[0]):
            fig.append_trace(get_ellipse(lda.mu_[k], lda.cov_), row=1, col=1)

        fig.append_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
            marker=dict(color=gaussian_pred, symbol=class_symbols[y])), row=1, col=2)
        for k in range(gaussian.classes_.shape[0]):
            fig.append_trace(get_ellipse(gaussian.mu_[k], np.diag(gaussian.vars_[k])), row=1, col=2)
        fig.append_trace(go.Scatter(x=gaussian.mu_[:, 0], y=gaussian.mu_[:, 1], mode="markers",
                                    marker=dict(color="Black", symbol="x")), row=1, col=2)
        # output_path = "C:/amiti_hamalka/iml"
        fig.write_html(str(f.split('.npy')[0]) + ".html")


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    gaus = GaussianNaiveBayes()
    S = {(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)}
    X = np.array([[1,1], [1,1], [2,1], [2,4], [3,3], [3,4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    gaus.fit(X, y)
    print(gaus.vars_)
    print(gaus.mu_[1])
