from IMLearn.metrics import loss_functions
from IMLearn.utils import split_train_test, split_set_to_X_y, utils
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df.drop(df[(df["id"] == 0)].index, inplace=True)
    df.drop(df[(df["price"] <= 0)].index, inplace=True)
    df.drop(df[(df["bedrooms"] <= 0)].index, inplace=True)
    features = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition",
                   "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15"]]
    features = pd.concat([features, pd.get_dummies(df['zipcode'], prefix="zipcode_")], axis=1)
    year_built = 2022 - df["yr_built"]
    year_renovated = 2022 - df["yr_renovated"]
    features[["year_since_last_remodel"]] = pd.concat([year_built, year_renovated], axis=1).min(axis=1)
    response = df[['price']]
    return features, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    # todo for bad feature choose one with pearson close to zero, for good pearson close to 1/-1
    for feature in X.columns.values:
        stdX = np.std(X[feature].tolist())
        y_list = y['price'].tolist()
        stdY = np.std(y_list)
        covXy = np.cov(X[feature], y_list, rowvar=False)
        pearson_corr = covXy / (stdX * stdY)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=X[feature], y=y_list, mode="markers", marker=dict(color="purple")))
        fig1.update_layout(title="Pearson Correlation between " + str(feature) + " and response is " +
                                 str(pearson_corr[0][1]) + ". feature is - " + str(feature), xaxis_title=str(feature)
                                                                                                         + " number",
                           yaxis_title="response result")
        fig1.write_html(output_path + "/" + str(feature) + ".html")


if __name__ == '__main__':
    np.random.seed(0)
    print(loss_functions.mean_square_error(np.array([279000, 432000, 326000, 333000, 437400, 555950]), np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])))
    # Question 1 - Load and preprocessing of housing prices dataset
    features, response = load_data("C:/amiti_hamalka/iml/IML.HUJI/datasets/house_prices.csv")
    # pd.DataFrame(features).to_csv("test_data.csv", index=False)
    # pd.DataFrame(response).to_csv("response.csv", index=False)

    # Question 2 - Feature evaluation with respect to response

    # feature_evaluation(features, response, "C:/amiti_hamalka/iml/IML.HUJI/exercises/plots")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    ave_loss = []
    ave_var = []
    joined = pd.concat([train_X, train_y], axis=1)
    fitter = LinearRegression(True)
    for p in range(10, 101, 1):
        p_loss = []
        for i in range(10):
            cur_data = joined.sample(frac=(p * 0.01))
            cur_X, cur_y = split_set_to_X_y(cur_data)
            fitter.fit(cur_X.to_numpy(), cur_y)
            cur_loss = fitter.loss(utils.include_intercept(test_X), test_y.to_numpy())
            p_loss.append(cur_loss)
        ave_var.append(np.var(p_loss))
        ave_loss.append(np.mean(p_loss))
    p = np.linspace(10, 100, 91)
    fig2 = go.Figure()
    upper_bound = list(np.array(ave_loss) + 2 * np.sqrt(np.array(ave_var)))
    lower_bound = list(np.array(ave_loss) - 2 * np.sqrt(np.array(ave_var)))
    print(upper_bound)
    print(lower_bound)
    fig2.add_trace(go.Scatter(name="loss", x=p, y=ave_loss, mode="lines", line=dict(color="purple")))
    fig2.add_trace(go.Scatter(name='Upper Bound', x=p, y=upper_bound, mode='lines', marker=dict(color="#444"),
                              line=dict(width=0), showlegend=False))
    fig2.add_trace(go.Scatter(name='Lower Bound', x=p, y=lower_bound, marker=dict(color="#444"), line=dict(width=0),
                              mode='lines', fillcolor='rgba(68, 68, 68, 0.3)', fill='tonexty', showlegend=False))
    fig2.update_layout(title="average loss as function of training size",
                       xaxis_title="percentage taken from training data", yaxis_title="average loss")
    fig2.show()
