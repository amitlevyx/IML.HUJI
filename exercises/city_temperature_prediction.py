import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test, utils

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df = df[df['Temp'] > -72]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    features_response = df[['Country', 'City', 'DayOfYear', 'Year', 'Month', 'Day', 'Temp']]
    return features_response


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    features_response = load_data("C:/amiti_hamalka/iml/IML.HUJI/datasets/City_Temperature.csv")

    #
    # Question 2 - Exploring data for specific country
    features_response_israel = features_response[(features_response['Country'] == 'Israel')]
    features_response_israel['Year'] = features_response_israel['Year'].astype(str)
    figYear = px.scatter(features_response_israel, x='DayOfYear', y='Temp', color='Year',
                         title="Daily temperature as a function of Day of year, color coded by year value")
    figYear.show()
    israel_groupby_month = features_response_israel.groupby(['Month'], as_index=False).agg({'Temp': 'std'})
    figMonth = px.bar(israel_groupby_month, x='Month', y='Temp',
                      title="Daily temperature std per month in Israel")
    figMonth.update_layout(xaxis_title="Month", yaxis_title="Temperature std")
    figMonth.show()

    #
    # Question 3 - Exploring differences between countries
    groupby_country_month = features_response.groupby(['Country', 'Month'], as_index=False).agg(
        Mean_temp=('Temp', 'mean'),
        STD_temp=('Temp', 'std'))
    figCountry = px.line(groupby_country_month, x='Month', y='Mean_temp'
                         , color='Country', error_y='STD_temp',
                         hover_data=['Month', 'Mean_temp', 'STD_temp', 'Country', 'STD_temp'],
                         title="mean temperature and standard deviation per month' represented for each sampled"
                               " country.")
    figCountry.show()
    #
    # Question 4 - Fitting model for different values of `k`
    train_X_israel, train_y_israel, test_X_israel, test_y_israel = utils.split_train_test(
        features_response_israel['DayOfYear'], features_response_israel['Temp'])
    loss_arr = []
    for k in range(1, 11, 1):
        k_fitter = PolynomialFitting(k)
        k_fitter.fit(train_X_israel.to_numpy(), train_y_israel.to_numpy())  # ???
        loss_arr.append(round(k_fitter.loss(test_X_israel.to_numpy(), test_y_israel), 2))
        print("Loss val for k=" + str(k) + " is " + str(loss_arr[-1]))

    figTestError = px.bar(x=np.linspace(1, 10, 10), y=loss_arr, labels={'x': 'k value', 'y': 'Test Error value'},
                          title='Error Test loss as a function of degree of polynomial (k)')
    figTestError.show()
    #
    # Question 5 - Evaluating fitted model on different countries
    # chose k
    five_israel_fitter = PolynomialFitting(5)
    five_israel_fitter.fit(features_response_israel['DayOfYear'].to_numpy(),
                           features_response_israel['Temp'].to_numpy())


    def calc_loss(df: pd.DataFrame):
        x = df['DayOfYear'].to_numpy()
        y = df['Temp'].to_numpy()
        return pd.Series({"Loss": round(five_israel_fitter.loss(x, y), 2)})


    groupby_country = features_response.groupby('Country', as_index=False).apply(calc_loss)
    figIsraelCountries = px.bar(groupby_country, x='Country', y='Loss',
                                title="Countries loss over model fitted for Israel.")
    figIsraelCountries.show()
