from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    univariate_gaussian = UnivariateGaussian()
    # Question 1 - Draw samples and print fitted model
    expectation = 10
    variance = 1
    vector = np.random.normal(expectation, variance, 1000)
    univariate_gaussian.fit(vector)
    tuple = (univariate_gaussian.mu_, univariate_gaussian.var_)
    print(tuple)


    # Question 2 - Empirically showing sample mean is consistent
    vector_arr = np.array(vector)
    est_mean = []
    temp_normal = UnivariateGaussian()
    for i in range(10, 1001, 10):
        temp_array = vector_arr[0:i]
        temp_normal.fit(temp_array)
        cur_dist = abs(expectation - temp_normal.mu_)
        est_mean.append(cur_dist)
    fig = go.Figure()
    sample_array = np.arange(10, 1001, 10).tolist()
    fig.add_trace(go.Scatter(x=sample_array, y=est_mean))
    fig.update_layout(title="Distance of sampled mean vs given expectation as a function of number of samples.",
                      xaxis_title="Number of samples", yaxis_title="Distance from actual expectation")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_results = univariate_gaussian.pdf(vector)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=vector_arr, y=pdf_results, mode="markers",
                              marker=dict(color="purple")))
    fig1.update_layout(title="Probability density function results as a function of sample value.",
                       xaxis_title="Sample value", yaxis_title="PDF result")
    fig1.show()
    print(UnivariateGaussian.log_likelihood(univariate_gaussian.mu_,
                                            np.sqrt(univariate_gaussian.var_), vector))


def test_multivariate_gaussian():
    multivariate_gaussian = MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov_arr = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    vector = np.random.multivariate_normal(mu, cov_arr)

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
