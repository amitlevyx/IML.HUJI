from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    np.random.seed(0)
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
    sample_array = np.linspace(10, 1001, 100)
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

    vector2 = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print("q3 in quiz when mu = 1: ", UnivariateGaussian.log_likelihood(1, 1, vector2))
    print("q3 in quiz when mu = 10: ", UnivariateGaussian.log_likelihood(10, 1, vector2))



def test_multivariate_gaussian():
    np.random.seed(0)
    multivariate_gaussian = MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    mu1 = np.array([0, 0, 4, 0])
    cov_arr = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    vector = np.random.multivariate_normal(mu1, cov_arr, 1000)
    multivariate_gaussian.fit(vector)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    arr = np.linspace(-10, 10, 200)
    f1 = arr
    f3 = arr
    log_arr = []
    for i in range(f1.size):
        cur_results = []
        for j in range(f3.size):
            cur_mu = np.array([arr[i], 0, arr[j], 0])
            cur_results.append(multivariate_gaussian.log_likelihood(cur_mu, cov_arr, vector))
        log_arr.append(cur_results)
    fig = go.Figure((go.Heatmap(x=f3, y=f1, z=log_arr)))
    fig.update_layout(title="Log-likelihood of given cov matrix as a function of different first and third variables",
                      yaxis_title="First variable value", xaxis_title="Third variable values")
    fig.show()


    # Question 6 - Maximum likelihood
    result = np.where(log_arr == np.amax(log_arr))
    print("f1 : ", arr[result[0]])
    print("f3 : ", arr[result[1]])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
