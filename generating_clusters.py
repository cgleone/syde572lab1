import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
np.random.seed(0)
from ellipse import ellipse_points

def generate_bivariate_data(mu, sigma, N=200):
    """
    Generate bivariate Gaussian data with mean mu and covariance sigma
    :param mu: np.ndarray of shape (2, 1)
    :param sigma: np.ndarray of shape (2, 2)
    :param N: int, number of samples
    :return: np.ndarray of shape (2, 200)
    """
    x = np.random.randn(2, N)
    x = (scipy.linalg.sqrtm(sigma) @ x) + mu
    return x

mu_a = np.array([5, 10]).reshape(2, 1)
mu_b = np.array([10, 15]).reshape(2, 1)
mu_c = mu_a
mu_d = mu_b
mu_e = np.array([10, 5]).reshape(2, 1)

sigma_a = np.array([[8, 0], [0, 4]])
sigma_b = sigma_a
sigma_c = np.array([[8, 4], [4, 40]])
sigma_d = np.array([[8, 0], [0, 8]])
sigma_e = np.array([[10, -5], [-5, 20]])

x_a = generate_bivariate_data(mu_a, sigma_a, N=200)
x_b = generate_bivariate_data(mu_b, sigma_b, N=200)
x_c = generate_bivariate_data(mu_c, sigma_c, N=100)
x_d = generate_bivariate_data(mu_d, sigma_d, N=200)
x_e = generate_bivariate_data(mu_e, sigma_e, N=150)

priors = [0.5, 0.5, 100/450, 200/450, 150/450]

class Cluster:
    def __init__(self, mu, sigma, prior, N=200, name=''):
        self.mu = mu
        self.sigma = sigma
        self.x = generate_bivariate_data(mu, sigma, N)
        self.name = name
        self.prior = prior


class_a = Cluster(mu_a, sigma_a, 0.5, N=200, name='A')
class_b = Cluster(mu_b, sigma_b, 0.5, N=200, name='B')
class_c = Cluster(mu_c, sigma_c, 100/450, N=100, name='C')
class_d = Cluster(mu_d, sigma_d, 200/450, N=200, name='D')
class_e = Cluster(mu_e, sigma_e, 150/450, N=150, name='E')

if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # Case 1
    axs[0].scatter(class_a.x[0, :], class_a.x[1, :], label='A')
    axs[0].scatter(class_b.x[0, :], class_b.x[1, :], label='B')
    axs[0].scatter(*ellipse_points(mu_a, sigma_a), c='C0', s=2)
    axs[0].scatter(*ellipse_points(mu_b, sigma_b), c='C1', s=2)
    axs[0].legend()
    axs[0].set_title("Case 1: A and B")

    # Case 2
    axs[1].scatter(class_c.x[0, :], class_c.x[1, :], label='C')
    axs[1].scatter(class_d.x[0, :], class_d.x[1, :], label='D')
    axs[1].scatter(class_e.x[0, :], class_e.x[1, :], label='E')
    axs[1].scatter(*ellipse_points(mu_c, sigma_c), c='C0', s=2)
    axs[1].scatter(*ellipse_points(mu_d, sigma_d), c='C1', s=2)
    axs[1].scatter(*ellipse_points(mu_e, sigma_e), c='C2', s=2)
    axs[1].legend()
    axs[1].set_title("Case 2: C, D, and E")
    plt.tight_layout()
    plt.show()