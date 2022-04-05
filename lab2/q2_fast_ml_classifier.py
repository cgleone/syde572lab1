from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from numpy.linalg import det

def ml_classifier_array(x, classes):
    """
    Classify a point (x, y) into classes using ML
    We choose the class that maximizes the likelihood P(x|A) = P(x|A)*P(A) / P(x)
    Since P(x) is the same marginal for all cases, we want max(P(x|C_i)*P(C_i)) for C_i in classes
    """
    likelihoods = []
    for c in classes:
        mean = (x.reshape(x.shape[0], 1, 2) - c.mu.reshape(1, 2)) 
        sigma = np.linalg.inv(c.sigma)
        res =  mean @ sigma.T
        res = res @ mean.reshape(x.shape[0], 2, 1)
        likelihood = np.exp(-0.5 * res.flatten()) / (2 * np.pi * np.sqrt(det(c.sigma)))
        likelihoods.append(likelihood)
    
    idx = np.argmax(np.array(likelihoods), axis=0)
    return idx

# Make classes
al_cluster = Cluster(al, mu_hat_al, sigma_hat_al, name = 'al')
bl_cluster = Cluster(bl, mu_hat_bl, sigma_hat_bl, name = 'bl')
cl_cluster = Cluster(cl, mu_hat_cl, sigma_hat_cl, name = 'cl')
classes = [al_cluster, bl_cluster, cl_cluster]

# Plots
ymin, ymax, xmin, xmax = 0, 500, 0, 500
step = 5
count = int((xmax-xmin+1)/step)
x = np.linspace(xmin, xmax, count)
y = np.linspace(ymin, ymax, count)
x_coords, y_coords = np.meshgrid(x, y, indexing='ij')
classified_array = np.zeros_like(x_coords)


# Make grid
xlims = (0, 500)
ylims = (0, 500)
dx = dy = 1

xx, yy = np.meshgrid(np.arange(*xlims, dx), np.arange(*ylims, dy))

# Create vectors to feed to classifier
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1, r2))


# Plot contour
y_pred = ml_classifier_array(grid, classes)
zz = np.array(y_pred).reshape(xx.shape)
plt.contour(xx, yy, zz, levels=list(range(4)), cmap='Paired')

plt.scatter(al_cluster.x[:, 0], al_cluster.x[:, 1], label='a', s=5)
plt.scatter(bl_cluster.x[:, 0], bl_cluster.x[:, 1], label='b', s=5)
plt.scatter(cl_cluster.x[:, 0], cl_cluster.x[:, 1], label='b', s=5)
plt.legend()
plt.show()