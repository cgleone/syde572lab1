import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt
import generating_clusters
from collections import Counter
import itertools



def create_boundary(classified_grid):

    boundary_xs = []
    boundary_ys = []
    xcoord = -10
    ycoord = 30
    for row in classified_grid:
        for i in range(len(row)-1):
            if row[i] != row[i+1]:
                boundary_ys.append(ycoord-0.5)
                boundary_xs.append(xcoord+0.5)
            xcoord += 1
        ycoord = ycoord - 1
        xcoord = -10

    plt.plot(np.array(boundary_xs), np.array(boundary_ys))
    plt.scatter(generating_clusters.class_a.x[0, :], generating_clusters.class_a.x[1, :], label='A')
    plt.scatter(generating_clusters.class_b.x[0, :], generating_clusters.class_b.x[1, :], label='B')
    plt.legend()
    plt.show()

def paint_by_numbers(classified_grid):
    plt.plot()
    pass


def get_classified_grid(x_coords, y_coords, classifier_function, classes):

    grid = classifier_function(x_coords, y_coords, classes)
    return grid

"""
xs = np.arange(-8, 8, 0.1)
ys = np.arange(-8, 8, 0.1)
xx, yy = np.meshgrid(xs, ys, indexing='ij')
out_grid = np.zeros((160, 160))
for i in range(xs.size):
    for j in range(ys.size):
        point = np.array([xx[i,j], yy[i, j]])
        out_grid[i, j] = nn.forward(point)  # Predict the class of this point
"""

def med(x, y, classes):
    """

    :param x: the x coord of the point to be classified
    :param y: the y coord of the point to be classified
    :param mus:
    :param sigmas:
    :return: 0, 1, or 2 for the class
    """
    point = np.array([x, y]).reshape(2, 1)
    distances = []
    for cluster in classes:
        prototype = cluster.mu
        d_E = np.sqrt((point-prototype).T @ (point - prototype))
        distances.append(d_E)

    index_of_min = min(range(len(distances)), key=distances.__getitem__)
    return index_of_min


def ged(x, y, classes):
    """

    :param x: x coord of grid point
    :param y: y coord of grid point
    :param classes: list of cluster objects with mu and sigma information
    :return: 0, 1, or 2 for the class
    """

    pattern = np.array([x,y]).reshape(2,1)

    def _distance_metric(x, mu, sigma):
        distance = ((x - mu).T@np.linalg.inv(sigma)@(x-mu))**(1/2)
        return distance

    class0_distance = _distance_metric(pattern, classes[0].mu, classes[0].sigma)
    class1_distance = _distance_metric(pattern, classes[1].mu, classes[1].sigma)

    if len(classes) > 2:
        class2_distance = _distance_metric(pattern, classes[2].mu, classes[2].sigma)
        if class2_distance < class1_distance and class2_distance < class0_distance:
            return 2
        elif class0_distance < class1_distance:
            return 0
        else:
            return 1

    else:
        if class0_distance < class1_distance:
            return 0
        else:
            return 1




def map_classifier(x, y, classes):
    """
    Classify whether x belongs to class K given the kth mean, sigma, and prior
    :param x: first dimension of pattern
    :param y: second dimension of pattern
    :return: 0, 1 for the binary classes
    """
    x = np.array([x, y]).reshape(2, 1)

    def _micd(x, c):
        return (x - c.mu).T @ np.linalg.inv(c.sigma) @ (x - c.mu)

    # Two class case
    if _micd(x, classes[0]) - _micd(x, classes[1]) >  \
            2 * np.log(classes[1].prior / classes[0].prior) + np.log(det(classes[0].sigma) / det(classes[1].sigma)):
        return 0
    else:
        return 1

def multi_class_map(x, y, classes):
    """
    Do MAP classification for multiple classes, pick whichever one is most common
    :param x: first dimension of pattern
    :param y: second dimension of pattern
    :param classes: list of cluster objects
    :return: int 0, 1, or 2 for the class
    """
    # x = np.array([x, y]).reshape(2, 1)
    class_names_to_index = {c.name: i for i, c in enumerate(classes)}
    counter = Counter()
    for combo in itertools.combinations(classes, 2):
        counter.update(combo[map_classifier(x, y, combo)].name)
    c_pred = counter.most_common(1)[0][0]
    return class_names_to_index[c_pred]



if __name__ == '__main__':

    case_1_classes = [generating_clusters.class_a, generating_clusters.class_b]
    case_2_classes = [generating_clusters.class_c, generating_clusters.class_d, generating_clusters.class_e]

    ymin, ymax, xmin, xmax = -10, 30, -10, 30
    step = 1
    x = np.linspace(xmin, xmax, int((xmax-xmin+1)/step))
    y = np.linspace(ymin, ymax, ymax-ymin+1)
    x_coords, y_coords = np.meshgrid(x, y, indexing='ij')
    classified_coords = get_classified_grid(x_coords, y_coords, med, case_1_classes)

    plt.contourf(x_coords, y_coords, classified_coords, cmap='Paired')

    # Plot points
    # Reset colours
    plt.gca().set_prop_cycle(None)
    for c in case_1_classes:
        plt.scatter(c.x[0, :], c.x[1, :], cmap='Paired', label=f'{c.name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
