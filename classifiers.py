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
    Classify a particular point on the grid using GED classification
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
    Classify a point (x, y) into classes using MAP
    We choose the class that maximizes the posterior P(A|x) = P(x|A)*P(A) / P(x)
    Since P(x) is the same marginal for all cases, we want max(P(x|C_i)*P(C_i)) for C_i in classes
    """
    x = np.array([x, y]).reshape(2, 1)
    posteriors = []
    for c in classes:
        liklihood = np.exp(-0.5 * (x - c.mu).T @ np.linalg.inv(c.sigma) @ (x - c.mu)) / (2 * np.pi * np.sqrt(det(c.sigma)))
        prior = c.prior
        posteriors.append(liklihood * prior)
    idx = np.argmax(posteriors)
    return idx


def nn(x, y, classes):
    """
    Classify whether x belongs to class K given its nearest neighbour
    :param x: first dimension of pattern
    :param y: second dimension of pattern
    :return: 0, 1 or 2 for the classes
    """
    x= np.array([x, y]).reshape(2,1)

    def _euclid_distance(x, c):
        d = []
        for i in range(len(c.x[0])):
            point = np.array([c.x[0][i], c.x[1][i]]).reshape(2,1)
            d_point = ((x-point).T@(x-point))**(1/2)
            d.append(d_point)
        return d
    
    min_distances = []
    
    for c in classes:
        distances = _euclid_distance(x, c)
        min_distance = min(distances)
        min_distances.append(min_distance)
    
    index = min_distances.index(min(min_distances))
    return index

def knn(x, y, classes):
    """
    Classify whether x belongs to class K based on K-nearest neighbour (K=5)
    :param x: first dimension of pattern
    :param y: second dimension of pattern
    :return: 0, 1, or 2 for the class
    """
    x= np.array([x, y]).reshape(2,1)
    
    def _euclid_distance(x, c):
        d = []
        for i in range(len(c.x[0])):
            point = np.array([c.x[0][i], c.x[1][i]]).reshape(2,1)
            d_point = ((x-point).T@(x-point))**(1/2)
            d.append(d_point)
        return d
    
    average_min_distances = []
    for c in classes:
        distances = _euclid_distance(x, c)
        distances.sort()
        top_5 = distances[0:5]
        average_min_distance = sum(top_5)/len(top_5)
        average_min_distances.append(average_min_distance)
    
    index = average_min_distances.index(min(average_min_distances))
    return index

    
def confusion_matrix(classes, classifier):
    """
    Generate confusion matrix of data in classes based on classifier
    :param classes: list of Cluster objects where Cluster.x is data to classify
    :return cm: np.ndarray confusion matrix of shape (len(classes), len(classes))
    """
    temp = []
    cm = np.zeros((len(classes), len(classes)))
    for c in classes:
        x = c.x
        output = []
        for i in range(x.shape[1]):
            output.append(classifier(x[0, i], x[1, i], classes))
        temp.append(Counter(output))
    for i, counter in enumerate(temp):
        for j, val in counter.items():
            cm[i, j] = val
    return cm

def confusion_matrix_nn(classes, classifier):
    
    # Split into training and testing points
    temp = []
    cm = np.zeros((len(classes), len(classes)))
    train_classes = []
    test_classes = []
    for c in classes:
        np.random.seed(0)
        train_class = generating_clusters.Cluster(c.mu, c.sigma, c.prior, N = int(len(c.x[0])*0.5), name='train')
        train_classes.append(train_class)
        np.random.seed(7)
        test_class = generating_clusters.Cluster(c.mu, c.sigma, c.prior, N = int(len(c.x[0])*0.5), name='test')
        test_classes.append(test_class)
    
    for i in range(len(classes)):
        x_test = test_classes[i].x
        output = []
        for i in range(x_test.shape[1]):
            output.append(classifier(x_test[0, i], x_test[1, i], train_classes))
        temp.append(Counter(output))
    
    
    for i, counter in enumerate(temp):
        for j, val in counter.items():
            cm[i, j] = val
    
    np.random.seed(0)
    
    return cm



def error_rate(cm):
    """Compute error rate based on confusion matrix.
    Correct elements are on diagonal, incorrect are off diagonal
    So trace(cm) / sum(cm) is proportion correct, 1-correct is error rate
    :param cm: numpy square array of confusion matrix
    :return: error rate, float between 0 and 1
    """
    return 1 - (cm.trace() / cm.sum())


    # all_points =[]
    # for c in classes:
    #     all_points.append(c.x)

    # closest = min(all_points, key= lambda i: _euclid_distance(x, i)
        
    
    return 



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
