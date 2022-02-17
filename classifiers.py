import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt
import generating_clusters
from collections import Counter
import itertools


def create_grid(xmin=-10, xmax=30, ymin=-10, ymax=30):
    """

    :param xmin: minimum x value
    :param xmax: maximum x value for grid
    :param ymin: minimum y value for grid
    :param ymax: maximum y value for grid
    :return: grid list of lists
    """
    x = np.linspace(xmin, xmax, xmax-xmin+1)
    y = np.linspace(ymin, ymax, ymax-ymin+1)

    grid = []
    for y_coord in y:
        row_list = []
        for x_coord in x:
            row_list.append(x_coord)
        grid.append(row_list)

    return grid

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


def get_classified_grid(grid, classifier_function, classes):

    classified_grid = []
    y_coord = 30  # start at the top
    for row in grid:
        row_list = []
        for x_coord in row:
            assigned_class = classifier_function(x_coord, y_coord, classes)
            row_list.append(assigned_class)
        classified_grid.append(row_list)
        y_coord = y_coord - 1

    return classified_grid

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

    
    
    
    return


    # all_points =[]
    # for c in classes:
    #     all_points.append(c.x)

    # closest = min(all_points, key= lambda i: _euclid_distance(x, i)
        
    
    return 



if __name__ == '__main__':

    case_1_classes = [generating_clusters.class_a, generating_clusters.class_b]
    case_2_classes = [generating_clusters.class_c, generating_clusters.class_d, generating_clusters.class_e]

    grid = create_grid()
    classified_grid = get_classified_grid(grid, ged, case_1_classes)

    for row in classified_grid:
        print(row)
    create_boundary(classified_grid)
