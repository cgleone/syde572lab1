import numpy as np
from numpy.linalg import det

def map_classifier(x, classes):
    """
    Classify whether x belongs to class K given the kth mean, sigma, and prior
    :param x: pattern to classify, shape (2, 1)
    :param mus: list of class means
    :param sigmas: list of class covariances
    :param priors: list of prior weights, between 0 and 1
    :return: 0, 1, or 2 for the class
    """
    def _micd(x, c):
        return (x - c.mu).T @ np.linalg.inv(c.sigma) @ (x - c.mu)

    # Two class case
    if _micd(x, classes[1]) - _micd(x, classes[0]) >  \
            2 * np.log(classes[1].prior / classes[0].prior) + np.log(det(classes[0].sigma) / det(classes[1].sigma)):
        return 0
    else:
        return 1

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


from generating_clusters import *

# 3 class case
classes = [class_c, class_d, class_e]

import itertools
from collections import Counter

if __name__ == '__main__':
    # x = class_c.x[:, 1].reshape(2, 1)
    # y = map_classifier(x, classes)
    # print(x, y)
    #
    # # Three class case, how to decide?
    # inds = np.random.choice(3, 20)
    # xs = [classes[i].x[:, j].reshape(2, 1) for j, i in enumerate(inds)]
    # y_pred = []
    # y_true = [classes[i].name for i in inds]
    #
    # for x in xs:
    #     counter = Counter()
    #     for combo in itertools.combinations(classes, 2):
    #         counter.update(combo[map_classifier(x, combo)].name)
    #     y_pred.append(counter.most_common(1)[0][0])
    # print(y_true)
    # print(y_pred)
    #

    case_1_classes = [class_a, class_b]
    case_2_classes = [class_c, class_d, class_e]

    grid = create_grid()
    classified_grid = get_classified_grid(grid, ged, case_1_classes)

    for row in classified_grid:
        print(row)
    # create_boundary(classified_grid)




