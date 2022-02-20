from generating_clusters import class_e, class_d, class_c, class_b, class_a

from classifiers import create_grid, get_classified_grid, nn, knn, create_boundary
from matplotlib import pyplot as plt
import numpy as np

case_1_classes = [class_a, class_b]
case_2_classes = [class_c, class_d, class_e]

grid = create_grid()
classified_grid = get_classified_grid(grid, knn, case_1_classes)

for row in classified_grid:
    print(row)
create_boundary(classified_grid)


