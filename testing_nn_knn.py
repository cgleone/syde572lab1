from generating_clusters import class_e, class_d, class_c, class_b, class_a, Cluster
from ellipse import ellipse_points
from classifiers import get_classified_grid, nn, knn, confusion_matrix_nn, error_rate
from matplotlib import pyplot as plt
import numpy as np
from ellipse import ellipse_points
from matplotlib.lines import Line2D

# case_1_classes = [class_a, class_b]
# case_2_classes = [class_c, class_d, class_e]

# grid = create_grid()
# classified_grid = get_classified_grid(grid, knn, case_1_classes)

# for row in classified_grid:
#     print(row)
# create_boundary(classified_grid)


xlims = (-10, 30)
ylims = (-10, 30)
dx = dy = 0.2

xx, yy = np.meshgrid(np.arange(*xlims, dx), np.arange(*ylims, dy))

# Create vectors to feed to classifier
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1, r2))
# classes = [class_c, class_d, class_e]
classes = [class_a, class_b]

#Confusion Matrix
cm = confusion_matrix_nn(classes, knn)
print(cm)
print(f'Error rate: {error_rate(cm)*100:.2f}%')

# Classify grid
# y_nn = []
# y_knn = []
# for i in range(grid.shape[0]):
#     x = grid[i, 0]
#     y = grid[i, 1]
#     y_nn.append(nn(x,y, classes))
#     y_knn.append(knn(x,y, classes))


# Plot filled contours
# fig, ax = plt.subplots(figsize=(8, 6))

# colours = ['darkviolet', 'dodgerblue']
# names = ['NN', 'KNN']
# for y_pred, colour in zip([y_nn, y_knn], colours):
#     zz = np.array(y_pred).reshape(xx.shape)
#     plt.contour(xx, yy, zz, levels=list(range(len(classes))), colors=colour, linewidths=3, zorder=3)

# custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colours]
# ax.legend(custom_lines, names)

# for c in classes:
#     color = next(ax._get_lines.prop_cycler)['color']
#     plt.scatter(c.x[0, :], c.x[1, :], c=color, label=f'{c.name}', s=20, alpha=0.4)
#     plt.plot(*ellipse_points(c.mu, c.sigma), c=color, lw=2, zorder=2, alpha=0.8)
# # plt.legend()
# plt.title('NN, KNN Classifiers: Case 2')
# # Case 1
# # plt.xlim(-5, 22)
# # plt.ylim(0, 20)
# # Case 2
# plt.xlim(-5, 25)
# plt.ylim(-10, 27)
# plt.tight_layout()
# plt.show()
