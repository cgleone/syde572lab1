from generating_clusters import class_e, class_d, class_c, class_b, class_a

from classifiers import map_classifier, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np

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

# Classify grid
y_pred = []
for i in range(grid.shape[0]):
    x = grid[i, 0]
    y = grid[i, 1]
    pred = map_classifier(x, y, classes)
    y_pred.append(pred)

# Plot filled contour
zz = np.array(y_pred).reshape(xx.shape)
plt.contour(xx, yy, zz, levels=[0, 1, 2], cmap='Paired')

# Plot points
# Reset colours
plt.gca().set_prop_cycle(None)
for c in classes:
    plt.scatter(c.x[0, :], c.x[1, :], cmap='Paired', label=f'{c.name}')
plt.legend()
plt.title('MAP Classifier')
plt.tight_layout()
plt.show()

cm = confusion_matrix(classes, map_classifier)
print(cm)