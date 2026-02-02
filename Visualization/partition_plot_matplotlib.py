
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



# 2d Plotting

def plot_tree_partitions(tree, X, ax=None, resolution=200, cmap='Pastel1'):
    """
    Plot 2D partitions of a CustomDecisionTree using only X (ignore y).
    
    Parameters
    ----------
    tree : CustomDecisionTree
    X : np.ndarray of shape (n_samples, 2)
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot on
    resolution : int
        Number of points along each axis for meshgrid
    cmap : str
        Colormap for regions
    """
    if X.shape[1] != 2:
        raise ValueError("This function only supports 2D input features.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))

    # create meshgrid for the feature space
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # assign each grid point to a leaf
    leaf_ids = []
    for point in grid_points:
        node = tree.root
        while not node.is_leaf:
            node = node.left_child if point[node.feature] <= node.threshold else node.right_child
        leaf_ids.append(node.node_id)

    Z = np.array(leaf_ids).reshape(xx.shape)

    # plot the partition
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

    # optionally overlay training points
    ax.scatter(X[:,0], X[:,1], c='k', s=15, edgecolor='w', alpha=0.7)

    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_title('Custom Tree 2D Partitions')
    return ax

def plot_tree_partitions_2d_rectangles(tree, X, ax=None, cmap='Pastel1'):
    if X.shape[1] != 2:
        raise ValueError("Only 2D features are supported.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))

    # xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    # ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1

    xmin, xmax = X[:,0].min() , X[:,0].max() 
    ymin, ymax = X[:,1].min() , X[:,1].max() 

    cuboids = []

    def recurse(node, bounds):
        if node.is_leaf:
            cuboids.append((node.node_id, tuple(bounds)))
            return

        f = node.feature
        t = node.threshold

        # convert bounds to list-of-lists to modify
        left_bounds = [list(b) for b in bounds]
        right_bounds = [list(b) for b in bounds]

        left_bounds[f] = [bounds[f][0], t]
        right_bounds[f] = [t, bounds[f][1]]

        recurse(node.left_child, left_bounds)
        recurse(node.right_child, right_bounds)

    recurse(tree.root, [[xmin, xmax], [ymin, ymax]])

    # Draw rectangles
    for leaf_id, (x_b, y_b) in cuboids:
        rect = Rectangle((x_b[0], y_b[0]),
                         x_b[1] - x_b[0],
                         y_b[1] - y_b[0],
                         facecolor=plt.cm.get_cmap(cmap)(leaf_id % 10),
                         edgecolor='k',
                         alpha=0.3)
        ax.add_patch(rect)

    # ax.scatter(X[:,0], X[:,1], c='k', s=15, edgecolor='w', alpha=0.7)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_title('Custom Tree 2D Partitions (Rectangles)')
    return ax

# 3d Plotting

from mpl_toolkits.mplot3d import Axes3D


def plot_tree_partitions_3d_scatter(
    tree,
    X,
    resolution=40,
    cmap="tab20",
):
    """
    Scatter plot of tree partitions in 3D.

    NOTE: resolution^3 points will be generated.
    Keep resolution <= 40.
    """

    if X.shape[1] != 3:
        raise ValueError("This function only supports 3D inputs.")

    x0 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, resolution)
    x1 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, resolution)
    x2 = np.linspace(X[:, 2].min() - 1, X[:, 2].max() + 1, resolution)

    xx, yy, zz = np.meshgrid(x0, x1, x2)

    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    leaf_ids = np.empty(len(grid), dtype=int)

    for i, p in enumerate(grid):
        node = tree.root
        while not node.is_leaf:
            node = (
                node.left_child
                if p[node.feature] <= node.threshold
                else node.right_child
            )
        leaf_ids[i] = node.node_id

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    sc = ax.scatter(
        grid[:, 0],
        grid[:, 1],
        grid[:, 2],
        c=leaf_ids,
        cmap=cmap,
        s=4,
        alpha=0.6,
    )

    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_zlabel("Feature 2")
    ax.set_title("3D Tree Partitions")

    plt.show()





from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# -------------------------------------------------------
# Compute cuboids for each leaf
# -------------------------------------------------------

def get_leaf_cuboids_3d(tree, X):
    """
    Return list of (leaf_id, bounds) where bounds is:
        [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
    """

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    cuboids = []

    def recurse(node, bounds):
        if node.is_leaf:
            cuboids.append((node.node_id, tuple(bounds)))
            return

        f = node.feature
        t = node.threshold

        # convert to list-of-lists for mutability
        left_bounds = [list(b) for b in bounds]
        right_bounds = [list(b) for b in bounds]

        # left: x_f <= t
        left_bounds[f] = [bounds[f][0], t]

        # right: x_f > t
        right_bounds[f] = [t, bounds[f][1]]

        recurse(node.left_child, left_bounds)
        recurse(node.right_child, right_bounds)


    initial_bounds = tuple((mins[i], maxs[i]) for i in range(3))

    recurse(tree.root, initial_bounds)

    return cuboids


# -------------------------------------------------------
# Draw one cuboid
# -------------------------------------------------------

def draw_cuboid(ax, bounds, color, alpha=0.25):

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    vertices = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],

        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ])

    faces = [
        [vertices[j] for j in [0,1,2,3]],
        [vertices[j] for j in [4,5,6,7]],
        [vertices[j] for j in [0,1,5,4]],
        [vertices[j] for j in [2,3,7,6]],
        [vertices[j] for j in [1,2,6,5]],
        [vertices[j] for j in [0,3,7,4]],
    ]

    poly = Poly3DCollection(faces, alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor("k")

    ax.add_collection3d(poly)


# -------------------------------------------------------
# Main plotting function
# -------------------------------------------------------

def plot_tree_partitions_3d_cubes(tree, X, cmap="tab20"):

    if X.shape[1] != 3:
        raise ValueError("This function only supports 3D features.")

    cuboids = get_leaf_cuboids_3d(tree, X)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    colors = plt.get_cmap(cmap)(
        np.linspace(0, 1, len(cuboids))
    )

    for (leaf_id, bounds), color in zip(cuboids, colors):
        draw_cuboid(ax, bounds, color)

    ax.set_xlim(X[:,0].min(), X[:,0].max())
    ax.set_ylim(X[:,1].min(), X[:,1].max())
    ax.set_zlim(X[:,2].min(), X[:,2].max())

    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_zlabel("Feature 2")

    ax.set_title("3D Decision Tree Partitions (Cuboids)")

    plt.tight_layout()
    plt.show()
