import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from CustomTree.customTree import CustomDecisionTree

def plot_tree_partitions_plotly(tree: CustomDecisionTree, X: np.ndarray, leaf_colorscale=None, save_path=None):
    """
    Plot partitions of a custom tree using Plotly.
    Works for 2D (rectangles) or 3D (cuboids) automatically based on X.shape[1].

    Parameters
    ----------
    tree : CustomDecisionTree
        The trained custom tree.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix. Must be 2D or 3D.
    leaf_colorscale : list of colors, optional
        Custom color palette for leaves.
    """
    n_features = X.shape[1]
    if n_features not in [2, 3]:
        print("Only 2D or 3D features are supported for plotting.")
        return
        # raise ValueError("Only 2D or 3D features are supported for plotting.")

    # Prepare colors
    leaves = tree.get_leaves()
    n_leaves = len(leaves)
    if leaf_colorscale is None:
        leaf_colorscale = px.colors.qualitative.Pastel
    colors = [leaf_colorscale[i % len(leaf_colorscale)] for i in range(n_leaves)]
    leaf_to_color = {leaf.node_id: colors[i] for i, leaf in enumerate(leaves)}

    # Function to recursively get leaf bounds
    def get_leaf_bounds(node, bounds):
        """
        Returns a dict of leaf_id -> bounds
        bounds: list of [min, max] for each dimension
        """
        if node.is_leaf:
            return {node.node_id: [b.copy() for b in bounds]}
        else:
            dim = node.feature
            thresh = node.threshold
            bounds_left = [b.copy() for b in bounds]
            bounds_right = [b.copy() for b in bounds]
            bounds_left[dim][1] = thresh
            bounds_right[dim][0] = thresh

            out = {}
            out.update(get_leaf_bounds(node.left_child, bounds_left))
            out.update(get_leaf_bounds(node.right_child, bounds_right))
            return out

    # Initialize bounds using the dataset
    bounds_init = [[X[:, i].min() , X[:, i].max() ] for i in range(n_features)]
    leaf_bounds = get_leaf_bounds(tree.root, bounds_init)

    # --- 2D Plot ---
    if n_features == 2:
        fig = go.Figure()
        for leaf_id, b in leaf_bounds.items():
            x0, x1 = b[0]
            y0, y1 = b[1]
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor=leaf_to_color[leaf_id],
                line=dict(color='black'),
                name=f'Leaf {leaf_id}',
                showlegend=False
            ))
        # Overlay points
        # fig.add_trace(go.Scatter(
        #     x=X[:,0], y=X[:,1], mode='markers', marker=dict(color='black', size=4),
        #     name='Samples'
        # ))
        fig.update_layout(
            title="Custom Tree 2D Partitions",
            xaxis_title="Feature 0",
            yaxis_title="Feature 1",
            width=700, height=600
        )
        fig.show()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
            fig.write_html(save_path.replace('.png', '.html'))

    # --- 3D Plot ---
    elif n_features == 3:
        fig = go.Figure()
        for leaf_id, b in leaf_bounds.items():
            x0, x1 = b[0]
            y0, y1 = b[1]
            z0, z1 = b[2]

            # Add a transparent cuboid using Mesh3d
            fig.add_trace(go.Mesh3d(
                x=[x0, x1, x1, x0, x0, x1, x1, x0],
                y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[z0, z0, z0, z0, z1, z1, z1, z1],
                color=leaf_to_color[leaf_id],
                opacity=0.3,
                alphahull=0,
                name=f'Leaf {leaf_id}',
                showlegend=False
            ))
        # Overlay points
        # fig.add_trace(go.Scatter3d(
        #     x=X[:,0], y=X[:,1], z=X[:,2],
        #     mode='markers',
        #     marker=dict(color='black', size=3),
        #     name='Samples'
        # ))
        fig.update_layout(
            title="Custom Tree 3D Partitions",
            scene=dict(
                xaxis_title='Feature 0',
                yaxis_title='Feature 1',
                zaxis_title='Feature 2'
            ),
            width=800,
            height=700
        )
        fig.show()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
            fig.write_html(save_path.replace('.png', '.html'))



