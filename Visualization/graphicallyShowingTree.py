import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib.patches import Rectangle

from CustomTree.customTree import CustomDecisionTree




def visualize_custom_tree(tree: CustomDecisionTree, figsize=(18, 10)):
    """
    Draw the custom decision tree using networkx + matplotlib.
    """

    G = nx.DiGraph()

    labels = {}

    # build graph
    for node in tree.nodes.values():

        if node.is_leaf:
            label = (
                f"Leaf {node.node_id}\n"
                f"samples={len(node.sample_indices)}\n"
                f"h={round(node.h, 3) if node.h is not None else None}"
            )
        else:
            label = (
                f"Node {node.node_id}\n"
                f"X[{node.feature}] <= {round(node.threshold, 3)}\n"
                f"samples={len(node.sample_indices)}"
            )

        labels[node.node_id] = label

        if node.left_child is not None:
            G.add_edge(node.node_id, node.left_child.node_id)

        if node.right_child is not None:
            G.add_edge(node.node_id, node.right_child.node_id)

    # hierarchical layout
    pos = hierarchy_pos(G, tree.root.node_id)

    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=2200,
        node_color="lightgray",
        font_size=8,
    )

    plt.title("Custom Decision Tree")
    plt.show()

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Recursively compute node positions for a tree layout.
    """

    def _hierarchy_pos(G, root, left, right, vert_loc, pos):

        pos[root] = ((left + right) / 2, vert_loc)

        children = list(G.successors(root))
        if len(children) == 0:
            return pos

        dx = (right - left) / len(children)

        nextx = left
        for child in children:
            pos = _hierarchy_pos(
                G,
                child,
                nextx,
                nextx + dx,
                vert_loc - vert_gap,
                pos,
            )
            nextx += dx

        return pos

    return _hierarchy_pos(G, root, 0, width, vert_loc, {})




