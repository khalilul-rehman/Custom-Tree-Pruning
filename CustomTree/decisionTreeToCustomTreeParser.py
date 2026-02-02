from sklearn.tree import DecisionTreeRegressor
import numpy as np
import copy
from CustomTree.customTree import CustomDecisionTree, CustomTreeNode

def sklearn_tree_to_custom_tree(model: DecisionTreeRegressor, X) -> CustomDecisionTree:
    tree = model.tree_
    custom_tree = CustomDecisionTree()
    node_map = {}

    # Create nodes
    for i in range(tree.node_count):
        is_leaf = tree.children_left[i] == -1
        node = CustomTreeNode(
            node_id=i,
            is_leaf=is_leaf,
            value=tree.value[i][0].copy(),
            feature=None if is_leaf else tree.feature[i],
            threshold=None if is_leaf else tree.threshold[i],
            impurity=tree.impurity[i],
            n_samples=tree.n_node_samples[i]
        )
        node_map[i] = node
        custom_tree.nodes[i] = node

    # Link nodes
    for i in range(tree.node_count):
        node = node_map[i]
        if not node.is_leaf:
            left = node_map[tree.children_left[i]]
            right = node_map[tree.children_right[i]]

            node.left_child = left
            node.right_child = right
            left.parent = node
            right.parent = node

    # Assign root
    custom_tree.root = node_map[0]

    # Compute depths correctly
    def assign_depths(node, depth=0):
        node.depth = depth
        # custom_tree.max_depth = max(custom_tree.max_depth, depth)
        if not node.is_leaf:
            assign_depths(node.left_child, depth + 1)
            assign_depths(node.right_child, depth + 1)

    assign_depths(custom_tree.root)

    #  Assigning the samples to nodes
    node_indicator = model.decision_path(X)

    for node_id, node in custom_tree.nodes.items():
        mask = node_indicator[:, node_id].toarray().ravel().astype(bool)
        node.sample_indices = np.where(mask)[0]

    return custom_tree



# After this, I can prune or modify tree_copy without affecting the original tree.
def copy_custom_tree(tree: CustomDecisionTree) -> CustomDecisionTree:
    new_tree = CustomDecisionTree()
    node_map = {}

    # First pass: copy all nodes
    for node_id, node in tree.nodes.items():
        new_node = CustomTreeNode(
            node_id=node.node_id,
            is_leaf=node.is_leaf,
            value=node.value.copy(),  # copy numpy array
            feature=node.feature,
            threshold=node.threshold,
            impurity=node.impurity,
            n_samples=node.n_samples
        )
        new_node.h = node.h
        new_node.M = None if node.M is None else node.M.copy()
        new_node.M_0 = None if node.M_0 is None else node.M_0.copy()
        new_node.sample_indices = node.sample_indices.copy()
        new_node.depth = node.depth

        node_map[node_id] = new_node
        new_tree.nodes[node_id] = new_node

    # Second pass: set parent/child links
    for node_id, node in tree.nodes.items():
        new_node = node_map[node_id]
        if node.left_child is not None:
            new_node.left_child = node_map[node.left_child.node_id]
            node_map[node.left_child.node_id].parent = new_node
        if node.right_child is not None:
            new_node.right_child = node_map[node.right_child.node_id]
            node_map[node.right_child.node_id].parent = new_node

    # Set root
    new_tree.root = node_map[tree.root.node_id]

    return new_tree
