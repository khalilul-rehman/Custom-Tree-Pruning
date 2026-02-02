import numpy as np

class CustomTreeNode:
    """
    Custom tree node for multivariate regression
    """
    def __init__(
        self,
        node_id,
        is_leaf,
        value,
        feature=None,
        threshold=None,
        impurity=None,
        n_samples=None,
    ):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.value = value              # ndarray (n_targets,)
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.n_samples = n_samples
        self.sample_indices = []
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.depth = 0

        self.h=None
        self.M=None
        self.M_0=None

    def __repr__(self):
        if self.is_leaf:
            return (
                f"Leaf(id={self.node_id}, "
                f"value={np.round(self.value, 3)}, "
                f"samples={self.n_samples})"
            )
        return (
            f"Node(id={self.node_id}, "
            f"feature={self.feature}, "
            f"threshold={self.threshold:.3f}, "
            f"samples={self.n_samples})"
        )


class CustomDecisionTree:
    def __init__(self):
        self.root = None
        self.nodes = {}
        # self.max_depth = 0

    def add_node(self, node):
        self.nodes[node.node_id] = node
        if node.parent is None:
            self.root = node
            node.depth = 0
        else:
            node.depth = node.parent.depth + 1
            self.max_depth = max(self.max_depth, node.depth)

    def get_leaves(self):
        return [n for n in self.nodes.values() if n.is_leaf]

    def num_nodes(self):
        return len(self.nodes)

    def num_leaves(self):
        return sum(1 for n in self.nodes.values() if n.is_leaf)
    
    def compute_max_depth(self):

        if self.root is None:
            return 0

        max_depth = 0
        stack = [(self.root, 0)]

        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)

            if not node.is_leaf:
                stack.append((node.left_child, depth + 1))
                stack.append((node.right_child, depth + 1))

        return max_depth

    def get_subtree_nodes(self, node):
        stack = [node]
        subtree = []
        while stack:
            current = stack.pop()
            subtree.append(current)
            if not current.is_leaf:
                stack.append(current.right_child)
                stack.append(current.left_child)
        return subtree

    def print_tree(self, node=None, indent=0):
        if node is None:
            node = self.root
        print("  " * indent + str(node))
        if not node.is_leaf:
            self.print_tree(node.left_child, indent + 1)
            self.print_tree(node.right_child, indent + 1)
