
from CustomTree.quadratic_constraint_optimization import gurobi_minimax
from Visualization.graphicallyShowingTree import visualize_custom_tree 


def get_prunable_parents(tree):
    """
    Return all nodes whose two children are leaves.
    """
    candidates = []

    for node in tree.nodes.values():
        if node.is_leaf:
            continue

        left = node.left_child
        right = node.right_child

        if left and right and left.is_leaf and right.is_leaf:
            candidates.append(node)

    return candidates




def collapse_node(tree, node, X, y):
    """
    Replace subtree rooted at node by a leaf.
    """

    # remove descendants
    for child in [node.left_child, node.right_child]:
        for sub in tree.get_subtree_nodes(child):
            tree.nodes.pop(sub.node_id, None)

    # detach
    node.left_child = None
    node.right_child = None

    node.is_leaf = True
    node.feature = None
    node.threshold = None

    # recompute leaf model
    M_val, m0_val, h_val = gurobi_minimax(
        X[node.sample_indices],
        y[node.sample_indices]
    )

    node.M = M_val
    node.M_0 = m0_val
    node.h = h_val




def global_greedy_prune(tree, X, y, alpha, plot_tree_each_iteration=False, verbose=False):

    iteration = 0

    while True:

        iteration += 1

        leaves = tree.get_leaves()
        keep_cost = sum(l.h for l in leaves) + alpha * len(leaves)

        best_delta = 0.0
        best_node = None
        best_h = None

        candidates = get_prunable_parents(tree)

        if not candidates:
            break

        for node in candidates:

            left, right = node.left_child, node.right_child

            # compute h for parent ON DEMAND
            M_val, m0_val, h_parent = gurobi_minimax(
                X[node.sample_indices],
                y[node.sample_indices]
            )

            # cost after prune
            new_cost = (
                keep_cost
                - left.h
                - right.h
                + h_parent
                - alpha          # k -> k-1 || this is happning as i am subtracting alpha
            )

            delta = new_cost - keep_cost

            if delta < best_delta:
                best_delta = delta
                best_node = node
                best_h = (M_val, m0_val, h_parent)

        # no improving prune
        if best_node is None:
            break

        if verbose:
            print(
                f"[ITER {iteration}] pruning node {best_node.node_id} "
                f"delta={best_delta:.4f}"
            )

        # apply best prune
        collapse_node(tree, best_node, X, y)
        # Graphically visulazing
        if plot_tree_each_iteration:
            visualize_custom_tree(tree)




# -- Pruning with custom cost functions -- #
def get_custom_cost_function(alpha=0.01, cost_type="basic"):
    """
    Returns a cost function that computes:
        cost = average(h of leaves) + alpha * number of leaves
    """
    if cost_type == "basic":
        def cost(leaf_h_values, num_leaves, alpha=0.01):
            if num_leaves == 0:
                return 0.0
            return sum(leaf_h_values) + alpha * num_leaves
        
    elif cost_type == "average":
        def cost(leaf_h_values, num_leaves, alpha=0.01):
            if num_leaves == 0:
                return 0.0
            return sum(leaf_h_values) / num_leaves + alpha * num_leaves
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")
    
    return cost




def global_greedy_prune_with_custom_cost(tree, X, y, cost_fn, plot_tree_each_iteration=False, alpha=0.01, verbose=False):
    """
    Greedy bottom-up pruning using a user-provided cost function.
    The cost function must accept:
        - leaf_h_values: list of h values of current leaves
        - num_leaves: number of leaves
        Returns a numeric cost.
    """
    iteration = 0

    while True:
        iteration += 1

        # Current leaves and their h values
        leaves = tree.get_leaves()
        leaf_h_values = [l.h for l in leaves]
        num_leaves = len(leaves)
        keep_cost = cost_fn(leaf_h_values, num_leaves, alpha)

        best_delta = 0.0
        best_node = None
        best_h = None

        candidates = get_prunable_parents(tree)
        if not candidates:
            break

        if verbose:
            print(f"[ITER {iteration}] candidates for pruning: {len(candidates)}")

        for node in candidates:
            left, right = node.left_child, node.right_child

            # compute h for parent on demand
            M_val, m0_val, h_parent = gurobi_minimax(
                X[node.sample_indices],
                y[node.sample_indices]
            )

            # simulate pruning: remove child leaves from the calculation
            simulated_leaf_h = [h for h in leaf_h_values if h not in (left.h, right.h)] + [h_parent]
            simulated_num_leaves = num_leaves - 1  # pruning two leaves into one

            new_cost = cost_fn(simulated_leaf_h, simulated_num_leaves, alpha)

            delta = new_cost - keep_cost
            if verbose:
                print(f"  Node {node.node_id}: delta={delta:.4f}")

            if delta < best_delta:
                best_delta = delta
                best_node = node
                best_h = (M_val, m0_val, h_parent)

        if best_node is None:
            if verbose:
                print("No more improving prunes.")
            break

        if verbose:
            print(f"[ITER {iteration}] Pruning node {best_node.node_id} with delta={best_delta:.4f}")

        # Apply best prune
        collapse_node(tree, best_node, X, y)

        if plot_tree_each_iteration:
            visualize_custom_tree(tree)