# CustomTree/pruning_scenario.py

import numpy as np
from CustomTree.quadratic_constraint_optimization import gurobi_minimax

try:
    # optional: only needed if you want plotting during pruning
    from Visualization.graphicallyShowingTree import visualize_custom_tree
except Exception:
    visualize_custom_tree = None


# ------------------------------------------------------------
# Scenario epsilon
# ------------------------------------------------------------
def scenario_epsilon(num_samples: int, d_eff: int, beta: float = 1e-6) -> float:
    """
    Conservative explicit scenario bound (common form):
        eps = (2/N) * (d_eff - 1 + log(1/beta))
    If N <= d_eff - 1 => bound is vacuous => eps = 1.

    Returns eps clipped to [0, 1].
    """
    N = int(num_samples)
    if N <= max(d_eff - 1, 0):
        return 1.0

    eps = (2.0 / N) * (d_eff - 1 + np.log(1.0 / beta))
    return float(np.clip(eps, 0.0, 1.0))


# ------------------------------------------------------------
# Utilities: candidates + metric computation
# ------------------------------------------------------------
def get_prunable_parents(tree):
    """
    Return internal nodes whose two children are leaves (a "cherry").
    This matches your current pruning move-set.
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


def compute_leaf_metrics(node, X, y, d_eff: int, beta: float = 1e-6):
    """
    Ensure node has (M, M_0, h) from gurobi_minimax and epsilon from scenario bound.
    Assumes node.sample_indices exists.
    """
    idx = node.sample_indices
    if idx is None or len(idx) == 0:
        # Degenerate leaf: no samples => no guarantee
        node.h = float("inf")
        node.epsilon = 1.0
        node.M = None
        node.M_0 = None
        return node

    M_val, m0_val, h_val = gurobi_minimax(X[idx], y[idx])
    node.M = M_val
    node.M_0 = m0_val
    node.h = float(h_val)
    node.epsilon = scenario_epsilon(len(idx), d_eff, beta)
    return node


def ensure_all_leaves_have_metrics(tree, X, y, d_eff: int, beta: float = 1e-6, verbose: bool = False):
    """
    If any leaf is missing h or epsilon, compute them on demand.
    """
    leaves = tree.get_leaves()
    for leaf in leaves:
        missing_h = not hasattr(leaf, "h") or leaf.h is None
        missing_eps = not hasattr(leaf, "epsilon") or leaf.epsilon is None
        if missing_h or missing_eps:
            if verbose:
                print(f"[METRICS] computing metrics for leaf {leaf.node_id} (missing_h={missing_h}, missing_eps={missing_eps})")
            compute_leaf_metrics(leaf, X, y, d_eff=d_eff, beta=beta)


# ------------------------------------------------------------
# Collapse (prune) operation
# ------------------------------------------------------------
def collapse_node(tree, node, X, y, d_eff: int, beta: float = 1e-6, precomputed=None):
    """
    Replace subtree rooted at `node` by a leaf.
    If `precomputed` is provided, it should be (M_val, m0_val, h_val, eps_val) for the parent.
    """
    # remove descendants
    for child in [node.left_child, node.right_child]:
        for sub in tree.get_subtree_nodes(child):
            tree.nodes.pop(sub.node_id, None)

    # detach & convert to leaf
    node.left_child = None
    node.right_child = None
    node.is_leaf = True
    node.feature = None
    node.threshold = None

    # compute or reuse leaf model and epsilon
    if precomputed is None:
        idx = node.sample_indices
        M_val, m0_val, h_val = gurobi_minimax(X[idx], y[idx])
        eps_val = scenario_epsilon(len(idx), d_eff, beta)
    else:
        M_val, m0_val, h_val, eps_val = precomputed

    node.M = M_val
    node.M_0 = m0_val
    node.h = float(h_val)
    node.epsilon = float(eps_val)


# ------------------------------------------------------------
# Cost function you requested
# cost = mean(h) + alpha * max(epsilon)
# ------------------------------------------------------------
def scenario_cost_from_leaves(leaves, alpha: float):
    hs = [float(l.h) for l in leaves]
    eps = [float(getattr(l, "epsilon", 1.0)) for l in leaves]
    if len(hs) == 0:
        return 0.0
    return (sum(hs) / len(hs)) + float(alpha) * max(eps)


def count_bad_eps(leaves, tol: float = 1e-12) -> int:
    eps = [float(getattr(l, "epsilon", 1.0)) for l in leaves]
    return sum(1 for e in eps if e >= 1.0 - tol)




def compute_d_eff_from_data(X, y, include_h=True):
    # d_eff should be number of decision variables in your minimax problem, e.g.:
    # d_eff = n_y * n_x + n_y + 1 if include_h else 0
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 1:
        n_x = 1
    else:
        n_x = X.shape[1]

    if y.ndim == 1:
        n_y = 1
    else:
        n_y = y.shape[1]

    d_eff = n_y * n_x + n_y + (1 if include_h else 0)
    return int(d_eff)

# ------------------------------------------------------------
# Main pruning routine:
# - ensures missing h/epsilon are computed
# - uses your new scenario cost
# - if any leaf has epsilon==1, it enters "repair mode" and prunes to reduce bad leaves
# ------------------------------------------------------------
def global_greedy_prune_with_scenario_cost(
    tree,
    X,
    y,
    alpha: float,
    # d_eff: int,
    beta: float = 1e-6,
    plot_tree_each_iteration: bool = False,
    verbose: bool = False,
):
    """
    Global greedy prune over prunable parents (cherries).
    Objective:
        mean(h) + alpha * max(epsilon)

    Plus a hard behavior:
        If any leaf has epsilon==1 (vacuous), prioritize prunes that reduce the
        number of such leaves ("repair mode").

    Notes:
    - Requires node.sample_indices to be correct at every node (union of subtree samples).
    - If plot_tree_each_iteration=True, requires visualize_custom_tree to be importable.
    """

    if plot_tree_each_iteration and visualize_custom_tree is None:
        raise RuntimeError("plot_tree_each_iteration=True but visualize_custom_tree could not be imported.")

    d_eff = compute_d_eff_from_data(X, y, include_h=True)

    iteration = 0

    while True:
        iteration += 1

        # Ensure all leaves have h/epsilon
        ensure_all_leaves_have_metrics(tree, X, y, d_eff=d_eff, beta=beta, verbose=verbose)

        leaves = tree.get_leaves()
        keep_cost = scenario_cost_from_leaves(leaves, alpha)
        bad_before = count_bad_eps(leaves)

        candidates = get_prunable_parents(tree)
        if not candidates:
            if verbose:
                print(f"[ITER {iteration}] no candidates, stopping.")
            break

        if verbose:
            print(f"[ITER {iteration}] leaves={len(leaves)} cost={keep_cost:.6f} bad_eps={bad_before} candidates={len(candidates)}")

        best_node = None
        best_delta = 0.0
        best_bad_reduction = 0
        best_parent_fit = None  # (M_val, m0_val, h_parent, eps_parent)

        # Build a quick lookup for current leaves by id (for safe simulation)
        leaf_by_id = {l.node_id: l for l in leaves}

        for node in candidates:
            left, right = node.left_child, node.right_child
            if left is None or right is None or (not left.is_leaf) or (not right.is_leaf):
                continue

            # ensure children have metrics (should already, but safe)
            if (not hasattr(left, "h")) or left.h is None or (not hasattr(left, "epsilon")) or left.epsilon is None:
                compute_leaf_metrics(left, X, y, d_eff=d_eff, beta=beta)
            if (not hasattr(right, "h")) or right.h is None or (not hasattr(right, "epsilon")) or right.epsilon is None:
                compute_leaf_metrics(right, X, y, d_eff=d_eff, beta=beta)

            # compute parent fit on demand
            idx = node.sample_indices
            M_val, m0_val, h_parent = gurobi_minimax(X[idx], y[idx])
            eps_parent = scenario_epsilon(len(idx), d_eff, beta)

            # simulate leaves list after pruning this node
            simulated = []
            for lid, lf in leaf_by_id.items():
                if lid in (left.node_id, right.node_id):
                    continue
                simulated.append(lf)

            # lightweight simulated parent leaf
            class _TmpLeaf:
                pass

            tmp = _TmpLeaf()
            tmp.node_id = node.node_id
            tmp.h = float(h_parent)
            tmp.epsilon = float(eps_parent)
            simulated.append(tmp)

            new_cost = scenario_cost_from_leaves(simulated, alpha)
            delta = new_cost - keep_cost

            bad_after = count_bad_eps(simulated)
            bad_reduction = bad_before - bad_after

            if verbose:
                print(f"  Node {node.node_id}: delta={delta:.6f} bad_reduction={bad_reduction}")

            # Selection rule:
            if bad_before > 0:
                # repair mode: maximize bad reduction, tie-break by delta
                if (bad_reduction > best_bad_reduction) or (
                    bad_reduction == best_bad_reduction and delta < best_delta
                ):
                    best_bad_reduction = bad_reduction
                    best_delta = delta
                    best_node = node
                    best_parent_fit = (M_val, m0_val, float(h_parent), float(eps_parent))
            else:
                # normal mode: best improvement in cost
                if delta < best_delta:
                    best_delta = delta
                    best_node = node
                    best_parent_fit = (M_val, m0_val, float(h_parent), float(eps_parent))

        # stopping conditions
        if best_node is None:
            if verbose:
                print(f"[ITER {iteration}] no improving prune found, stopping.")
            break

        if bad_before > 0 and best_bad_reduction <= 0:
            if verbose:
                print(f"[ITER {iteration}] bad_eps remain but cannot reduce with cherry-pruning, stopping.")
            break

        if verbose:
            print(f"[ITER {iteration}] PRUNE node={best_node.node_id} delta={best_delta:.6f} bad_reduction={best_bad_reduction}")

        # Apply prune; reuse best_parent_fit to avoid re-solving minimax
        collapse_node(tree, best_node, X, y, d_eff=d_eff, beta=beta, precomputed=best_parent_fit)

        if plot_tree_each_iteration:
            visualize_custom_tree(tree)

    return tree
