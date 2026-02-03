import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

from CustomTree.customTree import CustomDecisionTree, CustomTreeNode


def cvxpy_minimax(X_leaf, y_leaf):
    """
    Multi-output minimax regression using CVXPY.
    Returns M (m x d), m0 (m,), h (float)
    """
    print("Using CVXPY minimax solver...")
    n, d = X_leaf.shape
    if y_leaf.ndim == 1:
        y_leaf = y_leaf.reshape(-1, 1)
    m = y_leaf.shape[1]

    # Decision variables
    M = cp.Variable((m, d))
    m0 = cp.Variable(m)
    h = cp.Variable(nonneg=True)

    constraints = []
    for i in range(n):
        x_i = X_leaf[i, :]
        y_i = y_leaf[i, :]
        pred = M @ x_i + m0
        residual = pred - y_i
        constraints.append(cp.sum_squares(residual) <= h)

    # Objective: minimize worst-case squared error h
    problem = cp.Problem(cp.Minimize(h), constraints)
    problem.solve(solver=cp.SCS)  # or ECOS, OSQP, GUROBI if licensed

    return M.value, m0.value, h.value

# Updated Working version
def gurobi_minimax(X_leaf, y_leaf, verbose=False):
    """
    Minimax regression (multi-output) with squared L2 error.
    Minimize h s.t. for every sample i: sum_k (m0_k + M_k·x_i - y_{ik})^2 <= h
    Returns: M (m x d), m0 (m,), h (float)
    """
    if verbose:
        print("Using Gurobi minimax solver...")
    # Shapes
    n, d = X_leaf.shape
    if y_leaf.ndim == 1:
        y_leaf = y_leaf.reshape(-1, 1)
    m = y_leaf.shape[1]

    try:
        model = gp.Model("minimax_regression")
        model.Params.OutputFlag = 0  # silent

        # Decision variables
        M = model.addVars(m, d, lb=-GRB.INFINITY, name="M")
        m0 = model.addVars(m, lb=-GRB.INFINITY, name="m0")
        h  = model.addVar(lb=0.0, name="h")

        # Quadratic constraints: for each sample i, sum_k (affine)^2 <= h
        for i in range(n):
            quad_terms = []
            for k in range(m):
                # expr = m0[k] + sum_j M[k,j] * X[i,j] - y[i,k]
                expr = m0[k]
                for j in range(d):
                    expr += M[k, j] * float(X_leaf[i, j])  # X is data (constant)
                expr -= float(y_leaf[i, k])
                quad_terms.append(expr * expr)  # (affine)^2 -> QuadExpr

            # Move h to LHS so all quadratic stays on LHS
            model.addQConstr(gp.quicksum(quad_terms) - h <= 0.0)

        # Objective: minimize h
        model.setObjective(h, GRB.MINIMIZE)
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or model.SolCount > 0:
            M_val  = np.array([[M[k, j].X for j in range(d)] for k in range(m)])
            m0_val = np.array([m0[k].X for k in range(m)])
            h_val  = float(h.X)
            return M_val, m0_val, h_val
        else:
            return None, None, np.inf

    except gp.GurobiError as e:
        # License-size fallback or other Gurobi errors
        if "Model too large for size-limited license" in str(e):
            # Define this in your codebase
            print("Gurobi encountered an error:", str(e))
            print("Gurobi encountered an error. Falling back to CVXPY solver.")
            return cvxpy_minimax(X_leaf, y_leaf)
        raise
    



#  Following functions are added to make predictions using the minimax models stored in the leaves of a custom tree.
# def get_samples_per_leaf(tree: CustomDecisionTree, X):
#     """
#     Traverse the custom tree for each sample in X and return:
#         1. sample_to_leaf: array of leaf IDs corresponding to each sample
#         2. leaf_to_samples: dict mapping leaf_id -> list of sample indices
#     """
#     n_samples = X.shape[0]
#     sample_to_leaf = np.zeros(n_samples, dtype=int)
#     leaf_to_samples = {}

#     for idx in range(n_samples):
#         node = tree.root
#         while not node.is_leaf:
#             feature = node.feature
#             threshold = node.threshold
#             if X[idx, feature] <= threshold:
#                 node = node.left_child
#             else:
#                 node = node.right_child

#         leaf_id = node.node_id
#         sample_to_leaf[idx] = leaf_id
#         leaf_to_samples.setdefault(leaf_id, []).append(idx)

#     return sample_to_leaf, leaf_to_samples


# def predict_with_minimax_model(custom_tree: CustomDecisionTree, X):
#     """
#     Predict using the minimax models stored in the leaves of a custom tree.
#     """
#     n_samples = X.shape[0]
#     # n_targets = custom_tree.get_leaves()[0].M.shape[0]  # Assuming all leaves have same output dim
#     # infer number of outputs
#     for leaf in custom_tree.get_leaves():
#         if leaf.M is not None:
#             n_targets = leaf.M.shape[0]
#             break
#     else:
#         raise ValueError("No leaf contains minimax model parameters.")

#     # y_pred = np.zeros((n_samples, n_targets))
#     y_pred = np.zeros((n_samples, n_targets))

#     # Get leaf assignment for each sample
#     sample_to_leaf, leaf_to_samples = get_samples_per_leaf(custom_tree, X)

#     for leaf in custom_tree.get_leaves():
#         indices = leaf_to_samples.get(leaf.node_id, [])
#         if len(indices) == 0:
#             continue

#         if leaf.M is None or leaf.M_0 is None:
#             print(f"Warning: Leaf {leaf.node_id} has no minimax model parameters.")
#             # raise RuntimeError(
#             #     f"Leaf {leaf.node_id} has no minimax model parameters."
#             # )
#             continue

#         X_leaf = X[indices, :]
#         y_pred_leaf = apply_minimax_model(X_leaf, leaf.M, leaf.M_0)
#         print(f"Shape of predictions for leaf {leaf.node_id}: {y_pred_leaf.shape}")
#         y_pred[indices, :] = y_pred_leaf

#     return y_pred


# def apply_minimax_model(X_leaf, M, m0):
#     """
#     Apply the minimax regression model to data in a leaf.
#     X_leaf: (n_leaf_samples, d)
#     M: (m, d)
#     m0: (m,)
#     Returns: predictions (n_leaf_samples, m)
#     """
#     print("Applying minimax model...")
#     print(f"X_leaf shape: {X_leaf.shape}, M shape: {M.shape}, m0 shape: {m0.shape}")
#     return X_leaf @ M.T + m0  # Broadcasting m0 over samples


def get_samples_per_leaf(tree: CustomDecisionTree, X):
    """
    Traverse the tree and assign each sample to a leaf.
    Returns:
        sample_to_leaf: ndarray (n_samples,)
        leaf_to_samples: dict leaf_id -> list[int]
    """
    n_samples = X.shape[0]

    sample_to_leaf = np.empty(n_samples, dtype=int)
    leaf_to_samples = {}

    for i in range(n_samples):
        node = tree.root
        while not node.is_leaf:
            if X[i, node.feature] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child

        leaf_id = node.node_id
        sample_to_leaf[i] = leaf_id
        leaf_to_samples.setdefault(leaf_id, []).append(i)

    return sample_to_leaf, leaf_to_samples


def predict_with_minmax_model(custom_tree: CustomDecisionTree, X):
    """
    Predict outputs for X using minimax models stored in leaves.
    """
    if custom_tree.root is None:
        raise ValueError("Tree has no root.")

    n_samples = X.shape[0]

    # infer output dimension
    for leaf in custom_tree.get_leaves():
        if leaf.M is not None:
            n_targets = leaf.M.shape[0]
            break
    else:
        raise ValueError("No leaf contains minimax model parameters.")

    y_pred = np.zeros((n_samples, n_targets))

    # route samples once
    _, leaf_to_samples = get_samples_per_leaf(custom_tree, X)

    for leaf in custom_tree.get_leaves():

        indices = leaf_to_samples.get(leaf.node_id)
        if not indices:
            continue

        if leaf.M is None or leaf.M_0 is None:
            raise RuntimeError(
                f"Leaf {leaf.node_id} has no minimax model parameters."
            )

        X_leaf = X[indices]
        y_pred_leaf = apply_minimax_model(X_leaf, leaf.M, leaf.M_0)

        assert y_pred_leaf.shape == (len(indices), n_targets)

        y_pred[indices] = y_pred_leaf

    return y_pred


def apply_minimax_model(X_leaf, M, m0):
    """
    X_leaf: (n, d)
    M: (m, d)
    m0: (m,)
    """
    return X_leaf @ M.T + m0
