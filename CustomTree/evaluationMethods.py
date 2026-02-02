import numpy as np

def normalized_root_mean_square_error(y_true, y_pred):
    """
    Computes the Normalized Root Mean Square Error (NRMSE) between y_true and y_pred.
    If the range of y_true is zero, it normalizes by the number of samples * outputs.

    Parameters:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_outputs)
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_outputs)

    Returns:
        float: NRMSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Compute range
    y_range = np.max(y_true) - np.min(y_true)
    
    if y_range != 0:
        # Normalize by range
        return rmse / y_range
    else:
        # Normalize by n_samples * n_outputs
        n_samples, n_outputs = y_true.shape
        return np.sqrt(np.sum((y_true - y_pred) ** 2) / (n_samples * n_outputs))
    
def mean_absolute_error(y_true, y_pred):
    """
    Computes the Mean Absolute Error (MAE) between y_true and y_pred.

    Parameters:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_outputs)
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_outputs)
    Returns:
        float: MAE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

