import logging

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List

from src.utils.preprocessing_utils import create_dataframe_of_predicted_labels


def get_reconstruction_error(model, data):
    return model.predict(data) - data


def compute_mahalanobis_params(reconstruction_error):
    """
    Computes the Mahalanobis parameters (mean and covariance matrix) from the reconstruction error.

    Args:
        reconstruction_error (np.ndarray): The reconstruction error from the 'v_N1' split.
            Shape: (num_instances, subsequence_length, num_features).

    Returns:
        dict: A dictionary containing the Mahalanobis parameters 'mean' and 'cov'.

    """
    # The purpose of reshaping is to align the data in a format that the np.cov function can process. It treats
    # each row as an observation and each column as a variable when calculating the covariance matrix. Reshaping the
    # reconstruction error in this way ensures that each instance's subsequence of feature values is treated as a
    # separate observation, allowing us to calculate the covariance matrix correctly.

    reshaped_error = reconstruction_error.reshape(-1, reconstruction_error.shape[-1])
    mean = np.mean(reshaped_error, axis=0)
    cov = np.cov(reshaped_error, rowvar=False)

    mahalanobis_params = {'mean': mean, 'cov': cov}
    return mahalanobis_params


def compute_mahalanobis_distance_inv(reconstruction_error, mah_params):
    """
    Compute the Mahalanobis distance for each instance in the reconstruction error.

    Args:
        reconstruction_error (np.ndarray): The reconstruction error.
            Shape: (num_instances, subsequence_length, num_features).
        mah_params (dict): A dictionary containing the Mahalanobis parameters.
            The dictionary should have keys 'mean' and 'cov'.

    Returns:
        np.ndarray: The Mahalanobis distances for each instance.
            Shape: (num_instances,).

    """
    reshaped_error = reconstruction_error.reshape(-1, reconstruction_error.shape[-1])
    mahalanobis_dist = np.array([
        distance.mahalanobis(error, mah_params['mean'], np.linalg.inv(mah_params['cov'])) for error in reshaped_error
    ])
    mahalanobis_dist = mahalanobis_dist.reshape(reconstruction_error.shape[:-1])
    mean_mahalanobis_dist = np.mean(mahalanobis_dist, axis=1)
    return mean_mahalanobis_dist


def anomaly_scoring(reconstruction_error, time_steps, mah_params):
    """Compute anomaly scores, find anomalies and return the anomalous data indices and the threshold."""
    # rec error same shape with data_seq i.e., (n_seq,timesteps,n_feat)
    anomaly_scores_inv = compute_mahalanobis_distance_inv(reconstruction_error, mah_params)
    return anomaly_scores_inv


def evaluate_fbeta(threshold: float, normal_scores: List[float], anomaly_scores: List[float],
                   beta: float = 1.0) -> float:
    """
    Evaluates the F-beta score for a given threshold in anomaly detection. When beta is set to 1, it calculates the F1
    score, which balances precision and recall equally.

    Args:
        threshold: The threshold value for classification.
        normal_scores: List of scores for normal instances.
        anomaly_scores: List of scores for anomaly instances.
        beta: Beta value for controlling the trade-off between precision and recall. Default is 1.0 (F1 score).

    Returns:
        The F-beta score.

    Raises:
        None.

    """
    true_positives = sum(score >= threshold for score in anomaly_scores)
    false_positives = sum(score >= threshold for score in normal_scores)
    false_negatives = sum(score < threshold for score in anomaly_scores)

    if true_positives + false_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * ((precision * recall) / ((beta_squared * precision) + recall))

    return fbeta


def compute_threshold(normal_scores: List[float], anomaly_scores: List[float], num_thresholds: int = 20,
                      beta: float = 1.0) -> float:
    """
    Computes the optimal threshold for anomaly detection using the given normal and anomaly scores.

    Args:
        normal_scores: List of scores for normal instances.
        anomaly_scores: List of scores for anomaly instances.
        num_thresholds: Number of thresholds to divide the range between the medians. Default is 20.
        beta: Beta value for controlling the trade-off between precision and recall. Default is 1.0 (F1 score).

    Returns:
        The optimal threshold value for anomaly detection.

    Raises:
        None.
    """
    lower = np.median(normal_scores)
    upper = np.median(anomaly_scores)
    delta = (upper - lower) / num_thresholds

    if lower >= upper:
        raise ValueError("Invalid input: lower median should be less than upper median.")

    # Calculate delta, ensuring it's not too small
    MIN_DELTA = 1e-6
    delta = max((upper - lower) / num_thresholds, MIN_DELTA)

    # Debugging information
    logging.debug("Lower:", lower)
    logging.debug("Upper:", upper)
    logging.debug("Delta:", delta)

    thresholds = np.arange(lower, upper, delta)
    fbeta_scores = [evaluate_fbeta(threshold, normal_scores, anomaly_scores, beta) for threshold in thresholds]

    max_index = np.argmax(fbeta_scores)
    threshold = thresholds[max_index]

    return threshold


def get_anomalies(reconstruction_error, data_seq, threshold, time_steps, mah_params):
    """Create a list of the anomalous data indices."""
    anomaly_scores = anomaly_scoring(reconstruction_error, time_steps, mah_params)  # shape: (n_seq,)
    # Detect all the samples which are anomalies.
    anomalies = anomaly_scores > threshold  # boolean, shape (n_seq,)
    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    # for data_idx in range(time_steps - 1, len(data_seq) - time_steps + 1):
    for data_idx in range(time_steps - 1, len(data_seq)):
        if np.all(anomalies[data_idx - time_steps + 1: data_idx + 1]):
            anomalous_data_indices.append(data_idx)
    return anomalous_data_indices


def pred_eval(y_true, y_pred):
    """Calculate accuracy, precision, recall and f1 scores to evaluate the model."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    msg = "Accuracy %.2f, Precision %.2f, Recall %.2f, F1 %.2f" % (accuracy, precision, recall, f1)
    print(msg)

    return accuracy, precision, recall, f1, cm


def get_eval_metrics(reconstruction_error, data_x, data_y, threshold, time_steps, mah_params):
    """Evaluate, i.e., compute the reconstruction error and compute mahalanobis to get the anomaly scores, filter them
     with the threshold and return the anomalous indices."""
    anomalous_data_indices = get_anomalies(reconstruction_error, data_x, threshold, time_steps, mah_params)
    # Sliding windows, take the first column element from each window until first from the end
    # Take the whole last window since each element is not included in other windows
    data_y_unseq = np.concatenate([data_y[:-1, 0], data_y[-1, :]])
    data_y_unseq = pd.DataFrame(data_y_unseq.reshape(-1, 1))
    y_pred = create_dataframe_of_predicted_labels(data_y_unseq, anomalous_data_indices)
    accuracy, precision, recall, f1, cm = pred_eval(data_y_unseq[time_steps: -time_steps], y_pred[time_steps: -time_steps])
    return accuracy, precision, recall, f1, cm
