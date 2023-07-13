import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List

from src.utils.preprocessing_utils import create_dataframe_of_predicted_labels


def get_reconstruction_error(model, data):
    return model.predict(data) - data


def compute_mahalanobis_params(reconstruction_error, normal_data):
    """Compute the covariance matrix and the mean that are needed to compute the mahalanobis distance.
    This is done only for the normal data (norm_mahal_x_seq)"""
    cov = np.cov(reconstruction_error.reshape(-1, normal_data.shape[-1]).T)
    mean = np.mean(reconstruction_error.reshape(-1, normal_data.shape[-1]), axis=0)
    mah_params = {"mean": mean, "cov": cov}
    return mah_params


def compute_mahalanobis_distance(reconstruction_error, time_steps, mah_params):
    """Compute mahalanobis distance."""
    # Reshape the reconstruction error from 3D (#sequences, sequence_length, #features)
    # to 2D (#sequences * sequence_length, #features)
    reshaped_error = reconstruction_error.reshape(-1, reconstruction_error.shape[-1])  # shape: (seq*timesteps, n_feat)
    return np.mean(
        np.array([distance.mahalanobis(mah_params['mean'], reshaped_error[i], mah_params['cov']) for i in
                  range(len(reshaped_error))]).reshape(-1, time_steps), axis=1)


# def compute_mahalanobis_(reconstruction_error, mean, cov, time_steps):
#     """Compute Mahalanobis distance. GPT"""
#
#     # Reshape the reconstruction error from 3D (#sequences, sequence_length, #features) to 2D (#sequences * sequence_length, #features)
#     reshaped_error = reconstruction_error.reshape(-1, reconstruction_error.shape[-1])
#
#     # Compute the Mahalanobis distance
#     inv_cov = np.linalg.inv(cov)
#     mahalanobis_dist = []
#     for i in range(0, reshaped_error.shape[0], time_steps):
#         error_slice = reshaped_error[i:i + time_steps]
#         diff = error_slice - mean
#         dist = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))
#         mahalanobis_dist.extend(dist)
#
#     return np.array(mahalanobis_dist)


# # Test the compute_mahalanobis function
# reconstruction_error = np.random.randn(100, 4, 10)  # Example reconstruction error array
# mean = np.random.randn(10)  # Example mean array
# cov = np.eye(10)  # Example covariance matrix
# time_steps = 4  # Example number of time steps
#
# distances = compute_mahalanobis(reconstruction_error, mean, cov, time_steps)
# print(distances)

def evaluate_fbeta(threshold: float, normal_scores: List[float], anomaly_scores: List[float],
                   beta: float = 1.0) -> float:
    """
    Evaluates the F-beta score for a given threshold in anomaly detection.

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


def compute_threshold(normal_scores: List[float], anomaly_scores: List[float], num_thresholds: int = 20) -> float:
    """
    Computes the optimal threshold for anomaly detection using the given normal and anomaly scores.

    Args:
        normal_scores: List of scores for normal instances.
        anomaly_scores: List of scores for anomaly instances.
        num_thresholds: Number of thresholds to divide the range between the medians. Default is 20.

    Returns:
        The optimal threshold value for anomaly detection.

    Raises:
        None.
    """

    lower = np.median(normal_scores)
    upper = np.median(anomaly_scores)
    delta = (upper - lower) / num_thresholds

    thresholds = np.arange(lower, upper, delta)
    fbeta_scores = [evaluate_fbeta(threshold, normal_scores, anomaly_scores) for threshold in thresholds]

    max_index = np.argmax(fbeta_scores)
    threshold = thresholds[max_index]

    return threshold


def anomaly_scoring(reconstruction_error, time_steps, mah_params):
    """Compute anomaly scores, find anomalies and return the anomalous data indices and the threshold."""
    # rec error same shape with data_seq i.e., (n_seq,timesteps,n_feat)
    anomaly_scores = compute_mahalanobis_distance(reconstruction_error, time_steps, mah_params)  # shape: (n_seq,)
    return anomaly_scores


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

    return accuracy, precision, recall, f1


def get_eval_metrics(reconstruction_error, data_x, data_y, threshold, time_steps, mah_params):
    """Evaluate, i.e., compute the reconstruction error and compute mahalanobis to get the anomaly scores, filter them
     with the threshold and return the anomalous indices."""
    anomalous_data_indices = get_anomalies(reconstruction_error, data_x, threshold, time_steps, mah_params)
    data_y_unseq = np.concatenate([data_y[:-1, 0], data_y[-1, :]])
    data_y_unseq = pd.DataFrame(data_y_unseq.reshape(-1, 1))
    y_pred = create_dataframe_of_predicted_labels(data_y_unseq, anomalous_data_indices)
    accuracy, precision, recall, f1 = pred_eval(data_y_unseq[time_steps: -time_steps], y_pred[time_steps: -time_steps])
    return accuracy, precision, recall, f1
