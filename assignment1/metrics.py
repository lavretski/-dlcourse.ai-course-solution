from collections import Counter

import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = np.sum(prediction & ground_truth) / np.sum(prediction)
    recall = np.sum(prediction & ground_truth) / np.sum(ground_truth)
    accuracy = np.sum(prediction == ground_truth) / len(prediction)
    f1 = 2 / (1/precision + 1/recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return np.sum(prediction == ground_truth) / len(prediction)