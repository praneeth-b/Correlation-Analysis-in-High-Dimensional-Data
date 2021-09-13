import numpy as np


class Metrics(object):
    """
    Implementation of Performance metrics to assess the estimated output correlation structure.

    """

    def __init__(self, truth, estimate):
        """
        Args:
            truth (matrix): The matrix containing the ground truth sturcture of the correlated features.  todo
            estimate (matrix): The estimated correlation structure matrix.  todo
        """

        self.corr_truth = truth
        self.corr_est = estimate

    def PrecisionRecall(self):
        """
        Computes the precision and recall metrics for the given ground truth and estimated correlation
         structure matrices
        Returns:
            Precision (float) : The precision value
            Recall (float): The recall value

        """
        true_positive = ((self.corr_truth + self.corr_est) == 2).sum()
        false_positives = ((self.corr_est - self.corr_truth) == 1).sum()
        false_negatives = ((self.corr_est - self.corr_truth) == -1).sum()
        true_negative = ((self.corr_truth + self.corr_est) == 0).sum()
        try:
            precision = true_positive / (true_positive + false_positives)
        except ZeroDivisionError:
            precision = 0

        try:
            recall = true_positive / (true_positive + false_negatives)
        except ZeroDivisionError:
            recall = 0

        return precision, recall
