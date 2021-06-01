import numpy as np


class Metrics(object):
    """
    implementation of various performance metrics
    """

    def __init__(self, truth, estimate):
        self.corr_truth = truth
        self.corr_est = estimate

    def PrecisionRecall(self):
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
