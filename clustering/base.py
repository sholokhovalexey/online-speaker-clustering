import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRecognizer(object):
    """
    The base class for recognizers implementing comparisons between 
    an input object and the set of known classes.
    """

    def __init__(self):
        super().__init__()
        self.representations = {}

    def get_classes(self):
        return list(self.representations.keys())

    def verify(self, class_id, x):
        """
        Compute the similarity of a feature vector to a class.

        Args:
            class_id (str or int): class label.
            x (torch.Tensor): (1, dim) input feature vector.
        """
        raise NotImplementedError

    def verify_all(self, x):
        classes = list(self.representations.keys())
        scores = []
        for c in classes:
            s = self.verify(c, x)
            scores += [s.view(-1, 1)]
        scores = torch.cat(scores, dim=1)
        return scores

    def __call__(self, X):
        raise NotImplementedError


class RecognizerCentroids(BaseRecognizer):
    """
    Implements a recognizer with classes represented by centroids.

    Args:
        similarity_score (callable): a function computing the similarity between.
        a pair of feature vectors.
    """

    def __init__(self, similarity_score):
        super().__init__()
        self.similarity_score = similarity_score

    def verify(self, class_id, x):
        centroid, n = self.representations[class_id]
        score = self.similarity_score(centroid, x).view(-1)
        return score


class RecognizerMemory(BaseRecognizer):
    """
    Implements a recognizer with classes represented by sets of vectors.

    Args:
        similarity_score (callable): a function computing the similarity between 
        a pair of feature vectors.
        average_scores (bool): if True, average similarity scores across all objects.
        representing a class; otherwise, it is assumed that ``similarity_score`` can
        handle this within itself.
    """

    def __init__(self, similarity_score, average_scores=True):
        super().__init__()
        self.similarity_score = similarity_score
        self.average_scores = average_scores

    def verify(self, class_id, x):
        X = self.representations[class_id]
        if self.average_scores:
            score = torch.mean(self.similarity_score(X, x), dim=0).view(-1)
        else:
            score = self.similarity_score(X, x).view(-1)
        return score
