import torch

from .training import train_plda_sph_em
from .scoring import get_centroid, score_plda_sph_centroids


class SphPLDA:
    """
    Spherical Probabilistic Linear Discriminant Analysis

    The model is designed for centered and length-normalized embeddings.
    """

    def __init__(self, dim=None):
        self.dim = dim
        self.b = torch.tensor([0.01])
        self.w = torch.tensor([0.01])
        self.is_fitted = False

    def __repr__(self):
        return f"PLDA(b={self.b.item()}, w={self.w.item()})"

    def init(self, b, w, dim=None):
        """
        Set values for the model parameters.

        Args:
            b (float): between-class variance
            w (float): within-class variance
            dim (int, optional): embeddings dimension. Defaults to None.
        """
        self.dim = dim
        self.b = torch.tensor([b]).float()
        self.w = torch.tensor([w]).float()
        self.is_fitted = True

    def save(self, fname):
        torch.save({"b": self.b.item(), "w": self.w.item()}, fname)

    @classmethod
    def load(cls, fname):
        state_dict = torch.load(fname)
        model = cls()
        model.init(**state_dict)
        return model

    def fit(self, X, y, n_iter=10):
        """
        Train a sph-PLDA model from labeled embeddings.

        Args:
            X (torch.Tensor): (N, d) observation matrix of N d-dimensional embedding vectors.
            y (torch.Tensor): (N,) a vector of integer class labels.
            n_iter (int, optional): number of iterations of the EM algorithm. Defaults to 10.
        """
        b, w = train_plda_sph_em(X, y, n_iter=n_iter)
        self.init(b, w, X.shape[1])

    def score(self, X1, X2):
        """
        Compute a single similarity score (LLR) between two sets of embeddings.

        Args:
            X1 (torch.Tensor): (N, d) N enrollment embedding vectors.
            X2 (torch.Tensor): (M, d) M test embedding vectors.

        Returns:
            torch.Tensor: (1,) set-to-set similarity score.
        """
        enroll = get_centroid(X1)
        test = get_centroid(X2)
        return self.score_vector(enroll, test).view(-1)

    def score_vector(self, enroll, test):
        """
        Compute a vector of N scores corresponding to N trials. 
        The inputs contain trials sides and are represented as either a pair of tensors (centroids, counts) 
        or as a matrix of shape (N, d). In the former case, 'centroid' is the average vector 
        of a set consisting of 'count' embedding vectors. That is, each input can be a tuple of
        tensors with the following shapes: ((N, d), (N,)). In the latter case, rows of the matrix 
        are interpreted as centroids, assuming count=1 for each trial.

        Args:
            enroll (tuple or torch.Tensor): enrollment data, N trial sides.
            test (tuple or torch.Tensor): test data, N trial sides.

        Returns:
            torch.Tensor: (N, 1) a vector of similarity scores (LLRs).
        """
        assert self.is_fitted, "Model is not fitted"
        return score_plda_sph_centroids(enroll, test, self.b, self.w, all_pairs=False)

    def score_matrix(self, enroll, test):
        """
        Compute a matrix of scores for all possible pairs between N enrollment and M test sets.
        Each input is represented as either a pair of tensors (centroids, counts) or as a matrix 
        with rows interpreted as centroids (count=1).

        Args:
            enroll (tuple or torch.Tensor): enrollment data, N trial sides.
            test (tuple or torch.Tensor): test data, M trial sides.

        Returns:
            torch.Tensor: (N, M) a matrix of similarity scores (LLRs).
        """
        assert self.is_fitted, "Model is not fitted"
        return score_plda_sph_centroids(enroll, test, self.b, self.w, all_pairs=True)

