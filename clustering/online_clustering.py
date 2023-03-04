import torch

from models.gaussian import OnlineRecognizerGaussian
from models.vonmises import OnlineRecognizerMisesFisher

from .base import RecognizerCentroids, RecognizerMemory


def name_generator(dtype=str):
    i = 0
    while True:
        i += 1
        yield dtype(i)


class OnlineClusteringCentroids(RecognizerCentroids):
    """
    Performs online clustering given a sequence of feature vectors.
    Clusters are represented by centroids.

    Args:
        similarity_score (callable): a function computing the similarity between 
        a pair of feature vectors.
        threshold (float): similarity threshold to detect a known class.
        threshold_update (float): similarity threshold to update a class.
    """

    def __init__(self, similarity_score, threshold, threshold_update=None, **params):
        super().__init__(similarity_score)
        self.threshold = threshold
        self.threshold_update = (
            threshold if threshold_update is None else threshold_update
        )
        self.name_generator = name_generator()
        self.params = params

    def add_class(self, X):
        class_id = next(self.name_generator)
        centroid = torch.mean(X, dim=0, keepdim=True)
        n = X.shape[0]
        self.representations[class_id] = (centroid, n)
        return class_id

    def update_class(self, class_id, x):
        centroid, n = self.representations[class_id]
        alpha = 1 / (n + 1)  # simple average
        centroid = (1 - alpha) * centroid + alpha * x
        self.representations[class_id] = (centroid, n + 1)

    def verify_all(self, x):
        classes = list(self.representations.keys())
        centroids = []
        for c in classes:
            centroid, n = self.representations[c]
            centroids += [centroid]
        centroids = torch.cat(centroids)
        scores = self.similarity_score(centroids, x)
        return scores

    def __call__(self, X_stream):
        """
        Online clustering of feature vectors.

        Args:
            X_stream (torch.Tensor): input sequence of feature vectors.

        Returns:
            list: predicted class labels.
        """

        if X_stream.dim() == 1:
            X_stream = X_stream.unsqueeze(0)

        class_id = self.add_class(X_stream[0:1])
        classes = list(self.representations.keys())

        predictions = [class_id]

        for i in range(1, len(X_stream)):
            x = X_stream[i : i + 1]
            scores = self.verify_all(x).view(-1)
            idx_max = torch.argmax(scores).item()
            s_max = scores[idx_max]
            classes = list(self.representations.keys())
            class_id = classes[idx_max]

            if s_max > self.threshold:  # known class
                predictions += [class_id]
                # update cluster
                if s_max > self.threshold_update:
                    self.update_class(class_id, x)

            else:  # unknown class
                # create a new cluster
                class_id = self.add_class(x)
                predictions += [class_id]

        return predictions


class OnlineClusteringCentroidsV2(OnlineClusteringCentroids):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def verify_all(self, x):
        classes = list(self.representations.keys())
        scores = []
        for c in classes:
            centroid, n = self.representations[c]
            score = self.similarity_score((centroid, n), x)
            scores += [score]
        scores = torch.cat(scores)
        return scores


class OnlineClusteringMemory(RecognizerMemory):
    """
    Performs online clustering given a sequence of feature vectors.
    Clusters are represented by sets of feature vectors.

    Args:
        similarity_score (callable): a function computing the similarity between 
        a pair of feature vectors.
        threshold (float): similarity threshold to detect a known class.
        threshold_update (float): similarity threshold to update a class.
        average_scores (bool): if True, average similarity scores across all objects
        representing a class; otherwise, it is assumed that ``similarity_score`` can
        handle this within itself.
    """

    def __init__(
        self,
        similarity_score,
        threshold,
        threshold_update=None,
        average_scores=True,
        **params
    ):
        super().__init__(similarity_score, average_scores)
        self.threshold = threshold
        self.threshold_update = (
            threshold if threshold_update is None else threshold_update
        )
        self.name_generator = name_generator(int)
        self.params = params

    def add_class(self, X):
        class_id = next(self.name_generator)
        self.representations[class_id] = X
        return class_id

    def update_class(self, class_id, x):
        X_class = self.representations[class_id]
        self.representations[class_id] = torch.cat([X_class, x])

    def verify_all(self, x):
        classes = list(self.representations.keys())
        scores = []
        for c in classes:
            scores += [self.verify(c, x)]
        scores = torch.cat(scores)
        assert len(classes) == scores.numel()
        return scores

    def __call__(self, X_stream):
        """
        Online clustering of feature vectors.

        Args:
            X_stream (torch.Tensor): input sequence of feature vectors.

        Returns:
            list: predicted class labels.
        """

        if X_stream.dim() == 1:
            X_stream = X_stream.unsqueeze(0)

        class_id = self.add_class(X_stream[0:1])
        classes = list(self.representations.keys())

        predictions = [class_id]

        for i in range(1, len(X_stream)):
            x = X_stream[i : i + 1]
            scores = self.verify_all(x).view(-1)
            idx_max = torch.argmax(scores).item()
            s_max = scores[idx_max]
            classes = list(self.representations.keys())
            class_id = classes[idx_max]

            if s_max > self.threshold:  # known class
                predictions += [class_id]
                # update cluster
                if s_max > self.threshold_update:
                    self.update_class(class_id, x)

            else:  # unknown class
                # create a new cluster
                class_id = self.add_class(x)
                predictions += [class_id]

        return predictions


class OnlineClusteringGaussian(OnlineRecognizerGaussian):
    """
    Performs online clustering given a sequence of feature vectors.
    Clusters are represented by Gaussian distributions. 
    See the parent class for details.
    """

    def __init__(self, b, w, dim, threshold):
        super().__init__(b, w, dim, threshold)
        self.name_generator = name_generator()
        self.representations = {}

    def add_class(self, X):
        class_id = next(self.name_generator)
        b_inv, w_inv = self.get_inv_params()
        summa = torch.sum(X, dim=0, keepdim=True)
        n = X.shape[0]
        Sigma = 1 / (b_inv + n * w_inv)
        mu = Sigma.view(-1, 1) * (w_inv * summa)
        self.representations[class_id] = (mu, Sigma)
        return class_id

    def __call__(self, X_stream):
        """
        Online clustering of feature vectors.

        Args:
            X_stream (torch.Tensor): input sequence of feature vectors.

        Returns:
            list: predicted class labels.
        """

        if X_stream.dim() == 1:
            X_stream = X_stream.unsqueeze(0)

        class_id = self.add_class(X_stream[0:1])
        classes = list(self.representations.keys())
        n_classes = len(classes)

        predictions = [class_id]

        mu0, kappa0 = [], []
        for c in classes:
            m, s = self.representations[c]
            mu0 += [m]
            kappa0 += [s.unsqueeze(0)]
        mu0 = torch.cat(mu0)
        kappa0 = torch.cat(kappa0).view(-1)

        eta, Lambda = self.standard_to_natural(mu0, kappa0)

        for i in range(1, len(X_stream)):
            x = X_stream[i : i + 1]
            mu, Sigma = self.natural_to_standard(eta, Lambda)
            probs = self.vb_posteriors(x, mu, Sigma)
            idx_max = torch.argmax(probs.view(-1))

            if idx_max < n_classes:  # known class
                classes = list(self.representations.keys())
                class_id = classes[idx_max]
                predictions += [class_id]

                # update all clusters
                eta, Lambda = self.update_incremental(
                    x, probs[:, :-1].view(-1), eta, Lambda
                )

            else:  # unknown class
                # create a new cluster
                class_id = self.add_class(x)
                predictions += [class_id]
                n_classes += 1

                m, s = self.representations[class_id]
                eta_x, Lambda_x = self.standard_to_natural(m, s)
                eta = torch.cat([eta, eta_x])
                Lambda = torch.cat([Lambda, Lambda_x])

        return predictions


class OnlineClusteringMises(OnlineRecognizerMisesFisher):
    """
    Performs online clustering given a sequence of feature vectors.
    Clusters are represented by von Mises-Fisher distributions. 
    See the parent class for details.
    """

    def __init__(self, b, w, mu, threshold):
        super().__init__(b, w, mu, threshold)
        self.name_generator = name_generator()
        self.representations = {}

    def add_class(self, X):
        class_id = next(self.name_generator)
        b, w = self.get_params()
        summa = torch.sum(X, dim=0, keepdim=True)
        n = X.shape[0]
        eta = w * summa + b * self.mu  # (K, dim)
        kappa = torch.norm(eta, dim=1)  # (K,)
        mu = eta / kappa.view(-1, 1)
        self.representations[class_id] = (mu, kappa)
        return class_id

    def __call__(self, X_stream):
        """
        Online clustering of feature vectors.

        Args:
            X_stream (torch.Tensor): input sequence of feature vectors.

        Returns:
            list: predicted class labels.
        """

        if X_stream.dim() == 1:
            X_stream = X_stream.unsqueeze(0)

        class_id = self.add_class(X_stream[0:1])
        classes = list(self.representations.keys())
        n_classes = len(classes)

        predictions = [class_id]

        mu0, kappa0 = [], []
        for c in classes:
            m, s = self.representations[c]
            mu0 += [m]
            kappa0 += [s.unsqueeze(0)]
        mu0 = torch.cat(mu0)
        kappa0 = torch.cat(kappa0).view(-1)

        eta = self.standard_to_natural(mu0, kappa0)

        for i in range(1, len(X_stream)):
            x = X_stream[i : i + 1]
            mu, kappa = self.natural_to_standard(eta)
            probs = self.vb_posteriors(x, mu, kappa)
            idx_max = torch.argmax(probs.view(-1))

            if idx_max < n_classes:  # known class
                classes = list(self.representations.keys())
                class_id = classes[idx_max]
                predictions += [class_id]

                # update all clusters
                eta = self.update_incremental(x, probs[:, :-1].view(-1), eta)

            else:  # unknown class
                # create a new cluster
                class_id = self.add_class(x)
                predictions += [class_id]
                n_classes += 1

                m, k = self.representations[class_id]
                eta_x = self.standard_to_natural(m, k)
                eta = torch.cat([eta, eta_x])

        return predictions

