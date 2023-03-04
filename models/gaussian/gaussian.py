import math
import torch

from utils import one_hot
from .gaussian_utils import euclidean_distance, logmvnpdf, logmvnpdf_noisy


class BaseRecognizerGaussian:
    """
    The base class for recognizers with classes modelled by isotropic Gaussian distributions
    with the shared variance. Classes are represented by the posterior distributions of their centers.

    Args:
        b (float): between-class variance.
        w (float): within-class variance.
        dim (int): dimension of the feature space.
        threshold (float): interpreted as log of a-priory probability for an unknown class.
    """

    def __init__(self, b, w, dim, threshold=0.0):
        # self.dim = dim
        self.b = torch.tensor([b]).float()
        self.w = torch.tensor([w]).float()
        threshold = torch.tensor([threshold])
        self.threshold = threshold

    def get_params(self):
        return self.b, self.w

    def get_inv_params(self):
        return 1 / self.b, 1 / self.w

    def init_classes(self, X, probs, variance_noise=None):
        """
        Initialize classes from feature vectors and their (soft) class assignments.

        Args:
            X (torch.Tensor): (N, d) matrix of feature vectors.
            probs (torch.Tensor): (N, K) soft assignments to K classes; one-hot encoding.
            is a special case.
            variance_noise (torch.Tensor): (N,) optional variances for uncertainty propagation. 
            Defaults to None.

        Returns:
            ((K, d), (K,)): posterior distributions (means and variances) for the class centers.
        """
        # same as update_classes(mu0=zeros, Sigma=b)
        b_inv, w_inv = self.get_inv_params()
        if variance_noise is None:
            counts = torch.sum(probs, dim=0)
            Sigma = 1 / (b_inv + counts * w_inv * 1)  # (K,)
            mu = 1 * Sigma.view(-1, 1) * (w_inv * torch.mm(probs.t(), X))  # (K, dim)
        else:
            w = 1 / w_inv
            variance_noise = variance_noise.view(-1, 1)
            w_inv = 1 / (w + variance_noise)
            Sigma = 1 / (b_inv + torch.sum(probs * w_inv, dim=0) * 1)  # (K,)
            mu = 1 * Sigma.view(-1, 1) * torch.mm(probs.t(), X * w_inv)
        return mu, Sigma

    def update_classes(self, X, probs, mu0, Sigma0, variance_noise=None):
        """
        Update classes given feature vectors and their (soft) class assignments.

        Args:
            X (torch.Tensor): (N, d) matrix of feature vectors.
            probs (torch.Tensor): (N, K) soft assignments to K classes; one-hot encoding.
            is a special case.
            mu0 (torch.Tensor): (K, d) prior means.
            Sigma0 (torch.Tensor): (K,) prior variances.
            variance_noise (torch.Tensor): (N,) optional variances for uncertainty propagation. 
            Defaults to None.

        Returns:
            ((K, d), (K,)): posterior distributions (means and variances) for the class centers.
        """
        b_inv, w_inv = self.get_inv_params()
        counts = torch.sum(probs, dim=0)
        if variance_noise is None:
            Sigma = 1 / (1 / Sigma0 + counts * w_inv * 1)  # (K,)
            mu = (
                1
                * Sigma.view(-1, 1)
                * (torch.mm(probs.t(), X * w_inv) + mu0 / Sigma0.view(-1, 1))
            )  # (K, dim)
        else:
            w = 1 / w_inv
            variance_noise = variance_noise.view(-1, 1)
            w_inv = 1 / (w + variance_noise)
            Sigma = 1 / (1 / Sigma0 + torch.sum(probs * w_inv, dim=0) * 1)  # (K,)
            mu = (
                1
                * Sigma.view(-1, 1)
                * (torch.mm(probs.t(), X * w_inv) + mu0 / Sigma0.view(-1, 1))
            )
        return mu, Sigma

    def update_classes_natural(self, X, probs, eta0, Lambda0, variance_noise=None):
        """
        Update classes using natural parameterization.
        """
        b_inv, w_inv = self.get_inv_params()
        if variance_noise is None:
            counts = torch.sum(probs, dim=0)
            Lambda = Lambda0 + counts * w_inv * 1  # (K,)
            eta = 1 * (w_inv * torch.mm(probs.t(), X) + eta0)  # (K, dim)
        else:
            w = 1 / w_inv
            variance_noise = variance_noise.view(-1, 1)
            w_inv = 1 / (w + variance_noise)
            Lambda = Lambda0 + torch.sum(probs * w_inv, dim=0) * 1  # (K,)
            eta = 1 * (torch.mm(probs.t(), X * w_inv) + eta0)
        return eta, Lambda

    def natural_to_standard(self, eta, Lambda):
        Sigma = 1 / Lambda
        mu = Sigma.view(-1, 1) * eta
        return mu, Sigma

    def standard_to_natural(self, mu, Sigma):
        Lambda = 1 / Sigma
        eta = Lambda.view(-1, 1) * mu
        return eta, Lambda

    def update(self, X, probs, mu0, Sigma0, variance_noise=None, n_iters=1):
        # X: (N, dim)
        # probs: (N, K)
        # assert n_iters > 1
        # mu, Sigma = self.natural_to_standard(eta, Lambda)
        for i in range(n_iters):

            # update classes
            mu, Sigma = self.update_classes(X, probs, mu0, Sigma0, variance_noise)

            # update soft assignments
            if i < n_iters - 1:
                probs = self.vb_posteriors(X, mu, Sigma, variance_noise)
                probs = probs[:, :-1]
        #
        eta, Lambda = self.standard_to_natural(mu, Sigma)
        return eta, Lambda

    def update_natural(self, X, probs, eta0, Lambda0, variance_noise=None, n_iters=1):
        # X: (N, dim)
        # probs: (N, K)
        # assert n_iters > 1
        # mu, Sigma = self.natural_to_standard(eta, Lambda)
        for i in range(n_iters):

            # update classes
            eta, Lambda = self.update_classes_natural(
                X, probs, eta0, Lambda0, variance_noise
            )

            # update soft assignments
            if i < n_iters - 1:
                mu, Sigma = self.natural_to_standard(eta, Lambda)
                probs = self.vb_posteriors(X, mu, Sigma, variance_noise)
                probs = probs[:, :-1]
        #
        return eta, Lambda

    def vb_posteriors(self, X, mu, Sigma, variance_noise=None):
        """
        Posterior distribution of the class assignments. The last column corresponds 
        to an unknown class.
        """
        # q(z|x) \propto \int q(y) log p(x,y) dy - VB rule
        n_classes = mu.shape[0]
        Fa = 1.0
        dim = X.shape[1]
        b_inv, w_inv = self.get_inv_params()
        b = 1 / b_inv
        w = 1 / w_inv
        b_plus_w_inv = 1 / (b + w)
        const = -0.5 * dim * math.log(2 * math.pi)
        log_post_outlier = -0.5 * b_plus_w_inv * torch.sum(X ** 2, dim=1, keepdim=True)
        log_post_outlier += 0.5 * dim * torch.log(b_plus_w_inv) + const
        # log_post_outlier = logmvnpdf(X, torch.zeros(1, dim), b + w) # alternative

        log_prior = torch.zeros(n_classes + 1)
        log_prior[-1] = self.threshold  # unknown class prior

        if variance_noise is not None:
            w = 1 / w_inv
            variance_noise = variance_noise.view(-1, 1)
            w_inv = 1 / (w + variance_noise)

        log_post = -0.5 * euclidean_distance(X, mu) * w_inv
        log_post += 0.5 * dim * torch.log(w_inv)
        log_post += -0.5 * dim * w_inv * Sigma + const
        # log_post = logmvnpdf(X, mu, w) - 0.5 * dim * w_inv * Sigma # alternative
        # log_post = logmvnpdf(X, mu, Sigma + w)

        log_post = torch.cat([log_post, log_post_outlier], 1)
        # log_post = log_post - log_post_outlier.repeat(1, log_post.shape[-1])
        log_post = Fa * log_post + log_prior.view(1, -1)
        post = torch.exp(log_post - torch.logsumexp(log_post, dim=1, keepdim=True))
        return post  # [:, :-1]

    # verify_all
    def predict(
        self, X, mu, Sigma, variance_noise=None, detection=False, prior_unk=0.1
    ):
        # p(x|X) \propto \int p(x|y) q(y|X) dy - predictive distribution
        # score_k = p(x|X_k) / p(x)
        # score_k = p(x|X_k) / (p(x) + \sum_i p(x|X_i)) - detection
        b, w = self.get_params()
        dim = X.shape[1]
        n_classes = mu.shape[0]

        log_predictive_outlier = logmvnpdf(X, torch.zeros(1, dim), b + w)

        if variance_noise is None:
            log_predictive = logmvnpdf(X, mu, Sigma + w)
        else:
            log_predictive = logmvnpdf_noisy(
                X, mu, Sigma + w, variance_noise
            )  # logmvnpdf(X, mu, Sigma + w + variance_noise)
        scores = log_predictive - log_predictive_outlier
        # scores = plda_score(X, mu, Sigma, variance_noise) # alternative

        # if detection: # scoring model from https://hal.archives-ouvertes.fr/hal-01927584/document
        if detection and n_classes > 1:
            n_classes = mu.shape[0]
            I = torch.eye(n_classes, device=scores.device).unsqueeze(0)
            mult = 1 - I
            log_prior_known = math.log(1 - prior_unk) - math.log(n_classes - 1)
            bias = math.log(prior_unk) * I + log_prior_known * (1 - I)
            denominator = mult * scores + bias
            scores = scores - torch.logsumexp(denominator, dim=1, keepdim=True)

        return scores  # (N, K)

    def scores_to_probs(self, scores, threshold):
        # scores_max, _ = torch.max(scores, dim=1)
        scores = torch.cat([scores, threshold * torch.ones(scores.shape[0], 1)], dim=1)
        probs = torch.exp(scores - torch.logsumexp(scores, dim=1, keepdim=True))
        return probs


class OnlineRecognizerGaussian(BaseRecognizerGaussian):
    """
    Performs open-set classification of a sequence of observations one-by-one 
    while updating the class parameters after each step. Each class is modelled 
    by an isotropic Gaussian distribution. See the parent class for details.
    """

    def __init__(self, b, w, dim, threshold=0.0):
        super().__init__(b, w, dim, threshold)

    def update_incremental(self, x, probs_x, eta, Lambda, variance_noise=None):
        """
        Update classes from a single feature vector and the (soft) class assignments.

        Args:
            x (torch.Tensor): (1, d) feature vector.
            probs_x (torch.Tensor): (K,) soft assignments to K classes.
            eta, Lambda (torch.Tensor): (K, d) prior parameters in the natural parametrization.
            variance_noise (torch.Tensor): (1,) optional variance for uncertainty propagation. 
            Defaults to None.

        Returns:
            ((K, d), (K,)): posterior distributions (in the natural parametrization) for the class centers.
        """
        # probs_x: (K,)
        b_inv, w_inv = self.get_inv_params()
        if variance_noise is not None:
            w = 1 / w_inv
            w_inv = 1 / (w + variance_noise)
        Lambda = w_inv * probs_x + Lambda  # (K,)
        eta = w_inv * probs_x.view(-1, 1) * x + eta  # (K, dim)
        return eta, Lambda

    def __call__(self, X_stream, X_enroll, y_enroll, variance_noise=None, n_iters=1):
        # X_stream: (T, dim)
        # X_enroll: (M, dim)
        # y_enroll: (M,)
        # variance_noise: (T,)

        mu0, Sigma0 = self.init_classes(X_enroll, one_hot(y_enroll))
        eta, Lambda = self.standard_to_natural(mu0, Sigma0)

        # incremental updates
        probs_history = []
        for i in range(len(X_stream)):
            x = X_stream[i : i + 1]

            # predict
            mu, Sigma = self.natural_to_standard(eta, Lambda)

            if variance_noise is None:
                probs = self.vb_posteriors(x, mu, Sigma)
            else:
                probs = self.vb_posteriors(x, mu, Sigma, variance_noise[i : i + 1])

            probs_history += [probs]
            # NOTE: in the current implementation predictions do NOT change if n_iters>1;
            # TODO: add support for that

            # update classes
            if n_iters == 1:
                if variance_noise is None:
                    eta, Lambda = self.update_incremental(x, probs[:, :-1], eta, Lambda)
                else:
                    eta, Lambda = self.update_incremental(
                        x, probs[:, :-1], eta, Lambda, variance_noise[i]
                    )
            else:
                if variance_noise is None:
                    eta, Lambda = self.update(
                        X_stream[: i + 1],
                        torch.cat(probs_history)[:, :-1],
                        mu0,
                        Sigma0,
                        n_iters=n_iters,
                    )
                else:
                    eta, Lambda = self.update(
                        X_stream[: i + 1],
                        torch.cat(probs_history)[:, :-1],
                        mu0,
                        Sigma0,
                        variance_noise[: i + 1],
                        n_iters=n_iters,
                    )
        #
        mu, Sigma = self.natural_to_standard(eta, Lambda)
        probs_history = torch.cat(probs_history)
        return probs_history, mu, Sigma

