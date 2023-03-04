import torch

from utils import one_hot
from .vonmises_utils import rho, log_vmf_normalizer


class BaseRecognizerMisesFisher:
    """
    The base class for recognizers with classes modelled by Mises-Fisher distributions 
    with the shared concentration (a reciprocal measure of dispersion). Classes are represented 
    by the posterior distributions of their mean directions.

    Args:
        b (float): between-class concentration.
        w (float): within-class concentration.
        mu (np.ndarray): mean direction vector.
        threshold (float): interpreted as log of a-priory probability for an unknown class.
    """

    def __init__(self, b, w, mu, threshold=0.0):
        # self.dim = mu.shape[-1]
        self.mu = torch.tensor(mu).float().reshape(1, -1)
        self.b = torch.tensor([b]).float()
        self.w = torch.tensor([w]).float()
        threshold = torch.tensor([threshold])
        self.threshold = threshold

    def get_params(self):
        return self.b, self.w

    def init_classes(self, X, probs):
        """
        Initialize classes from feature vectors and their (soft) class assignments.

        Args:
            X (torch.Tensor): (N, d) matrix of feature vectors.
            probs (torch.Tensor): (N, K) soft assignments to K classes; one-hot encoding.
            is a special case.

        Returns:
            ((K, d), (K,)): posterior distributions (means and concentrations) for 
            the class mean directions.
        """
        b, w = self.get_params()
        eta = w * torch.mm(probs.t(), X) + b * self.mu  # (K, dim)
        kappa = torch.norm(eta, dim=1)  # (K,)
        mu = eta / kappa.view(-1, 1)
        return mu, kappa

    def update_classes(self, X, probs, mu0, kappa0, precision_noise=None):
        """
        Update classes given feature vectors and their (soft) class assignments.

        Args:
            X (torch.Tensor): (N, d) matrix of feature vectors.
            probs (torch.Tensor): (N, K) soft assignments to K classes; one-hot encoding. 
            is a special case.
            mu0 (torch.Tensor): (K, d) prior means.
            kappa0 (torch.Tensor): (K,) prior concentrations.
            precision_noise (torch.Tensor): (N,) optional precisions for uncertainty propagation. 
            Defaults to None.

        Returns:
            ((K, d), (K,)): posterior distributions (means and concentrations) for 
            the class mean directions.
        """
        b, w = self.get_params()
        if precision_noise is None:
            eta = w * torch.mm(probs.t(), X) + kappa0.view(-1, 1) * mu0  # (K, dim)
        else:
            precision_noise = precision_noise.view(-1, 1)
            w = 1 / (1 / w + 1 / precision_noise)
            eta = torch.mm(probs.t(), X * w) + kappa0.view(-1, 1) * mu0  # (K, dim)
        kappa = torch.norm(eta, dim=1)  # (K,)
        mu = eta / kappa.view(-1, 1)
        return mu, kappa

    def update_classes_natural(self, X, probs, eta0):
        """
        Update classes using natural parameterization.
        """
        b, w = self.get_params()
        eta = w * torch.mm(probs.t(), X) + eta0  # (K, dim)
        return eta

    def natural_to_standard(self, eta):
        kappa = torch.norm(eta, dim=1)  # (K,)
        mu = eta / kappa.view(-1, 1)
        return mu, kappa

    def standard_to_natural(self, mu, kappa):
        eta = kappa.view(-1, 1) * mu
        return eta

    def update(self, X, probs, mu0, kappa0, precision_noise=None, n_iters=3):
        # X: (N, dim)
        # probs: (N, K)
        for _ in range(n_iters):

            # update classes
            mu, kappa = self.update_classes(X, probs, mu0, kappa0, precision_noise)

            # update soft assignments
            probs = self.vb_posteriors(X, mu, kappa, precision_noise)
            probs = probs[:, :-1]

        eta = self.standard_to_natural(mu, kappa)
        return eta

    def vb_posteriors(self, X, mu, kappa, precision_noise=None):
        """
        Posterior distribution of the class assignments. The last column corresponds 
        to an unknown class.
        """
        n_classes, dim = mu.shape
        Fa = 1
        b, w = self.get_params()

        r = torch.norm(w * X + b * self.mu, dim=1)  # (K,)
        log_post_outlier = (
            log_vmf_normalizer(b.double().numpy(), dim)
            + log_vmf_normalizer(w.double().numpy(), dim)
            - log_vmf_normalizer(r.double().numpy(), dim)
        )
        log_post_outlier = torch.tensor(log_post_outlier).view(-1, 1).float()

        log_prior = torch.zeros(n_classes + 1)
        log_prior[-1] = self.threshold  # unknown class prior

        A = rho(kappa.double().numpy(), dim)
        A = torch.tensor(A).float()
        logC = log_vmf_normalizer(w.double().numpy(), dim)  # const
        logC = torch.tensor(logC).float()
        mu_exp = A.view(-1, 1) * mu

        if precision_noise is None:
            log_post = logC + w * torch.mm(X, mu_exp.t())
        else:
            precision_noise = precision_noise.view(-1, 1)
            w = 1 / (1 / w + 1 / precision_noise)
            log_post = logC + torch.mm(w * X, mu_exp.t())

        log_post = torch.cat([log_post, log_post_outlier], 1)
        log_post = Fa * log_post + log_prior.view(1, -1)
        post = torch.exp(log_post - torch.logsumexp(log_post, dim=1, keepdim=True))
        return post


class OnlineRecognizerMisesFisher(BaseRecognizerMisesFisher):
    """
    Performs open-set classification of a sequence of observations one-by-one 
    while updating the class parameters after each step. Each class is represented 
    by a von Mises-Fisher distribution. See the parent class for details.
    """

    def __init__(self, b, w, mu, threshold=0.0):
        super().__init__(b, w, mu, threshold)

    def update_incremental(self, x, probs_x, eta, precision_noise=None):
        """
        Update classes from a single feature vector and the (soft) class assignments.

        Args:
            x (torch.Tensor): (1, d) feature vector.
            probs_x (torch.Tensor): (K,) soft assignments to K classes.
            eta (torch.Tensor): (K, d) prior parameters in the natural parametrization.
            precision_noise (torch.Tensor): (1,) optional precision for uncertainty propagation. 
            Defaults to None.

        Returns:
            (K, d): posterior distributions (in the natural parametrization) for 
            the class mean directions.
        """
        b, w = self.get_params()
        if precision_noise is not None:
            w = 1 / (1 / w + 1 / precision_noise)
        eta = w * probs_x.view(-1, 1) * x + eta
        return eta

    def __call__(self, X_stream, X_enroll, y_enroll, precision_noise=None, n_iters=1):

        mu0, kappa0 = self.init_classes(X_enroll, one_hot(y_enroll))
        eta = self.standard_to_natural(mu0, kappa0)

        # incremental updates
        probs_history = []
        for i in range(len(X_stream)):
            x = X_stream[i : i + 1]

            # predict
            mu, kappa = self.natural_to_standard(eta)

            if precision_noise is None:
                probs = self.vb_posteriors(x, mu, kappa)
            else:
                probs = self.vb_posteriors(x, mu, kappa, precision_noise[i : i + 1])

            # scores = self.predict(x, mu, kappa, precision_noise[i:i+1], detection=True)
            # probs = self.scores_to_probs(scores, threshold=0.0)

            probs_history += [probs]

            # update classes
            if n_iters == 1:
                if precision_noise is None:
                    eta = self.update_incremental(x, probs[:, :-1], eta)
                else:
                    eta = self.update_incremental(
                        x, probs[:, :-1], eta, precision_noise[i]
                    )
            else:
                if precision_noise is None:
                    eta = self.update(
                        X_stream[: i + 1],
                        torch.cat(probs_history)[:, :-1],
                        mu0,
                        kappa0,
                        n_iters=n_iters,
                    )
                else:
                    eta = self.update(
                        X_stream[: i + 1],
                        torch.cat(probs_history)[:, :-1],
                        mu0,
                        kappa0,
                        precision_noise[: i + 1],
                        n_iters=n_iters,
                    )

        #
        mu, kappa = self.natural_to_standard(eta)
        probs_history = torch.cat(probs_history)
        return probs_history, mu, kappa

