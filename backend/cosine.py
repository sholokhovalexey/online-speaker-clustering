import torch


def cosine_similarity(a, b, numpy=False):
    if numpy:
        a, b = torch.tensor(a), torch.tensor(b)
    a, b = (
        a / torch.norm(a, dim=1, keepdim=True),
        b / torch.norm(b, dim=1, keepdim=True),
    )
    if numpy:
        return torch.mm(a, b.t()).numpy()
    else:
        return torch.mm(a, b.t())


def score_cosine_embeddings_averaging(X1, X2):
    centroid1 = torch.mean(X1, dim=0, keepdim=True)
    centroid2 = torch.mean(X2, dim=0, keepdim=True)
    score = cosine_similarity(centroid1, centroid2)
    return score


def score_cosine_scores_averaging(X1, X2):
    if X2.shape[0] > 1:
        score = torch.mean(cosine_similarity(X1, X2))
    else:
        score = torch.mean(cosine_similarity(X1, X2), dim=0)
    return score

