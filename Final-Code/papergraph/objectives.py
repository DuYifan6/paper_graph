"""Explicit supplementary ranking loss and validation score."""
import numpy as np
import torch
from torch.nn import functional as F


def ranking_loss(prediction, target):
    """Top 15% versus remaining targets; top count clipped to [5,25]."""
    n = prediction.numel()
    if n < 10:
        return prediction.sum()*0
    top = min(max(5, round(n*.15)), 25, n-1)
    order = torch.argsort(target, descending=True)
    difference = prediction[order[:top], None]-prediction[order[top:]][None, :]
    return F.softplus(-difference).mean()


def weighted_topk_score(detail):
    """Mean daily Top-K target returns, then K weights .05/.45/.30/.20.

    This validation score is separate from the cost-aware portfolio simulation.
    Only validation labels enter checkpoint selection.
    """
    frame = detail[np.isfinite(detail.y_true) & np.isfinite(detail.y_pred)]
    scores = []
    for _, day in frame.groupby('date', sort=True):
        ranked = day.sort_values(['y_pred', 'stock'], ascending=[False, True], kind='stable')
        scores.append(sum(weight*float(ranked.y_true.iloc[:min(k, len(ranked))].mean())
                          for k, weight in ((1, .05), (3, .45), (5, .30), (10, .20))))
    if not scores:
        raise ValueError('No validation observations for the supplementary score')
    return float(np.mean(scores))
