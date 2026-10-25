"""Two-layer gate, MLP prediction head, and target-independent forward paths."""
import numpy as np
import torch
from torch import nn


def validate_input_mask(input_available, n, device):
    if input_available is None:
        raise ValueError('An independent decision-time input availability mask is required')
    mask = torch.as_tensor(input_available, device=device)
    if mask.dtype != torch.bool or mask.shape != (n,):
        raise ValueError('input_available must be a Boolean vector with one entry per stock')
    return mask


def edge_tensors(graph, device, valid=None):
    if graph is None:
        raise ValueError('An empirical or randomized graph is required')
    u = torch.as_tensor(graph.u, dtype=torch.long, device=device)
    v = torch.as_tensor(graph.v, dtype=torch.long, device=device)
    w = torch.as_tensor(graph.weight, dtype=torch.float32, device=device)
    if valid is not None:
        keep = valid[u] & valid[v]
        u, v, w = u[keep], v[keep], w[keep]
    return u, v, w


def aggregate(h, graph, input_available):
    """Eq. (15): row-weighted mean on the input-available induced graph.

    Node inclusion and the denominator never depend on observed targets.
    Each undirected edge is stored once and contributes in both directions.
    """
    valid = validate_input_mask(input_available, len(h), h.device)
    u, v, w = edge_tensors(graph, h.device, valid)
    w = w.to(h.dtype)
    clean = torch.where(valid[:, None], h, torch.zeros_like(h))
    result = torch.zeros_like(h)
    degree = h.new_zeros(len(h))
    result.index_add_(0, u, w[:, None] * clean[v])
    result.index_add_(0, v, w[:, None] * clean[u])
    degree.index_add_(0, u, w)
    degree.index_add_(0, v, w)
    return result / (degree[:, None] + 1e-6)


class GraphMix(nn.Module):
    def __init__(self, hidden=128, dropout=.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden))
        self.message = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, graph, input_available, identity=False):
        valid = validate_input_mask(input_available, len(h), h.device)
        h = torch.where(valid[:, None], h, torch.zeros_like(h))
        neighbor = h if identity else aggregate(h, graph, valid)
        joined = torch.cat((h, neighbor), dim=-1)
        adjustment = self.dropout(torch.relu(self.message(joined)))
        return h + torch.sigmoid(self.gate(joined)) * adjustment


class SelfOnlyEncoder(nn.Module):
    """Within-stock 20x48 history -> 144 summaries -> learned representation.

    The summaries are 48 time-of-day means, standard deviations, and latest-day
    values. At H=128, the encoder plus three active fusion scalars has 82,304
    parameters. No other stock's data enter this branch.
    """
    def __init__(self, hidden=128, dropout=.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(144, 301), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(301, hidden), nn.ReLU())
        self.temporal_scale = nn.Parameter(torch.ones(()))
        self.hf_scale = nn.Parameter(torch.ones(()))
        self.fusion_bias = nn.Parameter(torch.zeros(()))

    def forward(self, temporal, history):
        if history.ndim == 2:
            if history.shape[1] % 48:
                raise ValueError('Self-only history must contain 48 returns per day')
            history = history.reshape(len(history), -1, 48)
        if history.ndim != 3 or history.shape[-1] != 48:
            raise ValueError('Expected a within-stock return history of shape [N, W, 48]')
        summary = torch.cat((history.mean(1), history.std(1, unbiased=False),
                             history[:, -1]), dim=1)
        return (self.temporal_scale * temporal + self.hf_scale * self.encoder(summary)
                + self.fusion_bias)


class Forecaster(nn.Module):
    def __init__(self, features=9, hidden=128, variant='graph2', hf_size=960, dropout=.1):
        super().__init__()
        if variant not in ('zero', 'identity', 'no_lap', 'graph1', 'graph2', 'graph3',
                            'random', '5m_nograph'):
            raise ValueError('Unknown forecasting variant')
        self.variant = variant
        self.temporal = nn.LSTM(features, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.graph = (SelfOnlyEncoder(hidden, dropout) if variant == '5m_nograph'
                      else GraphMix(hidden, dropout))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, sequence, graph=None, hf=None, *, input_available):
        valid = validate_input_mask(input_available, len(sequence), sequence.device)
        if sequence.ndim != 3:
            raise ValueError('Expected [N, L, F] temporal inputs')
        sequence = torch.where(valid[:, None, None], sequence, torch.zeros_like(sequence))
        if not torch.isfinite(sequence).all():
            raise ValueError('Available inputs must use a fixed past-fitted imputer')
        _, (state, _) = self.temporal(sequence)
        h = self.dropout(state[-1])
        if self.variant == '5m_nograph':
            if hf is None:
                raise ValueError('Self-only history is required')
            shape = (len(valid),) + (1,) * (hf.ndim - 1)
            hf = torch.where(valid.reshape(shape), hf, torch.zeros_like(hf))
            if not torch.isfinite(hf).all():
                raise ValueError('Available self-only history is nonfinite')
            h = self.graph(h, hf)
        elif self.variant != 'zero':
            h = self.graph(h, graph, valid, identity=self.variant == 'identity')
        return self.head(h).squeeze(-1)


class NeighborGCN(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.own = nn.Linear(inputs, outputs)
        self.neighbor = nn.Linear(inputs, outputs)
        self.norm = nn.LayerNorm(outputs)
        self.dropout = nn.Dropout(.1)

    def forward(self, x, graph, input_available):
        z = self.own(x) + self.neighbor(aggregate(x, graph, input_available))
        z = self.dropout(torch.relu(self.norm(z)))
        return torch.where(input_available[:, None], z, torch.zeros_like(z))


class GCNForecaster(nn.Module):
    variant = 'gcn_lstm'

    def __init__(self, features=9, hidden=128, dropout=.1):
        super().__init__()
        self.graph = nn.ModuleList([NeighborGCN(features, 64), NeighborGCN(64, 64)])
        self.temporal = nn.LSTM(64, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
                                  nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, sequence, graphs, hf=None, *, input_available):
        valid = validate_input_mask(input_available, len(sequence), sequence.device)
        sequence = torch.where(valid[:, None, None], sequence, torch.zeros_like(sequence))
        if len(graphs) != sequence.shape[1] or not torch.isfinite(sequence).all():
            raise ValueError('GCN sequence or historical graphs are invalid')
        states = []
        for i, graph in enumerate(graphs):
            h = sequence[:, i]
            for layer in self.graph:
                h = layer(h, graph, valid)
            states.append(h)
        _, (state, _) = self.temporal(torch.stack(states, dim=1))
        return self.head(state[-1]).squeeze(-1)


def laplacian_loss(pred, graph, supervised_valid):
    """Eqs. (18)-(19), induced loss subgraph, zero isolated-node diagonal.

    This supervised regularization mask is never used in a forward prediction.
    """
    valid = validate_input_mask(supervised_valid, len(pred), pred.device)
    n = valid.sum()
    if int(n) == 0:
        raise ValueError('No valid targets for the supervised loss')
    u, v, w = edge_tensors(graph, pred.device, valid)
    w = w.to(pred.dtype)
    degree = torch.zeros_like(pred)
    degree.index_add_(0, u, w)
    degree.index_add_(0, v, w)
    normalized = pred / torch.sqrt(torch.where(degree > 0, degree, torch.ones_like(degree)))
    return 2 * (w * (normalized[u] - normalized[v]).square()).sum() / n


def parameter_audit(model, sequence, graph=None, hf=None, *, input_available):
    model.zero_grad(set_to_none=True)
    pred = model(sequence, graph, hf, input_available=input_available)
    pred[input_available].square().sum().backward()
    audit = {
        'registered': sum(p.numel() for p in model.parameters()),
        'active': sum(p.numel() for p in model.parameters() if p.grad is not None),
        'temporal_registered': sum(p.numel() for p in model.temporal.parameters()),
        'graph_registered': sum(p.numel() for p in model.graph.parameters()),
        'head_registered': sum(p.numel() for p in model.head.parameters()),
    }
    model.zero_grad(set_to_none=True)
    return audit
