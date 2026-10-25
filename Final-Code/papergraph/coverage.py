"""Common-stock-day coverage comparisons; target checks are evaluation-only."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .config import write_json
from .statistics import dm_test


def compare_coverage(original_graph, original_zero, expanded_graph, expanded_zero, output):
    paths = [original_graph, original_zero, expanded_graph, expanded_zero]
    if any(p is None for p in paths):
        raise ValueError('Supply all four original/expanded Graph-2/ZeroGraph prediction files')
    labels = ['original_graph', 'original_zero', 'expanded_graph', 'expanded_zero']
    common = None
    for label, path in zip(labels, paths):
        frame = pd.read_csv(path, dtype={'stock': str})
        if not {'date', 'stock', 'y_true', 'y_pred'}.issubset(frame):
            raise ValueError('Coverage evaluation needs date, stock, y_true and y_pred')
        if frame.duplicated(['date', 'stock']).any():
            raise ValueError('Duplicate stock-day predictions')
        frame['date'] = pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')
        frame = frame[np.isfinite(frame.y_true) & np.isfinite(frame.y_pred)]
        frame = frame[['date', 'stock', 'y_true', 'y_pred']].rename(
            columns={'y_true': 'y_'+label, 'y_pred': label})
        common = frame if common is None else common.merge(frame, on=['date', 'stock'],
                                                          how='inner', validate='one_to_one')
    common = common.sort_values(['date', 'stock']).reset_index(drop=True)
    if common.empty:
        raise ValueError('No common evaluation observations')
    y = common['y_original_graph'].to_numpy()
    for label in labels[1:]:
        if not np.array_equal(y, common['y_'+label].to_numpy()):
            raise ValueError('Realized targets differ on common observations')
    denominator = float(np.square(y-y.mean()).sum())
    if denominator <= 0:
        raise ValueError('R-squared denominator is zero')
    metrics = []
    for label in labels:
        error = np.square(y-common[label].to_numpy())
        common[label+'_loss'] = error
        metrics.append({'model': label, 'stock_days': len(y), 'mse': float(error.mean()),
                        'r2': 1-float(error.sum())/denominator})
    daily = common.groupby('date', sort=True)[[x+'_loss' for x in labels]].mean()
    daily['original_advantage'] = daily.original_zero_loss-daily.original_graph_loss
    daily['expanded_advantage'] = daily.expanded_zero_loss-daily.expanded_graph_loss
    daily['advantage_change'] = daily.expanded_advantage-daily.original_advantage
    daily['graph_loss_change'] = daily.expanded_graph_loss-daily.original_graph_loss
    daily['zero_loss_change'] = daily.expanded_zero_loss-daily.original_zero_loss
    inference = {name: dm_test(daily[name], 2, reference='normal') for name in
                 ('original_advantage', 'expanded_advantage', 'advantage_change',
                  'graph_loss_change', 'zero_loss_change')}
    summary = {'common_stock_days': len(common), 'evaluation_days': len(daily),
               'target_fingerprint': hashlib.sha256(y.tobytes()).hexdigest(),
               'metrics': metrics, 'daily_inference': inference,
               'scope': 'single-seed paired comparison on identical observed targets'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output/'common_metrics.csv', index=False)
    daily.to_csv(output/'common_daily_losses.csv')
    write_json(output/'coverage_comparison.json', summary)
    return summary
