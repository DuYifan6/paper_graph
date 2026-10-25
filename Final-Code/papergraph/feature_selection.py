"""Quarterly LASSO with date-blocked CV and the one-standard-error rule."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import write_json


def screen_features(candidate_file, features, output, folds=5, alphas=100,
                    first_quarter='2022Q2', last_quarter='2024Q2', epsilon=1e-8):
    """CSV: date, asof, y, and explicitly named already-lagged predictors.

    asof must precede the target date. Imputation and standardization are fitted
    within each training fold, then refitted on the quarter for final coefficients.
    The intersection is data-dependent; the code never forces nine survivors.
    """
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import TimeSeriesSplit
    frame = pd.read_csv(candidate_file)
    if not features or len(set(features)) != len(features):
        raise ValueError('Declare distinct candidate predictor columns')
    if not {'date', 'asof', 'y', *features}.issubset(frame):
        raise ValueError('Candidate file lacks date, asof, y or predictor columns')
    frame['date'] = pd.to_datetime(frame.date).dt.normalize()
    frame['asof'] = pd.to_datetime(frame['asof'])
    if frame['asof'].isna().any() or (frame['asof'] >= frame.date).any():
        raise ValueError('Candidate predictors must be known before the target date')
    quarters = pd.period_range(first_quarter, last_quarter, freq='Q')
    if any(q.year >= 2025 for q in quarters):
        raise ValueError('Predictor screening excludes the 2025 evaluation period')
    selections, rows = [], []

    def standardize(train, validation):
        train = np.where(np.isfinite(train), train, np.nan)
        if np.isnan(train).all(axis=0).any():
            raise ValueError('A candidate is entirely missing in a training fold')
        median = np.nanmedian(train, axis=0)
        train = np.where(np.isfinite(train), train, median)
        mean, scale = train.mean(0), train.std(0)
        scale = np.where(scale > 1e-12, scale, 1.)
        return (train-mean)/scale, (np.where(np.isfinite(validation), validation, median)-mean)/scale

    for quarter in quarters:
        sample = frame[(frame.date.dt.to_period('Q') == quarter) & np.isfinite(frame.y)].sort_values('date')
        dates = pd.DatetimeIndex(sample.date.unique())
        if len(dates) <= folds or folds < 2:
            raise ValueError('Insufficient distinct trading dates for quarterly time-ordered CV')
        x, y = sample[features].to_numpy(dtype=float), sample.y.to_numpy(dtype=float)
        partitions = []
        for train_dates, validation_dates in TimeSeriesSplit(n_splits=folds).split(dates):
            train = sample.date.isin(dates[train_dates]).to_numpy()
            validation = sample.date.isin(dates[validation_dates]).to_numpy()
            xt, xv = standardize(x[train], x[validation])
            partitions.append((xt, y[train], xv, y[validation]))
        xt, yt, _, _ = partitions[0]
        largest = max(float(np.abs(xt.T @ (yt-yt.mean())).max()/len(yt)), 1e-8)
        grid = np.geomspace(largest, largest*.001, alphas)
        losses = np.empty((folds, alphas))
        for fold, (xt, yt, xv, yv) in enumerate(partitions):
            for i, alpha in enumerate(grid):
                model = Lasso(alpha=float(alpha), max_iter=20000, tol=1e-6)
                model.fit(xt, yt)
                losses[fold, i] = np.square(model.predict(xv)-yv).mean()
        mean = losses.mean(0)
        best = int(mean.argmin())
        limit = mean[best]+losses[:, best].std(ddof=1)/np.sqrt(folds)
        chosen = int(np.flatnonzero(mean <= limit)[0])
        full, _ = standardize(x, x)
        model = Lasso(alpha=float(grid[chosen]), max_iter=20000, tol=1e-6).fit(full, y)
        active = {f for f, beta in zip(features, model.coef_) if abs(beta) > epsilon}
        selections.append(active)
        for f, beta in zip(features, model.coef_):
            rows.append({'quarter': str(quarter), 'feature': f, 'coefficient': float(beta),
                         'selected': f in active, 'alpha_1se': float(grid[chosen])})
    stable = sorted(set.intersection(*selections))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'quarterly_lasso.csv', index=False)
    write_json(output/'stable_features.json', {'features': stable, 'quarters': len(quarters),
               'criterion': 'selected in every quarter', 'cv': 'chronological date blocks',
               'alpha_rule': 'largest alpha within one standard error of minimum mean CV MSE'})
    return stable
