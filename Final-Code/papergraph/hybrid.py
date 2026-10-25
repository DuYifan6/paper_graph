"""Reconstruct the latest raw step at 14:55 using a frozen reference scaler."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import FEATURES
from .preprocessing import FrozenTransform


def build_latest(raw_path, prices_path, dates, stocks, preprocessing, output,
                 return_scale=100.):
    """Raw NPZ follows preprocessing.py; quote CSV columns:

    date (signal day), stock, price1455, day_open, previous_close.
    Quote prices must share the raw return indicators' corporate-action basis.
    Retained intraday indicators are already observed by 14:30. Full-day flow
    indicators are shifted an additional feature session. Raw percentage-return
    indicators use return_scale=100 before the frozen standardization.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    with np.load(raw_path, allow_pickle=False) as z:
        feature_dates = pd.DatetimeIndex(pd.to_datetime(z['dates'])).normalize()
        if z['stocks'].astype(str).tolist() != list(stocks):
            raise ValueError('Raw stock identifiers differ')
        names = z['feature_names'].astype(str).tolist()
        if len(names) != 9 or set(names) != set(FEATURES):
            raise ValueError('Raw feature declarations differ')
        raw = np.asarray(z['X_raw'], dtype=float)[:, :, [names.index(f) for f in FEATURES]]
    if feature_dates.has_duplicates or not feature_dates.is_monotonic_increasing:
        raise ValueError('Raw feature sessions must be unique and increasing')
    quotes = pd.read_csv(prices_path, dtype={'stock': str})
    required = {'date', 'stock', 'price1455', 'day_open', 'previous_close'}
    if not required.issubset(quotes):
        raise ValueError('Quote CSV lacks pre-cutoff prices')
    quotes['date'] = pd.to_datetime(quotes.date).dt.normalize()
    if quotes.duplicated(['date', 'stock']).any():
        raise ValueError('Duplicate signal-day quotes')
    quotes = quotes.set_index(['date', 'stock'])
    latest = np.full((len(dates), len(stocks), 9), np.nan, dtype=float)
    asof = np.full(len(dates), np.datetime64('NaT'), dtype='datetime64[ns]')
    pct = FEATURES.index('D_pct_chg')
    intraday = FEATURES.index('D_intraday_return')
    slow = [FEATURES.index('D_mainforce_strength'), FEATURES.index('D_bull_bear_ratio')]
    for t, target in enumerate(dates):
        p = int(feature_dates.searchsorted(target, side='left'))-1
        if p < 1:
            continue
        signal = feature_dates[p]
        if t and signal != dates[t-1]:
            continue
        latest[t] = raw[p]
        latest[t, :, slow[0]] = raw[p-1, :, slow[0]]
        latest[t, :, slow[1]] = raw[p-1, :, slow[1]]
        latest[t, :, [pct, intraday]] = np.nan
        asof[t] = (signal+pd.Timedelta(hours=14, minutes=55)).to_datetime64()
        for j, stock in enumerate(stocks):
            if (signal, stock) not in quotes.index:
                continue
            q = quotes.loc[(signal, stock)]
            price, opening, previous = float(q.price1455), float(q.day_open), float(q.previous_close)
            if np.isfinite(price) and price > 0:
                if np.isfinite(previous) and previous > 0:
                    latest[t, j, pct] = return_scale*(price/previous-1)
                if np.isfinite(opening) and opening > 0:
                    latest[t, j, intraday] = return_scale*(price/opening-1)
    transform = FrozenTransform.load(preprocessing)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    common = {'dates': dates.to_numpy(), 'stocks': np.asarray(stocks, dtype=str),
              'feature_names': np.asarray(FEATURES), 'asof': asof}
    np.savez_compressed(output/'hybrid_raw.npz', X_latest_raw=latest, **common)
    np.savez_compressed(output/'hybrid_latest.npz', X_latest_hybrid_np=transform.transform(latest), **common)
    return latest
