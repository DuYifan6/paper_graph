"""Training-only median imputation and scaling, reusable across universes."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import FEATURES


class FrozenTransform:
    def __init__(self, median, mean, scale):
        self.median = np.asarray(median, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        if (self.median.shape != (9,) or self.mean.shape != (9,) or self.scale.shape != (9,)
                or not np.isfinite(np.r_[self.median, self.mean, self.scale]).all()
                or (self.scale <= 0).any()):
            raise ValueError('Invalid frozen preprocessing parameters')

    @classmethod
    def fit(cls, fitting_inputs):
        x = np.asarray(fitting_inputs, dtype=float).reshape(-1, 9)
        x = np.where(np.isfinite(x), x, np.nan)
        if not len(x) or np.isnan(x).all(axis=0).any():
            raise ValueError('Each predictor needs observed fitting-period values')
        median = np.nanmedian(x, axis=0)
        filled = np.where(np.isfinite(x), x, median)
        sd = filled.std(axis=0)
        return cls(median, filled.mean(axis=0), np.where(sd > 1e-12, sd, 1.))

    def transform(self, values):
        x = np.asarray(values, dtype=float)
        if x.shape[-1] != 9:
            raise ValueError('Expected the nine predictors in declared feature order')
        return ((np.where(np.isfinite(x), x, self.median)-self.mean)/self.scale).astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, median=self.median, mean=self.mean, scale=self.scale,
                            feature_names=np.asarray(FEATURES))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if z['feature_names'].astype(str).tolist() != list(FEATURES):
                raise ValueError('Preprocessing feature order differs')
            return cls(z['median'], z['mean'], z['scale'])


def prepare_cache(raw_path, dates, stocks, output, fit_start, fit_end,
                  preprocessing=None, fit_preprocessing=False, targets=None):
    """Align unlagged raw daily features to strictly later target dates.

    The input availability archive must still be built independently from raw
    predictors. A finite imputed value is never treated as availability evidence.
    Expanded-universe runs should load the reference preprocessing archive.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    with np.load(raw_path, allow_pickle=False) as z:
        raw_dates = pd.DatetimeIndex(pd.to_datetime(z['dates'])).normalize()
        raw = np.asarray(z['X_raw'], dtype=float)
        raw_stocks = z['stocks'].astype(str).tolist()
        features = z['feature_names'].astype(str).tolist()
    if (raw_stocks != list(stocks) or len(features) != 9 or set(features) != set(FEATURES)
            or raw_dates.has_duplicates or not raw_dates.is_monotonic_increasing
            or raw.shape != (len(raw_dates), len(stocks), 9)):
        raise ValueError('Raw predictor identifiers or dimensions differ')
    raw = raw[:, :, [features.index(f) for f in FEATURES]]
    positions = raw_dates.searchsorted(dates, side='left')-1
    aligned = np.full((len(dates), len(stocks), 9), np.nan, dtype=float)
    for t, p in enumerate(positions):
        if p >= 0 and (t == 0 or raw_dates[p] == dates[t-1]):
            aligned[t] = raw[p]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if fit_preprocessing:
        fit = (dates >= pd.Timestamp(fit_start)) & (dates <= pd.Timestamp(fit_end))
        transform = FrozenTransform.fit(aligned[fit])
        transform.save(output/'preprocessing.npz')
    else:
        if preprocessing is None:
            raise ValueError('Load a frozen reference transform or explicitly request fitting')
        transform = FrozenTransform.load(preprocessing)
    y = np.full((len(dates), len(stocks)), np.nan, dtype=np.float32)
    observed = np.zeros_like(y, dtype=bool)
    if targets is not None:
        with np.load(targets, allow_pickle=False) as z:
            if (not pd.DatetimeIndex(pd.to_datetime(z['dates'])).normalize().equals(dates)
                    or z['stocks'].astype(str).tolist() != list(stocks)):
                raise ValueError('Target identifiers differ')
            y = z['y_np'].astype(np.float32)
            if y.shape != observed.shape:
                raise ValueError('Target array shape differs')
            observed = np.isfinite(y)
    np.savez_compressed(output/'common.npz', X_np=transform.transform(aligned),
                        y_np=y, target_observed_np=observed)
    np.savez_compressed(output/'metadata.npz', all_dates=dates.to_numpy(),
                        ts_codes_order=np.asarray(stocks, dtype=str), feature_names=np.asarray(FEATURES))
    return transform
