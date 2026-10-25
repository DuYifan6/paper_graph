"""Frozen predictor loading with separate input and target availability."""
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
from .config import FEATURES, sha256
from .masks import load_input_mask


@dataclass
class Panel:
    X: np.ndarray
    y: np.ndarray
    target_observed: np.ndarray
    input_available: np.ndarray
    dates: pd.DatetimeIndex
    codes: list
    identity: dict
    train_start: int = 0
    latest: np.ndarray | None = None
    returns_offset: int = 0

    def validate(self):
        if self.X.ndim != 3:
            raise ValueError('Expected X[T,N,9]')
        t, n, f = self.X.shape
        if f != 9 or self.y.shape != (t, n):
            raise ValueError('Predictor/target dimensions disagree')
        for name in ('target_observed', 'input_available'):
            a = getattr(self, name)
            if a.dtype != np.bool_ or a.shape != (t, n):
                raise ValueError(name + ' must be Boolean [T,N]')
        if len(self.dates) != t or len(self.codes) != n or len(set(self.codes)) != n:
            raise ValueError('Panel identifiers disagree')
        if self.dates.has_duplicates or not self.dates.is_monotonic_increasing:
            raise ValueError('Panel dates must be unique and increasing')
        if not np.isfinite(self.X).all():
            raise ValueError('Supply a predictor tensor transformed by the frozen imputer/scaler')
        if not np.isfinite(self.y[self.target_observed]).all():
            raise ValueError('A target marked observed is nonfinite')
        if self.latest is not None and (self.latest.shape != self.X.shape
                                       or not np.isfinite(self.latest).all()):
            raise ValueError('Hybrid latest tensor must be finite and aligned')
        return self

    def sequence(self, t, lookback, device):
        if t < lookback - 1:
            raise ValueError('Insufficient temporal history')
        x = np.array(self.X[t-lookback+1:t+1], dtype=np.float32, copy=True)
        if self.latest is not None:
            x[-1] = self.latest[t]
        return torch.as_tensor(x.transpose(1, 0, 2), device=device)

    def supervised_mask(self, t, input_available=None):
        available = self.input_available[t] if input_available is None else input_available
        return available & self.target_observed[t] & np.isfinite(self.y[t])


def read_metadata(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].tolist() for k in z.files}
    if path.suffix.lower() in ('.pkl', '.pickle'):
        # Compatibility for the author's trusted metadata only; pickle executes code.
        with path.open('rb') as f:
            return pickle.load(f)
    raise ValueError('Use JSON/NPZ metadata, or a trusted original pickle')


def load_panel(cache, meta, hybrid=None, *, input_mask, protocol='full_day', lookback=20):
    m = read_metadata(meta)
    with np.load(cache, allow_pickle=False) as z:
        X = z['X_np'].astype(np.float32)
        y = (z['y_np'].astype(np.float32) if 'y_np' in z
             else np.full(X.shape[:2], np.nan, dtype=np.float32))
        key = 'target_observed_np' if 'target_observed_np' in z else 'mask_np'
        observed = np.array(z[key], copy=True) if key in z else np.isfinite(y)
    if observed.dtype != np.bool_:
        raise ValueError('Target observation flags must be Boolean')
    dates = pd.DatetimeIndex(pd.to_datetime(m['all_dates'])).normalize()
    codes = [str(c) for c in m['ts_codes_order']]
    features = m.get('feature_names', m.get('selected_features'))
    if features is None or len(features) != 9 or set(features) != set(FEATURES):
        raise ValueError('Metadata must declare the nine named predictors and their stored order')
    order = [list(features).index(name) for name in FEATURES]
    X = X[:, :, order]
    available, availability_identity = load_input_mask(input_mask, dates, codes, protocol, lookback)
    identity = {'cache_sha256': sha256(cache), 'metadata_sha256': sha256(meta),
                'dates': [str(d.date()) for d in dates], 'stocks': codes,
                'features': list(FEATURES), 'preprocessing': 'loaded frozen; not re-estimated',
                **availability_identity}
    latest = None
    if protocol == 'hybrid1455':
        if hybrid is None:
            raise ValueError('Supply the frozen pre-cutoff latest-step tensor')
        with np.load(hybrid, allow_pickle=False) as z:
            latest_order = order
            if 'feature_names' in z:
                latest_features=z['feature_names'].astype(str).tolist()
                if len(latest_features)!=9 or set(latest_features)!=set(FEATURES):
                    raise ValueError('Hybrid latest feature declarations differ')
                latest_order=[latest_features.index(name) for name in FEATURES]
            latest = z['X_latest_hybrid_np'].astype(np.float32)[:, :, latest_order]
            if 'dates' in z and not pd.DatetimeIndex(pd.to_datetime(z['dates'])).normalize().equals(dates):
                raise ValueError('Hybrid latest dates differ')
            if 'stocks' in z and z['stocks'].astype(str).tolist() != codes:
                raise ValueError('Hybrid latest stock order differs')
        # mask_hybrid_np, when present in old files, is never a forward mask.
        identity['hybrid_sha256'] = sha256(hybrid)
    elif hybrid is not None:
        raise ValueError('Use hybrid1455 protocol with a latest-step tensor')
    return Panel(X, y, observed, available, dates, codes, identity,
                 int(m.get('train_start_idx', 0)), latest).validate()


def split_dates(panel, config):
    d = panel.dates
    if not (pd.Timestamp(config.fit_start) <= pd.Timestamp(config.fit_end)
            < pd.Timestamp(config.validation_start) <= pd.Timestamp(config.validation_end)
            < pd.Timestamp(config.test_start) <= pd.Timestamp(config.test_end)):
        raise ValueError('Chronological split overlaps')
    usable = np.arange(len(d)) >= max(config.lookback - 1,
                                     config.window - panel.returns_offset, panel.train_start)
    split = {'fit': np.flatnonzero(usable & (d >= config.fit_start) & (d <= config.fit_end)),
             'validation': np.flatnonzero(usable & (d >= config.validation_start)
                                         & (d <= config.validation_end)),
             'test': np.flatnonzero(usable & (d >= config.test_start) & (d <= config.test_end))}
    if any(len(v) == 0 for v in split.values()):
        raise ValueError('Empty fitting/validation/test date split')
    if config.expected_test_days and len(split['test']) != config.expected_test_days:
        raise ValueError('The evaluation date count differs from the configured protocol')
    return split


def load_returns(path, panel):
    with np.load(path, allow_pickle=False) as z:
        r = z['returns'].astype(np.float32)
        key = 'bar_available' if 'bar_available' in z else 'available'
        valid = np.array(z[key], copy=True)
        dates = pd.DatetimeIndex(pd.to_datetime(z['dates']))
        codes = z['stocks'].astype(str).tolist()
    offset = len(dates) - len(panel.dates)
    if (offset < 0 or r.shape != (len(dates), len(panel.codes), 48)
            or valid.dtype != np.bool_ or valid.shape not in (r.shape, r.shape[:2])):
        raise ValueError('Return arrays must provide 48 returns and independent availability')
    if (dates.has_duplicates or not dates.is_monotonic_increasing
            or not dates[offset:].equals(panel.dates) or codes != panel.codes):
        raise ValueError('Return identifiers differ; only preceding extra dates are permitted')
    if panel.latest is not None and valid.ndim != 3:
        raise ValueError('Hybrid14:55 requires per-return availability, not full-day flags')
    if not np.isfinite(r[valid]).all():
        raise ValueError('An available return is nonfinite')
    panel.returns_offset = offset
    panel.identity['pre_panel_return_dates'] = offset
    return r, valid


def build_returns(minute_dir, panel, output, prehistory_dates=()):
    """Fixed 49-price grid; per-return flags keep the post-14:55 bar separate."""
    directory = Path(minute_dir)
    dates = pd.DatetimeIndex(pd.to_datetime(list(prehistory_dates))).append(panel.dates)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Prehistory must contain unique sessions before the panel')
    r = np.zeros((len(dates), len(panel.codes), 48), dtype=np.float32)
    valid = np.zeros_like(r, dtype=bool)
    for j, code in enumerate(panel.codes):
        found = [directory / f'{code}{s}' for s in ('_min.csv', '_5m.csv', '_5分钟.csv', '.csv')]
        found = [p for p in found if p.is_file()]
        if not found:
            continue
        if len(found) != 1:
            raise ValueError('Ambiguous intraday source: use one file per security')
        frame = pd.read_csv(found[0], usecols=['trade_time', 'close'])
        frame['trade_time'] = pd.to_datetime(frame['trade_time'], errors='raise')
        frame['close'] = pd.to_numeric(frame['close'], errors='coerce')
        if frame['trade_time'].duplicated().any():
            raise ValueError('Duplicate intraday timestamps')
        prices = frame.set_index('trade_time')['close']
        for t, day in enumerate(dates):
            text = str(day.date())
            grid = pd.date_range(text + ' 09:30', text + ' 11:30', freq='5min').append(
                pd.date_range(text + ' 13:05', text + ' 15:00', freq='5min'))
            values = prices.reindex(grid).to_numpy(dtype=float)
            ok = np.isfinite(values) & (values > 0)
            pair = ok[:-1] & ok[1:]
            safe = np.where(ok, values, 1.)
            returns = np.clip(np.diff(np.log(safe)), -.08, .08)
            r[t, j] = np.where(pair, returns, 0.)
            valid[t, j] = pair
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, returns=r, bar_available=valid,
                        dates=dates.to_numpy(), stocks=np.asarray(panel.codes, dtype=str))


def hf_for_target(returns, available, t, lookback, device):
    if t < lookback:
        raise ValueError('Insufficient self-only history')
    a = np.array(returns[t-lookback:t], dtype=np.float32, copy=True).transpose(1, 0, 2)
    flags = available[t-lookback:t]
    ok = flags.all(axis=(0, 2)) if flags.ndim == 3 else flags.all(axis=0)
    flat = np.clip(a.reshape(len(a), -1), -.08, .08)
    flat = np.clip((flat - flat.mean(1, keepdims=True)) /
                   (flat.std(1, keepdims=True) + 1e-6), -8., 8.)
    flat[~ok] = 0
    return torch.as_tensor(flat.reshape(a.shape), device=device)
