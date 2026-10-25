"""Build decision-time availability from pre-label raw predictors, never targets."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import sha256


def cutoff_times(signal_dates, protocol):
    dates = pd.DatetimeIndex(signal_dates).normalize()
    minutes = 14 * 60 + 55 if protocol == 'hybrid1455' else 15 * 60
    return dates + pd.Timedelta(minutes=minutes)


def load_input_mask(path, dates, stocks, protocol, lookback):
    if path is None:
        raise ValueError('Supply a separately constructed input mask; target mask fallback is disabled')
    with np.load(path, allow_pickle=False) as z:
        required = {'input_available_np', 'dates', 'stocks', 'signal_dates', 'input_asof',
                    'protocol', 'basis', 'lookback'}
        if not required.issubset(z.files):
            raise ValueError('Input mask lacks independent as-of provenance fields')
        valid = np.array(z['input_available_np'], copy=True)
        target_dates = pd.DatetimeIndex(pd.to_datetime(z['dates']))
        signal_dates = pd.DatetimeIndex(pd.to_datetime(z['signal_dates']))
        asof = pd.DatetimeIndex(pd.to_datetime(z['input_asof']))
        codes = z['stocks'].astype(str).tolist()
        if str(z['basis'].item()) != 'lagged_raw_predictors_before_label_filter':
            raise ValueError('Input availability must be built independently of target records')
        if str(z['protocol'].item()) != protocol or int(z['lookback'].item()) != lookback:
            raise ValueError('Input mask protocol or lookback does not match the run')
    if valid.dtype != np.bool_ or valid.shape != (len(dates), len(stocks)):
        raise ValueError('input_available_np must be Boolean [T, N]')
    if not target_dates.equals(dates) or codes != list(stocks):
        raise ValueError('Input mask dates or stock order do not match the predictors')
    if len(signal_dates) != len(dates) or len(asof) != len(dates):
        raise ValueError('Input as-of metadata dimensions disagree')
    active = valid.any(axis=1)
    limits = cutoff_times(signal_dates, protocol)
    if (signal_dates[active].isna().any() or asof[active].isna().any()
            or np.any(signal_dates[active] >= dates[active])
            or np.any(asof[active] > limits[active])):
        raise ValueError('Input availability uses information after the decision cutoff')
    return valid, {'input_mask_sha256': sha256(path), 'input_mask_basis':
                   'lagged_raw_predictors_before_label_filter', 'input_protocol': protocol}


def build_input_mask(raw_path, target_dates, stocks, output, lookback=20,
                     protocol='full_day', minimum_observed_features=9, latest_raw=None):
    """Raw NPZ: X_raw[S,N,9], dates[S], stocks[N], optional eligible[S,N].

    X_raw must precede target-based filtering and imputation. Its dates label
    feature days, not prediction targets. A stock needs the requested history;
    the observed-feature minimum is explicit rather than inferred after filling.
    For Hybrid14:55, latest_raw supplies X_latest_raw[T,N,9], dates, stocks and
    asof[T]. Only 19 completed historical steps plus that latest step are tested.
    No target values or target-observation masks are accepted by this builder.
    """
    if protocol not in ('full_day', 'hybrid1455') or lookback < 1:
        raise ValueError('Invalid mask protocol or history length')
    if not 1 <= minimum_observed_features <= 9:
        raise ValueError('minimum_observed_features must be between 1 and 9')
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()
    if target_dates.has_duplicates or not target_dates.is_monotonic_increasing:
        raise ValueError('Target dates must be unique and increasing')
    with np.load(raw_path, allow_pickle=False) as z:
        raw = np.asarray(z['X_raw'])
        feature_dates = pd.DatetimeIndex(pd.to_datetime(z['dates'])).normalize()
        if z['stocks'].astype(str).tolist() != list(stocks):
            raise ValueError('Raw predictor stock order differs')
        eligible = np.array(z['eligible'], dtype=bool) if 'eligible' in z else None
    if raw.shape != (len(feature_dates), len(stocks), 9):
        raise ValueError('Raw predictors must have shape [feature dates, stocks, 9]')
    if feature_dates.has_duplicates or not feature_dates.is_monotonic_increasing:
        raise ValueError('Raw feature dates must be unique and increasing')
    observed = np.isfinite(raw).sum(axis=2) >= minimum_observed_features
    if eligible is not None:
        if eligible.shape != observed.shape:
            raise ValueError('As-of eligibility shape differs')
        observed &= eligible
    latest_ok = None
    if protocol == 'hybrid1455':
        if latest_raw is None:
            raise ValueError('Hybrid mask construction requires raw pre-cutoff latest steps')
        with np.load(latest_raw, allow_pickle=False) as z:
            latest = np.asarray(z['X_latest_raw'])
            if (not pd.DatetimeIndex(pd.to_datetime(z['dates'])).normalize().equals(target_dates)
                    or z['stocks'].astype(str).tolist() != list(stocks)):
                raise ValueError('Hybrid latest identifiers differ')
            latest_asof = pd.DatetimeIndex(pd.to_datetime(z['asof']))
        if latest.shape != (len(target_dates), len(stocks), 9) or len(latest_asof) != len(target_dates):
            raise ValueError('Hybrid raw latest dimensions disagree')
        latest_ok = np.isfinite(latest).sum(axis=2) >= minimum_observed_features
    mask = np.zeros((len(target_dates), len(stocks)), dtype=bool)
    signals = np.full(len(target_dates), np.datetime64('NaT'), dtype='datetime64[ns]')
    asofs = signals.copy()
    for t, target in enumerate(target_dates):
        end = int(feature_dates.searchsorted(target, side='left'))
        if end == 0:
            continue
        signal = feature_dates[end - 1]
        signals[t] = signal.to_datetime64()
        limit = cutoff_times([signal], protocol)[0]
        asofs[t] = limit.to_datetime64()
        if t and signal != target_dates[t - 1]:
            continue  # missing the immediately preceding exchange session
        if end < lookback:
            continue
        if protocol == 'hybrid1455':
            if pd.isna(latest_asof[t]) or latest_asof[t] > limit:
                raise ValueError('Hybrid latest predictors exceed their decision cutoff')
            historical = observed[end - lookback:end - 1].all(axis=0)
            mask[t] = historical & latest_ok[t]
            if eligible is not None:
                mask[t] &= eligible[end - 1]
        else:
            mask[t] = observed[end - lookback:end].all(axis=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, input_available_np=mask, dates=target_dates.to_numpy(),
                        stocks=np.asarray(stocks, dtype=str), signal_dates=signals,
                        input_asof=asofs, protocol=np.array(protocol), lookback=np.array(lookback),
                        basis=np.array('lagged_raw_predictors_before_label_filter'),
                        minimum_observed_features=np.array(minimum_observed_features),
                        raw_sha256=np.array(sha256(raw_path)))
    return mask
