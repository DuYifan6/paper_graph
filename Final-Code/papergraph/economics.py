"""Pre-2025 isotonic calibration and conditional price-proxy portfolios.

    Execution CSV columns: date, stock, entry_time, exit_time, entry_price,
    exit_price, entry_executable. Prices and timestamps describe actual assumed
    fills; they never determine signal ranks. Optional entry_capacity_shares is
    explicitly in SHARES. Risk CSV: date, market_overnight_vol.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import write_json


def hurdle(buy=.0001, sell=.0011, slippage=.0010):
    if min(buy, sell, slippage) < 0 or sell + slippage >= 1:
        raise ValueError('Invalid costs')
    return (1+buy)*(1+slippage)/(1-sell-slippage)-1


def calibrate_file(predictions, output, fraction=.20, target_scale=.01):
    """Input y_true/y_pred follow Eq. (1), in percentage points.

    Only the calibration target is converted to decimal returns. The forecast
    remains on its original scale at the isotonic map's input.
    """
    from sklearn.isotonic import IsotonicRegression
    if predictions is None or not 0 < fraction <= 1 or target_scale <= 0:
        raise ValueError('Supply calibration predictions and valid scale/fraction')
    frame = pd.read_csv(predictions, dtype={'stock': str})
    if not {'date', 'stock', 'y_true', 'y_pred'}.issubset(frame):
        raise ValueError('Calibration needs dated, stock-level forecasts and realized targets')
    if frame.duplicated(['date', 'stock']).any():
        raise ValueError('Duplicate calibration stock-days')
    frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
    if (frame.date >= pd.Timestamp('2025-01-01')).any():
        raise ValueError('Calibration input must contain pre-2025 predictions only')
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    count = max(1, int(np.ceil(len(dates)*fraction)))
    if not len(dates):
        raise ValueError('Empty pre-2025 prediction dates')
    used_dates = dates[-count:]
    subset = frame[frame.date.isin(used_dates)]
    subset = subset[np.isfinite(subset.y_pred) & np.isfinite(subset.y_true)]
    if len(subset) < 2 or subset.y_pred.nunique() < 2:
        raise ValueError('Insufficient calibration variation')
    model = IsotonicRegression(increasing=True, out_of_bounds='clip')
    model.fit(subset.y_pred.to_numpy(), subset.y_true.to_numpy()*target_scale)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'calibration.npz', x=model.X_thresholds_, y=model.y_thresholds_,
                        output_unit=np.array('decimal_return'), target_scale=np.array(target_scale),
                        end=np.array(str(used_dates[-1].date())))
    metadata = {'start': str(used_dates[0].date()), 'end': str(used_dates[-1].date()),
                'dates': count, 'stock_days': len(subset), 'fraction': fraction,
                'input_unit': 'percentage_points', 'output_unit': 'decimal_return',
                'target_conversion': target_scale, 'hurdle': hurdle(),
                'prediction_provenance': 'caller-supplied selected-checkpoint predictions',
                'independence_from_checkpoint_selection': 'not established by this file'}
    write_json(output/'calibration_audit.json', metadata)
    return metadata


def calibrated_returns(values, calibration):
    with np.load(calibration, allow_pickle=False) as z:
        if str(z['output_unit'].item()) != 'decimal_return':
            raise ValueError('Calibration output must match the hurdle decimal units')
        if pd.Timestamp(str(z['end'].item())) >= pd.Timestamp('2025-01-01'):
            raise ValueError('Calibration is not frozen before evaluation')
        return np.interp(np.asarray(values), z['x'], z['y'])


def risk_exposure(risk):
    if not {'date', 'market_overnight_vol'}.issubset(risk):
        raise ValueError('Supply date and market_overnight_vol')
    risk = risk.copy()
    risk['date'] = pd.to_datetime(risk.date).dt.normalize()
    if risk.date.duplicated().any():
        raise ValueError('Duplicate risk dates')
    risk = risk.sort_values('date').set_index('date')
    if (not np.isfinite(risk.market_overnight_vol).all()
            or (risk.market_overnight_vol < 0).any()):
        raise ValueError('Invalid market volatility')
    trailing = risk.market_overnight_vol.rolling(20, min_periods=20).mean().shift(1)
    baseline = float(trailing[trailing.index < '2025-01-01'].mean())
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('Need a positive pre-2025 volatility reference')
    return (baseline/trailing.clip(lower=1e-12)).clip(upper=1)


def conditional_portfolio(predictions, executions, exposures, k, slippage_bp=0,
                          initial_nav=100000., capacity_fraction=None, round_lots=False):
    """Cash ledger with fixed ranks and weights, no replacement or renormalizing.

    A filled position requires an exit price and time. Delayed proceeds cannot
    be reused before that time. Unsettled positions are valued at entry price
    between exits; this book-value convention is explicit, not an execution claim.
    """
    if k < 1 or slippage_bp < 0 or initial_nav <= 0:
        raise ValueError('Invalid portfolio settings')
    if capacity_fraction is not None and not 0 < capacity_fraction <= 1:
        raise ValueError('Invalid capacity fraction')
    pred = predictions.copy()
    exe = executions.copy()
    pred['date'] = pd.to_datetime(pred.date).dt.normalize()
    exe['date'] = pd.to_datetime(exe.date).dt.normalize()
    if pred.duplicated(['date', 'stock']).any() or exe.duplicated(['date', 'stock']).any():
        raise ValueError('Duplicate prediction or execution stock-day')
    for name in ('entry_time', 'exit_time'):
        exe[name] = pd.to_datetime(exe[name], errors='coerce')
    if not exe.entry_executable.isin([True, False, 0, 1]).all():
        raise ValueError('entry_executable must be a Boolean or 0/1, not a string')
    exe = exe.set_index(['date', 'stock'])
    cash, positions, rows, trades = float(initial_nav), [], [], []
    slip = slippage_bp/10000
    entry_factor = (1+.0001)*(1+slip)
    exit_factor = 1-.0011-slip

    def settle(timestamp):
        nonlocal cash, positions
        remaining = []
        for p in positions:
            if p['exit_time'] <= timestamp:
                cash += p['quantity']*p['exit_price']*exit_factor
            else:
                remaining.append(p)
        positions = remaining

    for date, frame in pred.groupby('date', sort=True):
        try:
            candidates = exe.loc[date]
        except KeyError:
            raise ValueError('Execution calendar missing an evaluation date') from None
        entry_days = pd.DatetimeIndex(candidates.entry_time.dropna()).normalize().unique()
        if len(entry_days) != 1 or entry_days[0] >= date:
            raise ValueError('Entry times must identify the preceding decision day')
        decision = entry_days[0] + pd.Timedelta(hours=14, minutes=55)
        settle(decision)
        nav = cash + sum(p['quantity']*p['entry_price'] for p in positions)
        if date not in exposures.index or not np.isfinite(exposures.loc[date]):
            raise ValueError('Missing strictly lagged risk exposure')
        selected = frame[np.isfinite(frame.y_pred) & (frame.u >= hurdle())].sort_values(
            ['y_pred', 'stock'], ascending=[False, True], kind='stable').head(k)
        weights = 1/np.arange(1, len(selected)+1, dtype=float)
        if len(weights):
            weights = weights/weights.sum()*float(exposures.loc[date])
        if k == 1 and positions:
            selected = selected.iloc[:0]  # locked Top-1 capital: skip, never replace
        for (_, signal), weight in zip(selected.iterrows(), weights):
            key = (date, signal.stock)
            if key not in exe.index:
                continue  # the submitted slot stays in cash
            fill = exe.loc[key]
            if not bool(fill.entry_executable) or not np.isfinite(fill.entry_price) or fill.entry_price <= 0:
                continue
            if (pd.isna(fill.entry_time) or fill.entry_time < decision
                    or pd.isna(fill.exit_time) or fill.exit_time <= fill.entry_time
                    or not np.isfinite(fill.exit_price) or fill.exit_price <= 0):
                raise ValueError('Filled order has an unresolved entry/exit; no loss is discarded')
            spend = min(nav*weight, cash)
            quantity = spend/(fill.entry_price*entry_factor)
            if capacity_fraction is not None:
                if 'entry_capacity_shares' not in fill or not np.isfinite(fill.entry_capacity_shares):
                    raise ValueError('Capacity cap requires entry_capacity_shares in explicit share units')
                quantity = min(quantity, capacity_fraction*max(float(fill.entry_capacity_shares), 0))
            if round_lots:
                quantity = np.floor(quantity/100)*100
            if quantity <= 0:
                continue
            cash -= quantity*fill.entry_price*entry_factor
            position = {'stock': signal.stock, 'quantity': quantity,
                        'entry_price': float(fill.entry_price), 'exit_price': float(fill.exit_price),
                        'exit_time': fill.exit_time}
            positions.append(position)
            trades.append({'date': str(date.date()), 'stock': signal.stock, 'submitted_weight': weight,
                           'quantity_shares': quantity, 'entry_price': float(fill.entry_price),
                           'exit_price': float(fill.exit_price), 'entry_time': str(fill.entry_time),
                           'exit_time': str(fill.exit_time)})
        settle(date+pd.Timedelta(hours=9, minutes=30))
        value = cash+sum(p['quantity']*p['entry_price'] for p in positions)
        rows.append({'date': str(date.date()), 'nav': value, 'cash': cash,
                     'unsettled_positions': len(positions)})
    if positions:
        raise ValueError('Positions remain open at the evaluation endpoint; specify terminal treatment')
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No portfolio evaluation dates')
    frame['return'] = frame.nav.pct_change()
    frame.loc[0, 'return'] = frame.loc[0, 'nav']/initial_nav-1
    return frame, pd.DataFrame(trades)


def backtest_file(predictions, calibration, execution, output, ks, slippages,
                  risk=None, capacity_fraction=None, round_lots=False):
    if any(x is None for x in (predictions, calibration, execution, risk)):
        raise ValueError('Supply predictions, frozen calibration, execution records and risk CSV')
    pred = pd.read_csv(predictions, dtype={'stock': str})
    if not {'date', 'stock', 'y_pred'}.issubset(pred):
        raise ValueError('Predictions need date, stock and y_pred')
    pred['u'] = calibrated_returns(pred.y_pred, calibration)
    executions = pd.read_csv(execution, dtype={'stock': str})
    required = {'date', 'stock', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'entry_executable'}
    if not required.issubset(executions):
        raise ValueError('Execution table lacks the required prices, fill flags or timestamps')
    exposures = risk_exposure(pd.read_csv(risk))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for k in ks:
        for slip in slippages:
            daily, trades = conditional_portfolio(pred, executions, exposures, k, slip,
                                                  capacity_fraction=capacity_fraction,
                                                  round_lots=round_lots)
            label = f'K{k}_slippage{slip:g}'
            daily.to_csv(output/(label+'_daily.csv'), index=False)
            trades.to_csv(output/(label+'_trades.csv'), index=False)
            summaries.append({'K': k, 'slippage_bp': slip, 'cumulative_return':
                              float(daily.nav.iloc[-1]/100000-1),
                              'capacity_fraction': capacity_fraction,
                              'round_lots': round_lots, 'status': 'conditional_price_proxy_simulation'})
    pd.DataFrame(summaries).to_csv(output/'portfolio_summary.csv', index=False)
    return summaries
