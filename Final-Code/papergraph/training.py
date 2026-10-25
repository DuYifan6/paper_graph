"""Chronological MSE selection; targets never enter forward inference."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import random
import time
import numpy as np
import pandas as pd
import torch
from .config import SCHEMA, write_json
from .data import hf_for_target, split_dates
from .models import Forecaster, GCNForecaster, laplacian_loss, parameter_audit
from .statistics import period_metrics, weighted_period_r2
from .objectives import ranking_loss, weighted_topk_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def checkpoint_update(value, best, epoch=None, state=None):
    if not np.isfinite(value):
        raise ValueError('Nonfinite validation MSE')
    return value < best


def predict_day(model, panel, returns, available, t, config, graph_getter, device):
    """No target values or target-observation flags are read in this function."""
    seq = panel.sequence(t, config.lookback, device)
    input_valid = np.array(panel.input_available[t], copy=True)
    graph = ([graph_getter(i) for i in range(t-config.lookback+1, t+1)]
             if model.variant == 'gcn_lstm' else graph_getter(t))
    hf = None
    if model.variant == '5m_nograph':
        end = t + panel.returns_offset
        flags = available[end-config.lookback:end]
        input_valid &= flags.all(axis=(0, 2)) if flags.ndim == 3 else flags.all(axis=0)
        hf = hf_for_target(returns, available, end, config.lookback, device)
    valid = torch.as_tensor(input_valid, dtype=torch.bool, device=device)
    pred = model(seq, graph, hf, input_available=valid)
    if not torch.isfinite(pred[valid]).all():
        raise FloatingPointError('Nonfinite forecast for an input-available stock')
    return pred, valid, graph


def evaluate(model, panel, returns, available, indices, config, graph_getter,
             device, details=False):
    model.eval()
    rows, detail = [], []
    sse = sy = sy2 = 0.
    nobs = 0
    with torch.no_grad():
        for t in indices:
            t = int(t)
            pred, input_valid, _ = predict_day(model, panel, returns, available, t,
                                              config, graph_getter, device)
            input_numpy = input_valid.cpu().numpy()
            score = panel.supervised_mask(t, input_numpy)
            if int(score.sum()) < config.min_nodes:
                raise ValueError('Too few scoreable targets on an evaluation date')
            actual = panel.y[t, score].astype(np.float64)
            forecast = pred[torch.as_tensor(score, device=device)].cpu().numpy().astype(np.float64)
            loss = float(np.square(actual-forecast).sum())
            n = len(actual)
            sse += loss
            sy += float(actual.sum())
            sy2 += float(actual @ actual)
            nobs += n
            date = str(panel.dates[t].date())
            rows.append({'date': date, 'mse': loss/n, 'sse': loss, 'nobs': n,
                         'predicted_nodes': int(input_numpy.sum()),
                         'sum_y': float(actual.sum()), 'sum_y2': float(actual @ actual),
                         'mask_fingerprint': hashlib.sha256(score.tobytes()).hexdigest(),
                         'target_fingerprint': hashlib.sha256(actual.tobytes()).hexdigest()})
            if details:
                # Preserve forecasts even if the future target is unavailable.
                predictions = pred.cpu().numpy()
                for j in np.flatnonzero(input_numpy):
                    observed = bool(panel.target_observed[t, j] and np.isfinite(panel.y[t, j]))
                    detail.append({'date': date, 'stock': panel.codes[j],
                                   'y_pred': float(predictions[j]), 'input_available': True,
                                   'target_observed': observed,
                                   'y_true': float(panel.y[t, j]) if observed else np.nan})
    if not nobs:
        raise ValueError('Empty supervised evaluation')
    sst = sy2 - sy*sy/nobs
    return {'mse': sse/nobs, 'r2': 1-sse/sst if sst > 0 else None,
            'observations': nobs, 'daily': pd.DataFrame(rows), 'detail': pd.DataFrame(detail)}


def forecast(model, panel, returns, available, indices, config, graph_getter, device):
    """Operational predictions can be produced without any observed targets."""
    model.eval()
    rows = []
    with torch.no_grad():
        for t in indices:
            pred, valid, _ = predict_day(model, panel, returns, available, int(t),
                                         config, graph_getter, device)
            values = pred.cpu().numpy()
            for j in np.flatnonzero(valid.cpu().numpy()):
                rows.append({'date': str(panel.dates[t].date()), 'stock': panel.codes[j],
                             'y_pred': float(values[j]), 'input_available': True})
    return pd.DataFrame(rows, columns=['date', 'stock', 'y_pred', 'input_available'])


def make_model(config, variant):
    if variant == 'gcn_lstm':
        return GCNForecaster(hidden=config.hidden, dropout=config.dropout)
    return Forecaster(hidden=config.hidden, variant=variant, dropout=config.dropout)


def fit_one(panel, returns, available, config, variant, seed, graph_getter, out,
            device='cpu', save_details=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'checkpoint.pt').exists() or (out/'summary.json').exists():
        raise FileExistsError('Existing completed fit is preserved; choose another output directory')
    split = split_dates(panel, config)
    set_seed(seed)
    device = torch.device(device)
    model = make_model(config, variant).to(device)
    first = int(split['fit'][0])
    seq = panel.sequence(first, config.lookback, device)
    audit_input = torch.as_tensor(panel.input_available[first], device=device)
    hf = None
    if variant == '5m_nograph':
        end = first + panel.returns_offset
        flags = available[end-config.lookback:end]
        ok = flags.all(axis=(0, 2)) if flags.ndim == 3 else flags.all(axis=0)
        audit_input = audit_input & torch.as_tensor(ok, device=device)
        hf = hf_for_target(returns, available, end, config.lookback, device)
    audit_graph = ([graph_getter(i) for i in range(first-config.lookback+1, first+1)]
                   if variant == 'gcn_lstm' else graph_getter(first))
    audit = parameter_audit(model, seq, audit_graph, hf, input_available=audit_input)
    set_seed(seed)
    optimizer_type = torch.optim.AdamW if variant == 'gcn_lstm' else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    best, best_state, best_epoch, bad = float('inf'), None, 0, 0
    history = []
    start = time.perf_counter()
    uses_lap = variant in ('graph1', 'graph2', 'graph3', 'random')
    for epoch in range(1, config.epochs+1):
        model.train()
        epoch_sse = lap_total = 0.
        nobs = 0
        for t in split['fit']:
            t = int(t)
            optimizer.zero_grad(set_to_none=True)
            pred, input_valid, graph = predict_day(model, panel, returns, available, t,
                                                  config, graph_getter, device)
            score = torch.as_tensor(panel.supervised_mask(t, input_valid.cpu().numpy()),
                                    dtype=torch.bool, device=device)
            if int(score.sum()) < config.min_nodes:
                raise ValueError('Too few fitting targets; review the independent input mask')
            target = torch.as_tensor(panel.y[t], device=device)
            mse = (pred[score]-target[score]).square().mean()
            lap = laplacian_loss(pred, graph, score) if uses_lap else pred.sum()*0
            rank = ranking_loss(pred[score],target[score]) if config.ranking else pred.sum()*0
            loss = mse + config.laplacian*lap + config.ranking*rank
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            n = int(score.sum())
            epoch_sse += float(mse.detach())*n
            nobs += n
            lap_total += float(lap.detach())
        validation = evaluate(model, panel, returns, available, split['validation'],
                              config, graph_getter, device,config.selection=='weighted_topk')
        value = (validation['mse'] if config.selection=='mse'
                 else -weighted_topk_score(validation['detail']))
        history.append({'epoch': epoch, 'training_mse': epoch_sse/nobs,
                        'laplacian_sum': lap_total, 'validation_mse': validation['mse'],
                        'validation_selection_value': value})
        if checkpoint_update(value, best):
            best, best_epoch, bad = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= config.patience:
            break
    if best_state is None:
        raise RuntimeError('No checkpoint selected')
    model.load_state_dict(best_state)
    validation_final = evaluate(model, panel, returns, available, split['validation'],
                                config, graph_getter, device, save_details)
    test = evaluate(model, panel, returns, available, split['test'], config,
                    graph_getter, device, save_details)
    development = evaluate(model, panel, returns, available,
                           np.r_[split['fit'], split['validation']], config,
                           graph_getter, device, save_details)
    monthly = period_metrics(test['daily'], 'M')
    quarterly = period_metrics(development['daily'], 'Q')
    pd.DataFrame(history).to_csv(out/'training.csv', index=False)
    test['daily'].to_csv(out/'daily_test.csv', index=False)
    development['daily'].to_csv(out/'daily_development.csv', index=False)
    monthly.to_csv(out/'test_monthly.csv', index=False)
    quarterly.to_csv(out/'development_quarterly.csv', index=False)
    if save_details:
        for label, result in [('test', test), ('validation', validation_final),
                              ('development', development)]:
            result['detail'].to_csv(out/f'predictions_{label}.csv', index=False)
    if config.calibration_start is not None:
        if not (pd.Timestamp(config.validation_end)<pd.Timestamp(config.calibration_start)
                <=pd.Timestamp(config.calibration_end)<pd.Timestamp(config.test_start)):
            raise ValueError('Supplementary calibration must follow validation and precede testing')
        indices=np.flatnonzero((panel.dates>=config.calibration_start)&
                               (panel.dates<=config.calibration_end))
        calibration=evaluate(model,panel,returns,available,indices,config,graph_getter,device,True)
        calibration['detail'].to_csv(out/'predictions_calibration.csv',index=False)
    torch.save({'state_dict': best_state, 'schema': SCHEMA, 'config': asdict(config),
                'variant': variant, 'seed': seed, 'data_identity': panel.identity,
                'graph_run': getattr(graph_getter, 'run_identity', {})}, out/'checkpoint.pt')
    summary = {'schema': SCHEMA, 'variant': variant, 'model_seed': seed,
               'data_identity': panel.identity, 'config': asdict(config), 'parameters': audit,
               'best_epoch': best_epoch, 'best_validation_mse': validation_final['mse'],
               'best_validation_selection_value': best,
               'test_r2': test['r2'], 'test_mse': test['mse'],
               'test_observations': test['observations'], 'test_days': len(test['daily']),
               'test_weighted_monthly_r2': weighted_period_r2(monthly),
               'development_weighted_quarterly_r2': weighted_period_r2(quarterly),
               'runtime_seconds': time.perf_counter()-start,
               'selection': 'minimum pooled validation MSE' if config.selection=='mse' else 'maximum weighted validation Top-K return',
               'objective': 'MSE' + (' + ranking' if config.ranking else '') + (' + normalized Laplacian' if uses_lap else ''),
               'ranking_loss': bool(config.ranking), 'forward_mask': 'decision-time input availability only',
               'printed_manuscript_results_reproduced': False}
    write_json(out/'summary.json', summary)
    return summary, test['daily']
