"""Ridge, LightGBM and Transformer on the common lagged predictor tensor."""
from pathlib import Path
import math
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .config import write_json
from .data import split_dates
from .training import set_seed


class TransformerRegressor(nn.Module):
    def __init__(self, features=9, hidden=128, lookback=20):
        super().__init__()
        self.project = nn.Linear(features, hidden)
        position = torch.arange(lookback, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, hidden, 2)*(-math.log(10000.)/hidden))
        encoding = torch.zeros(lookback, hidden)
        encoding[:, 0::2] = torch.sin(position*scale)
        encoding[:, 1::2] = torch.cos(position*scale[:encoding[:, 1::2].shape[1]])
        self.register_buffer('encoding', encoding[None])
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden*4, .1,
                                            activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        z = self.encoder(self.project(x)+self.encoding[:, :x.shape[1]])
        return self.head(z[:, -1]).squeeze(-1)


def collect(panel, indices, lookback, output, name, supervised):
    masks = [(int(t), panel.supervised_mask(int(t)) if supervised
              else panel.input_available[int(t)]) for t in indices]
    count = sum(int(m.sum()) for _, m in masks)
    if not count:
        raise ValueError('Empty benchmark split')
    x = np.lib.format.open_memmap(output/(name+'_X.npy'), mode='w+', dtype=np.float32,
                                 shape=(count, lookback, 9))
    y = np.lib.format.open_memmap(output/(name+'_y.npy'), mode='w+', dtype=np.float32,
                                 shape=(count,))
    ids = []
    start = 0
    for t, mask in masks:
        n = int(mask.sum())
        x[start:start+n] = panel.sequence(t, lookback, 'cpu').numpy()[mask]
        y[start:start+n] = np.where(panel.target_observed[t, mask], panel.y[t, mask], np.nan)
        ids.extend((str(panel.dates[t].date()), panel.codes[j]) for j in np.flatnonzero(mask))
        start += n
    x.flush()
    y.flush()
    return x, y, ids


def batched_predict(model, x, device, batch=4096):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            values = torch.as_tensor(np.array(x[start:start+batch], copy=True), device=device)
            output.append(model(values).cpu().numpy())
    return np.concatenate(output)


def run_benchmarks(panel, returns, available, config, output, device='cpu', seed=0,
                   models=('ridge', 'lightgbm', 'transformer')):
    if not set(models).issubset({'ridge', 'lightgbm', 'transformer'}):
        raise ValueError('Unknown external benchmark')
    if 'lightgbm' in models:
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise ImportError('LightGBM benchmark requires the lightgbm package') from error
    output = Path(output)/'benchmarks'
    output.mkdir(parents=True, exist_ok=True)
    split = split_dates(panel, config)
    xf, yf, _ = collect(panel, split['fit'], config.lookback, output, 'fit', True)
    xv, yv, _ = collect(panel, split['validation'], config.lookback, output, 'validation', True)
    xt, yt, ids = collect(panel, split['test'], config.lookback, output, 'test', False)
    summary = []
    for name in models:
        destination = output/name
        destination.mkdir(exist_ok=True)
        if (destination/'summary.json').exists():
            raise FileExistsError('Existing benchmark result is preserved')
        set_seed(seed)
        if name == 'ridge':
            from sklearn.linear_model import Ridge
            best, selected = float('inf'), None
            for alpha in (.1, 1., 10., 100.):
                model = Ridge(alpha=alpha, solver='lsqr')
                model.fit(xf.reshape(len(xf), -1), yf)
                value = float(np.square(model.predict(xv.reshape(len(xv), -1))-yv).mean())
                if value < best:
                    best, selected, chosen = value, model, {'alpha': alpha}
            prediction = selected.predict(xt.reshape(len(xt), -1))
            with (destination/'model.pkl').open('wb') as f:
                pickle.dump(selected, f)
        elif name == 'lightgbm':
            best, selected = float('inf'), None
            for leaves, depth, child in ((31, -1, 20), (63, -1, 20), (31, 8, 50), (63, 8, 50)):
                model = lgb.LGBMRegressor(objective='regression', n_estimators=1200,
                    learning_rate=.03, num_leaves=leaves, max_depth=depth,
                    min_child_samples=child, subsample=.8, colsample_bytree=.8,
                    reg_alpha=0., reg_lambda=1., random_state=seed, verbosity=-1)
                model.fit(xf.reshape(len(xf), -1), yf,
                          eval_set=[(xv.reshape(len(xv), -1), yv)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
                value = float(np.square(model.predict(xv.reshape(len(xv), -1))-yv).mean())
                if value < best:
                    best, selected = value, model
                    chosen = {'num_leaves': leaves, 'max_depth': depth,
                              'min_child_samples': child, 'iteration': model.best_iteration_}
            prediction = selected.predict(xt.reshape(len(xt), -1))
            selected.booster_.save_model(str(destination/'model.txt'))
        else:
            model = TransformerRegressor(hidden=config.hidden, lookback=config.lookback).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
            loader = DataLoader(TensorDataset(torch.from_numpy(xf), torch.from_numpy(yf)),
                                batch_size=4096, shuffle=True)
            best, state, bad = float('inf'), None, 0
            for epoch in range(1, config.epochs+1):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(xb)-yb).square().mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()
                value = float(np.square(batched_predict(model, xv, device)-yv).mean())
                if value < best:
                    best, bad = value, 0
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    chosen = {'epoch': epoch}
                else:
                    bad += 1
                if bad >= config.patience:
                    break
            model.load_state_dict(state)
            prediction = batched_predict(model, xt, device)
            torch.save(state, destination/'state_dict.pt')
        valid = np.isfinite(yt)
        error = np.square(np.asarray(yt[valid], dtype=float)-prediction[valid])
        denominator = float(np.square(yt[valid]-yt[valid].mean()).sum())
        metrics = {'model': name, 'validation_mse': best,
                   'test_mse': float(error.mean()), 'test_r2':
                   1-float(error.sum())/denominator if denominator > 0 else None,
                   'selected': chosen, 'test_observations': int(valid.sum()),
                   'selection': 'minimum pooled validation MSE'}
        detail = pd.DataFrame(ids, columns=['date', 'stock'])
        detail['y_pred'] = prediction
        detail['y_true'] = yt
        detail['target_observed'] = valid
        detail.to_csv(destination/'predictions_test.csv', index=False)
        write_json(destination/'summary.json', metrics)
        summary.append(metrics)
    return summary
