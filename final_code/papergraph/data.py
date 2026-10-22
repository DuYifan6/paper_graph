from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from .config import FEATURES, sha256, write_json

@dataclass
class Panel:
    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    dates: pd.DatetimeIndex
    codes: list
    identity: dict
    train_start: int = 0
    latest: np.ndarray | None = None
    returns_offset: int = 0
    def validate(self):
        t,n,f=self.X.shape
        if f!=9 or self.y.shape!=(t,n) or self.mask.shape!=(t,n):raise ValueError('Expected X[T,N,9], y/mask[T,N]')
        if len(self.dates)!=t or len(self.codes)!=n or len(set(self.codes))!=n:raise ValueError('Panel identifiers disagree')
        if self.dates.has_duplicates or not self.dates.is_monotonic_increasing:raise ValueError('Dates not unique and increasing')
        if not np.isfinite(self.X).all():raise ValueError('Supply already imputed/scaled frozen predictors')
        if not np.isfinite(self.y[self.mask]).all():raise ValueError('Nonfinite target marked valid')
        if self.latest is not None and (self.latest.shape!=self.X.shape or not np.isfinite(self.latest).all()):
            raise ValueError('Hybrid latest tensor invalid')
        return self
    def sequence(self,t,lookback,device):
        if t<lookback-1:raise ValueError('Insufficient temporal history')
        x=np.array(self.X[t-lookback+1:t+1],dtype=np.float32,copy=True)
        if self.latest is not None:x[-1]=self.latest[t]
        return torch.as_tensor(x.transpose(1,0,2),device=device)

def load_panel(cache,meta,hybrid=None):
    # Metadata must be the author's trusted local pickle, never an untrusted download.
    with Path(meta).open('rb') as f:m=pickle.load(f)
    with np.load(cache,allow_pickle=False) as z:
        X=z['X_np'].astype(np.float32);y=z['y_np'].astype(np.float32);mask=z['mask_np'].astype(bool)
    dates=pd.DatetimeIndex(pd.to_datetime(m['all_dates'])).normalize()
    codes=[str(c) for c in m['ts_codes_order']]
    features=m.get('feature_names',m.get('selected_features'))
    order=None
    if features is not None:
        if len(features)!=9 or set(features)!=set(FEATURES):raise ValueError('Feature set differs from paper')
        order=[list(features).index(name) for name in FEATURES]
        X=X[:,:,order]
    identity={'cache_sha256':sha256(cache),'metadata_sha256':sha256(meta),
              'dates':[str(d.date()) for d in dates],'stocks':codes,
              'feature_order_verified':features is not None,
              'preprocessing':'loaded unchanged; not re-fit by this package'}
    latest=None
    if hybrid:
        with np.load(hybrid,allow_pickle=False) as z:
            latest=z['X_latest_hybrid_np'].astype(np.float32)
            hm=z['mask_hybrid_np'].astype(bool)
            if hm.shape!=mask.shape:raise ValueError('Hybrid mask shape differs')
            if np.any(hm & ~mask):raise ValueError('Hybrid mask adds unavailable targets')
            mask &= hm
        identity['hybrid_sha256']=sha256(hybrid)
        if order is not None:latest=latest[:,:,order]
    return Panel(X,y,mask,dates,codes,identity,int(m.get('train_start_idx',0)),latest).validate()

def split_dates(panel,config):
    d=panel.dates
    if not (pd.Timestamp(config.fit_end)<pd.Timestamp(config.validation_start)
            <=pd.Timestamp(config.validation_end)<pd.Timestamp(config.test_start)<=pd.Timestamp(config.test_end)):
        raise ValueError('Chronological split overlaps')
    usable=np.arange(len(d))>=max(config.lookback-1,config.window-panel.returns_offset,panel.train_start)
    splits={'fit':np.flatnonzero(usable & (d<=config.fit_end)),
            'validation':np.flatnonzero(usable & (d>=config.validation_start)&(d<=config.validation_end)),
            'test':np.flatnonzero(usable & (d>=config.test_start)&(d<=config.test_end))}
    if any(len(v)==0 for v in splits.values()):raise ValueError('Empty fitting/validation/test split')
    if config.expected_test_days and len(splits['test'])!=config.expected_test_days:
        raise ValueError(f"Expected {config.expected_test_days} test dates, got {len(splits['test'])}")
    return splits

def load_returns(path,panel):
    with np.load(path,allow_pickle=False) as z:
        r=z['returns'].astype(np.float32);v=z['available'].astype(bool)
        dates=z['dates'].astype('datetime64[ns]');codes=z['stocks'].astype(str).tolist()
    offset=len(dates)-len(panel.dates)
    if offset<0 or r.shape!=(len(dates),panel.y.shape[1],48) or v.shape!=r.shape[:2]:
        raise ValueError('Expected 48 aligned returns per day')
    raw_dates=pd.DatetimeIndex(dates)
    if raw_dates.has_duplicates or not raw_dates.is_monotonic_increasing:raise ValueError('Return dates not increasing')
    if not np.array_equal(dates[offset:],panel.dates.to_numpy()) or codes!=panel.codes:
        raise ValueError('Return identifiers differ; extra rows may only precede the common panel')
    panel.returns_offset=offset
    panel.identity['pre_panel_return_dates']=offset
    if not np.isfinite(r[v]).all():raise ValueError('Valid returns contain NaN/Inf')
    return r,v

def build_returns(minute_dir,panel,output,prehistory_dates=()):
    """Convert the existing aligned 49-close files, recording missing stock-days."""
    directory=Path(minute_dir);N=panel.y.shape[1]
    raw_dates=pd.DatetimeIndex(pd.to_datetime(list(prehistory_dates))).append(panel.dates)
    if raw_dates.has_duplicates or not raw_dates.is_monotonic_increasing:
        raise ValueError('Prehistory dates must be unique trading dates before the panel')
    T=len(raw_dates)
    r=np.zeros((T,N,48),dtype=np.float32);available=np.zeros((T,N),dtype=bool)
    date_index={d:i for i,d in enumerate(raw_dates)};audit=[]
    for j,code in enumerate(panel.codes):
        exact=[directory/f'{code}{s}' for s in ('_min.csv','_5m.csv','_5分钟.csv','.csv')]
        found=[p for p in exact if p.is_file()]
        if not found:found=sorted(directory.glob(f'{code}*.csv'))
        if not found:audit.append({'stock':code,'status':'missing_file'});continue
        frame=pd.read_csv(found[0],usecols=['trade_time','close'])
        frame['trade_time']=pd.to_datetime(frame['trade_time'],errors='raise')
        frame['close']=pd.to_numeric(frame['close'],errors='coerce')
        if frame['trade_time'].duplicated().any():raise ValueError(f'Duplicate timestamps in {found[0]}')
        frame=frame.sort_values('trade_time')
        for day,group in frame.groupby(frame['trade_time'].dt.normalize()):
            if day not in date_index:continue
            ti=date_index[day];last=group.tail(49)
            if len(last)!=49 or not np.isfinite(last['close']).all() or (last['close']<=0).any():continue
            if last['trade_time'].dt.strftime('%H:%M').iloc[-2:].tolist()!=['14:55','15:00']:
                raise ValueError(f'Expected final closes 14:55/15:00 for {code} on {day}')
            r[ti,j]=np.clip(np.diff(np.log(last['close'].to_numpy(dtype=float))),-.08,.08)
            available[ti,j]=True
        audit.append({'stock':code,'status':'read','valid_days':int(available[:,j].sum()),'source':str(found[0])})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,returns=r,available=available,dates=raw_dates.to_numpy(),stocks=np.array(panel.codes))
    write_json(output.with_suffix('.audit.json'),{'source':str(directory),'stocks':audit,
        'semantics':'49 aligned closes -> 48 intraday returns; no overnight bridge; clipped to +/-0.08'})

def hf_for_target(returns,available,t,lookback,device):
    if t<lookback:raise ValueError('Insufficient self-only HF history')
    a=np.asarray(returns[t-lookback:t],dtype=np.float32).transpose(1,0,2).reshape(returns.shape[1],-1).copy()
    ok=np.asarray(available[t-lookback:t],dtype=bool).all(axis=0)
    # Self-only preprocessing. No cross-sectional/market aggregation here.
    a=np.clip(a,-.08,.08);mean=a.mean(axis=1,keepdims=True);sd=a.std(axis=1,keepdims=True)
    a=(a-mean)/np.maximum(sd,1e-6);a[~ok]=0
    return torch.as_tensor(a,device=device)
