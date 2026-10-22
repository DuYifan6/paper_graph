from dataclasses import replace
from pathlib import Path
import pickle,sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from papergraph.data import load_panel,load_returns,split_dates,build_returns
from papergraph.config import FEATURES,Config
from papergraph.statistics import period_metrics,weighted_period_r2

def inputs(tmp_path):
    dates=pd.bdate_range('2024-05-01','2025-01-31');shape=(len(dates),2)
    X=np.broadcast_to(np.arange(9,dtype=np.float32),(*shape,9)).copy()
    np.savez(tmp_path/'common.npz',X_np=X,y_np=np.ones(shape),mask_np=np.ones(shape,bool))
    meta={'all_dates':dates,'ts_codes_order':['A','B'],'feature_names':list(reversed(FEATURES))}
    with (tmp_path/'meta.pkl').open('wb') as f:pickle.dump(meta,f)
    return load_panel(tmp_path/'common.npz',tmp_path/'meta.pkl')

def test_feature_names_reordered_explicitly_not_assumed(tmp_path):
    panel=inputs(tmp_path)
    assert np.array_equal(panel.X[0,0],np.arange(9)[::-1])
    assert panel.identity['feature_order_verified']

def test_prefix_returns_preserve_first_complete_training_window(tmp_path):
    panel=inputs(tmp_path);raw_dates=pd.DatetimeIndex(['2024-04-30']).append(panel.dates)
    np.savez(tmp_path/'returns.npz',returns=np.zeros((len(raw_dates),2,48)),available=np.ones((len(raw_dates),2),bool),
             dates=raw_dates.to_numpy(),stocks=np.array(panel.codes))
    load_returns(tmp_path/'returns.npz',panel)
    assert panel.returns_offset==1
    c=replace(Config(),lookback=4,window=4,expected_test_days=0)
    assert split_dates(panel,c)['fit'][0]==3

def test_old_feature_cache_does_not_get_refit(tmp_path):
    panel=inputs(tmp_path)
    assert panel.identity['preprocessing']=='loaded unchanged; not re-fit by this package'

def test_period_r2_uses_within_period_sst_and_observation_weights():
    daily=pd.DataFrame({'date':['2025-01-02','2025-02-03'],'sse':[1.,2.],
        'nobs':[2,4],'sum_y':[1.,2.],'sum_y2':[1.,4.]})
    periods=period_metrics(daily,'M')
    assert periods.r2.tolist()==pytest.approx([-1.,1-2/3])
    assert weighted_period_r2(periods)==pytest.approx((-1*2+(1-2/3)*4)/6)

def test_build_returns_records_raw_timestamp_cutoff(tmp_path):
    panel=inputs(tmp_path);day=panel.dates[0]
    morning=pd.date_range(str(day.date())+' 09:30',str(day.date())+' 11:30',freq='5min')
    afternoon=pd.date_range(str(day.date())+' 13:05',str(day.date())+' 15:00',freq='5min')
    times=morning.append(afternoon)
    for code in panel.codes:pd.DataFrame({'trade_time':times,'close':np.linspace(10,10.5,49)}).to_csv(tmp_path/f'{code}.csv',index=False)
    build_returns(tmp_path,panel,tmp_path/'raw.npz')
    r,v=load_returns(tmp_path/'raw.npz',panel)
    assert v[0].all();assert np.allclose(r[0,0],np.diff(np.log(np.linspace(10,10.5,49))))
