from dataclasses import replace
from pathlib import Path
import json,pickle,sys
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from papergraph.config import Config
from papergraph.data import Panel
from papergraph.graphs import Graph
from papergraph.training import fit_one
from papergraph.cli import random_experiment, parser, run

def synthetic():
    rng=np.random.default_rng(14);dates=pd.bdate_range('2024-01-02',periods=34);T=len(dates);N=12
    x=rng.normal(size=(T,N,9)).astype(np.float32);y=(.2*x[:,:,0]+rng.normal(0,.1,(T,N))).astype(np.float32)
    panel=Panel(x,y,np.ones((T,N),bool),dates,[f'S{i}' for i in range(N)],{'synthetic':True}).validate()
    returns=rng.normal(0,.005,(T,N,48)).astype(np.float32);available=np.ones((T,N),bool)
    config=replace(Config(),lookback=2,window=2,neighbors=2,threshold=.01,hidden=8,epochs=2,patience=1,min_nodes=5,
        fit_end=str(dates[15].date()),validation_start=str(dates[16].date()),validation_end=str(dates[23].date()),
        test_start=str(dates[24].date()),test_end=str(dates[-1].date()),expected_test_days=10)
    pairs=sorted({tuple(sorted((i,(i+d)%N))) for i in range(N) for d in (1,3)})
    graph=Graph(N,np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]),np.ones(len(pairs)),np.ones(N,bool)).validate()
    return panel,returns,available,config,graph

@pytest.mark.parametrize('variant',['zero','identity','no_lap','graph1','graph2','graph3','random','5m_nograph','gcn_lstm'])
def test_fit_validation_checkpoint_test_for_all_variants(tmp_path,variant):
    torch.set_num_threads(1);panel,r,v,c,g=synthetic()
    summary,daily=fit_one(panel,r,v,c,variant,0,lambda t:g,tmp_path/variant,save_details=True)
    assert summary['test_days']==10;assert summary['test_observations']==120
    assert summary['ranking_loss'] is False;assert np.isfinite(summary['test_mse'])
    history=pd.read_csv(tmp_path/variant/'training.csv')
    assert summary['best_epoch']==int(history.loc[history.validation_mse.idxmin(),'epoch'])
    assert (tmp_path/variant/'checkpoint.pt').exists()
    assert len(pd.read_csv(tmp_path/variant/'predictions_test.csv'))==120

def test_validation_does_not_consult_test_targets(tmp_path):
    torch.set_num_threads(1);panel,r,v,c,g=synthetic()
    one,_=fit_one(panel,r,v,c,'graph2',3,lambda t:g,tmp_path/'a')
    panel.y[24:]+=10000
    two,_=fit_one(panel,r,v,c,'graph2',3,lambda t:g,tmp_path/'b')
    assert one['best_validation_mse']==two['best_validation_mse'];assert one['best_epoch']==two['best_epoch']
    a=torch.load(tmp_path/'a'/'checkpoint.pt',weights_only=False)['state_dict']
    b=torch.load(tmp_path/'b'/'checkpoint.pt',weights_only=False)['state_dict']
    assert all(torch.equal(a[k],b[k]) for k in a)

def test_small_five_seed_topology_pipeline(tmp_path):
    torch.set_num_threads(1);panel,r,v,c,g=synthetic();panel.identity['returns_sha256']='synthetic'
    frame,inference=random_experiment(panel,r,v,replace(c,epochs=1),tmp_path,'cpu',list(range(5)),list(range(20)))
    assert len(frame)==5;assert inference['days']==10
    report=json.loads((tmp_path/'topology_pooled_inference.json').read_text())
    assert report['empirical_model_fits']==5;assert report['random_model_fits']==100
    assert len(pd.read_csv(tmp_path/'topology_pooled_daily.csv'))==10

def test_cli_reads_existing_cache_layout(tmp_path):
    torch.set_num_threads(1);panel,r,v,c,g=synthetic()
    common=tmp_path/'common.npz';meta=tmp_path/'meta.pkl';returns=tmp_path/'returns.npz'
    np.savez(common,X_np=panel.X,y_np=panel.y,mask_np=panel.mask)
    with meta.open('wb') as f:pickle.dump({'all_dates':panel.dates,'ts_codes_order':panel.codes},f)
    np.savez(returns,returns=r,available=v,dates=panel.dates.to_numpy(),stocks=np.array(panel.codes))
    args=parser().parse_args(['train','--common',str(common),'--meta',str(meta),'--returns',str(returns),
                             '--out',str(tmp_path/'run'),'--variant','graph2'])
    run(args,config_override=c)
    summary=json.loads((tmp_path/'run'/'results'/'graph2'/'seed_0'/'summary.json').read_text())
    assert summary['test_days']==10
