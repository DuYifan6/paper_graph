from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from papergraph.config import Config
from papergraph.graphs import Graph, empty_graph, positive_graph, graph_for_target, rewire, GraphStore
from papergraph.models import Forecaster, GraphMix, aggregate, laplacian_loss, parameter_audit
from papergraph.data import Panel, split_dates, hf_for_target
from papergraph.statistics import dm_test, paired_daily, pool_seeds
from papergraph.training import checkpoint_update

def fixture_graph():
    return Graph(5,np.array([0,0,1]),np.array([1,2,3]),np.array([.5,1.,2.]),np.ones(5,dtype=bool)).validate()

def adjacency(g):
    a=np.zeros((g.n,g.n));a[g.u,g.v]=g.weight;a[g.v,g.u]=g.weight
    return a

def test_union_degree_can_exceed_s():
    rng=np.random.default_rng(125)
    # Many draws are fixed here to search a heterogeneous but deterministic hub graph.
    graphs=[positive_graph(rng.normal(size=(12,80)),np.ones(12,dtype=bool),1,.001) for _ in range(5)]
    assert any(g.degree().max()>1 for g in graphs)
    for g in graphs:
        a=adjacency(g);assert np.array_equal(a,a.T);assert not np.diag(a).any()

def test_negative_correlations_not_absolute_edges():
    x=np.arange(50,dtype=float)/1000
    g=positive_graph(np.array([x,-x]),np.ones(2,dtype=bool),1,.01)
    assert len(g.u)==0

def test_clip_market_removal_pearson_and_threshold():
    rng=np.random.default_rng(4);x=rng.normal(0,.1,(8,60));valid=np.ones(8,dtype=bool)
    g=positive_graph(x,valid,2,.1,binary=False)
    clipped=np.clip(x,-.08,.08);residual=clipped-clipped.mean(axis=0)
    corr=np.maximum(np.corrcoef(residual),0)
    for u,v,w in zip(g.u,g.v,g.weight):assert w==pytest.approx(corr[u,v],abs=1e-6)

def test_graph_has_no_current_or_future_returns():
    rng=np.random.default_rng(7);r=rng.normal(0,.01,(30,8,48));v=np.ones((30,8),dtype=bool)
    config=replace(Config(),window=4,neighbors=2)
    g=graph_for_target(r,v,15,config)
    r[15:]=rng.normal(size=r[15:].shape)*100
    other=graph_for_target(r,v,15,config)
    assert g.fingerprint()==other.fingerprint()

def test_graph3_uses_last_24_returns():
    rng=np.random.default_rng(44);r=rng.normal(0,.01,(10,8,48));v=np.ones((10,8),dtype=bool)
    config=replace(Config(),window=3,neighbors=2)
    before=graph_for_target(r,v,8,config,'graph3')
    r[:,:,:24]*=-7
    assert before.fingerprint()==graph_for_target(r,v,8,config,'graph3').fingerprint()

def test_hybrid_excludes_only_latest_post_cutoff_return():
    rng=np.random.default_rng(8);r=rng.normal(0,.01,(10,8,48));v=np.ones((10,8),dtype=bool)
    config=replace(Config(),window=3,neighbors=2,protocol='hybrid1455')
    before=graph_for_target(r,v,8,config)
    r[7,:,-1]=100
    assert before.fingerprint()==graph_for_target(r,v,8,config).fingerprint()

def test_invalid_nodes_and_missing_windows_removed():
    rng=np.random.default_rng(8);r=rng.normal(0,.01,(10,8,48));v=np.ones((10,8),dtype=bool)
    v[6,2]=False
    graph=graph_for_target(r,v,8,replace(Config(),window=3,neighbors=2))
    assert not graph.valid[2];assert 2 not in graph.u;assert 2 not in graph.v

def test_undirected_rewiring_exact_degrees():
    n=30;pairs=sorted({tuple(sorted((i,(i+d)%n))) for i in range(n) for d in (1,3,7)})
    g=Graph(n,np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]),np.ones(len(pairs)),np.ones(n,bool)).validate()
    other,audit=rewire(g,42,swaps_per_edge=3)
    assert np.array_equal(g.degree(),other.degree())
    assert np.array_equal(adjacency(other),adjacency(other).T)
    assert len(g.u)==len(other.u);assert audit['rewiring_complete'];assert audit['edge_overlap']<.8
    assert other.fingerprint()==rewire(g,42,swaps_per_edge=3)[0].fingerprint()

def test_unmixable_graph_fails_explicitly():
    g=Graph(3,np.array([0,0,1]),np.array([1,2,2]),np.ones(3),np.ones(3,bool)).validate()
    with pytest.raises(RuntimeError,match='incomplete'):rewire(g,1,1)

def test_laplacian_matches_dense_matrix_and_mask():
    graph=fixture_graph();valid=torch.tensor([True,True,True,False,True])
    pred=torch.tensor([1.,2.,-3.,99.,6.],requires_grad=True)
    actual=laplacian_loss(pred,graph,valid)
    a=adjacency(graph);a[3,:]=0;a[:,3]=0;degree=a.sum(axis=1)
    inv=np.zeros(5);positive=degree>0;inv[positive]=1/np.sqrt(degree[positive])
    L=np.diag(positive.astype(float))-inv[:,None]*a*inv[None,:]
    expected=2/int(valid.sum())*float(pred.detach().numpy()@L@pred.detach().numpy())
    assert float(actual.detach())==pytest.approx(expected,rel=1e-6)
    actual.backward();assert pred.grad[3]==0;assert pred.grad[4]==0

def test_laplacian_no_padded_slot_denominator():
    g=Graph(2,np.array([0]),np.array([1]),np.ones(1),np.ones(2,bool)).validate()
    pred=torch.tensor([1.,3.]);value=laplacian_loss(pred,g,torch.ones(2,dtype=torch.bool))
    assert float(value)==pytest.approx(4.) # 2/2 * (1-3)^2

def test_normalized_laplacian_smooths_degree_scaled_values():
    g=fixture_graph();degree=adjacency(g).sum(axis=1)
    y=torch.tensor(np.sqrt(degree),dtype=torch.float32)
    assert float(laplacian_loss(y,g,torch.ones(5,dtype=torch.bool)))==pytest.approx(0.,abs=1e-12)

def test_graphmix_is_equation_16():
    torch.manual_seed(2);g=fixture_graph();h=torch.randn(5,8);m=GraphMix(8)
    cat=torch.cat((h,aggregate(h,g)),dim=1)
    expected=h+torch.sigmoid(m.gate(cat))*torch.relu(m.message(cat))
    assert torch.equal(m(h,g),expected)

def test_graphmix_uses_raw_edge_weights_not_sym_norm_weights():
    h=torch.arange(10,dtype=torch.float32).view(5,2);g=fixture_graph()
    agg=aggregate(h,g)
    assert torch.allclose(agg[0],(.5*h[1]+h[2])/(1.5+1e-6))
    assert torch.equal(agg[4],torch.zeros(2))

def test_formula_architecture_parameter_counts_and_bypass():
    torch.manual_seed(3);seq=torch.randn(5,20,9);g=fixture_graph()
    full=Forecaster();zero=Forecaster(variant='zero');self_model=Forecaster(variant='5m_nograph')
    full_audit=parameter_audit(full,seq,g)
    zero_audit=parameter_audit(zero,seq,g)
    self_audit=parameter_audit(self_model,seq,g,torch.randn(5,960))
    assert full_audit['registered']==137089;assert full_audit['active']==137089
    assert full_audit['graph_registered']==65792;assert full_audit['head_registered']==129
    assert zero_audit['registered']==137089;assert zero_audit['active']==71297
    assert self_audit['registered']==full_audit['registered'];assert self_audit['active']==137089
    assert len(torch.unique(self_model.graph.index))==960

def test_identity_never_reads_other_stocks():
    torch.manual_seed(2);model=Forecaster(hidden=8,variant='identity').eval()
    x=torch.randn(5,20,9);before=model(x,fixture_graph()).detach()
    x[2]*=5;after=model(x,fixture_graph()).detach()
    assert torch.equal(before[[0,1,3,4]],after[[0,1,3,4]])

def test_self_encoder_never_reads_other_stocks():
    torch.manual_seed(3);m=Forecaster(hidden=8,variant='5m_nograph',hf_size=96).eval()
    x=torch.randn(5,2,9);hf=torch.randn(5,96);before=m(x,None,hf).detach();hf[1]*=99
    after=m(x,None,hf).detach();assert torch.equal(before[[0,2,3,4]],after[[0,2,3,4]])

def test_same_seed_graph_random_initial_weights_match():
    torch.manual_seed(1);a=Forecaster(hidden=8,variant='graph2')
    torch.manual_seed(1);b=Forecaster(hidden=8,variant='random')
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())

def test_cache_version_and_identity_guard(tmp_path):
    store=GraphStore(tmp_path,{'data':'a'},Config(),'graph2')
    original=fixture_graph();restored=store.get(1,lambda:original)
    assert restored.fingerprint()==store.get(1,lambda:None).fingerprint()
    with pytest.raises(ValueError,match='Incompatible'):GraphStore(tmp_path,{'data':'b'},Config(),'graph2')

def test_graph_validation_rejects_duplicate_directed_or_invalid_edges():
    with pytest.raises(ValueError):Graph(2,np.array([1]),np.array([0]),np.ones(1),np.ones(2,bool)).validate()
    with pytest.raises(ValueError):Graph(2,np.array([0,0]),np.array([1,1]),np.ones(2),np.ones(2,bool)).validate()

def daily(offset=0):
    return pd.DataFrame({'date':['a','b','c'],'mask_fingerprint':['x']*3,
        'target_fingerprint':['y']*3,'nobs':[10]*3,'mse':np.array([3.,4.,5.])+offset})

def test_random_loss_averaging_then_seed_pooling():
    a=paired_daily(daily(),[daily(1),daily(3)])
    b=paired_daily(daily(),[daily(2),daily(4)])
    pooled=pool_seeds([a,b]);assert len(pooled)==3
    assert np.allclose(pooled.loss_advantage,2.5)

def test_matched_stock_day_target_checks():
    bad=daily(1);bad.loc[1,'nobs']=9
    with pytest.raises(ValueError,match='differ'):paired_daily(daily(),[bad])
    bad=daily(1);bad.loc[1,'target_fingerprint']='changed'
    with pytest.raises(ValueError,match='differ'):paired_daily(daily(),[bad])

def test_dm_matches_independent_hac_formula():
    from scipy.stats import t
    rng=np.random.default_rng(7);x=rng.normal(.03,.2,211);centered=x-x.mean();n=len(x)
    cov=np.correlate(centered,centered,mode='full')[n-1:n+2]/n
    lrv=cov[0]+2*(2/3*cov[1]+1/3*cov[2]);se=np.sqrt(lrv/n)
    result=dm_test(x,2)
    assert result['standard_error']==pytest.approx(se)
    assert result['one_sided_p']==pytest.approx(t.sf(x.mean()/se,210))
    assert result['days']==211

def test_checkpoint_uses_mse_only():
    assert checkpoint_update(.1,.2,3,None)
    assert not checkpoint_update(.3,.2,3,None)
    with pytest.raises(ValueError):checkpoint_update(float('nan'),.2,3,None)

def test_chronological_splits_and_latest_step_only():
    dates=pd.bdate_range('2024-05-01','2025-01-31');shape=(len(dates),5)
    x=np.zeros((*shape,9),dtype=np.float32);h=np.ones_like(x)
    panel=Panel(x,np.zeros(shape),np.ones(shape,bool),dates,list('abcde'),{},latest=h).validate()
    config=replace(Config(),lookback=4,window=4,expected_test_days=0)
    split=split_dates(panel,config)
    assert dates[split['fit'][-1]]<=pd.Timestamp(config.fit_end)
    assert dates[split['validation'][0]]>=pd.Timestamp(config.validation_start)
    assert dates[split['test'][0]]>=pd.Timestamp('2025-01-02')
    seq=panel.sequence(10,4,'cpu');assert torch.all(seq[:,:-1]==0);assert torch.all(seq[:,-1]==1)

def test_self_hf_strict_lag():
    r=np.arange(12*5*48,dtype=np.float32).reshape(12,5,48)/1e6;v=np.ones((12,5),bool)
    before=hf_for_target(r,v,8,4,'cpu');r[8:]*=-10
    assert torch.equal(before,hf_for_target(r,v,8,4,'cpu'))
