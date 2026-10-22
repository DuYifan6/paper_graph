from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from .config import SCHEMA, write_json
from .data import hf_for_target, split_dates
from .models import Forecaster, GCNForecaster, laplacian_loss, parameter_audit
from .statistics import period_metrics, weighted_period_r2

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def checkpoint_update(value,best,epoch,state):
    """Select on pooled validation MSE only. No portfolio score or ranking term."""
    if not np.isfinite(value):raise ValueError('Nonfinite validation MSE')
    return value<best

def predict_day(model,panel,returns,available,t,config,graph_getter,device):
    seq=panel.sequence(t,config.lookback,device)
    graph=([graph_getter(i) for i in range(t-config.lookback+1,t+1)]
           if model.variant=='gcn_lstm' else graph_getter(t))
    hf=hf_for_target(returns,available,t+panel.returns_offset,config.lookback,device) if model.variant=='5m_nograph' else None
    pred=model(seq,graph,hf)
    mask=torch.as_tensor(panel.mask[t],device=device)
    if not torch.isfinite(pred[mask]).all():raise FloatingPointError(f'Nonfinite prediction on date {panel.dates[t]}')
    return pred,mask,graph

def evaluate(model,panel,returns,available,indices,config,graph_getter,device,details=False):
    model.eval();rows=[];sse=sy=sy2=0.;nobs=0;detail=[]
    with torch.no_grad():
        for t in indices:
            if panel.mask[t].sum()<config.min_nodes:raise ValueError(f'Too few valid targets on {panel.dates[t]}')
            pred,mask,_=predict_day(model,panel,returns,available,int(t),config,graph_getter,device)
            actual=panel.y[t,panel.mask[t]].astype(np.float64)
            forecast=pred[mask].cpu().numpy().astype(np.float64)
            loss=float(np.square(actual-forecast).sum());n=len(actual)
            sse+=loss;sy+=float(actual.sum());sy2+=float(actual@actual);nobs+=n
            date=str(panel.dates[t].date())
            rows.append({'date':date,'mse':loss/n,'sse':loss,'nobs':n,
                'sum_y':float(actual.sum()),'sum_y2':float(actual@actual),
                'mask_fingerprint':hashlib.sha256(panel.mask[t].tobytes()).hexdigest(),
                'target_fingerprint':hashlib.sha256(actual.tobytes()).hexdigest()})
            if details:
                detail.extend({'date':date,'stock':panel.codes[j],'y_true':float(y),'y_pred':float(p)}
                    for j,y,p in zip(np.flatnonzero(panel.mask[t]),actual,forecast))
    if not nobs:raise ValueError('Empty evaluation')
    sst=sy2-sy*sy/nobs
    return {'mse':sse/nobs,'r2':1-sse/sst if sst>0 else None,'observations':nobs,
            'daily':pd.DataFrame(rows),'detail':pd.DataFrame(detail)}

def fit_one(panel,returns,available,config,variant,seed,graph_getter,out,device='cpu',save_details=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(f'Completed output exists: {out}; choose a new output root')
    split=split_dates(panel,config);set_seed(seed);device=torch.device(device)
    model=(GCNForecaster(hidden=config.hidden) if variant=='gcn_lstm' else
           Forecaster(hidden=config.hidden,variant=variant,hf_size=config.lookback*48)).to(device)
    first=int(split['fit'][0]);seq=panel.sequence(first,config.lookback,device)
    hf=hf_for_target(returns,available,first+panel.returns_offset,config.lookback,device) if variant=='5m_nograph' else None
    audit_graph=([graph_getter(i) for i in range(first-config.lookback+1,first+1)]
                 if variant=='gcn_lstm' else graph_getter(first))
    audit=parameter_audit(model,seq,audit_graph,hf)
    # Audit has no optimizer step; reset RNG so audits never change dropout/training randomness.
    set_seed(seed)
    optimizer_class=torch.optim.AdamW if variant=='gcn_lstm' else torch.optim.Adam
    optimizer=optimizer_class(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    best=float('inf');best_state=None;best_epoch=0;bad=0;history=[];start=time.perf_counter()
    uses_lap=variant in ('graph1','graph2','graph3','random')
    for epoch in range(1,config.epochs+1):
        model.train();epoch_sse=0.;nobs=0;lap_total=0.
        for t in split['fit']:
            if panel.mask[t].sum()<config.min_nodes:raise ValueError(f'Too few fitting targets at {panel.dates[t]}')
            optimizer.zero_grad(set_to_none=True)
            pred,valid,graph=predict_day(model,panel,returns,available,int(t),config,graph_getter,device)
            target=torch.as_tensor(panel.y[t],device=device)
            mse=(pred[valid]-target[valid]).square().mean()
            lap=laplacian_loss(pred,graph,valid) if uses_lap else pred.sum()*0
            loss=mse+config.laplacian*lap
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step()
            n=int(valid.sum());epoch_sse+=float(mse.detach())*n;nobs+=n;lap_total+=float(lap.detach())
        validation=evaluate(model,panel,returns,available,split['validation'],config,graph_getter,device)
        value=validation['mse'];history.append({'epoch':epoch,'training_mse':epoch_sse/nobs,
                'laplacian_sum':lap_total,'validation_mse':value})
        if checkpoint_update(value,best,epoch,None):
            best=value;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=config.patience:break
    if best_state is None:raise RuntimeError('No checkpoint selected')
    model.load_state_dict(best_state)
    validation_final=evaluate(model,panel,returns,available,split['validation'],config,graph_getter,device,save_details)
    test=evaluate(model,panel,returns,available,split['test'],config,graph_getter,device,save_details)
    development=evaluate(model,panel,returns,available,np.r_[split['fit'],split['validation']],
                         config,graph_getter,device,False)
    monthly=period_metrics(test['daily'],'M');quarterly=period_metrics(development['daily'],'Q')
    history=pd.DataFrame(history);history.to_csv(out/'training.csv',index=False)
    test['daily'].to_csv(out/'daily_test.csv',index=False)
    development['daily'].to_csv(out/'daily_development.csv',index=False)
    monthly.to_csv(out/'test_monthly.csv',index=False)
    quarterly.to_csv(out/'development_quarterly.csv',index=False)
    if save_details:
        test['detail'].to_csv(out/'predictions_test.csv',index=False)
        validation_final['detail'].to_csv(out/'predictions_validation.csv',index=False)
    torch.save({'state_dict':best_state,'schema':SCHEMA,'config':asdict(config),'variant':variant,'seed':seed,
                'data_identity':panel.identity},out/'checkpoint.pt')
    summary={'schema':SCHEMA,'variant':variant,'model_seed':seed,'data_identity':panel.identity,
        'config':asdict(config),'parameters':audit,'best_epoch':best_epoch,'best_validation_mse':best,
        'test_r2':test['r2'],'test_mse':test['mse'],'test_observations':test['observations'],
        'test_weighted_monthly_r2':weighted_period_r2(monthly),
        'development_weighted_quarterly_r2':weighted_period_r2(quarterly),
        'test_days':len(test['daily']),'runtime_seconds':time.perf_counter()-start,
        'selection':'minimum pooled validation MSE','objective':'MSE + lambda*normalized Laplacian' if uses_lap else 'MSE',
        'ranking_loss':False,'results_status':'newly computed; does not certify the manuscript printed values'}
    write_json(out/'summary.json',summary)
    return summary,test['daily']
