import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import norm

def dm_test(values,lag=2,reference='t'):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)<3:raise ValueError('Need finite consecutive daily losses')
    n=len(x);lag=min(int(lag),n-1)
    if lag<0:raise ValueError('Negative HAC lag')
    centered=x-x.mean();lrv=float(centered@centered/n)
    for k in range(1,lag+1):lrv+=2*(1-k/(lag+1))*float(centered[k:]@centered[:-k]/n)
    se=float(np.sqrt(max(lrv,0)/n));mean=float(x.mean())
    if se==0:
        return {'days':n,'mean':mean,'standard_error':0.,'statistic':None,
                'one_sided_p':None,'two_sided_p':None,'ci95':None,
                'lag':lag,'df':n-1,'status':'zero_variance; inference undefined'}
    if reference not in ('t','normal'):raise ValueError('Choose t or normal reference')
    distribution=student_t(df=n-1) if reference=='t' else norm
    stat=mean/se;crit=distribution.ppf(.975)
    return {'days':n,'mean':mean,'standard_error':se,'statistic':stat,
            'one_sided_p':float(distribution.sf(stat)),
            'two_sided_p':float(2*distribution.sf(abs(stat))),
            'ci95':[float(mean-crit*se),float(mean+crit*se)],'lag':lag,'df':n-1,
            'reference':reference+'; Newey-West Bartlett kernel; mean variance = LRV/T',
            'units':'percentage points squared'}

def paired_daily(real,random_frames):
    keys=['date','mask_fingerprint','target_fingerprint','nobs']
    real=real.sort_values('date').reset_index(drop=True)
    if real['date'].duplicated().any():raise ValueError('Duplicate real dates')
    losses=[]
    for frame in random_frames:
        frame=frame.sort_values('date').reset_index(drop=True)
        if not real[keys].equals(frame[keys]):raise ValueError('RandomGraph sample or targets differ')
        losses.append(frame['mse'].to_numpy())
    if not losses:raise ValueError('No randomized losses')
    result=real[keys].copy();result['loss_advantage']=np.mean(losses,axis=0)-real['mse'].to_numpy()
    return result

def pool_seeds(daily_frames):
    reference=daily_frames[0].sort_values('date').reset_index(drop=True)
    keys=['date','mask_fingerprint','target_fingerprint','nobs']
    losses=[]
    for frame in daily_frames:
        frame=frame.sort_values('date').reset_index(drop=True)
        if not reference[keys].equals(frame[keys]):raise ValueError('Initialization sample or targets differ')
        losses.append(frame['loss_advantage'].to_numpy())
    pooled=reference[keys].copy();pooled['loss_advantage']=np.mean(losses,axis=0)
    return pooled

def period_metrics(daily,frequency):
    frame=daily.copy();frame['period']=pd.to_datetime(frame['date']).dt.to_period(frequency).astype(str)
    total=frame.groupby('period',sort=True)[['sse','nobs','sum_y','sum_y2']].sum()
    total['mse']=total['sse']/total['nobs']
    sst=total['sum_y2']-total['sum_y']**2/total['nobs']
    total['r2']=np.where(sst>0,1-total['sse']/sst,np.nan)
    return total.reset_index()

def weighted_period_r2(periods):
    if not np.isfinite(periods['r2']).all():raise ValueError('Undefined period R2')
    return float(np.average(periods['r2'],weights=periods['nobs']))
