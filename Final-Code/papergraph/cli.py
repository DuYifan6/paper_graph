"""Command-line entry points for main experiments and paired controls."""
import argparse
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from .config import Config, SCHEMA, sha256, write_json
from .data import load_panel, load_returns, build_returns, read_metadata
from .masks import build_input_mask
from .graphs import GraphStore, graph_for_target, rewire, has_alternative_rewiring
from .statistics import dm_test, paired_daily, pool_seeds
from .training import fit_one, make_model, forecast

VARIANTS=['zero','identity','no_lap','graph1','graph2','graph3','5m_nograph','gcn_lstm']

def parser():
    p=argparse.ArgumentParser(description='Manuscript equations, undirected graphs and chronological MSE selection')
    p.add_argument('command',choices=['screen-features','prepare-inputs','build-hybrid','build-input-mask','build-returns','build-graphs','build-random-graphs','train','predict','ablation','five-seeds','randomgraph','robustness','benchmarks','compare-coverage','calibrate','backtest'])
    p.add_argument('--common',type=Path,help='Frozen X_np and optional y_np/target_observed_np NPZ')
    p.add_argument('--meta',type=Path,help='JSON/NPZ metadata; use original pickles only when trusted')
    p.add_argument('--returns',type=Path,help='Aligned returns/bar_available/dates/stocks NPZ')
    p.add_argument('--input-mask',type=Path,help='Independent input availability NPZ; never legacy mask_np')
    p.add_argument('--raw-inputs',type=Path,help='Pre-label, pre-imputation X_raw/dates/stocks NPZ')
    p.add_argument('--candidate-data',type=Path)
    p.add_argument('--candidate-features',nargs='+')
    p.add_argument('--latest-raw',type=Path,help='Hybrid pre-cutoff X_latest_raw/dates/stocks/asof NPZ')
    p.add_argument('--targets',type=Path,help='Optional dated y_np/dates/stocks NPZ, targets in percentage points')
    p.add_argument('--preprocessing',type=Path,help='Frozen reference median/mean/scale NPZ')
    p.add_argument('--quotes1455',type=Path,help='Signal-day price1455/day_open/previous_close CSV')
    p.add_argument('--fit-preprocessing',action='store_true')
    p.add_argument('--minimum-observed-features',type=int,default=9)
    p.add_argument('--minute-dir',type=Path)
    p.add_argument('--prehistory-dates',type=Path,help='Optional CSV with date column of exchange sessions before first panel target')
    p.add_argument('--out',type=Path,default=Path('runs'))
    p.add_argument('--variant',choices=VARIANTS,default='graph2')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['zero','identity','no_lap','graph2','5m_nograph'])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--model-seeds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--graph-seeds',type=int,nargs='+',default=list(range(20)))
    p.add_argument('--device',default='cpu');p.add_argument('--save-details',action='store_true')
    p.add_argument('--protocol',choices=['full_day','hybrid1455'],default='full_day')
    p.add_argument('--experiment',choices=['primary','supplementary-coverage'],default='primary')
    p.add_argument('--hybrid-cache',type=Path)
    p.add_argument('--window',type=int,default=20);p.add_argument('--neighbors',type=int,default=40)
    p.add_argument('--threshold',type=float,default=.10)
    for name in ('fit-start','fit-end','validation-start','validation-end','test-start','test-end'):
        p.add_argument('--'+name)
    p.add_argument('--expected-test-days',type=int,default=211)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=4)
    p.add_argument('--min-nodes',type=int,default=50)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--prediction-start')
    p.add_argument('--prediction-end')
    p.add_argument('--original-graph',type=Path)
    p.add_argument('--original-zero',type=Path)
    p.add_argument('--expanded-graph',type=Path)
    p.add_argument('--expanded-zero',type=Path)
    p.add_argument('--calibration-predictions',type=Path)
    p.add_argument('--calibration-fraction',type=float,default=.20)
    p.add_argument('--calibration-map',type=Path)
    p.add_argument('--predictions',type=Path)
    p.add_argument('--execution',type=Path)
    p.add_argument('--risk',type=Path,help='Dated market_overnight_vol CSV, including pre-2025 history')
    p.add_argument('--target-scale',type=float,default=.01,help='Eq. (1) percentage points to decimal returns')
    p.add_argument('--capacity-fraction',type=float)
    p.add_argument('--round-lots',action='store_true')
    p.add_argument('--k',type=int,nargs='+',default=[1,3,5,10])
    p.add_argument('--slippage-bp',type=float,nargs='+',default=[0,5,10])
    p.add_argument('--benchmark-models',nargs='+',choices=['ridge','lightgbm','transformer'],
                   default=['ridge','lightgbm','transformer'])
    return p

def empirical_getter(panel,returns,available,config,variant,root):
    graph_variant=variant if variant in ('graph1','graph3') else 'graph2'
    store=GraphStore(root/'graphs'/config.fingerprint()/graph_variant,
                     graph_identity(panel),config,graph_variant)
    @lru_cache(maxsize=32)
    def getter(t):return store.get(t,lambda:graph_for_target(returns,available,t+panel.returns_offset,config,graph_variant))
    getter.run_identity={'graph_variant':graph_variant,'randomized':False}
    return getter

def graph_identity(panel):
    # Graph caching depends on historical returns and identifiers, never labels.
    return {'returns_sha256':panel.identity['returns_sha256'],
            'dates':[str(d.date()) for d in panel.dates],'stocks':list(panel.codes)}

def randomized_getter(real_getter,panel,config,root,model_seed,graph_seed):
    identity=graph_identity(panel)|{'model_seed':model_seed,'graph_seed':graph_seed,
        'rewiring':'undirected double-edge swaps','successful_swaps_per_edge':5}
    directory=root/'random_graphs'/config.fingerprint()/f'model_{model_seed}'/f'graph_{graph_seed}'
    store=GraphStore(directory,identity,config,'random')
    @lru_cache(maxsize=32)
    def getter(t):
        def builder():
            # Cross-model randomizations are independently seeded; model RNG is separate.
            seed=int(np.random.SeedSequence([model_seed,graph_seed,int(t)]).generate_state(1)[0])
            real=real_getter(t)
            nv=int(real.valid.sum())
            unique_fiber=len(real.u)<2 or len(real.u)==nv*(nv-1)//2
            if not unique_fiber and len(real.u)<=256:
                unique_fiber=not has_alternative_rewiring(real)
            graph,audit=rewire(real,seed,swaps_per_edge=0 if unique_fiber else 5,
                               strict=not unique_fiber)
            audit['degenerate_fixed_degree_reference']=unique_fiber
            audit['note']=('No alternative graph in this trivial fixed-degree reference; retained unchanged.'
                           if unique_fiber else 'Undirected degree-preserving swaps completed.')
            write_json(directory/f'day_{t:04d}.audit.json',audit)
            return graph
        return store.get(t,builder)
    getter.run_identity={'randomized':True,'model_seed':model_seed,'graph_seed':graph_seed}
    return getter

def one(panel,returns,available,config,variant,seed,root,device,details,getter=None,suffix=None):
    if getter is None:getter=empirical_getter(panel,returns,available,config,variant,root)
    location=root/'results'/(suffix or variant)/f'seed_{seed}'
    return fit_one(panel,returns,available,config,variant,seed,getter,location,device,details)

def random_experiment(panel,returns,available,config,root,device,model_seeds,graph_seeds,details=False):
    if len(set(model_seeds))!=len(model_seeds) or len(set(graph_seeds))!=len(graph_seeds):
        raise ValueError('Duplicate seeds would double-count runs')
    if not model_seeds or len(graph_seeds)<2:raise ValueError('Need model seeds and >=2 randomizations')
    empirical=empirical_getter(panel,returns,available,config,'graph2',root)
    seed_rows=[];daily=[];runs=[]
    for seed in model_seeds:
        real,real_daily=one(panel,returns,available,config,'graph2',seed,root,device,details,
                            getter=empirical,suffix='topology/empirical')
        random_rows=[];random_daily=[]
        for graph_seed in graph_seeds:
            getter=randomized_getter(empirical,panel,config,root,seed,graph_seed)
            summary,losses=one(panel,returns,available,config,'random',seed,root,device,details,getter,
                               suffix=f'topology/random_graph_{graph_seed}')
            summary=summary|{'graph_seed':graph_seed};random_rows.append(summary);random_daily.append(losses)
            runs.append({'model_seed':seed,'graph_seed':graph_seed,'test_r2':summary['test_r2'],
                         'best_epoch':summary['best_epoch'],'test_observations':summary['test_observations']})
            getter.cache_clear()
        diff=paired_daily(real_daily,random_daily);daily.append(diff)
        inference=dm_test(diff['loss_advantage'],2)
        random_r2=np.array([row['test_r2'] for row in random_rows],dtype=float)
        seed_rows.append({'model_seed':seed,'empirical_r2':real['test_r2'],
            'random_r2_mean':float(random_r2.mean()),'random_r2_sd':float(random_r2.std(ddof=1)),
            'delta_r2':float(real['test_r2']-random_r2.mean()),'one_sided_dm_p':inference['one_sided_p']})
        write_json(root/f'topology_seed_{seed}_inference.json',inference)
        diff.to_csv(root/f'topology_seed_{seed}_daily.csv',index=False)
    pooled=pool_seeds(daily);inference=dm_test(pooled['loss_advantage'],2)
    frame=pd.DataFrame(seed_rows);frame.to_csv(root/'topology_by_model_seed.csv',index=False)
    pd.DataFrame(runs).to_csv(root/'topology_all_random_runs.csv',index=False)
    pooled.to_csv(root/'topology_pooled_daily.csv',index=False)
    write_json(root/'topology_pooled_inference.json',inference|{
        'model_seeds':model_seeds,'graph_seeds_per_model':graph_seeds,
        'random_model_fits':len(runs),'empirical_model_fits':len(seed_rows),
        'mean_delta_r2':float(frame['delta_r2'].mean()),
        'interpretation':'Trading dates are the inference unit; initialization/randomization losses are averaged, not extra dates.'})
    return frame,inference

def run(args,config_override=None):
    config=config_override or Config(window=args.window,neighbors=args.neighbors,threshold=args.threshold,protocol=args.protocol)
    if config_override is None and args.experiment=='supplementary-coverage':
        config=replace(config,protocol='hybrid1455',fit_end='2023-12-07',
                       validation_start='2023-12-08',validation_end='2024-06-05',
                       calibration_start='2024-06-06',calibration_end='2024-12-31',
                       ranking=.05,selection='weighted_topk')
    if config_override is None:
        values={key:getattr(args,key) for key in ('fit_start','fit_end','validation_start','validation_end','test_start','test_end') if getattr(args,key) is not None}
        config=replace(config,**values,epochs=args.epochs,patience=args.patience,
                       min_nodes=args.min_nodes,expected_test_days=args.expected_test_days)
    if args.command=='screen-features':
        from .feature_selection import screen_features
        if args.candidate_data is None:raise ValueError('--candidate-data is required')
        return screen_features(args.candidate_data,args.candidate_features,args.out)
    if args.command=='compare-coverage':
        from .coverage import compare_coverage
        return compare_coverage(args.original_graph,args.original_zero,args.expanded_graph,args.expanded_zero,args.out)
    if args.command in ('calibrate','backtest'):
        from .economics import calibrate_file,backtest_file
        if args.command=='calibrate':
            return calibrate_file(args.calibration_predictions,args.out,args.calibration_fraction,args.target_scale)
        return backtest_file(args.predictions,args.calibration_map,args.execution,args.out,args.k,args.slippage_bp,args.risk,args.capacity_fraction,args.round_lots)
    if args.command in ('prepare-inputs','build-hybrid','build-input-mask','build-returns'):
        if args.meta is None:raise ValueError('--meta is required')
        meta=read_metadata(args.meta)
        dates=pd.DatetimeIndex(pd.to_datetime(meta['all_dates'])).normalize()
        codes=[str(c) for c in meta['ts_codes_order']]
        if args.command=='build-hybrid':
            from .hybrid import build_latest
            if any(x is None for x in (args.raw_inputs,args.quotes1455,args.preprocessing)):
                raise ValueError('--raw-inputs, --quotes1455 and --preprocessing are required')
            return build_latest(args.raw_inputs,args.quotes1455,dates,codes,args.preprocessing,args.out)
        if args.command=='prepare-inputs':
            from .preprocessing import prepare_cache
            if args.raw_inputs is None:raise ValueError('--raw-inputs is required')
            return prepare_cache(args.raw_inputs,dates,codes,args.out,config.fit_start,config.fit_end,
                                 args.preprocessing,args.fit_preprocessing,args.targets)
        if args.command=='build-input-mask':
            if args.raw_inputs is None or args.input_mask is None:raise ValueError('--raw-inputs and --input-mask are required')
            return build_input_mask(args.raw_inputs,dates,codes,args.input_mask,config.lookback,
                                    config.protocol,args.minimum_observed_features,args.latest_raw)
        if args.minute_dir is None or args.returns is None:raise ValueError('--minute-dir and --returns are required')
        prehistory=pd.read_csv(args.prehistory_dates)['date'].tolist() if args.prehistory_dates else []
        return build_returns(args.minute_dir,SimpleNamespace(dates=dates,codes=codes),args.returns,prehistory)
    if args.command=='predict':
        if args.checkpoint is None:raise ValueError('--checkpoint is required')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if checkpoint.get('schema')!=SCHEMA:raise ValueError('Checkpoint belongs to a different model/mask schema')
        config=Config(**checkpoint['config'])
        args.variant=checkpoint['variant']
    if config.protocol=='hybrid1455' and not args.hybrid_cache:raise ValueError('Hybrid latest-step cache required')
    if config.protocol=='full_day' and args.hybrid_cache:raise ValueError('Specify --protocol hybrid1455 with hybrid data')
    if any(x is None for x in (args.common,args.meta,args.returns,args.input_mask)):
        raise ValueError('--common, --meta, --returns and --input-mask are required')
    if config.protocol=='hybrid1455' and (args.variant not in ('zero','graph2') or args.command not in ('train','predict','build-graphs')):
        raise ValueError('The economic input protocol is specified for Graph-2 and ZeroGraph')
    panel=load_panel(args.common,args.meta,args.hybrid_cache,input_mask=args.input_mask,
                     protocol=config.protocol,lookback=config.lookback)
    returns,available=load_returns(args.returns,panel)
    panel.identity['returns_sha256']=sha256(args.returns)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=='predict':
        model=make_model(config,args.variant).to(args.device)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        getter=empirical_getter(panel,returns,available,config,args.variant,args.out)
        if args.variant=='random':
            identity=checkpoint.get('graph_run',{})
            if not identity.get('randomized'):raise ValueError('RandomGraph checkpoint lacks its graph seed')
            getter=randomized_getter(getter,panel,config,args.out,identity['model_seed'],identity['graph_seed'])
        start=args.prediction_start or config.test_start
        end=args.prediction_end or config.test_end
        indices=np.flatnonzero((panel.dates>=start)&(panel.dates<=end)&(np.arange(len(panel.dates))>=max(config.lookback-1,config.window-panel.returns_offset)))
        if not len(indices):raise ValueError('No dates with sufficient history in prediction interval')
        output=forecast(model,panel,returns,available,indices,config,getter,args.device)
        output.to_csv(args.out/'forecasts.csv',index=False)
        return output
    if args.command=='benchmarks':
        from .benchmarks import run_benchmarks
        return run_benchmarks(panel,returns,available,config,args.out,args.device,args.seed,args.benchmark_models)
    if args.command=='build-graphs':
        if args.variant not in ('graph1','graph2','graph3'):raise ValueError('Choose empirical graph variant')
        getter=empirical_getter(panel,returns,available,config,args.variant,args.out)
        audit=[{'date':str(d.date())}|getter(i).audit() for i,d in enumerate(panel.dates)]
        write_json(args.out/f'{args.variant}_graph_audit.json',audit);return
    if args.command=='train':
        one(panel,returns,available,config,args.variant,args.seed,args.out,args.device,args.save_details);return
    if args.command=='build-random-graphs':
        empirical=empirical_getter(panel,returns,available,config,'graph2',args.out)
        for seed in args.model_seeds:
            for graph_seed in args.graph_seeds:
                getter=randomized_getter(empirical,panel,config,args.out,seed,graph_seed)
                for t in range(len(panel.dates)):getter(t)
                getter.cache_clear()
        return
    if args.command=='ablation':
        rows=[]
        for variant in args.variants:
            s,_=one(panel,returns,available,config,variant,args.seed,args.out,args.device,args.save_details)
            rows.append({'variant':variant,'test_r2':s['test_r2'],'registered':s['parameters']['registered'],
                         'active':s['parameters']['active'],'best_epoch':s['best_epoch']})
        pd.DataFrame(rows).to_csv(args.out/'ablation_summary.csv',index=False);return
    if args.command=='randomgraph':
        return random_experiment(panel,returns,available,config,args.out,args.device,args.model_seeds,args.graph_seeds,args.save_details)
    if args.command=='five-seeds':
        from scipy.stats import t as student_t
        rows=[]
        for seed in args.model_seeds:
            z,_=one(panel,returns,available,config,'zero',seed,args.out,args.device,args.save_details,suffix='five_seeds/zero')
            g,_=one(panel,returns,available,config,'graph2',seed,args.out,args.device,args.save_details,suffix='five_seeds/graph2')
            rows.append({'seed':seed,'zero_r2':z['test_r2'],'graph2_r2':g['test_r2'],'delta_r2':g['test_r2']-z['test_r2']})
        df=pd.DataFrame(rows);df.to_csv(args.out/'five_seeds.csv',index=False)
        if len(df)<2:raise ValueError('Need >=2 seeds for seed-variation interval')
        mean=float(df.delta_r2.mean());sd=float(df.delta_r2.std(ddof=1));half=student_t.ppf(.975,len(df)-1)*sd/np.sqrt(len(df))
        write_json(args.out/'five_seeds_interval.json',{'mean':mean,'sd':sd,'ci95':[mean-half,mean+half],
            'replicates':len(df),'scope':'initialization variation conditional on this fixed dataset'});return
    if args.command=='robustness':
        settings=[('baseline',config)]+[(f'W_{w}',replace(config,window=w)) for w in (10,40)]
        settings += [(f's_{s}',replace(config,neighbors=s)) for s in (20,60)]
        settings += [(f'tau_{tau}',replace(config,threshold=tau)) for tau in (.05,.15)]
        rows=[]
        for label,c in settings:
            for variant in ('zero','graph2'):
                s,_=one(panel,returns,available,c,variant,args.seed,args.out,args.device,args.save_details,suffix=f'robustness/{label}/{variant}')
                rows.append({'setting':label,'variant':variant,'pooled_test_r2':s['test_r2']})
        pd.DataFrame(rows).to_csv(args.out/'robustness_pooled.csv',index=False)

def main(argv=None):
    args=parser().parse_args(argv);run(args)

if __name__=='__main__':main()
