"""Scientific invariants and small synthetic end-to-end fits; no private inputs."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from papergraph.config import Config, FEATURES
from papergraph.data import Panel, load_panel, load_returns
from papergraph.graphs import Graph, graph_for_target, positive_graph, rewire
from papergraph.masks import build_input_mask, load_input_mask
from papergraph.models import Forecaster, aggregate, laplacian_loss, parameter_audit
from papergraph.training import predict_day, evaluate, forecast, fit_one
from papergraph.cli import parser, run
from papergraph.economics import calibrate_file, calibrated_returns, conditional_portfolio, risk_exposure
from papergraph.preprocessing import FrozenTransform
from papergraph.coverage import compare_coverage


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(2)


def synthetic():
    rng = np.random.default_rng(14)
    dates = pd.bdate_range('2024-01-02', periods=34)
    n = 12
    x = rng.normal(size=(len(dates), n, 9)).astype(np.float32)
    y = (.2*x[:, :, 0]+rng.normal(0, .1, (len(dates), n))).astype(np.float32)
    panel = Panel(x, y, np.ones(y.shape, bool), np.ones(y.shape, bool), dates,
                  [f'S{i}' for i in range(n)], {'synthetic': True}).validate()
    returns = rng.normal(0, .005, (len(dates), n, 48)).astype(np.float32)
    flags = np.ones(returns.shape, bool)
    config = replace(Config(), lookback=2, window=2, neighbors=2, hidden=8,
                     epochs=2, patience=1, min_nodes=3, fit_start=str(dates[2].date()),
                     fit_end=str(dates[15].date()), validation_start=str(dates[16].date()),
                     validation_end=str(dates[23].date()), test_start=str(dates[24].date()),
                     test_end=str(dates[-1].date()), expected_test_days=10)
    pairs = sorted({tuple(sorted((i, (i+d) % n))) for i in range(n) for d in (1, 3)})
    graph = Graph(n, np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]),
                  np.ones(len(pairs)), np.ones(n, bool)).validate()
    return panel, returns, flags, config, graph


@pytest.mark.parametrize('variant', ['zero', 'identity', 'no_lap', 'graph1', 'graph2',
                                    'graph3', 'random', '5m_nograph', 'gcn_lstm'])
def test_target_values_and_observation_flags_do_not_change_forecasts(variant):
    from papergraph.training import make_model
    panel, r, flags, config, graph = synthetic()
    model = make_model(config, variant).eval()
    first = forecast(model, panel, r, flags, [24, 25], config, lambda t: graph, 'cpu')
    panel.target_observed[:] = False
    panel.y[:] = np.nan
    second = forecast(model, panel, r, flags, [24, 25], config, lambda t: graph, 'cpu')
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_evaluation_sample_changes_but_predictions_do_not():
    panel, r, flags, config, graph = synthetic()
    model = Forecaster(hidden=8).eval()
    first = evaluate(model, panel, r, flags, [24], config, lambda t: graph, 'cpu', True)
    panel.target_observed[24, 1] = False
    panel.y[24, 1] = np.nan
    second = evaluate(model, panel, r, flags, [24], config, lambda t: graph, 'cpu', True)
    assert second['observations'] == first['observations']-1
    assert len(first['detail']) == len(second['detail']) == 12
    assert np.array_equal(first['detail'].y_pred, second['detail'].y_pred)


def test_missing_input_mask_is_rejected():
    model = Forecaster(hidden=8)
    with pytest.raises(TypeError):
        model(torch.randn(3, 2, 9))
    with pytest.raises(ValueError, match='independent'):
        model(torch.randn(3, 2, 9), input_available=None)


@pytest.mark.parametrize('variant', ['graph2', 'identity', 'zero', '5m_nograph'])
def test_unavailable_sender_features_cannot_change_available_predictions(variant):
    panel, r, flags, config, graph = synthetic()
    panel.input_available[24, 1] = False
    model = Forecaster(hidden=8, variant=variant).eval()
    first = predict_day(model, panel, r, flags, 24, config, lambda t: graph, 'cpu')[0]
    panel.X[23:25, 1] = np.nan
    second = predict_day(model, panel, r, flags, 24, config, lambda t: graph, 'cpu')[0]
    assert torch.equal(first, second)


def test_manuscript_architecture_counts_and_zero_bypass():
    seq = torch.randn(12, 20, 9)
    _, _, _, _, graph = synthetic()
    valid = torch.ones(12, dtype=torch.bool)
    full = parameter_audit(Forecaster(), seq, graph, input_available=valid)
    zero = parameter_audit(Forecaster(variant='zero'), seq, graph, input_available=valid)
    own = parameter_audit(Forecaster(variant='5m_nograph'), seq, graph,
                          torch.randn(12, 20, 48), input_available=valid)
    assert full['registered'] == full['active'] == 170113
    assert full['temporal_registered'] == 71168
    assert full['graph_registered'] == 82304
    assert full['head_registered'] == 16641
    assert zero['registered'] == 170113 and zero['active'] == 87809
    assert own['registered'] == own['active'] == 170113


def test_laplacian_matches_dense_induced_subgraph_with_isolates():
    graph = Graph(5, np.array([0, 0, 1]), np.array([1, 2, 3]),
                  np.array([.5, 1., 2.]), np.ones(5, bool)).validate()
    valid = torch.tensor([True, True, True, False, True])
    pred = torch.tensor([1., 2., -3., 99., 6.], requires_grad=True)
    actual = laplacian_loss(pred, graph, valid)
    a = np.zeros((5, 5))
    a[graph.u, graph.v] = graph.weight
    a[graph.v, graph.u] = graph.weight
    a[3, :] = a[:, 3] = 0
    d = a.sum(1)
    inverse = np.zeros(5)
    inverse[d > 0] = 1/np.sqrt(d[d > 0])
    lap = inverse[:, None]*(np.diag(d)-a)*inverse[None, :]
    y = pred.detach().numpy()
    assert float(actual.detach()) == pytest.approx(2/4*(y @ lap @ y), rel=1e-6)
    actual.backward()
    assert pred.grad[3] == pred.grad[4] == 0


def test_positive_union_is_symmetric_and_can_exceed_neighbor_limit():
    rng = np.random.default_rng(125)
    graphs = [positive_graph(rng.normal(size=(12, 80)), np.ones(12, bool), 1, .001)
              for _ in range(5)]
    assert any(g.degree().max() > 1 for g in graphs)
    for g in graphs:
        assert np.all(g.u < g.v)
        assert len(np.unique(g.u*g.n+g.v)) == len(g.u)


def test_graphs_do_not_use_target_day_returns_or_availability():
    panel, r, flags, config, graph = synthetic()
    first = graph_for_target(r, flags, 24, config)
    r[24:] = np.nan
    flags[24:] = False
    assert first.fingerprint() == graph_for_target(r, flags, 24, config).fingerprint()


def test_hybrid_ignores_post_cutoff_values_and_availability():
    panel, r, flags, config, graph = synthetic()
    config = replace(config, protocol='hybrid1455')
    first = graph_for_target(r, flags, 24, config)
    r[23, :, -1] = np.nan
    flags[23, :, -1] = False
    assert first.fingerprint() == graph_for_target(r, flags, 24, config).fingerprint()


def test_rewiring_preserves_undirected_node_degrees():
    *_, graph = synthetic()
    randomized, audit = rewire(graph, 31, swaps_per_edge=2)
    assert np.array_equal(graph.degree(), randomized.degree())
    assert audit['rewiring_complete']
    assert len(graph.u) == len(randomized.u)


def make_raw(tmp_path):
    dates = pd.bdate_range('2024-01-02', periods=10)
    codes = ['A', 'B', 'C']
    path = tmp_path/'raw.npz'
    np.savez(path, X_raw=np.ones((10, 3, 9)), dates=dates.to_numpy(),
             stocks=np.array(codes), feature_names=np.array(FEATURES))
    return path, dates, codes


def test_input_mask_is_built_before_imputation_and_rejects_late_provenance(tmp_path):
    path, dates, codes = make_raw(tmp_path)
    output = tmp_path/'inputs.npz'
    mask = build_input_mask(path, dates[2:], codes, output, lookback=2)
    assert mask.all()
    restored, _ = load_input_mask(output, dates[2:], codes, 'full_day', 2)
    assert np.array_equal(mask, restored)
    with np.load(output) as z:
        values = {key: z[key] for key in z.files}
    values['input_asof'] = dates[2:].to_numpy()
    np.savez(output, **values)
    with pytest.raises(ValueError, match='after the decision cutoff'):
        load_input_mask(output, dates[2:], codes, 'full_day', 2)


def test_legacy_target_mask_cannot_substitute_for_input_availability(tmp_path):
    path, dates, codes = make_raw(tmp_path)
    meta = tmp_path/'meta.npz'
    common = tmp_path/'common.npz'
    np.savez(meta, all_dates=dates.to_numpy(), ts_codes_order=np.array(codes),
             feature_names=np.array(FEATURES))
    np.savez(common, X_np=np.ones((10, 3, 9)), y_np=np.ones((10, 3)), mask_np=np.ones((10, 3), bool))
    with pytest.raises(ValueError, match='separately constructed'):
        load_panel(common, meta, input_mask=None)


def test_hybrid_input_mask_does_not_use_latest_full_day_raw_features(tmp_path):
    path, dates, codes = make_raw(tmp_path)
    target = dates[2:]
    latest = tmp_path/'latest.npz'
    asof = dates[1:-1]+pd.Timedelta(hours=14, minutes=55)
    np.savez(latest, X_latest_raw=np.ones((8, 3, 9)), dates=target.to_numpy(),
             stocks=np.array(codes), asof=asof.to_numpy())
    first = build_input_mask(path, target, codes, tmp_path/'first.npz', 2, 'hybrid1455', 9, latest)
    raw = np.ones((10, 3, 9))
    raw[1] = np.nan
    np.savez(path, X_raw=raw, dates=dates.to_numpy(), stocks=np.array(codes))
    second = build_input_mask(path, target, codes, tmp_path/'second.npz', 2, 'hybrid1455', 9, latest)
    assert np.array_equal(first[0], second[0])


@pytest.mark.parametrize('variant', ['zero', 'identity', 'no_lap', 'graph1', 'graph2',
                                    'graph3', 'random', '5m_nograph', 'gcn_lstm'])
def test_synthetic_training_and_selection(tmp_path, variant):
    panel, r, flags, config, graph = synthetic()
    summary, daily = fit_one(panel, r, flags, config, variant, 0, lambda t: graph,
                             tmp_path/variant, save_details=True)
    history = pd.read_csv(tmp_path/variant/'training.csv')
    assert summary['best_epoch'] == int(history.loc[history.validation_mse.idxmin(), 'epoch'])
    assert summary['test_observations'] == 120
    assert len(daily) == 10
    assert (tmp_path/variant/'checkpoint.pt').is_file()


def test_test_targets_cannot_change_selected_training_weights(tmp_path):
    panel, r, flags, config, graph = synthetic()
    first, _ = fit_one(panel, r, flags, config, 'graph2', 3, lambda t: graph, tmp_path/'a')
    panel.y[24:] += 10000
    second, _ = fit_one(panel, r, flags, config, 'graph2', 3, lambda t: graph, tmp_path/'b')
    assert first['best_epoch'] == second['best_epoch']
    a = torch.load(tmp_path/'a/checkpoint.pt', weights_only=True)['state_dict']
    b = torch.load(tmp_path/'b/checkpoint.pt', weights_only=True)['state_dict']
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_cli_load_train_and_target_free_prediction(tmp_path):
    panel, r, flags, config, graph = synthetic()
    meta = tmp_path/'meta.npz'
    common = tmp_path/'common.npz'
    mask = tmp_path/'mask.npz'
    raw = tmp_path/'raw.npz'
    returns = tmp_path/'returns.npz'
    raw_dates = pd.bdate_range(end=panel.dates[-1], periods=len(panel.dates)+2)
    np.savez(raw, X_raw=np.ones((len(raw_dates), 12, 9)), dates=raw_dates.to_numpy(),
             stocks=np.array(panel.codes))
    build_input_mask(raw, panel.dates, panel.codes, mask, lookback=2)
    np.savez(meta, all_dates=panel.dates.to_numpy(), ts_codes_order=np.array(panel.codes),
             feature_names=np.array(FEATURES))
    np.savez(common, X_np=panel.X, y_np=panel.y, mask_np=panel.target_observed)
    np.savez(returns, returns=r, bar_available=flags, dates=panel.dates.to_numpy(), stocks=np.array(panel.codes))
    shared = ['--common', str(common), '--meta', str(meta), '--returns', str(returns),
              '--input-mask', str(mask), '--out', str(tmp_path/'runs')]
    args = parser().parse_args(['train']+shared)
    run(args, config_override=config)
    np.savez(common, X_np=panel.X)
    args = parser().parse_args(['predict']+shared+['--checkpoint', str(tmp_path/'runs/results/graph2/seed_0/checkpoint.pt')])
    predicted = run(args)
    assert len(predicted) == 120
    assert 'y_true' not in predicted


def test_frozen_transform_is_independent_of_future_values():
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(20, 4, 9))
    transform = FrozenTransform.fit(raw[:10])
    first = transform.transform(raw[:10])
    raw[10:] = 1e9
    assert np.array_equal(first, transform.transform(raw[:10]))


def test_calibration_converts_percentage_points_exactly_once(tmp_path):
    frame = pd.DataFrame({'date': pd.bdate_range('2024-01-02', periods=10),
                          'stock': ['A']*10, 'y_pred': np.arange(10)/10,
                          'y_true': np.arange(10)/10})
    file = tmp_path/'pred.csv'
    frame.to_csv(file, index=False)
    calibrate_file(file, tmp_path/'cal', fraction=1.)
    actual = calibrated_returns([.3], tmp_path/'cal/calibration.npz')
    assert actual[0] == pytest.approx(.003)
    frame.loc[0, 'date'] = pd.Timestamp('2025-01-02')
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match='pre-2025'):
        calibrate_file(file, tmp_path/'bad')


def test_lagged_exposure_does_not_use_target_day_returns():
    risk = pd.DataFrame({'date': pd.bdate_range('2024-10-01', '2025-01-31'),
                         'market_overnight_vol': .01})
    first = risk_exposure(risk)
    risk.loc[risk.date >= '2025-01-02', 'market_overnight_vol'] = 100
    second = risk_exposure(risk)
    assert first.loc['2025-01-02'] == second.loc['2025-01-02']


def test_blocked_top_rank_order_is_not_replaced():
    predictions = pd.DataFrame({'date': ['2025-01-02']*2, 'stock': ['A', 'B'],
                                'y_pred': [1., .9], 'u': [.01, .009]})
    executions = pd.DataFrame({'date': ['2025-01-02']*2, 'stock': ['A', 'B'],
        'entry_time': ['2024-12-31 15:00']*2, 'exit_time': ['2025-01-02 09:30']*2,
        'entry_price': [10., 10.], 'exit_price': [11., 11.], 'entry_executable': [False, True]})
    exposure = pd.Series([1.], index=pd.to_datetime(['2025-01-02']))
    daily, trades = conditional_portfolio(predictions, executions, exposure, 1)
    assert trades.empty and daily.nav.iloc[-1] == 100000


def test_delayed_exit_locks_top_one_capital():
    dates = ['2025-01-02', '2025-01-03']
    predictions = pd.DataFrame({'date': dates, 'stock': ['A', 'B'], 'y_pred': [1., 1.], 'u': [.01, .01]})
    executions = pd.DataFrame({'date': dates, 'stock': ['A', 'B'],
        'entry_time': ['2024-12-31 15:00', '2025-01-02 15:00'],
        'exit_time': ['2025-01-03 09:30', '2025-01-03 09:30'],
        'entry_price': [10., 10.], 'exit_price': [11., 20.], 'entry_executable': [True, True]})
    exposure = pd.Series([1., 1.], index=pd.to_datetime(dates))
    daily, trades = conditional_portfolio(predictions, executions, exposure, 1)
    assert trades.stock.tolist() == ['A']
    assert daily.unsettled_positions.tolist() == [1, 0]


def test_common_coverage_rejects_changed_targets(tmp_path):
    frame = pd.DataFrame({'date': ['2025-01-02', '2025-01-03', '2025-01-06'],
                          'stock': ['A']*3, 'y_true': [1., 2., 3.], 'y_pred': [.9, 1.9, 2.9]})
    paths = []
    for i in range(4):
        path = tmp_path/f'p{i}.csv'
        frame.to_csv(path, index=False)
        paths.append(path)
    compare_coverage(*paths, tmp_path/'good')
    frame.loc[0, 'y_true'] = 9
    frame.to_csv(paths[-1], index=False)
    with pytest.raises(ValueError, match='targets differ'):
        compare_coverage(*paths, tmp_path/'bad')


def test_supplementary_ranking_and_checkpoint_selection(tmp_path):
    panel, r, flags, config, graph = synthetic()
    config = replace(config, ranking=.05, selection='weighted_topk')
    summary, _ = fit_one(panel, r, flags, config, 'graph2', 0, lambda t: graph,
                         tmp_path/'supplementary')
    history = pd.read_csv(tmp_path/'supplementary/training.csv')
    assert summary['ranking_loss'] is True
    assert summary['best_epoch'] == int(history.loc[history.validation_selection_value.idxmin(), 'epoch'])


def test_topology_pipeline_averages_randomizations_before_inference(tmp_path):
    from papergraph.cli import random_experiment
    panel, r, flags, config, graph = synthetic()
    panel.identity['returns_sha256'] = 'synthetic'
    config = replace(config, epochs=1)
    frame, inference = random_experiment(panel, r, flags, config, tmp_path/'topology',
                                         'cpu', [0, 1], [0, 1])
    assert len(frame) == 2 and inference['days'] == 10
    assert len(pd.read_csv(tmp_path/'topology/topology_all_random_runs.csv')) == 4


def test_ridge_and_transformer_benchmarks(tmp_path):
    from papergraph.benchmarks import run_benchmarks
    panel, r, flags, config, graph = synthetic()
    summary = run_benchmarks(panel, r, flags, config, tmp_path/'bench',
                              'cpu', 0, models=['ridge', 'transformer'])
    assert len(summary) == 2
    assert all(np.isfinite(row['test_mse']) for row in summary)


def test_quarterly_screen_uses_only_pre_target_inputs(tmp_path):
    from papergraph.feature_selection import screen_features
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2022-04-01', periods=30)
    x = rng.normal(size=len(dates))
    frame = pd.DataFrame({'date': dates, 'asof': dates-pd.Timedelta(days=1),
                          'y': 3*x+rng.normal(0, .001, len(x)), 'signal': x})
    path = tmp_path/'candidates.csv'
    frame.to_csv(path, index=False)
    selected = screen_features(path, ['signal'], tmp_path/'screen', folds=3, alphas=8,
                                first_quarter='2022Q2', last_quarter='2022Q2')
    assert selected == ['signal']


def test_hybrid_builder_ignores_latest_full_close_and_flow_fields(tmp_path):
    from papergraph.hybrid import build_latest
    path, dates, stocks = make_raw(tmp_path)
    transform = FrozenTransform(np.zeros(9), np.zeros(9), np.ones(9))
    scaler = tmp_path/'scaler.npz'
    transform.save(scaler)
    quotes = pd.DataFrame([{'date': day, 'stock': stock, 'price1455': 10.1,
                            'day_open': 10., 'previous_close': 10.}
                           for day in dates for stock in stocks])
    quote_path = tmp_path/'quotes.csv'
    quotes.to_csv(quote_path, index=False)
    first = build_latest(path, quote_path, dates[2:], stocks, scaler, tmp_path/'a')
    raw = np.ones((10, 3, 9))
    raw[1, :, FEATURES.index('D_pct_chg')] = 999.
    raw[1, :, FEATURES.index('D_intraday_return')] = 999.
    raw[1, :, FEATURES.index('D_mainforce_strength')] = 999.
    raw[1, :, FEATURES.index('D_bull_bear_ratio')] = 999.
    np.savez(path, X_raw=raw, dates=dates.to_numpy(), stocks=np.array(stocks), feature_names=np.array(FEATURES))
    second = build_latest(path, quote_path, dates[2:], stocks, scaler, tmp_path/'b')
    assert np.array_equal(first[0], second[0])
