"""No-network equivalence and causality tests; fixtures are not Cloud results."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest

from cloud_campaign import MODEL_SCHEMA, order_candidates
import hybrid_controller as original
from optimized_controller import (DEFAULT_UTILITY_POLICY, UtilityPolicy,
    cached_hybrid_forecast, utility_forecast, utility_policy_manifest)
from simulation import Config, plant


def model(key, frequency, completed=0):
    return {'candidate_id': key, 'model': {'schema': MODEL_SCHEMA,
        'scientific_status': 'simulated_sensor_data',
        'completed_episode_index': completed, 'frequency_hz': frequency,
        'quality_score': .8}}


def reference(c, *, noise=.015, seed=9021):
    t = np.arange(round(c.duration/c.dt))*c.dt
    return np.sin(2*np.pi*1.73*t+.4)+np.random.default_rng(seed).normal(0., noise, len(t))


def candidates():
    return [model(str(i), f) for i, f in enumerate((1.5, 1.72, 1.72, 1.95, 4., 1.73))]


@pytest.mark.parametrize('mode', ['online', 'full', 'hybrid'])
@pytest.mark.parametrize('variant', ['ordinary', 'zero_weight', 'zero_blend', 'clipped', 'quiet', 'changed_regime'])
def test_cached_outputs_and_every_decision_are_bitwise_equivalent(mode, variant):
    c = Config(dt=.01, duration=2.5); signal = reference(c)
    policy = original.DEFAULT_POLICY; blend = .6
    if variant == 'zero_weight': policy = replace(policy, max_memory_weight=0.)
    if variant == 'zero_blend': blend = 0.
    if variant == 'clipped': c = replace(c, command_limit=.4); signal += 10.
    if variant == 'quiet': signal[:] = 0.
    if variant == 'changed_regime': signal[141:] *= -1.
    rows = candidates()+[model('future', 1.73, 30)]
    expected, expected_decisions = original.hybrid_forecast(signal, c, mode, rows,
        blend=blend, episode_index=30, policy=policy)
    diagnostics = {}
    actual, decisions = cached_hybrid_forecast(signal, c, mode, rows,
        blend=blend, episode_index=30, policy=policy, diagnostics=diagnostics)
    np.testing.assert_array_equal(actual, expected)
    assert decisions == expected_decisions
    assert diagnostics['cloud_calls'] == 0
    assert diagnostics['latency_wall_s'] >= 0


def test_shared_holdout_estimator_removes_repeated_searches_without_changing_decisions(monkeypatch):
    c = Config(dt=.01, duration=2.5); signal = reference(c)
    rows = [model(str(i), f) for i, f in enumerate(np.linspace(1.5, 1.95, 12).tolist())]
    # Zero weight forces v1 to evaluate every compatible prior. It is a
    # contract fixture, not a claim that every live workload has this speedup.
    policy = replace(original.DEFAULT_POLICY, max_memory_weight=0.)
    calls = []; estimator = original.estimate_frequency
    def counted(*args, **kwargs):
        calls.append(1); return estimator(*args, **kwargs)
    monkeypatch.setattr(original, 'estimate_frequency', counted)
    expected, expected_decisions = original.hybrid_forecast(signal, c, 'hybrid', rows,
        episode_index=30, policy=policy)
    diagnostics = {}
    actual, decisions = cached_hybrid_forecast(signal, c, ordered=rows,
        episode_index=30, policy=policy, diagnostics=diagnostics)
    np.testing.assert_array_equal(actual, expected)
    assert decisions == expected_decisions
    assert diagnostics['frequency_estimator_scans'] <= 2*len(decisions)
    assert diagnostics['frequency_estimator_scans'] < len(calls)
    assert diagnostics['direct_harmonic_fit_calls'] <= 2*len(decisions)




@pytest.mark.parametrize('controller', [cached_hybrid_forecast, utility_forecast])
def test_future_suffix_and_append_only_input_cannot_change_earlier_control(controller):
    c = Config(dt=.01, duration=4.); signal = reference(c, noise=.08, seed=88319)
    rows = [model('near', 1.73), model('wrong', 1.95)]
    changed = signal.copy(); changed[143:] = 20.
    a, da = controller(signal, c, ordered=rows, episode_index=30)
    b, db = controller(changed, c, ordered=rows, episode_index=30)
    short, _ = controller(signal[:143], c, ordered=rows, episode_index=30)
    np.testing.assert_array_equal(a[:143], b[:143])
    np.testing.assert_array_equal(short, a[:143])
    assert [d for d in da if d['step'] < 143] == [d for d in db if d['step'] < 143]
    for decision in da:
        for entry in decision['candidate_evaluations']:
            if 'training_end_step' in entry:
                assert entry['training_end_step'] < entry['validation_start_step'] <= entry['validation_end_step'] <= decision['step']


def test_utility_requires_advantage_and_does_not_force_memory_when_online_is_exact():
    c = Config(dt=.01, duration=2.5); signal = reference(c, noise=0.)
    expected, _ = original.hybrid_forecast(signal, c, 'online', episode_index=30)
    actual, decisions = utility_forecast(signal, c, ordered=[model('identical', 1.73),
        model('wrong', 1.95)], episode_index=30)
    np.testing.assert_array_equal(actual, expected)
    assert all(d['memory_weight'] == 0 and not d['attributable_candidate_ids'] for d in decisions)
    assert all(d['reason'] == 'no_memory_with_observed_conservative_advantage' for d in decisions)


def test_utility_evaluates_all_distinct_frequencies_and_can_override_a_worse_odvl_prior():
    c = Config(dt=.01, duration=4.); signal = reference(c, noise=.08, seed=88319)
    rows = [model('higher-ranked-wrong', 1.95), model('useful', 1.73), model('duplicate', 1.73)]
    diagnostics = {}
    ff, decisions = utility_forecast(signal, c, ordered=rows, episode_index=30, diagnostics=diagnostics)
    used = [d for d in decisions if d['memory_contributed']]
    assert used and all(d['selected_candidate_id'] == 'useful' for d in used)
    for decision in decisions:
        assert decision['evaluated_distinct_frequencies_hz'] == [1.95, 1.73]
        if decision['memory_contributed']:
            assert 0 < decision['memory_weight'] <= .30
            assert decision['online_weight'] >= .70
            assert decision['selected_conservative_holdout_gain'] > 0
            selected = next(e for e in decision['candidate_evaluations'] if e['candidate_id'] == 'useful')
            assert selected['observed_holdout_gain'] > selected['required_advantage_margin']
    online, _ = utility_forecast(signal, c, 'online', episode_index=30)
    assert np.max(np.abs(ff-online)) > 1e-7
    assert diagnostics['duplicate_candidates_skipped'] == len(decisions)
    assert diagnostics['frequency_estimator_scans'] <= 2*len(decisions)


def test_supplied_odvl_order_breaks_practically_equal_positive_utility_ties():
    c = Config(dt=.01, duration=4.); signal = reference(c, noise=.08, seed=88319)
    rows = [model('a', 1.73), model('b', 1.73000001)]
    _, left = utility_forecast(signal, c, ordered=rows, episode_index=30)
    _, right = utility_forecast(signal, c, ordered=rows[::-1], episode_index=30)
    assert left[-1]['selected_candidate_id'] == 'a'
    assert right[-1]['selected_candidate_id'] == 'b'
    assert abs(left[-1]['best_conservative_holdout_gain']-left[-1]['selected_conservative_holdout_gain']) < left[-1]['prior_tie_tolerance']


@pytest.mark.parametrize('variant', ['zero_blend', 'clipped'])
def test_utility_does_not_credit_memory_without_a_bounded_feedforward_effect(variant):
    c = Config(dt=.01, duration=4., command_limit=.4)
    signal = reference(c, noise=.08, seed=88319)
    blend = 0. if variant == 'zero_blend' else .6
    if variant == 'clipped': signal += 10.
    ff, decisions = utility_forecast(signal, c, ordered=[model('prior', 1.73)],
        blend=blend, episode_index=30)
    assert all(not d['memory_contributed'] and not d['attributable_candidate_ids'] for d in decisions)
    trace = plant(signal, np.zeros_like(signal), ff, c, (1.4, 110.))
    assert np.max(np.abs(ff)) <= c.command_limit+1e-12
    assert np.max(np.abs(trace['command'])) <= c.command_limit+1e-12
    assert np.max(np.abs(np.diff(np.r_[0., trace['command']]))) <= c.slew_per_second*c.dt+1e-12


def test_utility_confidence_does_not_read_outcome_labels_or_future_episode_memory():
    c = Config(dt=.01, duration=4.); signal = reference(c, noise=.08, seed=88319)
    rows = [model('useful', 1.73), model('future', 1.73, 31)]
    altered = copy.deepcopy(rows)
    altered[0]['model'].update(quality_score=-1000., observed_success=False,
                               generator_future_frequency_hz=7.)
    a, da = utility_forecast(signal, c, ordered=rows, episode_index=30)
    b, db = utility_forecast(signal, c, ordered=altered, episode_index=30)
    np.testing.assert_array_equal(a, b); assert da == db
    assert all('future' not in d['attributable_candidate_ids'] for d in da)


def test_policy_is_frozen_and_residual_extension_is_explicitly_unsupported():
    with pytest.raises(FrozenInstanceError): DEFAULT_UTILITY_POLICY.max_memory_weight = .9
    with pytest.raises(ValueError): UtilityPolicy(max_memory_weight=.31)
    manifest = utility_policy_manifest()
    assert manifest['parameters']['minimum_relative_advantage'] == .05
    assert manifest['parameters']['prior_tie_fraction'] == .01
    assert 'not statistical confidence bounds' in manifest['choice_status']
    with pytest.raises(NotImplementedError, match='contract_not_supplied'):
        utility_forecast(np.zeros(20), Config(), episode_index=30,
                         contextual_residual_correction={'unreviewed': True})
