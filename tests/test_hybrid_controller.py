"""Causal controller tests on analytic fixtures; never live Cloud evidence."""
import copy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from cloud_campaign import MODEL_SCHEMA, selected_forecast
from hybrid_controller import DEFAULT_POLICY, HybridPolicy, hybrid_forecast, policy_manifest
from simulation import Config, plant


def model(key, frequency, completed=0):
    return {'candidate_id': key, 'retrieval_order': 0, 'tsst_score': .8,
            'model': {'schema': MODEL_SCHEMA,
                      'scientific_status': 'simulated_sensor_data',
                      'completed_episode_index': completed,
                      'frequency_hz': frequency, 'quality_score': .9}}


def measured_fixture(c, *, noise=True):
    t = np.arange(round(c.duration/c.dt))*c.dt
    ref = np.sin(2*np.pi*1.73*t+.4)
    if noise:
        ref += np.random.default_rng(88193).normal(0., .015, len(t))
    return ref


@pytest.fixture
def config():
    return Config(dt=.01, duration=2.5)


def test_future_observations_cannot_change_earlier_ff_or_decisions(config):
    reference = measured_fixture(config)
    changed = reference.copy(); cut = 141
    changed[cut:] = np.random.default_rng(99391).normal(0., 20., len(changed)-cut)
    candidates = [model('past-a', 1.72), model('past-b', 1.9)]
    a, decisions_a = hybrid_forecast(reference, config, 'hybrid', candidates, episode_index=20)
    b, decisions_b = hybrid_forecast(changed, config, 'hybrid', candidates, episode_index=20)
    np.testing.assert_array_equal(a[:cut], b[:cut])
    assert [d for d in decisions_a if d['step'] < cut] == [d for d in decisions_b if d['step'] < cut]
    for decision in decisions_a:
        assert decision['max_observed_step'] == decision['step']
        for candidate in decision['candidate_evaluations']:
            if 'training_end_step' in candidate:
                assert candidate['training_end_step'] < candidate['validation_start_step']
                assert candidate['validation_start_step'] <= candidate['validation_end_step'] <= decision['step']


def test_append_only_future_preserves_existing_ff_prefix(config):
    reference = measured_fixture(config)
    candidates = [model('past', 1.72)]
    short, _ = hybrid_forecast(reference[:143], config, 'hybrid', candidates, episode_index=20)
    long, _ = hybrid_forecast(reference, config, 'hybrid', candidates, episode_index=20)
    np.testing.assert_array_equal(short, long[:len(short)])


def test_compatible_memory_really_changes_ff_and_bounded_plant_command(config):
    reference = measured_fixture(config)
    online, _ = hybrid_forecast(reference, config, 'online', episode_index=20)
    hybrid, decisions = hybrid_forecast(reference, config, 'hybrid',
                                       [model('past', 1.72)], episode_index=20)
    used = [d for d in decisions if d['memory_contributed']]
    assert used and any(d['feedforward_contribution_max_abs'] > 1e-7 for d in used)
    assert not np.array_equal(online, hybrid)
    for decision in used:
        assert 0 < decision['memory_weight'] <= .30
        assert .70 <= decision['online_weight'] < 1.
        assert decision['attributable_candidate_ids'] == ['past']
    noise = np.zeros_like(reference)
    online_trace = plant(reference, noise, online, config, (1.4, 110.))
    hybrid_trace = plant(reference, noise, hybrid, config, (1.4, 110.))
    assert np.max(np.abs(online_trace['command']-hybrid_trace['command'])) > 1e-7


def test_wrong_but_physically_qualified_memory_loses_weight_via_holdout_error(config):
    reference = measured_fixture(config)
    _, good = hybrid_forecast(reference, config, 'hybrid', [model('near', 1.72)], episode_index=20)
    _, wrong = hybrid_forecast(reference, config, 'hybrid', [model('wrong', 1.95)], episode_index=20)
    a = good[0]['candidate_evaluations'][0]
    b = wrong[0]['candidate_evaluations'][0]
    # Both survive the SAME instantaneous fit gate; the observed holdout is
    # what reduces confidence in the erroneous prior, not a prior-ID blacklist.
    assert min(a['current_window_explained'], b['current_window_explained']) >= config.minimum_explained
    assert b['memory_holdout_mse'] > a['memory_holdout_mse']
    assert b['relative_innovation_compatibility'] < a['relative_innovation_compatibility']
    assert b['weight'] < a['weight']


def test_first_compatible_identical_prior_corroborates_without_seeking_a_perturbation(config):
    reference = measured_fixture(config, noise=False)
    online, _ = hybrid_forecast(reference, config, 'online', episode_index=20)
    hybrid, decisions = hybrid_forecast(reference, config, 'hybrid',
        [model('first-identical', 1.73), model('second-different', 1.95)], episode_index=20)
    np.testing.assert_array_equal(hybrid, online)
    assert all(d.get('corroborating_candidate_id') == 'first-identical' for d in decisions if d['accepted'])
    assert all(not d['memory_contributed'] and not d['attributable_candidate_ids'] for d in decisions)
    assert all(len(d['candidate_evaluations']) == 1 for d in decisions if d['accepted'])


def test_zero_weight_and_empty_history_preserve_online_exactly_and_get_no_credit(config):
    reference = measured_fixture(config)
    online, _ = hybrid_forecast(reference, config, 'online', episode_index=20)
    for candidates, policy in (([], DEFAULT_POLICY),
            ([model('prior', 1.72)], replace(DEFAULT_POLICY, max_memory_weight=0.))):
        hybrid, decisions = hybrid_forecast(reference, config, 'hybrid', candidates,
                                            episode_index=20, policy=policy)
        np.testing.assert_array_equal(hybrid, online)
        assert all(d['memory_weight'] == 0 and d['selected_candidate_id'] is None for d in decisions)
        assert all(not d['memory_contributed'] and not d['attributable_candidate_ids'] for d in decisions)


def test_zero_blend_has_no_memory_attribution_even_with_distinct_predictions(config):
    ff, decisions = hybrid_forecast(measured_fixture(config), config, 'hybrid',
        [model('prior', 1.72)], blend=0., episode_index=20)
    np.testing.assert_array_equal(ff, np.zeros_like(ff))
    assert all(not d['memory_contributed'] and d['memory_weight'] == 0 for d in decisions)


def test_clipping_cannot_manufacture_credit_for_an_ineffective_prior(config):
    c = replace(config, command_limit=.4)
    reference = 10.+measured_fixture(c)
    online, _ = hybrid_forecast(reference, c, 'online', episode_index=20)
    hybrid, decisions = hybrid_forecast(reference, c, 'hybrid',
        [model('distinct-but-clipped', 1.72)], episode_index=20)
    np.testing.assert_array_equal(hybrid, online)
    assert all(not d['memory_contributed'] and not d['attributable_candidate_ids'] for d in decisions)
    assert all(d['selected_candidate_id'] is None for d in decisions)


def test_same_actual_candidate_order_can_change_hybrid_without_changing_online(config):
    reference = measured_fixture(config)
    candidates = [model('first', 1.65), model('second', 1.9)]
    a, decisions_a = hybrid_forecast(reference, config, 'hybrid', candidates, episode_index=20)
    b, decisions_b = hybrid_forecast(reference, config, 'hybrid', candidates[::-1], episode_index=20)
    assert decisions_a[0]['selected_candidate_id'] == 'first'
    assert decisions_b[0]['selected_candidate_id'] == 'second'
    assert not np.array_equal(a, b)
    left, _ = hybrid_forecast(reference, config, 'online', candidates, episode_index=20)
    right, _ = hybrid_forecast(reference, config, 'online', candidates[::-1], episode_index=20)
    np.testing.assert_array_equal(left, right)


def test_current_future_and_unconfirmed_provenance_cannot_enter_hybrid(config):
    reference = measured_fixture(config)
    candidates = [model('current', 1.72, 20), model('future', 1.72, 21), model('foreign', 1.72)]
    candidates[2]['model']['scientific_status'] = 'unconfirmed'
    online, _ = hybrid_forecast(reference, config, 'online', episode_index=20)
    hybrid, decisions = hybrid_forecast(reference, config, 'hybrid', candidates, episode_index=20)
    np.testing.assert_array_equal(hybrid, online)
    assert all(d['selected_candidate_id'] is None for d in decisions)
    assert {r['reason'] for r in decisions[0]['candidate_evaluations']} == {
        'not_completed_before_current_episode', 'unsupported_model_provenance'}


def test_outcome_labels_and_tsst_score_do_not_determine_numerical_confidence(config):
    reference = measured_fixture(config)
    candidates = [model('prior', 1.72)]
    altered = copy.deepcopy(candidates)
    altered[0]['tsst_score'] = -9000.
    altered[0]['model'].update(quality_score=-9000., observed_success=False,
        simulator_future_frequency_hz=7.5, rms_uT=0.)
    before, decisions = hybrid_forecast(reference, config, 'hybrid', candidates, episode_index=20)
    after, changed = hybrid_forecast(reference, config, 'hybrid', altered, episode_index=20)
    np.testing.assert_array_equal(before, after)
    assert decisions == changed


def test_full_online_wrappers_preserve_frozen_comparison_and_do_not_mutate_inputs(config):
    reference = measured_fixture(config)
    candidates = [model('prior', 1.72)]
    unchanged = copy.deepcopy(candidates)
    for arm, expected_candidates in (('online', []), ('full', candidates)):
        actual, _ = hybrid_forecast(reference, config, arm, candidates, episode_index=20)
        expected, _ = selected_forecast(reference, config, arm, expected_candidates)
        np.testing.assert_array_equal(actual, expected)
    assert candidates == unchanged


def test_feedforward_and_final_commands_keep_common_bounds_and_slew(config):
    c = replace(config, command_limit=.4, slew_per_second=3.)
    reference = 15*measured_fixture(c)
    ff, _ = hybrid_forecast(reference, c, 'hybrid', [model('prior', 1.72)], episode_index=20)
    trace = plant(reference, np.zeros_like(reference), ff, c, (1.4, 110.))
    assert np.max(np.abs(ff)) <= c.command_limit+1e-12
    assert np.max(np.abs(trace['command'])) <= c.command_limit+1e-12
    assert np.max(np.abs(np.diff(np.r_[0., trace['command']]))) <= c.slew_per_second*c.dt+1e-12


def test_policy_is_frozen_a_priori_and_cannot_silently_replace_the_online_estimator():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.max_memory_weight = .8
    with pytest.raises(ValueError, match='70_percent_online'):
        HybridPolicy(max_memory_weight=.31)
    manifest = policy_manifest()
    assert manifest['parameters']['max_memory_weight'] == .30
    assert 'no claim of optimality' in manifest['choice_status']
