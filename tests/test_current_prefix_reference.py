"""Analytic observed-channel fixtures, independent of prospective experiments."""
import numpy as np
import pytest

import current_prefix_reference as current


CONTEXT = {'device_id': 'analytic-current-prefix', 'control_mode': 'pi-baseline'}


def measured_fixture(*, gain=.65, noise=.001, count=500):
    rng = np.random.default_rng(719)
    command = np.repeat(rng.uniform(-1, 1, (count+4)//5), 5)[:count]
    contrast = np.zeros(count)
    for k in range(1, count):
        contrast[k] = .8*contrast[k-1]+gain*.2*(command[k-3] if k >= 3 else 0)+.002
    reference = np.sin(np.arange(count)*.043)
    measured = reference+contrast+rng.normal(0, noise, count)
    return reference, measured, command


def call(channels=None, **kwargs):
    ref, measured, command = measured_fixture() if channels is None else channels
    args = {'current_reference': ref, 'current_measured_residual': measured,
        'current_applied_command': command, 'current_context': CONTEXT,
        'current_episode_index': 7, 'calibration_id': 'analytic-prefix-7',
        'prefix_samples': 500, 'nominal_feedforward': np.r_[np.zeros(500), np.ones(100)],
        'dt_s': .002, 'command_limit': 8}
    args.update(kwargs)
    return current.fresh_prefix_correction(**args)


def test_uses_current_index_and_completed_prefix_without_claiming_past_episode():
    ff, decision = call()
    assert decision['applied'] and decision['scale'] == 1.2
    assert decision['fit']['model']['ep'] == decision['current_episode_index'] == 7
    assert decision['fit']['diagnostics']['episode_id'] == 'analytic-prefix-7'
    assert decision['fit']['diagnostics']['baseline_slice'] == [0, 500]
    assert decision['scientific_status'] == 'current_prefix_estimate_not_cloud_memory'
    assert decision['uses_cloud_memory'] is False
    assert decision['calibration_provenance']['completed'] is True
    assert decision['calibration_provenance']['completed_unit'] == 'observed_PI_calibration_prefix'
    assert decision['calibration_provenance']['entire_episode_completed'] is False
    assert decision['max_observed_step'] == 499 and decision['correction_start_step'] == 500
    assert decision['fit']['diagnostics']['training_end_exclusive'] == 325
    assert decision['fit']['diagnostics']['validation_end_exclusive'] == 500
    np.testing.assert_array_equal(ff[:500], 0.)
    np.testing.assert_array_equal(ff[500:], 1.2)


def test_only_prefix_arrays_reach_fitter_and_future_sensors_cannot_change_choice(monkeypatch):
    original = current.fit_residual_experience
    received = []
    def capture(*channels, **kwargs):
        received.append((tuple(len(v) for v in channels), kwargs))
        return original(*channels, **kwargs)
    monkeypatch.setattr(current, 'fit_residual_experience', capture)
    channels = measured_fixture(); before_ff, before = call(channels)
    future = tuple(np.r_[v, np.full(100, np.nan)] for v in channels)
    after_ff, after = call(future)
    assert received[0][0] == received[1][0] == (500, 500, 500)
    assert received[1][1]['episode_index'] == 7
    assert received[1][1]['baseline_slice'] == (0, 500)
    assert before == after
    np.testing.assert_array_equal(before_ff, after_ff)


def test_future_nominal_actions_do_not_select_model_or_scale():
    _, before = call()
    _, after = call(nominal_feedforward=np.r_[np.zeros(500), np.linspace(-9, 9, 100)])
    for key in ('fit', 'scale', 'reason', 'applied', 'effective_gain', 'approximate_gain_interval'):
        assert before[key] == after[key]


def test_input_vectors_not_mutated():
    channels = measured_fixture(); saved = [v.copy() for v in channels]
    nominal = np.r_[np.zeros(500), np.ones(100)]
    call(channels, nominal_feedforward=nominal)
    for actual, expected in zip(channels, saved): np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(nominal, np.r_[np.zeros(500), np.ones(100)])


def test_gain_interval_including_unity_abstains():
    ff, decision = call(measured_fixture(gain=1.))
    assert decision['fit']['accepted']
    assert not decision['applied'] and decision['reason'] == 'gain_correction_direction_uncertain'
    np.testing.assert_array_equal(ff, np.r_[np.zeros(500), np.ones(100)])


def test_temporal_validation_failure_abstains_instead_of_retuning():
    ref, measured, command = measured_fixture()
    measured[325:] += np.random.default_rng(829).normal(0, 3, 175)
    ff, decision = call((ref, measured, command))
    assert not decision['applied']
    assert 'heldout_response_prediction_failed' in decision['fit']['reasons']
    np.testing.assert_array_equal(ff, np.r_[np.zeros(500), np.ones(100)])


def test_saturated_or_unexcited_prefix_cannot_create_correction():
    ref, measured, command = measured_fixture(); command[:50] = 8
    _, decision = call((ref, measured, command))
    assert not decision['applied'] and 'saturated_baseline' in decision['fit']['reasons']
    _, decision = call((ref, measured, np.zeros_like(command)))
    assert not decision['applied'] and 'insufficient_command_excitation' in decision['fit']['reasons']


def test_low_and_high_gain_scales_bounded_and_future_ff_clipped():
    ff, decision = call(nominal_feedforward=np.r_[np.zeros(500), np.ones(100)*7.9])
    assert decision['scale'] == 1.2 and np.max(np.abs(ff)) <= 8
    ff, decision = call(measured_fixture(gain=1.5))
    assert decision['applied'] and decision['scale'] == .8
    np.testing.assert_array_equal(ff[500:], .8)


def test_selected_scale_without_nominal_ff_reports_zero_effect():
    ff, decision = call(nominal_feedforward=np.zeros(600))
    assert decision['applied'] and decision['scale'] == 1.2
    assert decision['effective_ff_max_abs_delta'] == 0 and decision['effective_ff_changed'] is False
    np.testing.assert_array_equal(ff, 0.)


def test_PI_prefix_and_explicit_observable_context_required():
    with pytest.raises(ValueError, match='PI_only'):
        call(nominal_feedforward=np.ones(600))
    with pytest.raises(ValueError, match='observable_context'):
        call(current_context={'device_id': 'A', 'hidden_gain': '.65'})
    with pytest.raises(ValueError, match='current_episode_identity'):
        call(current_episode_index=True)
    with pytest.raises(ValueError, match='calibration_identity'):
        call(calibration_id=' ')


def test_short_nonfinite_and_invalid_metadata_fail_closed():
    channels = measured_fixture()
    with pytest.raises(ValueError, match='insufficient_current'):
        call(tuple(v[:100] for v in channels))
    channels[0][10] = np.nan
    with pytest.raises(ValueError, match='finite_observed'):
        call(channels)
    for update in ({'dt_s': float('nan')}, {'command_limit': 0}, {'prefix_samples': True},
                   {'nominal_feedforward': np.r_[np.zeros(500), np.nan]}):
        with pytest.raises(ValueError): call(**update)


def test_fit_source_hash_binds_observed_calibration_and_metadata():
    _, before = call(); channels = measured_fixture(); channels[1][10] += .001
    _, after = call(channels)
    assert before['fit']['diagnostics']['source_sha256'] != after['fit']['diagnostics']['source_sha256']
    _, renamed = call(calibration_id='other-calibration')
    assert before['fit']['diagnostics']['source_sha256'] != renamed['fit']['diagnostics']['source_sha256']
