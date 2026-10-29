"""Current PI calibration reference, using no Cloud or remembered response.

Only the already observed calibration segment is complete when this function
runs. The surrounding episode is not complete. The frozen ARX fitter is reused
on that segment; no older episode identity is fabricated to pass a memory gate.
"""
from __future__ import annotations

import math

import numpy as np

from residual_experience import POLICY, fit_residual_experience


SCIENTIFIC_STATUS = 'current_prefix_estimate_not_cloud_memory'


def fresh_prefix_correction(*, current_reference, current_measured_residual,
        current_applied_command, nominal_feedforward, prefix_samples, dt_s,
        command_limit, current_context, current_episode_index, calibration_id):
    """Fit observed PI data, then apply a fixed, bounded future FF scale.

    All sensor/command vectors are sliced before fit or finite-value validation.
    Future nominal FF is checked for transport integrity and transformed only
    after selection; it never selects the ARX model or its scale. The caller
    still applies the plant's ordinary command and slew limits.

    ``decision['applied']`` means a nonunit scale was selected. A zero/clipped
    nominal FF can still have no effect: inspect ``effective_ff_max_abs_delta``.
    This reference does not validate a stored model, retrieve memory, or claim
    that a fitted effective response guarantees future performance.
    """
    nominal = np.asarray(nominal_feedforward, dtype=float)
    if (nominal.ndim != 1 or not np.all(np.isfinite(nominal))
            or type(prefix_samples) is not int or not 1 <= prefix_samples <= len(nominal)):
        raise ValueError('invalid_feedforward_or_prefix')
    if (type(dt_s) not in (int, float) or not math.isfinite(dt_s) or dt_s <= 0
            or type(command_limit) not in (int, float)
            or not math.isfinite(command_limit) or command_limit <= 0):
        raise ValueError('invalid_numerical_metadata')
    if type(current_episode_index) is not int or current_episode_index < 0:
        raise ValueError('current_episode_identity_required')
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError('current_calibration_identity_required')
    if any(prefix_samples > len(v) for v in
           (current_reference, current_measured_residual, current_applied_command)):
        raise ValueError('insufficient_current_observed_prefix')
    if np.any(nominal[:prefix_samples] != 0):
        raise ValueError('current_prefix_not_PI_only')

    result = nominal.copy()
    provenance = {'scientific_status': 'simulated_sensor_data', 'completed': True,
        'baseline_policy': 'pi_only', 'source': 'observed_sensor_trace'}
    # The fitter receives only the complete prefix, never the unfinished episode.
    prefix = [v[:prefix_samples] for v in
              (current_reference, current_measured_residual, current_applied_command)]
    fit = fit_residual_experience(*prefix, dt_s=dt_s, context=current_context,
        episode_index=current_episode_index, episode_id=calibration_id,
        baseline_slice=(0, prefix_samples), command_limit=command_limit,
        provenance=provenance)
    decision = {'schema': 'current-prefix-reference/v1',
        'scientific_status': SCIENTIFIC_STATUS, 'data_status': 'simulated_sensor_data',
        'uses_cloud_memory': False, 'uses_future_observations': False,
        'isolated_physical_benefit_claimed': False,
        'current_episode_index': current_episode_index, 'calibration_id': calibration_id,
        'calibration_provenance': {**provenance,
            'completed_unit': 'observed_PI_calibration_prefix',
            'entire_episode_completed': False,
            'fitter_episode_index_semantics': 'current episode containing the completed prefix'},
        'calibration_interval': {'start_step': 0, 'end_exclusive': prefix_samples,
            'start_time_s': 0., 'last_observed_time_s': (prefix_samples-1)*float(dt_s)},
        'max_observed_step': prefix_samples-1, 'correction_start_step': prefix_samples,
        'applied': False, 'scale': 1., 'reason': 'current_prefix_identification_rejected',
        'effective_ff_max_abs_delta': 0., 'effective_ff_changed': False,
        'fit': fit}
    if fit['accepted'] is not True:
        return result, decision

    model = fit['model']; gain, low, high = (float(model[k]) for k in ('g', 'lo', 'hi'))
    decision.update(effective_gain=gain, approximate_gain_interval=[low, high],
        delay_samples=model['d'], calibration_source_sha256=model['src'])
    if (not all(math.isfinite(v) for v in (gain, low, high))
            or not .25 <= gain <= 4 or not 0 <= low <= gain <= high
            or max(gain-low, high-gain)/gain > POLICY.maximum_relative_gain_half_width):
        decision['reason'] = 'invalid_or_uncertain_current_prefix_gain'
        return result, decision
    if low <= 1 <= high:
        decision['reason'] = 'gain_correction_direction_uncertain'
        return result, decision

    scale = float(np.clip(1/gain, 1-POLICY.maximum_relative_correction,
                          1+POLICY.maximum_relative_correction))
    # No phase, delay, bias or PI adaptation: a fixed future FF multiplier only.
    result[prefix_samples:] = np.clip(scale*nominal[prefix_samples:],
                                     -command_limit, command_limit)
    delta = float(np.max(np.abs(result-nominal)))
    decision.update(applied=True, scale=scale,
        reason='current_prefix_gain_validated_on_temporal_holdout',
        effective_ff_max_abs_delta=delta, effective_ff_changed=delta > 0.)
    return result, decision
