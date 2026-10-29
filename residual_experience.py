"""Small causal ARX experience learned from completed, observed PI prefixes.

The observable target is measured_residual - measured_reference. With matched
reference/residual sensors it approximates the sensed actuator contribution.
Its fitted gain/delay are effective response parameters, not hidden plant truth.
The caller authenticates Cloud receipts; this module never reads a memory store.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ResidualPolicy:
    training_fraction: float = .65
    maximum_delay_s: float = .06
    minimum_training_samples: int = 64
    minimum_validation_samples: int = 32
    minimum_command_std_fraction: float = .0025
    minimum_sensor_contrast_std: float = .005
    maximum_saturation_fraction: float = .02
    maximum_condition: float = 1e5
    maximum_validation_nmse: float = .35
    maximum_relative_gain_half_width: float = .35
    maximum_current_gain_ratio_deviation: float = .15
    maximum_current_bias_in_std: float = .20
    maximum_relative_correction: float = .20


POLICY = ResidualPolicy()
CONTEXT_KEYS = frozenset({'device_id', 'reference_sensor_id', 'residual_sensor_id',
                          'control_mode', 'observable_regime', 'installation_id'})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _context_hash(context):
    if (not isinstance(context, Mapping) or not context.get('device_id')
            or set(context) - CONTEXT_KEYS or
            any(not isinstance(v, str) or not v.strip() or len(v) > 80
                for v in context.values())):
        raise ValueError('explicit_observable_context_required')
    return hashlib.sha256(_json(dict(context)).encode()).hexdigest()


def _vector(values):
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or not np.all(np.isfinite(out)):
        raise ValueError('finite_observed_vector_required')
    return out


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError('invalid_numerical_metadata')
    return float(value)


def _excitation(z, command, command_limit):
    reasons = []
    if np.std(command) < POLICY.minimum_command_std_fraction * command_limit:
        reasons.append('insufficient_command_excitation')
    if np.std(z) < POLICY.minimum_sensor_contrast_std:
        reasons.append('insufficient_sensor_contrast')
    saturation = float(np.mean(np.abs(command) >= command_limit * (1-1e-8)))
    if saturation > POLICY.maximum_saturation_fraction:
        reasons.append('saturated_baseline')
    return reasons, saturation


def _rollout(z, command, a, b, bias, delay, start, stop):
    """Seed once with an observed past state, then predict from past commands."""
    state = float(z[start-1]); prediction = []
    for k in range(start, stop):
        state = a*state + b*float(command[k-delay]) + bias
        prediction.append(state)
    return np.asarray(prediction)


def _fit_candidate(z, command, delay, start, stop):
    design = np.column_stack((z[start-1:stop-1], command[start-delay:stop-delay],
                              np.ones(stop-start)))
    scales = np.std(design[:, :2], axis=0)
    if np.min(scales) <= 1e-12:
        return None
    standardized = np.column_stack(((design[:, :2]-np.mean(design[:, :2], axis=0))/scales,
                                    np.ones(stop-start)))
    condition = float(np.linalg.cond(standardized))
    if not math.isfinite(condition) or condition > POLICY.maximum_condition:
        return None
    coef, _, rank, _ = np.linalg.lstsq(design, z[start:stop], rcond=None)
    a, b, bias = map(float, coef)
    if rank != 3 or not (0 <= a < .995 and b > 0):
        return None
    gain = b/(1-a)
    if not .25 <= gain <= 4:
        return None
    error = design @ coef-z[start:stop]
    variance = float(np.sum(error*error)/max(1, len(error)-3))
    covariance = variance*np.linalg.pinv(design.T @ design)
    gradient = np.asarray([b/(1-a)**2, 1/(1-a), 0.])
    half_width = 1.96*math.sqrt(max(0., float(gradient @ covariance @ gradient)))
    return {'a': a, 'b': b, 'c': bias, 'd': delay, 'g': gain,
            'lo': max(0., gain-half_width), 'hi': gain+half_width,
            'training_mse': float(np.mean(error*error)), 'condition': condition}


def fit_residual_experience(reference, measured_residual, applied_command, *,
                            dt_s, context, episode_index, episode_id,
                            baseline_slice, command_limit, provenance):
    """Fit only the declared PI prefix after its entire episode has completed.

    Delay/model selection uses training data only. The later block is used once
    for acceptance, with recursive prediction rather than observed future states.
    The interval combines approximate regression uncertainty and nearby delay
    ambiguity; it is not a calibrated physical confidence guarantee.
    """
    dt = _number(dt_s, positive=True); limit = _number(command_limit, positive=True)
    ctx = _context_hash(context)
    if type(episode_index) is not int or episode_index < 0 or not isinstance(episode_id, str) or not episode_id:
        raise ValueError('completed_episode_identity_required')
    required = {'scientific_status': 'simulated_sensor_data', 'completed': True,
                'baseline_policy': 'pi_only', 'source': 'observed_sensor_trace'}
    if (not isinstance(provenance, Mapping) or provenance.get('completed') is not True
            or any(provenance.get(k) != v for k, v in required.items())):
        raise ValueError('completed_observed_PI_provenance_required')
    if (not isinstance(baseline_slice, (tuple, list)) or len(baseline_slice) != 2
            or any(type(v) is not int for v in baseline_slice)):
        raise ValueError('explicit_baseline_slice_required')
    begin, end = baseline_slice
    if begin < 0 or end <= begin or any(end > len(v) for v in (reference, measured_residual, applied_command)):
        raise ValueError('invalid_observed_baseline_slice')
    # Slicing precedes every fit/finite check: later samples cannot select a model.
    ref, residual, command = [_vector(v[begin:end]) for v in
                              (reference, measured_residual, applied_command)]
    z = residual-ref
    split = int(len(z)*POLICY.training_fraction)
    maximum_delay = max(1, int(math.floor(POLICY.maximum_delay_s/dt)))
    source_hash = hashlib.sha256()
    for values in (ref, residual, command):
        source_hash.update(np.asarray(values, dtype='<f8').tobytes())
    source_hash.update(_json({'context': dict(context), 'episode_index': episode_index,
                             'episode_id': episode_id, 'baseline_slice': [begin, end],
                             'dt_s': dt, 'provenance': required}).encode())
    reasons, saturation = _excitation(z, command, limit)
    diagnostics = {'schema': 'observed-residual-arx/v1', 'scientific_status': 'simulated_sensor_data',
        'episode_id': episode_id, 'completed_episode_index': episode_index,
        'baseline_slice': [begin, end], 'training_end_exclusive': begin+split,
        'validation_start': begin+split, 'validation_end_exclusive': end,
        'sample_count': len(z), 'command_std': float(np.std(command)),
        'sensor_contrast_std': float(np.std(z)), 'saturation_fraction': saturation,
        'context': dict(context), 'source_sha256': source_hash.hexdigest(),
        'policy': asdict(POLICY), 'identification': 'effective sensed response, not hidden plant truth',
        'correction_effectiveness_established': False}
    if split-maximum_delay < POLICY.minimum_training_samples or len(z)-split < POLICY.minimum_validation_samples:
        reasons.append('insufficient_baseline_samples')
    if reasons:
        return {'accepted': False, 'reasons': reasons, 'model': None, 'diagnostics': diagnostics}
    candidates = [fit for delay in range(1, maximum_delay+1)
                  if (fit := _fit_candidate(z, command, delay, maximum_delay, split))]
    if not candidates:
        return {'accepted': False, 'reasons': ['unidentified_or_ill_conditioned_response'],
                'model': None, 'diagnostics': diagnostics}
    chosen = min(candidates, key=lambda f: (f['training_mse'], f['d']))
    plausible = [f for f in candidates if f['training_mse'] <= chosen['training_mse']*1.05+1e-8]
    low, high = min(f['lo'] for f in plausible), max(f['hi'] for f in plausible)
    uncertainty = max(chosen['g']-low, high-chosen['g'])/chosen['g']
    prediction = _rollout(z, command, chosen['a'], chosen['b'], chosen['c'], chosen['d'], split, len(z))
    mse = float(np.mean((prediction-z[split:])**2))
    null_mse = float(np.mean((z[split:]-float(np.mean(z[:split])))**2))
    nmse = mse/max(null_mse, POLICY.minimum_sensor_contrast_std**2)
    if uncertainty > POLICY.maximum_relative_gain_half_width:
        reasons.append('gain_or_delay_uncertain')
    if nmse > POLICY.maximum_validation_nmse:
        reasons.append('heldout_response_prediction_failed')
    diagnostics.update(training_mse=chosen['training_mse'], condition=chosen['condition'],
        validation_mse=mse, validation_null_mse=null_mse, validation_nmse=nmse,
        approximate_gain_interval=[low, high], relative_gain_half_width=uncertainty,
        plausible_delays_samples=[f['d'] for f in plausible],
        observed_contrast_bias=float(np.mean(z)), fitted_equilibrium_bias=chosen['c']/(1-chosen['a']))
    model = {'v': 1, **{k: round(chosen[k], 8) for k in ('a', 'b', 'c', 'g')},
             'd': chosen['d'], 'lo': round(low, 8), 'hi': round(high, 8),
             'dt': dt, 'ep': episode_index, 'ctx': ctx, 'src': source_hash.hexdigest(),
             'q': round(nmse, 8)}
    return {'accepted': not reasons, 'reasons': reasons, 'model': model if not reasons else None,
            'diagnostics': diagnostics}


def export_compact_experience(experience):
    """JSON payload for an authenticated enclosing observed episode assertion."""
    if experience.get('accepted') is not True or not isinstance(experience.get('model'), dict):
        raise ValueError('only_accepted_experience_can_be_exported')
    model = dict(experience['model'])
    if len(_json(model)) > 350:
        raise ValueError('residual_model_exceeds_350_chars')
    return model


def propose_residual_correction(experience, *, current_reference,
        current_measured_residual, current_applied_command, current_context,
        current_episode_index, nominal_feedforward, prefix_samples, dt_s,
        command_limit):
    """Choose a bounded FF scale using only a current, already observed PI prefix.

    Prefix commands remain unchanged. Future nominal FF is transformed only
    after the choice; it cannot influence acceptance or the chosen coefficient.
    Final actuator clipping and slew limiting remain the caller's responsibility.
    """
    nominal = np.asarray(nominal_feedforward, dtype=float)
    if (nominal.ndim != 1 or not np.all(np.isfinite(nominal))
            or type(prefix_samples) is not int or not 1 <= prefix_samples <= len(nominal)):
        raise ValueError('invalid_feedforward_or_prefix')
    result = nominal.copy()
    decision = {'applied': False, 'scale': 1., 'reason': '', 'max_observed_step': prefix_samples-1,
                'correction_start_step': prefix_samples, 'uses_future_observations': False,
                'isolated_physical_benefit_claimed': False}
    def reject(reason):
        decision['reason'] = reason
        return result, decision
    try:
        dt = _number(dt_s, positive=True); limit = _number(command_limit, positive=True)
        ctx = _context_hash(current_context)
        model = experience.get('model') if 'accepted' in experience else experience
        if 'accepted' in experience and experience['accepted'] is not True:
            return reject('experience_not_accepted')
        if not isinstance(model, Mapping) or type(model.get('v')) is not int or model['v'] != 1:
            return reject('invalid_experience_schema')
        if model['ctx'] != ctx:
            return reject('observable_context_mismatch')
        if (type(current_episode_index) is not int or type(model['ep']) is not int
                or not 0 <= model['ep'] < current_episode_index):
            return reject('experience_not_completed_before_current_episode')
        if not math.isclose(model['dt'], dt, rel_tol=0, abs_tol=1e-12):
            return reject('sample_interval_mismatch')
        a,b,bias,g,low,high,q = [_number(model[k]) for k in ('a','b','c','g','lo','hi','q')]
        delay = model['d']
        if (not 0 <= a < .995 or b <= 0 or not .25 <= g <= 4 or not 0 <= low <= g <= high
                or q < 0 or q > POLICY.maximum_validation_nmse
                or not math.isclose(g,b/(1-a),rel_tol=1e-5,abs_tol=1e-6)
                or type(delay) is not int or not 1 <= delay <= max(1,math.floor(POLICY.maximum_delay_s/dt))
                or max(g-low,high-g)/g > POLICY.maximum_relative_gain_half_width
                or not isinstance(model['src'],str) or len(model['src']) != 64
                or any(ch not in '0123456789abcdef' for ch in model['src'])):
            return reject('invalid_or_uncertain_experience')
        if any(prefix_samples > len(v) for v in (current_reference, current_measured_residual, current_applied_command)):
            return reject('insufficient_current_observed_prefix')
        ref,residual,command = [_vector(v[:prefix_samples]) for v in
                               (current_reference,current_measured_residual,current_applied_command)]
        if not np.all(np.isfinite(nominal[:prefix_samples])) or np.any(nominal[:prefix_samples] != 0):
            return reject('current_prefix_not_PI_only')
        z=residual-ref; reasons,_ = _excitation(z,command,limit)
        if reasons:
            return reject('current_'+reasons[0])
        if prefix_samples-delay < POLICY.minimum_training_samples+POLICY.minimum_validation_samples:
            return reject('insufficient_current_observed_prefix')
        prediction = _rollout(z,command,a,b,bias,delay,delay,prefix_samples)
        observed = z[delay:]; error = prediction-observed
        mse = float(np.mean(error*error)); variance = max(float(np.var(observed)),POLICY.minimum_sensor_contrast_std**2)
        centered=prediction-np.mean(prediction)
        ratio=float(centered @ (observed-np.mean(observed))/max(centered @ centered,1e-12))
        normalized_bias=abs(float(np.mean(error)))/math.sqrt(variance)
        decision.update(current_prediction_nmse=mse/variance, current_response_gain_ratio=ratio,
                        current_bias_in_std=normalized_bias, effective_gain=g,
                        approximate_gain_interval=[low,high], delay_samples=delay,
                        experience_source_sha256=model['src'])
        if (mse/variance > POLICY.maximum_validation_nmse or
                abs(ratio-1) > POLICY.maximum_current_gain_ratio_deviation or
                normalized_bias > POLICY.maximum_current_bias_in_std):
            return reject('old_response_inconsistent_with_current_PI_prefix')
        if low <= 1 <= high:
            return reject('gain_correction_direction_uncertain')
        scale=float(np.clip(1/g,1-POLICY.maximum_relative_correction,1+POLICY.maximum_relative_correction))
        # The future nominal array is used only for this fixed transformation.
        result[prefix_samples:]=np.clip(scale*nominal[prefix_samples:],-limit,limit)
        decision.update(applied=True,scale=scale,reason='validated_past_response_and_current_PI_prefix')
        return result,decision
    except (KeyError,TypeError,ValueError,OverflowError):
        return reject('invalid_experience_or_observed_prefix')
