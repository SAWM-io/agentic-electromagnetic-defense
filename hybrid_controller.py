"""Causal online control regularized by an already validated Cloud memory list.

This module makes no API calls and never opens a memory store or results file.
The caller supplies the same real, receipt-validated TSST candidates in their
observed ODVL order to the full and hybrid arms. Rank is a prior preference,
not a confidence score: recent measured prediction errors determine its weight.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from cloud_campaign import MODEL_SCHEMA, estimate_frequency, selected_forecast
from simulation import harmonic_fit


ARMS = ('online', 'full', 'hybrid')


@dataclass(frozen=True)
class HybridPolicy:
    """A priori conservative choices; these are not validated optima."""
    max_memory_weight: float = .30
    validation_fraction: float = .25
    minimum_training_samples: int = 24
    minimum_validation_samples: int = 8

    def __post_init__(self):
        if not math.isfinite(self.max_memory_weight) or not 0 <= self.max_memory_weight <= .30:
            raise ValueError('memory_weight_must_preserve_at_least_70_percent_online')
        if not math.isfinite(self.validation_fraction) or not 0 < self.validation_fraction < .5:
            raise ValueError('invalid_validation_fraction')
        for value in (self.minimum_training_samples, self.minimum_validation_samples):
            if type(value) is not int or value < 4:
                raise ValueError('invalid_validation_sample_count')


DEFAULT_POLICY = HybridPolicy()


def policy_manifest(policy=DEFAULT_POLICY):
    """Serializable protocol, frozen before collecting the next campaign."""
    return {
        'schema': 'magnetic-hybrid-controller/v1',
        'parameters': asdict(policy),
        'selection': 'first compatible memory in supplied actual TSST+ODVL order',
        'online_estimator': 'unchanged coarse/refined harmonic estimator on current trailing samples',
        'validation': 'blocked trailing holdout wholly observed before the command; not a future episode or a prospective held-out test',
        'innovation_floor': 'minimum_std squared from the common configuration',
        'memory_weight': 'max_memory_weight * max(0,1-memory_mse/max(training_mean_baseline_mse,floor)) * min(1,(online_mse+floor)/(memory_mse+floor))',
        'final_prediction': '(1-weight)*online_prediction + weight*memory_prediction; both refit on all current observed samples',
        'failure_behavior': 'invalid/weak online model abstains; incompatible memory gives online-only control',
        'choice_status': 'a priori 70/30 maximum split; no claim of optimality and no tuning on previous campaign outcomes',
        'bounds': 'same feedforward clipping, blend, 40ms transition and bounded/slew-limited plant as the other arms',
        'memory_authentication': 'caller must validate actual committed server receipts and entity scope; this numerical controller cannot certify transport',
        'credit': 'only a positive-weight change surviving FF clipping, blend and ramp is attributable; no online fallback or identical action is credited, and this does not prove plant benefit',
    }


def _predict(frequency, coefficients, times):
    a, b, dc = coefficients
    return a*np.sin(2*np.pi*frequency*times) + b*np.cos(2*np.pi*frequency*times) + dc


def _admissible(ordered, episode_index):
    """Defensive causal metadata check; not a substitute for receipt validation."""
    accepted, rejected, seen = [], [], set()
    for row in ordered:
        key = row.get('candidate_id')
        model = row.get('model', {})
        reason = None
        if not isinstance(key, str) or not key or key in seen:
            reason = 'missing_or_duplicate_id'
        elif model.get('schema') != MODEL_SCHEMA:
            reason = 'foreign_model_schema'
        elif model.get('scientific_status') != 'simulated_sensor_data':
            reason = 'unsupported_model_provenance'
        elif (type(model.get('completed_episode_index')) is not int or
              model['completed_episode_index'] < 0 or
              model['completed_episode_index'] >= episode_index):
            reason = 'not_completed_before_current_episode'
        else:
            frequency = model.get('frequency_hz')
            if (type(frequency) not in (int, float) or not math.isfinite(frequency)
                    or not .4 <= frequency <= 9.):
                reason = 'unsupported_frequency'
        if reason:
            rejected.append({'candidate_id': key, 'reason': reason})
            continue
        seen.add(key)
        accepted.append(row)
    return accepted, rejected


def _validation(past, times, memory_frequency, c, policy):
    """Fit on the earlier observed block, predict the later observed block."""
    count = max(policy.minimum_validation_samples,
                round(len(past)*policy.validation_fraction))
    split = len(past)-count
    if split < policy.minimum_training_samples:
        return {'weight': 0., 'reason': 'insufficient_past_validation',
                'training_count': max(0, split), 'validation_count': count}
    train, validation = past[:split], past[split:]
    estimated = estimate_frequency(train, c.dt)
    if estimated is None:
        return {'weight': 0., 'reason': 'validation_online_model_unavailable',
                'training_count': split, 'validation_count': count}
    online_frequency = estimated['frequency_hz']
    coefficients, _ = harmonic_fit(times[:split], train,
                                   [online_frequency, memory_frequency])
    online_error = float(np.mean((_predict(online_frequency, coefficients[0],
                                           times[split:])-validation)**2))
    memory_error = float(np.mean((_predict(memory_frequency, coefficients[1],
                                           times[split:])-validation)**2))
    baseline_error = float(np.mean((validation-float(np.mean(train)))**2))
    floor = c.minimum_std**2
    skill = float(np.clip(1-memory_error/max(baseline_error, floor), 0., 1.))
    compatibility = float(min(1., (online_error+floor)/(memory_error+floor)))
    weight = policy.max_memory_weight*skill*compatibility
    return {'weight': weight, 'reason': 'past_holdout_innovation',
            'training_count': split, 'validation_count': count,
            'validation_online_frequency_hz': online_frequency,
            'online_holdout_mse': online_error, 'memory_holdout_mse': memory_error,
            'training_mean_baseline_holdout_mse': baseline_error,
            'innovation_floor_uT2': floor, 'memory_validation_skill': skill,
            'relative_innovation_compatibility': compatibility}


def hybrid_forecast(ref, c, mode, ordered=(), blend=.6, *, episode_index,
                    policy=DEFAULT_POLICY):
    """Return FF/decisions for online, full or hybrid on the same measured trace.

    Only ref[start:k+1] participates in a decision at k. No generator labels,
    true disturbance, plant residual truth, future outcomes, or stored quality
    score are inputs. Predictions over the next interval are extrapolations.
    The common ``plant`` remains responsible for final command slew limiting.
    """
    if mode not in ARMS:
        raise ValueError('unsupported_hybrid_comparison_arm')
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError('invalid_episode_index')
    if not math.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError('invalid_feedforward_blend')
    for value in (c.dt, c.update_period, c.window, c.command_limit, c.minimum_std):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('invalid_controller_configuration')
    if (not math.isfinite(c.forecast_lead) or c.forecast_lead < 0 or
            not math.isfinite(c.minimum_explained) or not 0 <= c.minimum_explained <= 1):
        raise ValueError('invalid_forecast_or_qualification_configuration')
    stride, window = round(c.update_period/c.dt), round(c.window/c.dt)
    if stride < 1 or window < 4 or not math.isfinite(c.prefix) or c.prefix < 0:
        raise ValueError('invalid_controller_schedule')
    # Even if ordered is supplied for fair provenance, online must not consume it.
    if mode == 'online':
        return selected_forecast(ref, c, 'online', (), blend=blend)
    candidates, invalid = _admissible(ordered, episode_index)
    if mode == 'full':
        return selected_forecast(ref, c, 'full', candidates, blend=blend)

    out = np.zeros(len(ref)); decisions = []
    for k in range(round(c.prefix/c.dt), len(ref), stride):
        start = max(0, k-window+1); stop = min(len(ref), k+stride)
        past = np.asarray(ref[start:k+1], dtype=float)
        if past.ndim != 1 or not np.all(np.isfinite(past)):
            raise ValueError('invalid_observed_history')
        times = (np.arange(start, k+1)-k)*c.dt
        variance = float(np.var(past))
        decision = {'step': k, 'max_observed_step': k, 'history_start_step': start,
            'accepted': False, 'source': 'abstain', 'selected_candidate_id': None,
            'memory_weight': 0., 'online_weight': 1., 'memory_contributed': False,
            'attributable_candidate_ids': [], 'rejected_memory_ids': [r['candidate_id'] for r in invalid],
            'candidate_evaluations': list(invalid)}
        if variance < c.minimum_std**2:
            decision['reason'] = 'insufficient_signal'; decisions.append(decision); continue
        estimated = estimate_frequency(past, c.dt)
        if estimated is None:
            decision['reason'] = 'online_model_unavailable'; decisions.append(decision); continue
        frequency = estimated['frequency_hz']
        coefficients, errors = harmonic_fit(times, past, [frequency])
        explained = float(1-errors[0]/variance)
        if explained < c.minimum_explained:
            decision['reason'] = 'online_model_not_qualified'; decisions.append(decision); continue
        future = (np.arange(k, stop)-k)*c.dt+c.forecast_lead
        ramp = np.minimum((np.arange(k, stop)-k)*c.dt/.04, 1.)
        online_prediction = _predict(frequency, coefficients[0], future)
        prediction = online_prediction
        decision.update(accepted=True, source='online_fallback', frequency_hz=frequency,
                        online_frequency_hz=frequency, explained=explained,
                        online_fit_mse=float(errors[0]), reason='recent_online_estimate')
        for candidate in candidates:
            key = candidate['candidate_id']; memory_frequency = float(candidate['model']['frequency_hz'])
            memory_coefficients, memory_errors = harmonic_fit(times, past, [memory_frequency])
            memory_explained = float(1-memory_errors[0]/variance)
            entry = {'candidate_id': key, 'memory_frequency_hz': memory_frequency,
                     'current_window_explained': memory_explained}
            if memory_explained < c.minimum_explained:
                entry.update(weight=0., reason='current_physical_fit_rejected')
                decision['candidate_evaluations'].append(entry)
                decision['rejected_memory_ids'].append(key); continue
            validation = _validation(past, times, memory_frequency, c, policy)
            entry.update(validation)
            entry['training_end_step'] = start+validation['training_count']-1
            entry['validation_start_step'] = start+validation['training_count']
            entry['validation_end_step'] = k
            memory_prediction = _predict(memory_frequency, memory_coefficients[0], future)
            delta = float(np.max(np.abs(memory_prediction-online_prediction)))
            weight = validation['weight']
            if weight <= 0:
                entry.update(effective_weight=0.)
                decision['candidate_evaluations'].append(entry)
                decision['rejected_memory_ids'].append(key); continue
            combined = (1-weight)*online_prediction+weight*memory_prediction
            # Attribute only a change that survives the common FF bounds and
            # transition. This is an FF contribution, not proof of benefit or
            # of a final plant-command difference (the plant also limits slew).
            effective_delta = float(np.max(np.abs(blend*ramp*(
                np.clip(combined, -c.command_limit, c.command_limit)-
                np.clip(online_prediction, -c.command_limit, c.command_limit)))))
            if delta <= 1e-12 or effective_delta <= 1e-12:
                entry.update(effective_weight=0., reason='bounded_prediction_identical_to_online')
                decision['candidate_evaluations'].append(entry)
                decision.update(corroborating_candidate_id=key,
                                reason='memory_corroborates_online_without_action_change')
                # Respect the first compatible prior. Do not search for a
                # lower-ranked, different model merely to manufacture an effect.
                break
            prediction = combined
            entry['effective_weight'] = weight
            decision['candidate_evaluations'].append(entry)
            decision.update(memory_weight=weight, online_weight=1-weight,
                memory_frequency_hz=memory_frequency, selected_candidate_id=key,
                source='hybrid_memory_online', reason='past_innovation_weighted_memory_prior',
                memory_contributed=True, attributable_candidate_ids=[key],
                weighted_prediction_max_abs_delta=weight*delta,
                feedforward_contribution_max_abs=effective_delta,
                validation={**validation, 'training_end_step': entry['training_end_step'],
                            'validation_start_step': entry['validation_start_step'],
                            'validation_end_step': k})
            break
        prediction = np.clip(prediction, -c.command_limit, c.command_limit)
        last = out[k-1] if k else 0.
        out[k:stop] = last*(1-ramp)-blend*prediction*ramp
        decisions.append(decision)
    return out, decisions
