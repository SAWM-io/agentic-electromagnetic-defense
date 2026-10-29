"""Local controller optimizations; no transport, credentials, or file access.

``cached_hybrid_forecast`` preserves hybrid v1 outputs AND decision records.
``utility_forecast`` is a separately declared policy: it considers every
distinct admissible memory frequency, and requires a conservative observed
holdout advantage over the recent online estimator before using memory.
Runtime diagnostics are returned through an optional separate dictionary so
wall-clock measurements never affect decisions or v1 equivalence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from cloud_campaign import estimate_frequency, selected_forecast
from hybrid_controller import DEFAULT_POLICY, HybridPolicy, _admissible, _predict
from simulation import harmonic_fit


@dataclass(frozen=True)
class UtilityPolicy(HybridPolicy):
    minimum_relative_advantage: float = .05
    prior_tie_fraction: float = .01

    def __post_init__(self):
        super().__post_init__()
        if (not math.isfinite(self.minimum_relative_advantage)
                or not 0 <= self.minimum_relative_advantage < 1):
            raise ValueError('invalid_minimum_relative_advantage')
        if not math.isfinite(self.prior_tie_fraction) or not 0 <= self.prior_tie_fraction < 1:
            raise ValueError('invalid_prior_tie_fraction')


DEFAULT_UTILITY_POLICY = UtilityPolicy()


def utility_policy_manifest(policy=DEFAULT_UTILITY_POLICY):
    return {'schema': 'magnetic-observed-utility-controller/v1', 'parameters': asdict(policy),
        'choice_status': 'fixed a priori; no tuning on campaign outcomes; margins are conservative heuristics, not statistical confidence bounds',
        'candidate_scope': 'all distinct currently qualified frequencies from the caller validated real TSST+ODVL list',
        'advantage': 'online_holdout_MSE - memory_holdout_MSE - minimum_relative_advantage*max(online_holdout_MSE, minimum_std_squared)',
        'weight': 'max_memory_weight * clip(conservative_advantage/max(online_holdout_MSE,minimum_std_squared),0,1)',
        'selection': 'largest positive conservative advantage; supplied ODVL order breaks ties within prior_tie_fraction*max(online_holdout_MSE,minimum_std_squared)',
        'abstention': 'no positive conservative advantage means online-only; no forced memory effect',
        'validation': 'same blocked observed-past holdout for every candidate; refit on current past only after selection',
        'bounds': 'at least70% recent online prediction; common FF clipping, blend, transition and plant bounds',
        'contextual_residual_correction': 'reserved interface; unsupported until an explicit causal contract is supplied'}


def _stats():
    return {'frequency_estimator_scans': 0, 'coarse_and_fine_search_passes': 0,
            'direct_harmonic_fit_calls': 0, 'direct_fit_frequency_rows': 0,
            'validation_cache_hits': 0, 'distinct_validation_frequencies': 0,
            'duplicate_candidates_skipped': 0, 'decision_latency_s': [],
            'cloud_calls': 0, 'counters_complete': True}


def _estimate(observed, dt, diagnostics):
    diagnostics['frequency_estimator_scans'] += 1
    # An insufficient-signal estimate exits before either scan.
    if len(observed) >= 4 and float(np.std(observed)) >= .02:
        diagnostics['coarse_and_fine_search_passes'] += 2
    return estimate_frequency(observed, dt)


def _fit(times, observed, frequencies, diagnostics):
    diagnostics['direct_harmonic_fit_calls'] += 1
    diagnostics['direct_fit_frequency_rows'] += len(frequencies)
    return harmonic_fit(times, observed, frequencies)


class _Window:
    """Per-decision cache only: nothing can survive into an unrelated history."""
    def __init__(self, past, times, candidates, c, policy, diagnostics):
        self.past, self.times, self.c, self.policy, self.diagnostics = past, times, c, policy, diagnostics
        self.frequencies = list(dict.fromkeys(float(row['model']['frequency_hz']) for row in candidates))
        self.online = _estimate(past, c.dt, diagnostics)
        self.fits = {}; self.validations = {}; self.validation_ready = False
        self.validation_failure = None
        if self.online is not None:
            frequencies = list(dict.fromkeys([self.online['frequency_hz'], *self.frequencies]))
            coefficients, errors = _fit(times, past, frequencies, diagnostics)
            self.fits = {frequency: (coefficients[i], float(errors[i]))
                         for i, frequency in enumerate(frequencies)}

    def validation(self, frequency):
        if frequency in self.validations:
            self.diagnostics['validation_cache_hits'] += 1
            return dict(self.validations[frequency])
        if not self.validation_ready:
            self._prepare_validation()
        if self.validation_failure is not None:
            return dict(self.validation_failure)
        self.diagnostics['validation_cache_hits'] += 1
        return dict(self.validations[frequency])

    def _prepare_validation(self):
        self.validation_ready = True
        c, policy, past, times = self.c, self.policy, self.past, self.times
        count = max(policy.minimum_validation_samples, round(len(past)*policy.validation_fraction))
        split = len(past)-count
        if split < policy.minimum_training_samples:
            self.validation_failure = {'weight': 0., 'reason': 'insufficient_past_validation',
                'training_count': max(0, split), 'validation_count': count}
            return
        train, validation = past[:split], past[split:]
        estimated = _estimate(train, c.dt, self.diagnostics)
        if estimated is None:
            self.validation_failure = {'weight': 0., 'reason': 'validation_online_model_unavailable',
                'training_count': split, 'validation_count': count}
            return
        online_frequency = estimated['frequency_hz']
        frequencies = list(dict.fromkeys([online_frequency, *self.frequencies]))
        coefficients, _ = _fit(times[:split], train, frequencies, self.diagnostics)
        fitted = {frequency: coefficients[i] for i, frequency in enumerate(frequencies)}
        online_error = float(np.mean((_predict(online_frequency, fitted[online_frequency],
                                               times[split:])-validation)**2))
        baseline_error = float(np.mean((validation-float(np.mean(train)))**2))
        floor = c.minimum_std**2
        for frequency in self.frequencies:
            memory_error = float(np.mean((_predict(frequency, fitted[frequency],
                                                   times[split:])-validation)**2))
            skill = float(np.clip(1-memory_error/max(baseline_error, floor), 0., 1.))
            compatibility = float(min(1., (online_error+floor)/(memory_error+floor)))
            self.validations[frequency] = {'weight': policy.max_memory_weight*skill*compatibility,
                'reason': 'past_holdout_innovation', 'training_count': split, 'validation_count': count,
                'validation_online_frequency_hz': online_frequency,
                'online_holdout_mse': online_error, 'memory_holdout_mse': memory_error,
                'training_mean_baseline_holdout_mse': baseline_error,
                'innovation_floor_uT2': floor, 'memory_validation_skill': skill,
                'relative_innovation_compatibility': compatibility}
        self.diagnostics['distinct_validation_frequencies'] += len(self.frequencies)


def _entry(candidate, window, variance, start, k):
    frequency = float(candidate['model']['frequency_hz'])
    _, error = window.fits[frequency]
    explained = float(1-error/variance)
    entry = {'candidate_id': candidate['candidate_id'], 'memory_frequency_hz': frequency,
             'current_window_explained': explained}
    if explained < window.c.minimum_explained:
        entry.update(weight=0., reason='current_physical_fit_rejected')
        return entry, None
    validation = window.validation(frequency)
    entry.update(validation)
    entry.update(training_end_step=start+validation['training_count']-1,
                 validation_start_step=start+validation['training_count'], validation_end_step=k)
    return entry, validation


def _combine(window, frequency, online_prediction, future, ramp, blend, weight):
    coefficients, _ = window.fits[frequency]
    memory_prediction = _predict(frequency, coefficients, future)
    delta = float(np.max(np.abs(memory_prediction-online_prediction)))
    combined = (1-weight)*online_prediction+weight*memory_prediction
    limit = window.c.command_limit
    effective = float(np.max(np.abs(blend*ramp*(np.clip(combined, -limit, limit)
                                               -np.clip(online_prediction, -limit, limit)))))
    return combined, delta, effective


def _accept(decision, candidate, entry, validation, weight, delta, effective, *, utility=False):
    key = candidate['candidate_id']
    if utility:
        validation = {**validation, 'compatibility_weight_v1': validation['weight'],
                      'weight': weight, 'weight_rule': 'conservative_past_utility'}
    decision.update(memory_weight=weight, online_weight=1-weight,
        memory_frequency_hz=float(candidate['model']['frequency_hz']), selected_candidate_id=key,
        source='hybrid_memory_online',
        reason=('conservative_past_utility_memory_prior' if utility else 'past_innovation_weighted_memory_prior'),
        memory_contributed=True, attributable_candidate_ids=[key],
        weighted_prediction_max_abs_delta=weight*delta, feedforward_contribution_max_abs=effective,
        validation={**validation, 'training_end_step': entry['training_end_step'],
                    'validation_start_step': entry['validation_start_step'],
                    'validation_end_step': entry['validation_end_step']})


def _cached_choice(window, candidates, decision, variance, start, k, online_prediction, future, ramp, blend):
    for candidate in candidates:
        key = candidate['candidate_id']; frequency = float(candidate['model']['frequency_hz'])
        entry, validation = _entry(candidate, window, variance, start, k)
        if validation is None:
            decision['candidate_evaluations'].append(entry)
            decision['rejected_memory_ids'].append(key); continue
        weight = validation['weight']
        if weight <= 0:
            entry.update(effective_weight=0.)
            decision['candidate_evaluations'].append(entry)
            decision['rejected_memory_ids'].append(key); continue
        combined, delta, effective = _combine(window, frequency, online_prediction, future, ramp, blend, weight)
        if delta <= 1e-12 or effective <= 1e-12:
            entry.update(effective_weight=0., reason='bounded_prediction_identical_to_online')
            decision['candidate_evaluations'].append(entry)
            decision.update(corroborating_candidate_id=key,
                            reason='memory_corroborates_online_without_action_change')
            break
        entry['effective_weight'] = weight
        decision['candidate_evaluations'].append(entry)
        _accept(decision, candidate, entry, validation, weight, delta, effective)
        return combined
    return online_prediction


def _utility_choice(window, candidates, decision, variance, start, k, online_prediction, future, ramp, blend):
    policy = window.policy; eligible = []; seen = set()
    decision.update(selection_policy='positive_conservative_past_utility',
                    evaluated_distinct_frequencies_hz=[])
    for rank, candidate in enumerate(candidates):
        frequency = float(candidate['model']['frequency_hz'])
        if frequency in seen:
            window.diagnostics['duplicate_candidates_skipped'] += 1
            continue
        seen.add(frequency); decision['evaluated_distinct_frequencies_hz'].append(frequency)
        entry, validation = _entry(candidate, window, variance, start, k)
        entry['supplied_odvl_rank'] = rank
        if validation is not None and 'online_holdout_mse' in validation:
            scale = max(validation['online_holdout_mse'], window.c.minimum_std**2)
            gain = validation['online_holdout_mse']-validation['memory_holdout_mse']
            margin = policy.minimum_relative_advantage*scale
            conservative = gain-margin
            weight = policy.max_memory_weight*float(np.clip(conservative/scale, 0., 1.))
            entry.update(observed_holdout_gain=gain, required_advantage_margin=margin,
                         conservative_holdout_gain=conservative, utility_scale=scale,
                         weight=weight, effective_weight=0.)
            if conservative > 0 and weight > 0:
                eligible.append((candidate, entry, validation, weight, conservative, scale))
            else:
                entry['reason'] = 'no_conservative_advantage_over_recent_online'
                decision['rejected_memory_ids'].append(candidate['candidate_id'])
        else:
            entry['effective_weight'] = 0.
            decision['rejected_memory_ids'].append(candidate['candidate_id'])
        decision['candidate_evaluations'].append(entry)
    if not eligible:
        decision['reason'] = 'no_memory_with_observed_conservative_advantage'
        return online_prediction
    best_gain = max(row[4] for row in eligible)
    tolerance = policy.prior_tie_fraction*eligible[0][5]
    # All candidates share the same online holdout and denominator. Preserve
    # supplied ODVL preference only inside this declared practical tie band.
    chosen = next(row for row in eligible if row[4] >= best_gain-tolerance)
    candidate, entry, validation, weight, gain, _ = chosen
    combined, delta, effective = _combine(window, entry['memory_frequency_hz'],
                                          online_prediction, future, ramp, blend, weight)
    decision.update(best_conservative_holdout_gain=best_gain,
                    selected_conservative_holdout_gain=gain, prior_tie_tolerance=tolerance)
    if delta <= 1e-12 or effective <= 1e-12:
        entry.update(effective_weight=0., reason='bounded_prediction_identical_to_online')
        decision.update(corroborating_candidate_id=candidate['candidate_id'],
                        reason='memory_corroborates_online_without_action_change')
        return online_prediction
    entry['effective_weight'] = weight
    _accept(decision, candidate, entry, validation, weight, delta, effective, utility=True)
    return combined


def _run(ref, c, mode, ordered, blend, episode_index, policy, diagnostics, correction, *, utility):
    started = time.perf_counter(); stats = _stats()
    if correction is not None:
        raise NotImplementedError('contextual_residual_correction_contract_not_supplied')
    if mode not in ('online', 'full', 'hybrid', 'utility') or (mode == 'utility' and not utility):
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
    stride, window_size = round(c.update_period/c.dt), round(c.window/c.dt)
    if stride < 1 or window_size < 4 or not math.isfinite(c.prefix) or c.prefix < 0:
        raise ValueError('invalid_controller_schedule')
    candidates, invalid = _admissible(ordered, episode_index) if mode != 'online' else ([], [])
    if mode in ('online', 'full'):
        result = selected_forecast(ref, c, mode, candidates, blend=blend)
        stats['counters_complete'] = False
    else:
        out = np.zeros(len(ref)); decisions = []
        for k in range(round(c.prefix/c.dt), len(ref), stride):
            decision_start = time.perf_counter()
            start = max(0, k-window_size+1); stop = min(len(ref), k+stride)
            past = np.asarray(ref[start:k+1], dtype=float)
            if past.ndim != 1 or not np.all(np.isfinite(past)):
                raise ValueError('invalid_observed_history')
            times = (np.arange(start, k+1)-k)*c.dt; variance = float(np.var(past))
            decision = {'step': k, 'max_observed_step': k, 'history_start_step': start,
                'accepted': False, 'source': 'abstain', 'selected_candidate_id': None,
                'memory_weight': 0., 'online_weight': 1., 'memory_contributed': False,
                'attributable_candidate_ids': [], 'rejected_memory_ids': [r['candidate_id'] for r in invalid],
                'candidate_evaluations': list(invalid)}
            if variance < c.minimum_std**2:
                decision['reason'] = 'insufficient_signal'
            else:
                window = _Window(past, times, candidates, c, policy, stats)
                if window.online is None:
                    decision['reason'] = 'online_model_unavailable'
                else:
                    frequency = window.online['frequency_hz']; coefficients, error = window.fits[frequency]
                    explained = float(1-error/variance)
                    if explained < c.minimum_explained:
                        decision['reason'] = 'online_model_not_qualified'
                    else:
                        future = (np.arange(k, stop)-k)*c.dt+c.forecast_lead
                        ramp = np.minimum((np.arange(k, stop)-k)*c.dt/.04, 1.)
                        online_prediction = _predict(frequency, coefficients, future)
                        decision.update(accepted=True, source='online_fallback', frequency_hz=frequency,
                            online_frequency_hz=frequency, explained=explained,
                            online_fit_mse=float(error), reason='recent_online_estimate')
                        choose = _utility_choice if utility else _cached_choice
                        prediction = choose(window, candidates, decision, variance, start, k,
                                            online_prediction, future, ramp, blend)
                        prediction = np.clip(prediction, -c.command_limit, c.command_limit)
                        last = out[k-1] if k else 0.
                        out[k:stop] = last*(1-ramp)-blend*prediction*ramp
            decisions.append(decision)
            stats['decision_latency_s'].append({'step': k, 'wall_s': time.perf_counter()-decision_start})
        result = out, decisions
    stats['latency_wall_s'] = time.perf_counter()-started
    stats['selected_candidate_ids'] = [d.get('selected_candidate_id') for d in result[1]]
    stats['memory_weights'] = [d.get('memory_weight', 0.) for d in result[1]]
    if diagnostics is not None:
        diagnostics.clear(); diagnostics.update(stats)
    return result


def cached_hybrid_forecast(ref, c, mode='hybrid', ordered=(), blend=.6, *, episode_index,
                           policy=DEFAULT_POLICY, diagnostics=None, contextual_residual_correction=None):
    """Hybrid v1 semantics with batched fits and one shared validation estimator."""
    return _run(ref, c, mode, ordered, blend, episode_index, policy, diagnostics,
                contextual_residual_correction, utility=False)


def utility_forecast(ref, c, mode='hybrid', ordered=(), blend=.6, *, episode_index,
                     policy=DEFAULT_UTILITY_POLICY, diagnostics=None, contextual_residual_correction=None):
    """A distinct conservative-utility policy; not represented as v1 equivalent."""
    if not isinstance(policy, UtilityPolicy):
        raise ValueError('utility_policy_required')
    return _run(ref, c, mode, ordered, blend, episode_index, policy, diagnostics,
                contextual_residual_correction, utility=True)
