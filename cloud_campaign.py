"""Public numerical demonstration helpers. No private memory engines are included."""
from __future__ import annotations
import hashlib
import numpy as np
from simulation import harmonic_fit
SEARCH_FREQUENCIES=np.arange(.4,9.001,.05)
MODEL_SCHEMA="magnetic-sequential-sensor-model/v1"

def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()



def estimate_frequency(observed, dt):
    """Estimate from supplied measured samples only, including a local refinement."""
    observed = np.asarray(observed, dtype=float)
    if len(observed) < 4 or float(np.std(observed)) < .02:
        return None
    t = np.arange(len(observed)) * dt
    _, errors = harmonic_fit(t, observed, SEARCH_FREQUENCIES)
    f = float(SEARCH_FREQUENCIES[int(np.argmin(errors))])
    fine = np.arange(max(.4, f-.05), min(9., f+.05)+.0001, .005)
    _, errors = harmonic_fit(t, observed, fine)
    selected = int(np.argmin(errors))
    return {'frequency_hz': round(float(fine[selected]), 6),
            'observed_fit_mse': float(errors[selected]),
            'explained': float(1-errors[selected]/max(np.var(observed), 1e-12))}



def query_from_prefix(prefix, c):
    fitted = estimate_frequency(prefix,c.dt)
    f = fitted['frequency_hz'] if fitted else None
    return ('magnetic compensation completed sensor observation harmonic model '
            'frequency_hz='+str(f)+' reference_rms_uT='+str(round(float(
                np.sqrt(np.mean(np.asarray(prefix)**2))),4))+
            ' retrieve earlier measured control outcomes and reusable frequency models')



def qualify_candidates(candidates, observed_prefix, c, episode_index):
    """Real recall candidates are physically checked locally, using past samples."""
    prefix = np.asarray(observed_prefix)
    variance = float(np.var(prefix))
    if variance < c.minimum_std**2:
        return [], []
    accepted, rejected, seen = [], [], set()
    times = np.arange(len(prefix))*c.dt
    for rank, row in enumerate(candidates):
        key = row.get('candidate_id')
        model = row.get('model',{})
        reason = None
        if not key or key in seen:
            reason = 'missing_or_duplicate_id'
        elif model.get('schema') != MODEL_SCHEMA:
            reason = 'foreign_model_schema'
        elif model.get('completed_episode_index',episode_index) >= episode_index:
            reason = 'not_completed_before_current_episode'
        elif not .4 <= float(model.get('frequency_hz',-1)) <= 9.:
            reason = 'unsupported_frequency'
        if reason:
            rejected.append({'candidate_id':key,'reason':reason}); continue
        seen.add(key)
        _, errors = harmonic_fit(times,prefix,[model['frequency_hz']])
        explained = float(1-errors[0]/variance)
        if explained < c.minimum_explained:
            rejected.append({'candidate_id':key,'reason':'local_physical_fit',
                             'explained':explained}); continue
        accepted.append({**row,'retrieval_order':rank,'fit_mse':float(errors[0]),
                         'prefix_explained':explained})
    return accepted,rejected



def order_candidates(qualified, mode, observations=()):
    if mode == 'full':
        state = {row['candidate_id']:row for row in observations}
        missing = [r['candidate_id'] for r in qualified if
                   not state.get(r['candidate_id'],{}).get('present',False)]
        if missing:
            raise RuntimeError('actual_odvl_nodes_missing: '+','.join(missing))
        def key(row):
            actual = state[row['candidate_id']]
            resonance = actual.get('normalized',actual.get('resonance'))
            if resonance is None or not np.isfinite(float(resonance)):
                raise RuntimeError('actual_odvl_resonance_missing')
            return (-float(resonance),row['retrieval_order'],row['fit_mse'])
        return sorted(qualified,key=key)
    if mode == 'tsst':
        return sorted(qualified,key=lambda r:r['retrieval_order'])
    return sorted(qualified,key=lambda r:(r['fit_mse'],
                  -float(r['model'].get('quality_score',0)),r['candidate_id']))



def selected_forecast(ref, c, mode, ordered=(), blend=.6):
    """Causal local controller. A retrieved model may be rejected after a change.

    The one cloud candidate list is fixed for this episode. All active arms get
    identical local decision times and the same fallback frequency estimator.
    Candidate order affects selection only among models fitting current samples.
    """
    out=np.zeros(len(ref)); decisions=[]
    if mode in ('none','pi'):
        return out,decisions
    stride=round(c.update_period/c.dt); window=round(c.window/c.dt)
    for k in range(round(c.prefix/c.dt),len(ref),stride):
        start=max(0,k-window+1); past=np.asarray(ref[start:k+1])
        stop=min(len(ref),k+stride); variance=float(np.var(past))
        decision={'step':k,'max_observed_step':k,'selected_candidate_id':None,
                  'source':'abstain','accepted':False,'rejected_memory_ids':[]}
        if variance < c.minimum_std**2:
            decision['reason']='insufficient_signal';decisions.append(decision);continue
        times=(np.arange(start,k+1)-k)*c.dt
        choice=None
        for candidate in ordered:
            f=float(candidate['model']['frequency_hz'])
            coef,errors=harmonic_fit(times,past,[f])
            if 1-errors[0]/variance >= c.minimum_explained:
                choice=(f,coef[0],float(errors[0]),candidate);break
            decision['rejected_memory_ids'].append(candidate['candidate_id'])
        if choice is None:
            estimated=estimate_frequency(past,c.dt)
            if estimated:
                f=estimated['frequency_hz'];coef,errors=harmonic_fit(times,past,[f])
                choice=(f,coef[0],float(errors[0]),None)
        if choice is None or 1-choice[2]/variance<c.minimum_explained:
            decision['reason']='no_qualified_model';decisions.append(decision);continue
        f,coef,error,candidate=choice
        a,b,dc=coef; future=(np.arange(k,stop)-k)*c.dt+c.forecast_lead
        prediction=a*np.sin(2*np.pi*f*future)+b*np.cos(2*np.pi*f*future)+dc
        prediction=np.clip(prediction,-c.command_limit,c.command_limit)
        ramp=np.minimum((np.arange(k,stop)-k)*c.dt/.04,1.)
        last=out[k-1] if k else 0.
        out[k:stop]=last*(1-ramp)-blend*prediction*ramp
        decision.update(accepted=True,frequency_hz=f,explained=float(1-error/variance),
            source='completed_memory' if candidate else 'online_fallback',
            selected_candidate_id=candidate['candidate_id'] if candidate else None,
            reason='locally_qualified_past_samples')
        decisions.append(decision)
    return out,decisions



def measured_outcome(trace, ref, c):
    """Feedback uses observed sensors and command only, never latent field truth."""
    start=round(c.prefix/c.dt); y=np.asarray(trace['measured'])[start:]
    r=np.asarray(ref)[start:]; u=np.asarray(trace['command'])[start:]
    rms=float(np.sqrt(np.mean(y*y))); ref_rms=float(np.sqrt(np.mean(r*r)))
    normalized=rms/max(ref_rms,.06)
    saturation=float(np.mean(np.abs(u)>=c.command_limit-1e-8))
    peak=float(np.max(np.abs(y)))
    success=bool(normalized<=.5 and saturation<=.1)
    return {'measured_rms_uT':rms,'normalized_measured_rms':normalized,
            'measured_peak_uT':peak,'command_effort_uT2_s':float(np.sum(u*u)*c.dt),
            'saturation_fraction':saturation,'observed_success':success,
            'quality_score':float(np.clip(1-normalized,0,1))}



def completed_model(ref, trace, c, episode_index, candidate_id, selected_id):
    """Learn a new frequency only after its episode finishes; outcomes are attached."""
    tail=np.asarray(ref)[-round(1.5/c.dt):]
    fitted=estimate_frequency(tail,c.dt)
    if fitted is None or fitted['explained']<c.minimum_explained:
        return None
    outcome=measured_outcome(trace,ref,c)
    model={'schema':MODEL_SCHEMA,'scientific_status':'simulated_sensor_data',
           'episode_id':candidate_id,'completed_episode_index':episode_index,
           'frequency_hz':fitted['frequency_hz'],
           'observed_fit_mse':round(fitted['observed_fit_mse'],8),
           'model_explained_variance':round(fitted['explained'],6),
           'model_sample_count':len(tail),
           'observation_sha256':array_sha(ref),
           'model_window_sha256':array_sha(tail),
           'measured_residual_sha256':array_sha(trace['measured']),
           'command_sha256':array_sha(trace['command']),
           'model_window':'completed_episode_last_1.5_seconds',
           'parent_candidate_id':selected_id,
           'outcome_scope':'episode_controller_not_counterfactual_model_success'}
    model.update({key:round(value,6) if isinstance(value,float) else value
                  for key,value in outcome.items()})
    return model
