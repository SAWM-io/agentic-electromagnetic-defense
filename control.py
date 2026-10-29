"""Public numerical demonstration helpers. No private memory engines are included."""
from __future__ import annotations
import copy, hashlib, json, math, time
import numpy as np
from simulation import sensor
from cloud_campaign import MODEL_SCHEMA,array_sha,estimate_frequency,measured_outcome,order_candidates,qualify_candidates
from hybrid_controller import hybrid_forecast
from optimized_controller import cached_hybrid_forecast,utility_forecast
from residual_experience import export_compact_experience,fit_residual_experience,propose_residual_correction
from candidate_validation import assert_current_candidates,feedback_gate
ARMS=('online','hybrid_v1','cached_hybrid','utility','optimized')
GAINS=(1.4,110.)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)



def content_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()



def causal_hold_last(values, valid):
    values = np.asarray(values, dtype=float); valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError('invalid_sensor_dropout_shape')
    out = np.empty_like(values); last = 0.
    for k in range(len(values)):
        if valid[k]:
            if not np.isfinite(values[k]):
                raise ValueError('valid_sensor_sample_not_finite')
            last = values[k]
        out[k] = last
    return out



def generate_observations(spec, c):
    truth = spec['generator_only']; t = np.arange(round(c.duration/c.dt))*c.dt
    d = truth['amplitude']*np.sin(2*np.pi*truth['frequency_hz']*t+truth['phase'])
    noise = np.random.default_rng(truth['noise_seed']).normal(0,truth['noise_std'],(2,len(t)))
    raw_reference = sensor(d, noise[0], c); valid = np.ones(len(t),dtype=bool)
    if spec['reference_dropout_s'] is not None:
        begin,end = spec['reference_dropout_s']; valid[(t>=begin)&(t<end)] = False
    n = round(c.prefix/c.dt)+1; tail = round(1.5/c.dt)
    if not np.all(valid[:n]) or not np.all(valid[-tail:]):
        raise ValueError('dropout_overlaps_PI_prefix_or_model_tail')
    return t,d,noise[1],causal_hold_last(raw_reference,valid),raw_reference,valid



def variable_plant(d, residual_noise, ff, c, *, hidden_gain, hidden_tau_s, gains=GAINS):
    """Old plant arithmetic, with hidden constant response parameters per episode."""
    if not all(np.isfinite(v) and v>0 for v in (hidden_gain,hidden_tau_s)):
        raise ValueError('invalid_simulator_response_parameters')
    n=len(d);field=np.zeros(n);measured=np.zeros(n);command=np.zeros(n)
    sensor_state=0.;coil=0.;previous_y=0.;previous_ff=0.;desired=0.
    filtered=np.zeros(n);delay=round(c.sensor_delay/c.dt)
    command_delay=round(c.command_delay/c.dt);alpha_s=math.exp(-c.dt/c.sensor_tau)
    kp,ki=gains
    for k in range(n):
        field[k]=d[k]+coil
        sensor_state=alpha_s*sensor_state+(1-alpha_s)*field[k];filtered[k]=sensor_state
        y=(filtered[k-delay] if k>=delay else 0.)+residual_noise[k];measured[k]=y
        desired += -kp*(y-previous_y)-ki*y*c.dt+(ff[k]-previous_ff)
        desired=float(np.clip(desired,-c.command_limit,c.command_limit))
        last=command[k-1] if k else 0.
        command[k]=float(np.clip(desired,last-c.slew_per_second*c.dt,last+c.slew_per_second*c.dt))
        desired=command[k];previous_y=y;previous_ff=ff[k]
        applied=command[k-command_delay] if k>=command_delay else 0.
        alpha=math.exp(-c.dt/hidden_tau_s)
        coil=alpha*coil+(1-alpha)*hidden_gain*applied
    return {'residual':field,'command':command,'measured':measured,'feedforward':ff}



def version_validators(versions, index, history_version):
    """Bind one refresh generation to a deep copy, never a growing live dict."""
    frozen=copy.deepcopy(versions)
    def recall(response, generation_history):
        if generation_history!=history_version:
            raise ValueError('refresh_history_version_mismatch')
        assert_current_candidates(response,frozen,index)
    def observe(response, ids, generation_history):
        if generation_history!=history_version:
            raise ValueError('refresh_history_version_mismatch')
        known={key for key,_ in frozen}
        if any(key not in known for key in ids):
            raise ValueError('observe_candidate_without_confirmed_receipt')
        actual=[r['candidate_id'] for r in response['observations']]
        if len(actual)!=len(set(actual)) or set(actual)!=set(ids) or any(r.get('present') is not True for r in response['observations']):
            raise ValueError('observe_identity_mismatch')
    return recall,observe



def choose_residual(models, observations, ref, pi_trace, nominal, c, context, index):
    """No committed-bank access: only caller's authenticated snapshot is supplied."""
    ranks={r['candidate_id']:float(r.get('normalized',r.get('resonance',0.))) for r in observations}
    candidates=[r for r in models if r['model'].get('context_id')==context['device_id']
        and isinstance(r['model'].get('residual_model'),dict)
        and type(r['model'].get('completed_episode_index')) is int
        and 0<=r['model']['completed_episode_index']<index
        and r['model']['residual_model'].get('ep')==r['model']['completed_episode_index']]
    candidates.sort(key=lambda r:(-r['model']['completed_episode_index'],-ranks.get(r['candidate_id'],0.),r['candidate_id']))
    attempted=[];n=round(c.prefix/c.dt)+1
    for row in candidates:
        ff,decision=propose_residual_correction(row['model']['residual_model'],current_reference=ref[:n],
            current_measured_residual=pi_trace['measured'][:n],current_applied_command=pi_trace['command'][:n],
            current_context=context,current_episode_index=index,nominal_feedforward=nominal,
            prefix_samples=n,dt_s=c.dt,command_limit=c.command_limit)
        attempted.append({'candidate_id':row['candidate_id'],'decision':decision})
        if decision['applied']:
            return ff,{**decision,'candidate_id':row['candidate_id'],'attempted':attempted,
                'effective_ff_max_abs_delta':float(np.max(np.abs(ff-nominal)))}
    return nominal.copy(),{'applied':False,'scale':1.,'candidate_id':None,'attempted':attempted,
        'reason':'no_returned_contextual_experience_validated_on_current_PI_prefix',
        'max_observed_step':n-1,'effective_ff_max_abs_delta':0.}



def local_controllers(ref, pi_trace, models, observations, c, context, index):
    """Pure controller boundary: no simulator truth, Cloud client or bank argument."""
    n=round(c.prefix/c.dt)+1
    qualified,rejected=qualify_candidates(models,ref[:n],c,index)
    ordered=order_candidates(qualified,'full',observations)
    results={}
    for arm in ARMS[:-1]:
        diagnostics={}
        if arm in ('online','hybrid_v1'):
            started=time.perf_counter()
            ff,decisions=hybrid_forecast(ref,c,'online' if arm=='online' else 'hybrid',ordered,episode_index=index)
            diagnostics={'latency_wall_s':time.perf_counter()-started,'cloud_calls':0}
        else:
            function=cached_hybrid_forecast if arm=='cached_hybrid' else utility_forecast
            ff,decisions=function(ref,c,'hybrid',ordered,episode_index=index,diagnostics=diagnostics)
        results[arm]={'ff':ff,'decisions':decisions,'diagnostics':diagnostics}
    if not np.array_equal(results['hybrid_v1']['ff'],results['cached_hybrid']['ff']) or results['hybrid_v1']['decisions']!=results['cached_hybrid']['decisions']:
        raise RuntimeError('cached_hybrid_does_not_exactly_match_frozen_v1')
    residual_started=time.perf_counter()
    ff,residual=choose_residual(models,observations,ref,pi_trace,results['utility']['ff'],c,context,index)
    residual_wall=time.perf_counter()-residual_started
    optimized_diagnostics=copy.deepcopy(results['utility']['diagnostics'])
    optimized_diagnostics.update(residual_choice_wall_s=residual_wall,
        latency_wall_s=optimized_diagnostics['latency_wall_s']+residual_wall,
        latency_scope='shared utility forecast computation plus residual selection; not an independent timed repetition')
    results['optimized']={'ff':ff,'decisions':copy.deepcopy(results['utility']['decisions']),
        'diagnostics':optimized_diagnostics,'residual_decision':residual}
    if any(np.any(result['ff'][:n]!=0) for result in results.values()):
        raise RuntimeError('controller_modified_common_PI_prefix')
    return results,ordered,qualified,rejected



def completed_compact_model(ref, trace, c, context, index, candidate_id, residual_fit):
    fitted=estimate_frequency(np.asarray(ref)[-round(1.5/c.dt):],c.dt)
    if fitted is None or fitted['explained']<c.minimum_explained:
        raise RuntimeError('completed_observation_has_no_qualified_frequency_model')
    outcome=measured_outcome(trace,ref,c)
    model={'schema':MODEL_SCHEMA,'scientific_status':'simulated_sensor_data','episode_id':candidate_id,
        'completed_episode_index':index,'frequency_hz':fitted['frequency_hz'],'context_id':context['device_id'],
        'observation_sha256':array_sha(ref),'measured_residual_sha256':array_sha(trace['measured']),
        'command_sha256':array_sha(trace['command']),
        'observed_success':outcome['observed_success'],'quality_score':round(outcome['quality_score'],6),
        'residual_diagnostics_sha256':content_sha(residual_fit)}
    if residual_fit['accepted']:
        model['residual_model']=export_compact_experience(residual_fit)
    # Same canonical envelope as the authenticated Cloud adapter.
    from cloud_memory import MODEL_SCHEMA as ENVELOPE_SCHEMA
    if len(canonical({'schema':ENVELOPE_SCHEMA,'model':model}))>1200:
        raise ValueError('compact_model_exceeds_public_1200_character_envelope')
    return model



def residual_feedback_gate(optimized, online):
    if optimized.get('residual_decision',{}).get('applied'):
        return [],{'rewarded':False,'reasons':['mixed_frequency_and_residual_attribution_unproven'],
            'stimulus':'empty advance; no score0 candidate','isolated_causal_success_claimed':False}
    return feedback_gate(optimized,online)
