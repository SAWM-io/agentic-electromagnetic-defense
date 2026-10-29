#!/usr/bin/env python3
"""Fresh, six-episode simulation using optional authenticated Cloud memory.

The experiment is a low-amplitude, one-axis control model, not an EMP or RF
solver. Offline mode runs numerical references only. Cloud mode stores and
retrieves actual synthetic experiences through the public API; no local model
is ever substituted for a missing Cloud response. Existing published evidence
comes from a longer inherited history and is deliberately not overwritten.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

import control
from async_cloud_memory import AsyncCloudMemory, SharedCallBudget
from candidate_validation import contributor_ids, validate_versioned_response
from cloud_memory import CloudMemory
from current_prefix_reference import fresh_prefix_correction
from simulation import Config, harmonic_fit, metrics, save_json, sha
from transport import DispatchPacer, PacedClient
from typed_memory_reader import CausalTypedReader, planning

ROOT=Path(__file__).resolve().parent
ARMS=(*control.ARMS,'fresh_current_response')
SEED=78006
MAX_CALLS=48


def schedule():
    rng=np.random.default_rng(SEED)
    fixed=[('coilA',.78,.030,1.61),('coilB',.72,.032,1.81),
           ('coilA',.78,.030,1.61),('coilB',.72,.032,1.81),
           ('coilA',.55,.050,1.61),('coilA',.55,.050,1.61)]
    return [{'episode_index':i,'observable_context':{'device_id':device},'reference_dropout_s':None,
             'generator_only':{'gain':gain,'tau_s':tau,'frequency_hz':frequency,'noise_std':.02,
                 'phase':float(rng.uniform(-np.pi,np.pi)),'amplitude':float(rng.uniform(.8,1.5)),
                 'noise_seed':SEED+10000+i*97}}
            for i,(device,gain,tau,frequency) in enumerate(fixed)]


def forecast_evaluation(ref,decisions,c):
    """Score future measurements only after a decision has already been fixed."""
    horizon=round(c.forecast_lead/c.dt);window=round(c.window/c.dt);points=[]
    for decision in decisions:
        k=decision['step'];target=k+horizon
        if not decision.get('accepted') or target>=len(ref):continue
        start=max(0,k-window+1);past=np.asarray(ref[start:k+1]);times=(np.arange(start,k+1)-k)*c.dt
        online=decision.get('online_frequency_hz') or decision['frequency_hz']
        frequencies=[online];weight=float(decision.get('memory_weight',0.))
        if weight>0:frequencies.append(decision['memory_frequency_hz'])
        coefficients,_=harmonic_fit(times,past,frequencies)
        def predict(i):
            a,b,dc=coefficients[i];f=frequencies[i]
            return float(a*np.sin(2*np.pi*f*c.forecast_lead)+b*np.cos(2*np.pi*f*c.forecast_lead)+dc)
        prediction=(1-weight)*predict(0)+weight*predict(1) if weight>0 else predict(0)
        prediction=float(np.clip(prediction,-c.command_limit,c.command_limit));observed=float(ref[target])
        points.append({'decision_step':k,'last_observed_step_used':k,'target_step':target,
                       'predicted_reference':prediction,'observed_future_reference':observed,'error':prediction-observed})
    return {'horizon_s':c.forecast_lead,'points':points,
            'RMSE':float(np.sqrt(np.mean([r['error']**2 for r in points]))) if points else None}


def evaluate(spec,c,models=(),observations=(),*,mode='offline'):
    """Generator truth is restricted to plant/evaluation, never memory/choice."""
    t,d,noise,ref,raw,valid=control.generate_observations(spec,c)
    truth=spec['generator_only'];index=spec['episode_index'];context=spec['observable_context'];n=round(c.prefix/c.dt)+1
    def plant(ff):return control.variable_plant(d,noise,ff,c,hidden_gain=truth['gain'],hidden_tau_s=truth['tau_s'])
    pi=plant(np.zeros_like(ref));prefix={key:pi[key][:n] for key in ('measured','command')}
    local,ordered,qualified,rejected=control.local_controllers(ref,prefix,models,observations,c,context,index)
    ff,fresh=fresh_prefix_correction(current_reference=ref[:n],current_measured_residual=prefix['measured'],
        current_applied_command=prefix['command'],nominal_feedforward=local['online']['ff'],prefix_samples=n,
        dt_s=c.dt,command_limit=c.command_limit,current_context=context,current_episode_index=index,
        calibration_id=f'current-prefix-e{index:02d}')
    local['fresh_current_response']={'ff':ff,'decisions':copy.deepcopy(local['online']['decisions']),
                                      'residual_decision':fresh,'diagnostics':{'uses_cloud_memory':False}}
    arms=ARMS if mode=='cloud' else ('online','fresh_current_response')
    results={};traces={'reference':ref,'time':t,'disturbance':d}
    for arm in arms:
        item=local[arm];trace=plant(item['ff'])
        if any(not np.array_equal(trace[key][:n],pi[key][:n]) for key in ('measured','command','residual')):
            raise RuntimeError('common_PI_prefix_mismatch')
        weights=[v.get('memory_weight',0.) for v in item['decisions']]
        row={'episode_index':index,'arm':arm,'observable_context':context,'metrics':metrics(trace,d,c),
             'measured_outcome':control.measured_outcome(trace,ref,c),
             'memory_contributor_ids':contributor_ids(item['decisions']),
             'mean_memory_weight':float(np.mean(weights)) if weights else 0.,
             'residual_decision':item.get('residual_decision',{}),'forecast_32ms':forecast_evaluation(ref,item['decisions'],c),
             'returned_ids':[r['candidate_id'] for r in models],
             'ranked_ids':[r['candidate_id'] for r in ordered],
             'qualified_ids':[r['candidate_id'] for r in qualified],'rejected_candidates':rejected,
             'decisions':item['decisions']}
        results[arm]={'row':row,'trace':trace,'decisions':item['decisions'],
                      'residual_decision':item.get('residual_decision',{})}
        for key,values in trace.items():traces[f'{arm}_{key}']=values
    return results,traces,ref


def _refresh(cache,queries,history_version,context_key):
    cache.start_refresh(queries,history_version=history_version,context_key=context_key)
    outcome=cache.wait_refresh(timeout=30.)
    if outcome is None or outcome.status!='published':raise RuntimeError('authenticated_refresh_unavailable')
    snapshot=cache.get_snapshot(context_key=context_key,min_history_version=history_version)
    if snapshot is None:raise RuntimeError('validated_snapshot_unavailable')
    return snapshot


def run(output,*,mode,credentials=None,confirm_dedicated_lab=False,clients=None,pacer=None):
    if mode not in {'offline','cloud'}:raise ValueError('mode_must_be_explicit')
    if mode=='cloud' and not confirm_dedicated_lab:raise ValueError('confirm_dedicated_lab_required')
    if mode=='offline' and (credentials is not None or clients is not None):raise ValueError('offline_mode_accepts_no_credentials')
    out=Path(output)
    if out.exists():raise FileExistsError('output_directory_must_be_new')
    out.mkdir(parents=True,mode=0o700);os.chmod(out,0o700)
    c=Config();run_id='em-'+uuid.uuid4().hex[:16];budget=SharedCallBudget(MAX_CALLS)
    protocol={'schema':'agentic-electromagnetic-defense-public-demo/v1','run_id':run_id,'seed':SEED,'mode':mode,
        'config':asdict(c),'schedule':schedule(),'max_http_attempts':MAX_CALLS,'SDK_retries':0,
        'initial_client_confirmed_history_count':0,'historical_evidence_reproduction':False,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.glob('*.py'))},
        'public_SDK_sha256':sha(ROOT/'vendor/basira_client.py'),
        'scientific_status':'simulated_sensor_data','physical_hardware':False,'SAWM_LLM_called':False,
        'history':'Cloud only: only optimized completed outcomes write new assertions. Other arms are shadow comparisons.',
        'causality':'Controllers see measured samples and prior authenticated Cloud models, not hidden generator values.',
        'first_episode':'No prior confirmed experience: controller runs without recalled models, then stores its first observation.',
        'entity':'Caller confirms a dedicated laboratory entity; no global assertion-inventory claim or deletion is made.'}
    save_json(out/'protocol.json',protocol)
    raw_clients=();fresh_client=None;cache=None;versions={};bank=[];rows=[];traces={};receipts=[];snapshots=[]
    pacer=pacer or DispatchPacer();started=time.perf_counter()
    def call(operation,function,*args,**kwargs):
        budget.reserve(operation)
        return function(*args,**kwargs)
    try:
        if mode=='cloud':
            if clients is not None:
                raw_clients=tuple(clients)
            else:
                if not credentials:raise ValueError('external_credentials_required')
                first=CloudMemory.from_credentials(credentials,out/'private-cloud-http.jsonl',max_retries=0)
                raw_clients=(first,)
                second=first.reconnect();raw_clients=(first,second)
            if len(raw_clients)!=2 or any(getattr(x,'_max_retries',None)!=0 for x in raw_clients):
                raise ValueError('two_clients_with_zero_retries_required')
            if any(x._attempt_counter()!=0 for x in raw_clients):raise ValueError('fresh_instrumented_clients_required')
            writer=PacedClient(raw_clients[0],pacer)
        for spec in protocol['schedule']:
            index=spec['episode_index'];context=spec['observable_context'];context_key=control.content_sha(context)
            captured={'models':[],'observations':[],'bootstrap':True}
            if mode=='cloud' and bank:
                readers=tuple(PacedClient(CausalTypedReader(x,versions,index),pacer) for x in raw_clients)
                vr,vo=control.version_validators(versions,index,len(bank))
                def require_active(response,ids,generation):
                    vo(response,ids,generation)
                    if response.get('mode')!='on':raise ValueError('active_ODVL_required')
                cache=AsyncCloudMemory(readers,validate_recall=vr,validate_observe=require_active,ttl_s=60.,
                    refresh_deadline_s=20.,budget=budget,journal_path=out/f'e{index:02d}-private-cache-events.jsonl')
                queries=planning(context,bank,current_index=index)
                snapshot=_refresh(cache,queries,len(bank),context_key);captured=snapshot.to_dict()
                snapshots.append({'episode_index':index,'typed_queries':[json.loads(q) for q in queries],
                                  'snapshot':captured})
                cache.start_refresh(queries,history_version=len(bank),context_key=context_key)
                if cache.get_snapshot(context_key=context_key,min_history_version=len(bank)) is None:
                    raise RuntimeError('snapshot_expired_before_control')
            results,episode_traces,ref=evaluate(spec,c,captured['models'],captured['observations'],mode=mode)
            if cache is not None:
                refreshed=cache.wait_refresh(timeout=30.)
                if refreshed is None or refreshed.status!='published':raise RuntimeError('background_refresh_unavailable')
                cache.stop();cache=None
            for key,value in episode_traces.items():traces[f'e{index:02d}_{key}']=value
            pending=[v['row'] for v in results.values()]
            save_json(out/'pending-episode.json',{'episode_index':index,'outcomes':pending})
            np.savez_compressed(out/'traces.npz',**traces)
            save_json(out/'private-snapshots.json',snapshots)
            if mode=='cloud':
                result=results['optimized'];candidate=f'{run_id}-e{index:02d}';n=round(c.prefix/c.dt)+1
                fit=control.fit_residual_experience(ref,result['trace']['measured'],result['trace']['command'],
                    dt_s=c.dt,context=context,episode_index=index,episode_id=candidate,baseline_slice=(0,n),
                    command_limit=c.command_limit,provenance={'scientific_status':'simulated_sensor_data',
                    'completed':True,'baseline_policy':'pi_only','source':'observed_sensor_trace'})
                save_json(out/f'e{index:02d}-residual-fit.json',fit)
                model=control.completed_compact_model(ref,result['trace'],c,context,index,candidate,fit)
                stored=call('remember',writer.remember_model,model,claim_key=candidate,candidate_id=candidate,
                    provenance_key=candidate,importance=0.,anchor=False,idempotency_key=candidate+'-store')
                record={'candidate_id':candidate,'claim_key':candidate,'revision':1,'known_at_ms':stored['known_at_ms'],
                        'provenance_key':candidate,'model':model,'receipt':stored}
                versions[(candidate,1)]=record;bank.append(record)
                receipt={'episode_index':index,'model':model,'remember':stored,'stimulus_status':'not_attempted'}
                receipts.append(receipt);save_json(out/'private-receipts.json',receipts)
                rewarded,feedback=control.residual_feedback_gate(result,results['online'])
                score=float(np.clip(result['row']['measured_outcome']['quality_score']*result['row']['mean_memory_weight'],0,1))
                receipt.update(stimulus_status='attempted',feedback_decision=feedback)
                save_json(out/'private-receipts.json',receipts)
                stimulus=call('stimulate',writer.stimulate,[{'candidate_id':key,'score':score,'importance':score,'anchor':False} for key in rewarded],
                    elapsed_units=.2,dt_units=.02,settle_steps=0,idempotency_key=candidate+'-outcome')
                receipt.update(stimulus_status='responded',outcome_stimulus=stimulus)
                save_json(out/'private-receipts.json',receipts)
                if stimulus.get('mode')!='on':raise RuntimeError('active_ODVL_required')
                receipt.update(stimulus_status='confirmed')
                save_json(out/'private-receipts.json',receipts)
            rows.extend(pending);save_json(out/'episodes.json',rows)
        verification=[]
        if mode=='cloud':
            fresh_client=raw_clients[0].reconnect()
            reader=PacedClient(CausalTypedReader(fresh_client,versions,6),pacer)
            for device in ('coilA','coilB'):
                query=planning({'device_id':device},bank,current_index=6)[0];target=json.loads(query)['expected_candidate']
                response=call('verify',reader.recall_models,query,limit=10)
                checked=validate_versioned_response(response,versions,current=True)
                if target not in checked['returned_ids']:raise RuntimeError('latest_memory_not_returned')
                verification.append({'device_id':device,'target_returned':True,'target_candidate':target})
            save_json(out/'verification.json',verification)
        attempts=sum(x._attempt_counter() for x in (*raw_clients,*([fresh_client] if fresh_client else [])))
        if attempts!=budget.attempts or attempts>MAX_CALLS:raise RuntimeError('HTTP_attempt_budget_mismatch')
        arms=ARMS if mode=='cloud' else ('online','fresh_current_response')
        summary={'schema':protocol['schema'],'mode':mode,'live_cloud':mode=='cloud','SAWM_LLM_called':False,
            'episodes':6,'controller_runs':len(rows),'actual_http_attempts':attempts,
            'reserved_attempts':budget.attempts,'new_assertions':len(receipts),
            'latest_context_verified':len(verification)==2 if mode=='cloud' else None,
            'mean_RMS_uT':{a:float(np.mean([r['metrics']['rms_uT'] for r in rows if r['arm']==a])) for a in arms},
            'total_wall_s':time.perf_counter()-started,'protocol_sha256':sha(out/'protocol.json'),
            'not_historical_reproduction':True,'scientific_status':'simulated_sensor_data'}
        save_json(out/'summary.json',summary);(out/'pending-episode.json').unlink(missing_ok=True)
        return summary
    except BaseException:
        save_json(out/'failure.json',{'status':'stopped_without_automatic_retry',
            'reserved_attempts':budget.attempts,'completed_episodes':len({r['episode_index'] for r in rows}),
            'mutation_replay':False,'action':'Inspect local private receipts before any manual recovery; do not replay writes blindly.'})
        raise
    finally:
        if cache is not None:cache.stop()
        for client in (*raw_clients,*([fresh_client] if fresh_client else [])):
            with contextlib.suppress(Exception):client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['offline','cloud'],required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--credentials')
    parser.add_argument('--confirm-dedicated-lab',action='store_true')
    args=parser.parse_args()
    try:result=run(args.output,mode=args.mode,credentials=args.credentials,confirm_dedicated_lab=args.confirm_dedicated_lab)
    except Exception:
        raise SystemExit('Demo stopped. Inspect the local failure/receipt files. No automatic mutation retry.') from None
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
