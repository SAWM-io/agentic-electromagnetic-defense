"""Public numerical demonstration helpers. No private memory engines are included."""
from __future__ import annotations
import copy
import numpy as np

def validate_versioned_response(response,versions,*,cutoff=None,current=False):
    """History includes all accepted revisions known by cutoff, not only latest."""
    if bool(response.get('flashback'))!=(cutoff is not None):
        raise ValueError('response_flashback_scope_mismatch')
    latest={}
    for (key,revision),value in versions.items():
        if cutoff is None or value['known_at_ms']<=cutoff:
            latest[key]=max(latest.get(key,0),revision)
    seen=set();checks=[]
    for row in response['models']:
        identity=(row['candidate_id'],row['revision'])
        if identity in seen:
            raise ValueError('duplicate_candidate_revision')
        seen.add(identity);accepted=versions.get(identity)
        if accepted is None:
            raise ValueError('revision_has_no_accepted_assertion')
        if row['claim_key']!=accepted['claim_key'] or row['model']!=accepted['model']:
            raise ValueError('recalled_revision_payload_mismatch')
        if cutoff is not None and accepted['known_at_ms']>cutoff:
            raise ValueError('revision_not_yet_known_at_cutoff')
        if current and row['revision']!=latest[row['candidate_id']]:
            raise ValueError('current_read_returned_nonlatest_revision')
        checks.append({'candidate_id':identity[0],'revision':identity[1],
            'accepted_known_at_ms':accepted['known_at_ms'],
            'completed_episode_index':accepted['model']['completed_episode_index']})
    return {'returned_known_completed_models':bool(checks),'version_checks':checks,
        'returned_ids':[row['candidate_id'] for row in response['models']],
        'returned_count':len(checks),'returned_unique_candidates':len({r['candidate_id'] for r in checks}),
        'all_returned_versions_verified':True,'all_returned_known_at_or_before_cutoff':True,
        'top_k_limit':10,'full_inventory_verified':False,'response':response}


def assert_current_candidates(response,versions,index):
    validate_versioned_response(response,versions,current=True)
    if any(r['model']['completed_episode_index']>=index for r in response['models']):
        raise RuntimeError('current_or_future_model_in_causal_retrieval')



def contributor_ids(decisions):
    identifiers=[]
    for row in decisions:
        if row.get('accepted') and row.get('memory_weight',0)>0 and row.get('memory_contributed',False):
            for key in row.get('attributable_candidate_ids',[]):
                if key not in identifiers:
                    identifiers.append(key)
    return identifiers



def feedback_gate(hybrid_result,online_result):
    """Post-episode feedback uses observed outcome and actions, no latent RMS."""
    row=hybrid_result['row'];hybrid=hybrid_result['trace'];online=online_result['trace']
    ff_delta=float(np.max(np.abs(hybrid['feedforward']-online['feedforward'])))
    command_delta=float(np.max(np.abs(hybrid['command']-online['command'])))
    reasons=[];contributors=row['memory_contributor_ids']
    if len(contributors)!=1:
        reasons.append('requires_exactly_one_actual_memory_contributor')
    if row['mean_memory_weight']<=0:
        reasons.append('no_positive_memory_weight')
    if ff_delta<=1e-10 or command_delta<=1e-10:
        reasons.append('no_nonzero_effect_after_feedforward_and_command_limits')
    if not row['measured_outcome']['observed_success']:
        reasons.append('measured_hybrid_outcome_did_not_pass_fixed_threshold')
    past=[];valid=True
    for decision in hybrid_result.get('decisions',[]):
        if not (decision.get('accepted') and decision.get('memory_contributed') and decision.get('memory_weight',0)>0):
            continue
        evidence=decision.get('validation',{})
        memory_error=evidence.get('memory_holdout_mse');online_error=evidence.get('online_holdout_mse')
        if (not isinstance(memory_error,(int,float)) or not isinstance(online_error,(int,float))
            or not np.isfinite(memory_error) or not np.isfinite(online_error)
            or memory_error<0 or online_error<0
            or not evidence.get('training_end_step',10**12)<evidence.get('validation_start_step',-1)
            or not evidence.get('validation_start_step',10**12)<=evidence.get('validation_end_step',-1)<=decision['step']):
            valid=False;break
        past.append((decision['memory_weight'],online_error,memory_error))
    if not past or not valid:
        reasons.append('missing_or_noncausal_observed_holdout_evidence')
        online_mean=memory_mean=gain=None
    else:
        mass=sum(v[0] for v in past)
        online_mean=sum(w*o for w,o,m in past)/mass
        memory_mean=sum(w*m for w,o,m in past)/mass
        gain=online_mean-memory_mean
        if gain<=0:
            reasons.append('memory_has_no_positive_weighted_past_holdout_advantage')
    return (contributors if not reasons else []),{
        'rewarded':not reasons,'reasons':reasons or ['measured_success_and_nonzero_post_limit_action_contribution'],
        'feedforward_max_abs_delta_vs_online':ff_delta,'command_max_abs_delta_vs_online':command_delta,
        'weighted_online_holdout_mse':online_mean,'weighted_memory_holdout_mse':memory_mean,
        'observed_holdout_advantage':gain,'contributing_validation_decisions':len(past),
        'threshold':1e-10,'isolated_causal_success_claimed':False}


class _RecallView:
    """Instance-local composition, not monkeypatching any class or old source."""
    def __init__(self,adapter,semantic):
        self.adapter=adapter;self.entity_id=adapter.entity_id;self.semantic=copy.deepcopy(semantic)
    def _scope(self,*args,**kwargs):return self.adapter._scope(*args,**kwargs)
    def _call(self,operation,payload,validate):
        if operation!='recall_tsst':raise ValueError('diagnostic_exposes_read_only_tsst')
        payload=copy.deepcopy(payload)
        if payload['semantic']!={'predicate':'declaration'}:raise ValueError('frozen_recall_semantic_contract_changed')
        payload['semantic']=copy.deepcopy(self.semantic)
        return self.adapter._call(operation,payload,validate)
