"""Ephemeral validated Cloud-memory snapshots with bounded background refresh.

No writes or stimuli are issued. Two persistent scoped clients perform reads;
caller-supplied validators bind each generation to confirmed historical receipts.
Snapshots contain prior evidence, not a guarantee of relevance to a new context.
TTL bounds local staleness; it cannot prove instantaneous offline revocation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from collections.abc import Mapping
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType


class CacheError(RuntimeError):
    pass


class RefreshInProgress(CacheError):
    pass


class BudgetExhausted(CacheError):
    pass


class SharedCallBudget:
    """Thread-safe conservative dispatch cap, also usable by external writes.

    A reservation counts even when a client rejects before transport. SDK retry
    count must be zero, so actual HTTP attempts cannot exceed reservations.
    Exact HTTP attempts remain in the validated SDK adapter's own receipts.
    """
    def __init__(self,max_calls,initial_attempts=0):
        if type(max_calls) is not int or type(initial_attempts) is not int or not 0<=initial_attempts<=max_calls or max_calls<1:
            raise ValueError('invalid_shared_call_budget')
        self.max_calls=max_calls;self._attempts=initial_attempts;self._lock=threading.Lock()

    def reserve(self,operation='external'):
        with self._lock:
            if self._attempts>=self.max_calls:
                raise BudgetExhausted('shared_call_budget_exhausted')
            self._attempts+=1
            return self._attempts

    @property
    def attempts(self):
        with self._lock:
            return self._attempts

    @property
    def remaining(self):
        return self.max_calls-self.attempts


def _freeze(value):
    if isinstance(value,Mapping):
        return MappingProxyType({str(k):_freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or type(value) in (str,bool,int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('snapshot_value_must_be_finite_json')


def _thaw(value):
    if isinstance(value,Mapping):
        return {k:_thaw(v) for k,v in value.items()}
    if isinstance(value,tuple):
        return [_thaw(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MemorySnapshot:
    version:int
    generation:int
    history_version:int
    scope:tuple
    service_fingerprint:str
    context_key:str
    query_digest:str
    refresh_started_at:float
    published_at:float
    expires_at:float
    models:tuple
    observations:tuple
    recall_responses:tuple
    observe_response:object

    def to_dict(self):
        return {name:_thaw(getattr(self,name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class RefreshOutcome:
    generation:int
    history_version:int
    status:str
    error_code:str|None
    snapshot_version:int|None
    wall_s:float


@dataclass(frozen=True)
class _Request:
    generation:int
    history_version:int
    queries:tuple
    context_key:str
    query_digest:str
    mode:str
    started_at:float
    deadline:float
    validate_recall:object
    validate_observe:object
    cancel:threading.Event=field(default_factory=threading.Event)
    wake:threading.Event=field(default_factory=threading.Event)


def _authorization_failure(error):
    for value in (getattr(error,'status_code',None),getattr(error,'status',None),
                  getattr(getattr(error,'response',None),'status_code',None)):
        if value in (401,403):
            return True
    # Actual adapter errors contain a bounded service code, not raw HTTP data.
    code=str(getattr(error,'code',str(error))).lower().strip()
    return code in {'401','403','unauthorized','forbidden','access_denied','permission_denied',
        'authentication_failed','invalid_api_key','api_key_invalid','api_key_revoked','key_revoked',
        'revoked_api_key','insufficient_scope','scope_denied','not_authorized','missing_api_key',
        'memory_authorization_denied','memory_access_denied'}


def _client_identity(client):
    declared=getattr(client,'_cloud_base',None)
    underlying=getattr(getattr(client,'_client',None),'base_url',None)
    if declared is not None:
        declared=str(declared).rstrip('/')
    if underlying is not None:
        underlying=str(underlying).rstrip('/')
    if declared is not None and underlying is not None and declared!=underlying:
        raise ValueError('adapter_and_sdk_base_mismatch')
    return (client.space_id,client.entity_id,declared or underlying,
            getattr(client,'sdk_source_sha256',None))


class AsyncCloudMemory:
    """A nonblocking cache accessor and one atomic refresh generation at a time.

    Validators: validate_recall(response, history_version), and
    validate_observe(response, candidate_ids_tuple, history_version). They must
    raise on any invalid scope/version/evidence, and capture immutable copies of
    the caller's confirmed history. Per-refresh overrides are bound immediately.

    get_snapshot uses a try-lock and never performs I/O or waits for refresh.
    Without context_key it may return still-valid evidence from an older query;
    the consumer MUST reapply current physical/context qualification locally.
    With context_key it requires an exact context match. ``now`` is monotonic
    wall time, never simulated physical time. TTL starts at refresh initiation.
    """
    def __init__(self,clients,*,validate_recall,validate_observe,ttl_s=60.,
                 refresh_deadline_s=5.,budget=None,journal_path=None,clock=time.monotonic):
        if len(clients)!=2 or clients[0] is clients[1]:
            raise ValueError('two_distinct_persistent_clients_required')
        scopes=[(c.space_id,c.entity_id) for c in clients]
        if scopes[0]!=scopes[1]:
            raise ValueError('client_scope_mismatch')
        instance_ids=[getattr(c,'client_instance_id',None) for c in clients]
        if instance_ids[0] is not None and instance_ids[0]==instance_ids[1]:
            raise ValueError('distinct_client_instances_required')
        if any(getattr(c,'_max_retries',0)!=0 for c in clients):
            raise ValueError('sdk_retries_must_be_disabled')
        identities=[_client_identity(c) for c in clients]
        if identities[0]!=identities[1]:
            raise ValueError('client_base_or_sdk_source_mismatch')
        if not callable(validate_recall) or not callable(validate_observe):
            raise ValueError('receipt_validators_required')
        if not all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in (ttl_s,refresh_deadline_s)):
            raise ValueError('positive_finite_ttl_and_deadline_required')
        self._clients=tuple(clients);self.scope=scopes[0]
        self._service_identity=identities[0];self.service_fingerprint=_digest(identities[0])
        self._validate_recall=validate_recall;self._validate_observe=validate_observe
        self.ttl_s=float(ttl_s);self.refresh_deadline_s=float(refresh_deadline_s)
        self.budget=budget or SharedCallBudget(50);self._clock=clock
        self._lock=threading.Lock();self._journal_lock=threading.Lock()
        self._journal_path=Path(journal_path) if journal_path else None
        if self._journal_path:
            self._journal_path.parent.mkdir(parents=True,exist_ok=True)
        self._io=ThreadPoolExecutor(max_workers=2,thread_name_prefix='validated-memory-io')
        self._coordinator=ThreadPoolExecutor(max_workers=1,thread_name_prefix='validated-memory-refresh')
        self._generation=0;self._version=0;self._snapshot=None;self._request=None;self._future=None
        self._stopped=False;self._last_error=None;self._refreshes=[];self._phases=[]
        self._callback_samples=deque(maxlen=4096)

    def _journal(self,event):
        if not self._journal_path:
            return
        # Deliberately excludes raw queries, response content, headers, exception
        # text, credential objects and arbitrary caller context strings.
        encoded=(json.dumps({'schema':'async-cloud-memory-events/v1','monotonic_at':self._clock(),
            'space_id':self.scope[0],'entity_id':self.scope[1],**event},sort_keys=True,
            separators=(',',':'),allow_nan=False)+'\n').encode()
        with self._journal_lock:
            fd=os.open(self._journal_path,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'ab') as stream:
                stream.write(encoded);stream.flush();os.fsync(stream.fileno())

    def start_refresh(self,queries,*,history_version=0,context_key=None,mode='async_parallel',
                      validate_recall=None,validate_observe=None):
        if not isinstance(queries,(list,tuple)) or len(queries)!=2 or any(not isinstance(q,str) or not q.strip() for q in queries):
            raise ValueError('exactly_two_nonempty_queries_required')
        if type(history_version) is not int or history_version<0:
            raise ValueError('invalid_history_version')
        if mode not in ('async_parallel','sync_sequential'):
            raise ValueError('unknown_refresh_mode')
        digest=_digest(list(queries));context_key=digest if context_key is None else context_key
        if not isinstance(context_key,str):
            raise ValueError('context_key_must_be_string')
        recall_validator=validate_recall or self._validate_recall
        observe_validator=validate_observe or self._validate_observe
        if not callable(recall_validator) or not callable(observe_validator):
            raise ValueError('receipt_validators_required')
        with self._lock:
            if self._stopped:
                raise CacheError('cache_stopped')
            if self._future is not None and not self._future.done() and self._request and not self._request.cancel.is_set():
                raise RefreshInProgress('refresh_already_active')
            self._generation+=1;now=self._clock()
            request=_Request(self._generation,history_version,tuple(queries),context_key,digest,
                mode,now,now+self.refresh_deadline_s,recall_validator,observe_validator)
            self._request=request
            self._future=self._coordinator.submit(self._refresh,request)
            return request.generation

    def _current(self,request):
        with self._lock:
            return not self._stopped and self._generation==request.generation and not request.cancel.is_set()

    def _invalidate_generation(self,request,error_code,*,global_auth=False):
        with self._lock:
            if global_auth or self._generation==request.generation:
                self._snapshot=None;self._last_error=error_code
                if global_auth:
                    self._generation+=1
                    if self._request:
                        self._request.cancel.set();self._request.wake.set()

    def _invoke(self,request,operation,client_index,args):
        if not self._current(request):
            raise CacheError('refresh_cancelled')
        if self._clock()>=request.deadline:
            raise CacheError('refresh_deadline_expired')
        if _client_identity(self._clients[client_index])!=self._service_identity:
            raise ValueError('client_identity_changed_after_cache_construction')
        attempt=self.budget.reserve(operation)
        self._journal({'event':'dispatch','generation':request.generation,'operation':operation,
            'attempt':attempt,'client_slot':client_index,'query_digest':request.query_digest,
            'history_version':request.history_version})
        started=self._clock();status='accepted'
        try:
            client=self._clients[client_index]
            if operation=='recall':
                result=client.recall_models(args,limit=10)
            else:
                result=client.observe(list(args))
            return result
        except Exception as error:
            status='authorization_error' if _authorization_failure(error) else 'request_error'
            self._invalidate_generation(request,status,global_auth=status=='authorization_error')
            raise CacheError(status) from None
        finally:
            phase={'generation':request.generation,'operation':operation,'attempt':attempt,
                'client_slot':client_index,'status':status,'client_call_s':self._clock()-started}
            with self._lock:
                self._phases.append(phase)
            self._journal({'event':'response',**phase})

    def _submit(self,request,operation,client_index,args):
        future=self._io.submit(self._invoke,request,operation,client_index,args)
        future.add_done_callback(lambda _future:request.wake.set())
        return future

    def _await(self,request,futures):
        while True:
            request.wake.clear()
            if not self._current(request):
                for future in futures:
                    future.cancel()
                with self._lock:
                    if self._last_error=='authorization_error':
                        raise CacheError('authorization_error')
                raise CacheError('refresh_cancelled')
            if all(future.done() for future in futures):
                return [future.result() for future in futures]
            remaining=request.deadline-self._clock()
            if remaining<=0:
                for future in futures:
                    future.cancel()
                raise CacheError('refresh_deadline_expired')
            request.wake.wait(remaining)

    @staticmethod
    def _fuse(responses):
        merged={}
        for response in responses:
            for row in response['models']:
                key=row['candidate_id'];score=row['tsst_score']
                if not isinstance(key,str) or not key or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
                    raise ValueError('invalid_validated_candidate')
                if key in merged:
                    prior=merged[key]
                    if prior['revision']!=row['revision'] or prior['model']!=row['model'] or prior['claim_key']!=row['claim_key']:
                        raise ValueError('inconsistent_candidate_between_reads')
                    prior['tsst_score']=max(prior['tsst_score'],score)
                else:
                    merged[key]=_thaw(_freeze(row))
        return sorted(merged.values(),key=lambda row:(-row['tsst_score'],row['candidate_id']))

    @staticmethod
    def _observe_identity(response,ids):
        rows=response['observations'];seen=[]
        for row in rows:
            key=row['candidate_id']
            if key in seen or key not in ids or row.get('present') is not True:
                raise ValueError('invalid_observed_candidate_membership')
            seen.append(key)
        if set(seen)!=set(ids):
            raise ValueError('missing_observed_candidate')

    def _refresh(self,request):
        outcome=None
        self._journal({'event':'refresh_started','generation':request.generation,'mode':request.mode,
            'history_version':request.history_version,'query_digest':request.query_digest,
            'context_digest':_digest(request.context_key)})
        try:
            if request.mode=='async_parallel':
                futures=[self._submit(request,'recall',i,query) for i,query in enumerate(request.queries)]
                responses=self._await(request,futures)
            else:
                responses=[]
                for i,query in enumerate(request.queries):
                    responses.extend(self._await(request,[self._submit(request,'recall',i,query)]))
            validated_at=self._clock()
            for response in responses:
                if request.validate_recall(response,request.history_version) is False:
                    raise ValueError('caller_rejected_recall')
            models=self._fuse(responses)
            ids=tuple(row['candidate_id'] for row in models)
            # Empty validated retrieval is honest absence, never a fabricated
            # ODVL response and never a usable memory snapshot.
            if not ids:
                self._invalidate_generation(request,'validated_memory_absent')
                outcome=RefreshOutcome(request.generation,request.history_version,'empty',None,None,self._clock()-request.started_at)
                return outcome
            observed=self._await(request,[self._submit(request,'observe',0,ids)])[0]
            if request.validate_observe(observed,ids,request.history_version) is False:
                raise ValueError('caller_rejected_observe')
            self._observe_identity(observed,ids)
            frozen_models=_freeze(models);frozen_observations=_freeze(observed['observations'])
            frozen_recalls=_freeze(responses);frozen_observe=_freeze(observed)
            now=self._clock();expires=request.started_at+self.ttl_s
            if now>=request.deadline or now>=expires:
                raise CacheError('refresh_deadline_expired' if now>=request.deadline else 'snapshot_expired_before_publication')
            with self._lock:
                if self._stopped or request.cancel.is_set() or request.generation!=self._generation:
                    raise CacheError('refresh_cancelled')
                self._version+=1
                snapshot=MemorySnapshot(self._version,request.generation,request.history_version,
                    self.scope,self.service_fingerprint,request.context_key,request.query_digest,request.started_at,now,expires,
                    frozen_models,frozen_observations,frozen_recalls,frozen_observe)
                self._snapshot=snapshot;self._last_error=None
            outcome=RefreshOutcome(request.generation,request.history_version,'published',None,snapshot.version,now-request.started_at)
            self._journal({'event':'atomic_publish','generation':request.generation,'snapshot_version':snapshot.version,
                'models':len(models),'post_recall_validation_and_observe_s':now-validated_at,
                'wall_s':outcome.wall_s,'expires_at':expires})
            return outcome
        except Exception as error:
            allowed={'refresh_cancelled','refresh_deadline_expired','snapshot_expired_before_publication',
                'authorization_error','request_error','shared_call_budget_exhausted'}
            code=str(error) if isinstance(error,CacheError) and str(error) in allowed else 'validation_failed'
            self._invalidate_generation(request,code)
            status='cancelled' if code=='refresh_cancelled' else 'failed'
            outcome=RefreshOutcome(request.generation,request.history_version,status,code,None,self._clock()-request.started_at)
            return outcome
        finally:
            if outcome:
                with self._lock:
                    self._refreshes.append(outcome)
                self._journal({'event':'refresh_finished','generation':outcome.generation,
                    'status':outcome.status,'error_code':outcome.error_code,'wall_s':outcome.wall_s})

    def get_snapshot(self,now=None,*,min_history_version=None,context_key=None):
        started=time.perf_counter_ns();reason='state_busy';snapshot=None
        if self._lock.acquire(blocking=False):
            try:
                candidate=self._snapshot;current=self._clock() if now is None else now
                if self._stopped:
                    reason='stopped'
                elif candidate is None:
                    reason='absent'
                elif not isinstance(current,(int,float)) or not math.isfinite(current) or current<candidate.published_at:
                    reason='invalid_clock'
                elif current>=candidate.expires_at:
                    reason='expired';self._snapshot=None
                elif min_history_version is not None and (type(min_history_version) is not int or min_history_version<0):
                    reason='invalid_history_version'
                elif min_history_version is not None and candidate.history_version<min_history_version:
                    reason='older_history'
                elif context_key is not None and candidate.context_key!=context_key:
                    reason='context_mismatch'
                else:
                    snapshot=candidate;reason='hit'
            finally:
                self._lock.release()
        # deque append is a bounded in-memory operation; no journal/Cloud I/O.
        self._callback_samples.append((time.perf_counter_ns()-started,reason))
        return snapshot

    def wait_refresh(self,timeout=None):
        """Only outside the control callback; timeout does not authorize stale use."""
        with self._lock:
            future=self._future
        if future is None:
            return None
        return future.result(timeout=timeout)

    def invalidate(self,reason='explicit_invalidation'):
        with self._lock:
            self._snapshot=None;self._generation+=1;self._last_error='explicit_invalidation'
            if self._request:
                self._request.cancel.set();self._request.wake.set()
        self._journal({'event':'invalidate','reason_digest':_digest(str(reason))})

    @property
    def refreshing(self):
        with self._lock:
            return self._future is not None and not self._future.done()

    def stats(self):
        with self._lock:
            phases=[dict(row) for row in self._phases]
            refreshes=[{name:getattr(row,name) for name in row.__dataclass_fields__} for row in self._refreshes]
            version=self._version;last_error=self._last_error;stopped=self._stopped
        samples=list(self._callback_samples.copy());durations=sorted(v[0]/1e6 for v in samples)
        return {'scope':self.scope,'service_fingerprint':self.service_fingerprint,
            'snapshot_versions_published':version,'stopped':stopped,
            'last_error_code':last_error,'budget_reserved_attempts':self.budget.attempts,
            'budget_remaining':self.budget.remaining,'phases':phases,'refreshes':refreshes,
            'callback_samples_retained':len(samples),'callback_hits':sum(r=='hit' for _,r in samples),
            'callback_misses_by_reason':{reason:sum(r==reason for _,r in samples) for reason in sorted({r for _,r in samples if r!='hit'})},
            'callback_max_ms':max(durations,default=0.),
            'callback_p95_ms':durations[min(len(durations)-1,int(.95*len(durations)))] if durations else 0.,
            'ttl_scope':'local monotonic age bound, not proof of instantaneous offline permission revocation'}

    elapsedstats=stats

    def stop(self):
        """Nonblocking shutdown; already dispatched reads may finish but cannot publish.

        Clients remain caller-owned. This cannot forcibly abort a running HTTP
        request; the SDK's own timeout bounds its transport lifetime.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped=True;self._snapshot=None;self._generation+=1
            if self._request:
                self._request.cancel.set();self._request.wake.set()
        self._coordinator.shutdown(wait=False,cancel_futures=True)
        self._io.shutdown(wait=False,cancel_futures=True)
        self._journal({'event':'stopped','inflight_results_discarded':True})
