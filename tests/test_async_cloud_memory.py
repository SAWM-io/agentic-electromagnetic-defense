"""Local cache/concurrency contracts. Fake transports are never Cloud results."""
import copy
import json
from pathlib import Path
import threading
import time

import pytest

from async_cloud_memory import (AsyncCloudMemory,BudgetExhausted,CacheError,
    RefreshInProgress,SharedCallBudget)


def row(key='a',score=.8,index=0):
    return {'candidate_id':key,'claim_key':key,'revision':1,'tsst_score':score,
        'model':{'completed_episode_index':index,'frequency_hz':1.68,'nested':[1,2]}}


class FakeClient:
    """Bounded, observable in-memory test transport only."""
    space_id='test-space';entity_id='test-entity';_max_retries=0
    def __init__(self,slot):
        self.client_instance_id=f'fake-client-{slot}';self.rows=[row()]
        self.recall_gate=None;self.observe_gate=None;self.barrier=None
        self.recall_entered=threading.Event();self.observe_entered=threading.Event()
        self.fail=None;self.invalid_observe=False;self.calls=[];self.closed=False
    def recall_models(self,query,limit=10):
        self.calls.append(('recall',query));self.recall_entered.set()
        if self.barrier:
            self.barrier.wait(timeout=1)
        if self.recall_gate:
            assert self.recall_gate.wait(2),'test gate not released'
        if self.fail:
            raise self.fail
        return {'scope':'test','models':copy.deepcopy(self.rows)}
    def observe(self,ids):
        self.calls.append(('observe',tuple(ids)));self.observe_entered.set()
        if self.observe_gate:
            assert self.observe_gate.wait(2),'test gate not released'
        chosen=ids[:-1] if self.invalid_observe else ids
        return {'scope':'test','observations':[{'candidate_id':key,'present':True,
            'normalized':.7,'resonance':.6} for key in chosen]}
    def close(self):
        self.closed=True


def validate_recall(response,version):
    assert response['scope']=='test'
    assert all(r['model']['completed_episode_index']<version for r in response['models'])


def validate_observe(response,ids,version):
    assert response['scope']=='test' and version>0


@pytest.fixture
def setup_cache():
    created=[]
    def make(**kwargs):
        clients=(FakeClient(0),FakeClient(1))
        cache=AsyncCloudMemory(clients,validate_recall=validate_recall,
            validate_observe=validate_observe,ttl_s=kwargs.pop('ttl_s',10),
            refresh_deadline_s=kwargs.pop('refresh_deadline_s',1),**kwargs)
        created.append((cache,clients));return cache,clients
    yield make
    for cache,clients in created:
        for client in clients:
            if client.recall_gate:client.recall_gate.set()
            if client.observe_gate:client.observe_gate.set()
        cache.stop()


def warm(cache,version=1,**kwargs):
    cache.start_refresh(('query-a','query-b'),history_version=version,**kwargs)
    outcome=cache.wait_refresh(timeout=2)
    assert outcome.status=='published',outcome
    return cache.get_snapshot()


def test_atomic_snapshot_requires_validated_grouped_observe(setup_cache):
    cache,clients=setup_cache();gate=threading.Event();clients[0].observe_gate=gate
    clients[1].rows=[row('b',.9)]
    cache.start_refresh(('a','b'),history_version=1)
    assert clients[0].observe_entered.wait(1)
    assert cache.get_snapshot() is None
    assert clients[0].calls[-1]==('observe',('b','a'))
    gate.set();assert cache.wait_refresh(1).status=='published'
    assert [r['candidate_id'] for r in cache.get_snapshot().models]==['b','a']


def test_snapshot_deeply_immutable_and_to_dict_is_detached(setup_cache):
    cache,clients=setup_cache();snapshot=warm(cache)
    with pytest.raises(TypeError):snapshot.models[0]['model']['frequency_hz']=8.
    with pytest.raises(TypeError):snapshot.models[0]['model']['nested'][0]=9
    plain=snapshot.to_dict();plain['models'][0]['model']['nested'][0]=9
    clients[0].rows[0]['model']['frequency_hz']=8.
    assert snapshot.models[0]['model']['nested'][0]==1
    assert snapshot.models[0]['model']['frequency_hz']==1.68


def test_300_cache_hits_do_not_wait_for_cloud_refresh(setup_cache):
    cache,clients=setup_cache();snapshot=warm(cache)
    gate=threading.Event()
    for client in clients:
        client.recall_gate=gate;client.recall_entered.clear()
    cache.start_refresh(('next-a','next-b'),history_version=1)
    assert all(client.recall_entered.wait(1) for client in clients)
    before=time.perf_counter()
    values=[cache.get_snapshot() for _ in range(300)]
    elapsed=time.perf_counter()-before
    assert all(value is snapshot for value in values)
    assert elapsed<.1  # Fake Cloud remains blocked until after all 300 decisions.
    assert cache.stats()['callback_hits']>=301
    gate.set();assert cache.wait_refresh(1).status=='published'


def test_busy_state_lock_is_a_nonblocking_miss(setup_cache):
    cache,_=setup_cache();warm(cache)
    cache._lock.acquire()
    try:
        started=time.perf_counter();assert cache.get_snapshot() is None
        assert time.perf_counter()-started<.02
    finally:cache._lock.release()


def test_ttl_clock_context_and_history_gates(setup_cache):
    clock=[100.];cache,_=setup_cache(clock=lambda:clock[0],ttl_s=2)
    snapshot=warm(cache,version=3,context_key='prior-context')
    assert cache.get_snapshot(min_history_version=4) is None
    assert cache.get_snapshot(context_key='new-context') is None
    assert cache.get_snapshot() is snapshot  # prior-only evidence, caller qualifies locally
    assert cache.get_snapshot(now=99) is None
    clock[0]=102.;assert cache.get_snapshot() is None


def test_parallel_mode_uses_distinct_clients_and_matches_sequential(setup_cache):
    cache,clients=setup_cache();clients[1].rows=[row('b',.9)]
    seq=warm(cache,mode='sync_sequential')
    barrier=threading.Barrier(2)
    for client in clients:client.barrier=barrier
    parallel=warm(cache,mode='async_parallel')
    assert seq.models==parallel.models and seq.observations==parallel.observations
    assert cache.budget.attempts==6


@pytest.mark.parametrize('status',[401,403])
def test_authorization_failure_invalidates_unexpired_cache(setup_cache,status):
    class HTTPFailure(Exception):
        status_code=status
    cache,clients=setup_cache();warm(cache)
    clients[0].fail=HTTPFailure('secret must not be logged')
    cache.start_refresh(('a','b'),history_version=1)
    outcome=cache.wait_refresh(1)
    assert outcome.status in ('failed','cancelled')
    assert cache.get_snapshot() is None
    assert cache.stats()['last_error_code']=='authorization_error'


def test_invalid_versions_and_observation_membership_never_publish(setup_cache):
    cache,clients=setup_cache();warm(cache)
    clients[0].rows[0]['model']['completed_episode_index']=2
    cache.start_refresh(('a','b'),history_version=1)
    assert cache.wait_refresh(1).status=='failed';assert cache.get_snapshot() is None
    clients[0].rows=[row()];clients[0].invalid_observe=True
    cache.start_refresh(('a','b'),history_version=1)
    assert cache.wait_refresh(1).status=='failed';assert cache.get_snapshot() is None


def test_deadline_discards_late_response_and_invalidates_old_cache(setup_cache):
    cache,clients=setup_cache(refresh_deadline_s=.05);warm(cache)
    gate=threading.Event()
    for client in clients:client.recall_gate=gate
    cache.start_refresh(('a','b'),history_version=1)
    outcome=cache.wait_refresh(1)
    assert outcome.error_code=='refresh_deadline_expired'
    assert cache.get_snapshot() is None
    gate.set()
    assert cache.get_snapshot() is None


def test_explicit_invalidation_cancels_generation_and_binds_new_validators(setup_cache):
    cache,clients=setup_cache();gate=threading.Event()
    for client in clients:client.recall_gate=gate
    cache.start_refresh(('old-a','old-b'),history_version=1)
    assert all(client.recall_entered.wait(1) for client in clients)
    with pytest.raises(RefreshInProgress):cache.start_refresh(('busy-a','busy-b'),history_version=1)
    cache.invalidate('new confirmed write epoch')
    for client in clients:client.rows=[row('new',index=1)]
    seen=[]
    def new_validator(response,version):
        seen.append(version);assert version==2
        assert all(r['candidate_id']=='new' for r in response['models'])
    cache.start_refresh(('new-a','new-b'),history_version=2,validate_recall=new_validator)
    gate.set();outcome=cache.wait_refresh(1)
    assert outcome.status=='published' and seen==[2,2]
    assert cache.get_snapshot().history_version==2
    assert cache.get_snapshot().models[0]['candidate_id']=='new'


def test_stop_is_nonblocking_and_cannot_publish_inflight_reads(setup_cache):
    cache,clients=setup_cache();gate=threading.Event()
    for client in clients:client.recall_gate=gate
    cache.start_refresh(('a','b'),history_version=1)
    assert all(client.recall_entered.wait(1) for client in clients)
    before=time.perf_counter();cache.stop();assert time.perf_counter()-before<.1
    assert cache.get_snapshot() is None
    with pytest.raises(CacheError,match='stopped'):cache.start_refresh(('a','b'),history_version=1)
    gate.set();cache.wait_refresh(1)
    assert cache.get_snapshot() is None and not any(c.closed for c in clients)


def test_shared_budget_counts_external_work_and_stops_before_dispatch(setup_cache):
    budget=SharedCallBudget(3);budget.reserve('external-validated-write')
    cache,clients=setup_cache(budget=budget)
    cache.start_refresh(('a','b'),history_version=1)
    assert cache.wait_refresh(1).error_code=='shared_call_budget_exhausted'
    assert budget.attempts==3
    assert not any(call[0]=='observe' for c in clients for call in c.calls)
    with pytest.raises(BudgetExhausted):budget.reserve('another-write')


def test_empty_validated_retrieval_is_absence_without_fake_observe(setup_cache):
    cache,clients=setup_cache()
    for client in clients:client.rows=[]
    cache.start_refresh(('a','b'),history_version=1)
    assert cache.wait_refresh(1).status=='empty'
    assert cache.get_snapshot() is None and cache.budget.attempts==2
    assert not any(call[0]=='observe' for c in clients for call in c.calls)


def test_journal_has_timing_but_no_queries_exception_text_or_context_secrets(setup_cache,tmp_path):
    path=tmp_path/'events.jsonl';cache,clients=setup_cache(journal_path=path)
    cache.start_refresh(('QUERY_SECRET_A','QUERY_SECRET_B'),history_version=1,context_key='CONTEXT_SECRET')
    assert cache.wait_refresh(1).status=='published'
    clients[0].fail=RuntimeError('ERROR_SECRET')
    cache.start_refresh(('QUERY_SECRET_A','QUERY_SECRET_B'),history_version=1)
    cache.wait_refresh(1)
    text=path.read_text()
    assert all(secret not in text for secret in ('QUERY_SECRET','CONTEXT_SECRET','ERROR_SECRET'))
    assert any(r['event']=='atomic_publish' for r in map(json.loads,text.splitlines()))


def test_scope_retry_and_instance_configuration_fail_closed():
    a,b=FakeClient(0),FakeClient(1);b.entity_id='other'
    with pytest.raises(ValueError,match='scope'):
        AsyncCloudMemory((a,b),validate_recall=validate_recall,validate_observe=validate_observe)
    b.entity_id=a.entity_id;b._max_retries=1
    with pytest.raises(ValueError,match='retries'):
        AsyncCloudMemory((a,b),validate_recall=validate_recall,validate_observe=validate_observe)
    with pytest.raises(ValueError,match='distinct'):
        AsyncCloudMemory((a,a),validate_recall=validate_recall,validate_observe=validate_observe)


@pytest.mark.parametrize('attribute,value',[('_cloud_base','https://other.test'),('sdk_source_sha256','other-sdk')])
def test_base_and_sdk_are_pinned_between_clients_and_during_refresh(setup_cache,attribute,value):
    a,b=FakeClient(0),FakeClient(1);setattr(b,attribute,value)
    with pytest.raises(ValueError,match='base_or_sdk'):
        AsyncCloudMemory((a,b),validate_recall=validate_recall,validate_observe=validate_observe)
    cache,clients=setup_cache();warm(cache)
    setattr(clients[0],attribute,value)
    cache.start_refresh(('a','b'),history_version=1)
    assert cache.wait_refresh(1).status=='failed'
    assert cache.get_snapshot() is None


def test_actual_adapter_and_sdk_mocktransport_auth_code_invalidates_cache(tmp_path,monkeypatch):
    """Actual SDK+adapter, HTTPX MockTransport only: zero network access."""
    import cloud_memory as cloud
    import httpx
    sdk,digest=cloud._load_sdk(cloud.DEFAULT_SDK_PATH)
    real_http_client=httpx.Client;deny=threading.Event();attempts=[]
    space='mem_test';entity='ent_'+'a'*32;secret='UNIT_TEST_KEY_NEVER_LOG'
    model={'scientific_status':'simulated_sensor_data','completed_episode_index':0,'frequency_hz':1.68}
    def transport(request):
        attempts.append(request.method)
        if deny.is_set():
            return httpx.Response(403,json={'detail':{'code':'memory_authorization_denied'}})
        body=json.loads(request.content)
        if 'candidate_ids' in body:
            ids=body['candidate_ids']
            payload={'schema':'basira-odvl-observation/v1','memory_space_id':space,'entity_id':entity,
                'mode':'on','available':True,'clock':'frozen','count':len(ids),'requested_count':len(ids),
                'observations':[{'candidate_id':key,'present':True,'resonance':.4,'normalized':.5,
                    'normalized_global':.4,'depth':.1,'proof':.3,'carried_by':'surface','cluster_id':None} for key in ids],
                'culminations':{'global':1.,'clusters':{}},'last_step_at':None}
        else:
            payload={'schema':'basira-tsst-evidence/v1','memory_space_id':space,'entity_id':entity,
                'count':1,'flashback':False,'evidence':[{'evidence_id':'ev_'+'c'*32,'candidate_id':'a',
                    'claim_key':'a','revision':1,'integrity_verified':True,'score':.8,'status':'observed',
                    'excerpt':'a declaration '+json.dumps({'schema':cloud.MODEL_SCHEMA,'model':model})}]}
        return httpx.Response(200,json=payload)
    monkeypatch.setattr(sdk.httpx,'Client',lambda **kwargs:real_http_client(
        transport=httpx.MockTransport(transport),**kwargs))
    credentials=tmp_path/'credentials.env'
    credentials.write_text(f'BASIRA_BASE=https://example.test/api/v1/basira\nBASIRA_SPACE={space}\nBASIRA_ENTITY={entity}\nBASIRA_KEY={secret}\n')
    credentials.chmod(0o600)
    clients=tuple(cloud.CloudMemory.from_credentials(credentials,tmp_path/f'http-{i}.jsonl') for i in range(2))
    assert attempts==[]
    def valid_recall(response,version):
        assert response['entity_id']==entity and response['memory_space_id']==space
        assert all(r['model']['completed_episode_index']<version for r in response['models'])
    def valid_observe(response,ids,version):
        assert response['entity_id']==entity and response['memory_space_id']==space
    journal=tmp_path/'cache-events.jsonl'
    cache=AsyncCloudMemory(clients,validate_recall=valid_recall,validate_observe=valid_observe,journal_path=journal)
    try:
        snapshot=warm(cache)
        assert snapshot.scope==(space,entity) and snapshot.models[0]['candidate_id']=='a'
        assert len(attempts)==3 and clients[0].sdk_source_sha256==digest
        deny.set();cache.start_refresh(('a','b'),history_version=1)
        cache.wait_refresh(2)
        assert cache.get_snapshot() is None
        assert cache.stats()['last_error_code']=='authorization_error'
        assert secret not in journal.read_text()
        records=[json.loads(line) for file in tmp_path.glob('http-*.jsonl') for line in file.read_text().splitlines()]
        failures=[r for r in records if r.get('accepted') is False]
        assert failures and all(r['error_code']=='memory_authorization_denied' for r in failures)
        assert all(r['http_attempts']==1 for r in records if r.get('phase')=='response')
    finally:
        cache.stop()
        for client in clients:client.close()
