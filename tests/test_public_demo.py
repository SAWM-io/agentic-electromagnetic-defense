"""Offline contract tests. Mock responses below are not live Cloud evidence."""
import copy
import json
import math
from pathlib import Path
import threading

import pytest

import run_demo
from cloud_memory import CloudMemory, CloudMemoryError


class NoWait:
    def call(self, operation, function, *args, **kwargs):
        return function(*args, **kwargs)


class FixtureStore:
    def __init__(self):
        self.records={};self.operations=[];self.lock=threading.Lock()
        self.fail_stimulus=False;self.mode='on';self.inject_foreign=False

    def client(self,path):
        sdk=FixtureSDK(self)
        raw=CloudMemory(sdk,'mem_fixture','ent_'+'a'*32,path,
                        sdk_source_sha256='fixture-only',_attempt_counter=lambda:sdk.calls)
        raw._reconnect_factory=lambda:self.client(path)
        return raw


class FixtureSDK:
    def __init__(self,store):self.store=store;self.calls=0
    def close(self):pass
    def base(self,schema):return {'schema':schema,'memory_space_id':'mem_fixture','entity_id':'ent_'+'a'*32}
    def call(self,operation):
        with self.store.lock:self.calls+=1;self.store.operations.append(operation)
    def write_assertion(self,**request):
        self.call('write');a=request['assertion'];key=a['candidate_id']
        assert key not in self.store.records,'A mutation must not be replayed'
        receipt={**self.base('basira-assertion/v1'),'status':'completed','claim_key':a['claim_key'],
                 'candidate_id':key,'revision':1,'epistemic_status':'observed','confidence':a['confidence'],
                 'odvl_applied':True,'known_at_ms':1000+len(self.store.records),'valid_time_ms':None,
                 'decision_time_ms':None,'belief_time_ms':None}
        self.store.records[key]={'request':copy.deepcopy(a),'receipt':copy.deepcopy(receipt)}
        return receipt
    def recall_tsst(self,**request):
        self.call('recall');axes=request['semantic'];rows=[]
        for i,(key,record) in enumerate(self.store.records.items(),1):
            a=record['request']
            if 'entity' in axes and a['claim_key']!=axes['entity']:continue
            if 'source_id' in axes and a['provenance_key']!=axes['source_id']:continue
            rows.append({'candidate_id':key,'claim_key':a['claim_key'],'revision':1,
                         'evidence_id':'ev_'+f'{i:032x}','integrity_verified':True,'status':'observed','score':.8,
                         'excerpt':a['claim_key']+' declaration '+a['content']})
        
        if self.store.inject_foreign and rows:
            rows[0]={**rows[0], 'candidate_id':'unconfirmed-foreign-model'}
        rows=rows[:request['limit']]
        return {**self.base('basira-tsst-evidence/v1'),'evidence':rows,'count':len(rows),'flashback':False}
    def observation(self,ids,schema,clock):
        rows=[{'candidate_id':k,'present':k in self.store.records,'resonance':1.,'normalized':.5,
               'normalized_global':.5,'depth':0.,'proof':1.} for k in ids]
        return {**self.base(schema),'available':True,'mode':self.store.mode,'clock':clock,
                'observations':rows,'count':len(rows),'requested_count':len(rows)}
    def odvl_observe(self,**request):
        self.call('observe')
        return self.observation(request['candidate_ids'],'basira-odvl-observation/v1','frozen')
    def odvl_stimulate(self,**request):
        self.call('stimulate')
        if self.store.fail_stimulus:raise CloudMemoryError('ip_rate_limited')
        ids=[x['candidate_id'] for x in request['stimuli']]
        return {**self.observation(ids,'basira-odvl-stimulation/v1','game'),'applied':True,
                'settle_steps':request['settle_steps'],'elapsed_units':request['elapsed_units'],
                'steps_applied':math.ceil(request['elapsed_units']/request['dt_units']-1e-10)}


def test_offline_has_no_cloud_or_secret_file_access(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Offline must not open Cloud credentials')
    monkeypatch.setattr(CloudMemory,'from_credentials',forbidden)
    out=tmp_path/'offline';summary=run_demo.run(out,mode='offline')
    assert summary['actual_http_attempts']==0 and summary['new_assertions']==0
    assert summary['controller_runs']==12 and summary['live_cloud'] is False
    rows=json.loads((out/'episodes.json').read_text())
    assert not any(r['memory_contributor_ids'] or r['returned_ids'] for r in rows)
    assert not list(out.glob('*http*'))
    with pytest.raises(FileExistsError):run_demo.run(out,mode='offline')


def test_cloud_requires_explicit_dedicated_lab_confirmation(tmp_path):
    with pytest.raises(ValueError,match='confirm_dedicated_lab'):
        run_demo.run(tmp_path/'cloud',mode='cloud',credentials='must-not-open')
    assert not (tmp_path/'cloud').exists()


def test_cloud_complete_contract_uses_returned_versions_and_persists_new_history(tmp_path):
    store=FixtureStore();clients=(store.client(tmp_path/'http.jsonl'),store.client(tmp_path/'http.jsonl'))
    out=tmp_path/'cloud';summary=run_demo.run(out,mode='cloud',confirm_dedicated_lab=True,clients=clients,pacer=NoWait())
    assert summary['actual_http_attempts']==44 and summary['reserved_attempts']==44
    assert summary['new_assertions']==6 and summary['latest_context_verified']
    assert summary['controller_runs']==36
    assert len(store.records)==6 and store.operations.count('write')==6
    snapshots=json.loads((out/'private-snapshots.json').read_text())
    for snapshot in snapshots:
        assert snapshot['snapshot']['models']
        assert all(r['model']['completed_episode_index']<snapshot['episode_index'] for r in snapshot['snapshot']['models'])
    assert all(json.loads(r['request']['content'])['model']['scientific_status']=='simulated_sensor_data' for r in store.records.values())


def test_cloud_failure_keeps_receipt_and_never_retries_mutation(tmp_path):
    store=FixtureStore();store.fail_stimulus=True
    clients=(store.client(tmp_path/'http.jsonl'),store.client(tmp_path/'http.jsonl'))
    out=tmp_path/'failure'
    with pytest.raises(CloudMemoryError,match='ip_rate_limited'):
        run_demo.run(out,mode='cloud',confirm_dedicated_lab=True,clients=clients,pacer=NoWait())
    assert store.operations==['write','stimulate']
    receipts=json.loads((out/'private-receipts.json').read_text())
    assert len(receipts)==1 and receipts[0]['remember']['status']=='completed'
    assert receipts[0]['stimulus_status']=='attempted'
    assert (out/'failure.json').exists() and not (out/'summary.json').exists()


def test_shadow_odvl_is_not_silently_called_active(tmp_path):
    store=FixtureStore();store.mode='shadow'
    clients=(store.client(tmp_path/'http.jsonl'),store.client(tmp_path/'http.jsonl'))
    with pytest.raises(RuntimeError,match='active_ODVL_required'):
        run_demo.run(tmp_path/'shadow',mode='cloud',confirm_dedicated_lab=True,clients=clients,pacer=NoWait())
    assert store.operations.count('write')==1


def test_public_sources_do_not_import_private_engines():
    root=Path(__file__).resolve().parents[1]
    for path in root.glob('*.py'):
        text=path.read_text()
        assert 'from services.' not in text
        assert 'odvl_adapter' not in text
        assert 'MAGNETIC_TEST_PRIVATE_SOURCE_ROOT' not in text
        assert '/Users/' not in text


def test_unconfirmed_cloud_memory_stops_before_next_action_or_write(tmp_path):
    store=FixtureStore();store.inject_foreign=True
    clients=(store.client(tmp_path/'http.jsonl'),store.client(tmp_path/'http.jsonl'))
    out=tmp_path/'foreign'
    with pytest.raises(RuntimeError,match='authenticated_refresh_unavailable'):
        run_demo.run(out,mode='cloud',confirm_dedicated_lab=True,clients=clients,pacer=NoWait())
    assert store.operations.count('write')==1
    assert not (out/'summary.json').exists()
