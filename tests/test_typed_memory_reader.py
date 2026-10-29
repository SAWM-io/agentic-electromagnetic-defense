"""Local fixtures/HTTPX MockTransport only; these are never live Cloud evidence."""
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from async_cloud_memory import AsyncCloudMemory, SharedCallBudget
from cloud_memory import CloudMemory, CloudMemoryError, MODEL_SCHEMA
from typed_memory_reader import (CausalTypedReader, planning, capture_http_baseline,
                                 measure_hot_reads_during_http)


SPACE = 'mem_fixture'
ENTITY = 'ent_' + 'a' * 32


def record(index, *, context='coilA', arx=False, revision=1):
    key = f'fixture-e{index:02d}'
    receipt = {'status': 'completed', 'epistemic_status': 'observed', 'candidate_id': key,
               'claim_key': 'claim-' + key, 'revision': revision, 'known_at_ms': 1000 + index,
               'memory_space_id': SPACE, 'entity_id': ENTITY}
    model = {'scientific_status': 'simulated_sensor_data', 'completed_episode_index': index,
             'context_id': context, 'frequency_hz': 1.63, 'nested': [1, 2]}
    if arx:
        model['residual_model'] = {'v': 1, 'ep': index, 'a': .8, 'b': .1}
    return {'candidate_id': key, 'claim_key': receipt['claim_key'], 'revision': revision,
            'known_at_ms': receipt['known_at_ms'], 'provenance_key': 'source-' + key,
            'model': model, 'receipt': receipt}


def versions_of(*rows):
    return {(row['candidate_id'], row['revision']): copy.deepcopy(row) for row in rows}


class FixtureSDK:
    def __init__(self, rows, http):
        self.rows = rows
        self.http = http
        self.calls = []
        self.edit = lambda response: response
        self.empty = False
        self.after_response_headers = None

    def recall_tsst(self, **request):
        self.calls.append(copy.deepcopy(request))
        rows = self.rows
        if 'entity' in request['semantic']:
            rows = [row for row in rows if row['claim_key'] == request['semantic']['entity']]
        if 'source_id' in request['semantic']:
            rows = [row for row in rows if row['provenance_key'] == request['semantic']['source_id']]
        evidence = [] if self.empty else [
            {'candidate_id': row['candidate_id'], 'claim_key': row['claim_key'], 'revision': row['revision'],
             'evidence_id': 'ev_' + f'{index:032x}', 'integrity_verified': True, 'status': 'observed', 'score': .8,
             'excerpt': row['claim_key'] + ' declaration ' + json.dumps({'schema': MODEL_SCHEMA, 'model': row['model']})}
            for index, row in enumerate(rows[:request['limit']], 1)]
        response = self.http.post('https://fixture.invalid/recall', json={'fixture': True})
        response.raise_for_status()
        if self.after_response_headers:
            assert self.after_response_headers.wait(2)
        return self.edit({'schema': 'basira-tsst-evidence/v1', 'memory_space_id': SPACE,
                          'entity_id': ENTITY, 'count': len(evidence), 'flashback': False, 'evidence': evidence})

    def odvl_observe(self, **request):
        self.calls.append(copy.deepcopy(request))
        self.http.post('https://fixture.invalid/observe', json={'fixture': True})
        ids = request['candidate_ids']
        return {'schema': 'basira-odvl-observation/v1', 'memory_space_id': SPACE, 'entity_id': ENTITY,
                'available': True, 'mode': 'on', 'clock': 'frozen', 'count': len(ids), 'requested_count': len(ids),
                'observations': [dict(candidate_id=key, present=True, resonance=.4, normalized=.5,
                                      normalized_global=.5, depth=.2, proof=.3) for key in ids]}

    def close(self):
        pass


@pytest.fixture
def factory(tmp_path):
    clients = []
    def make(rows, gate=None):
        attempts = [0]
        entered = threading.Event()
        def count(_request):
            attempts[0] += 1
        def transport(_request):
            entered.set()
            if gate is not None:
                assert gate.wait(2), 'local fixture gate not released'
            return httpx.Response(200, json={'fixture': True})
        http = httpx.Client(transport=httpx.MockTransport(transport), event_hooks={'request': [count]})
        sdk = FixtureSDK(copy.deepcopy(rows), http)
        raw = CloudMemory(sdk, SPACE, ENTITY, tmp_path / f'receipts-{len(clients)}.jsonl', _attempt_counter=lambda: attempts[0])
        raw._owned_http_client = http
        clients.append(raw)
        return raw, sdk, entered
    yield make
    for raw in clients:
        raw.close()


def query_for(row):
    return planning({'device_id': row['model']['context_id']}, [row], current_index=row['model']['completed_episode_index'] + 1)[0]


def test_planning_latest_context_then_distinct_latest_accepted_ARX():
    rows = [record(40, arx=True), record(41, context='coilB', arx=True), record(43)]
    first, second = map(json.loads, planning({'device_id': 'coilA'}, rows, current_index=44))
    assert first['expected_candidate'] == 'fixture-e43'
    assert second['expected_candidate'] == 'fixture-e40'
    assert first['semantic'] == {'predicate': 'declaration', 'entity': 'claim-fixture-e43',
                                 'entity_type': 'entity', 'source_id': 'source-fixture-e43'}


def test_latest_ARX_same_candidate_uses_broad_second_query_and_no_guessed_source():
    row = record(43, arx=True)
    del row['provenance_key']
    first, second = map(json.loads, planning({'device_id': 'coilA'}, [row]))
    assert 'source_id' not in first['semantic']
    assert second['expected_candidate'] is None and second['semantic'] == {'predicate': 'declaration'}


@pytest.mark.parametrize('fault', ['future', 'unaccepted', 'wrong_receipt'])
def test_planning_rejects_unconfirmed_or_future_metadata(fault):
    row = record(43)
    if fault == 'future':
        row['model']['completed_episode_index'] = 44
    elif fault == 'unaccepted':
        row['receipt']['status'] = 'pending'
    else:
        row['receipt']['candidate_id'] = 'other'
    with pytest.raises(ValueError):
        planning({'device_id': 'coilA'}, [row], current_index=44)


def test_typed_cloud_read_uses_frozen_validator_and_causal_index_above44(factory):
    row = record(46, arx=True)
    raw, sdk, _ = factory([row])
    reader = CausalTypedReader(raw, versions_of(row), 47)
    response = reader.recall_models(query_for(row))
    assert response['models'][0]['model'] == row['model']
    assert sdk.calls[0]['semantic']['source_id'] == row['provenance_key']
    assert raw._attempt_counter() == 1
    assert reader._cloud_base == raw._cloud_base and reader.client_instance_id == raw.client_instance_id
    assert not hasattr(reader, 'remember_model') and not hasattr(reader, 'stimulate')
    with pytest.raises(AttributeError):
        reader.current_index = 99


def test_reader_deep_freezes_versions_and_rejects_current_observation(factory):
    row = record(43)
    raw, _, _ = factory([row])
    versions = versions_of(row)
    reader = CausalTypedReader(raw, versions, 44)
    versions[(row['candidate_id'], 1)]['model']['nested'][0] = 999
    assert reader.recall_models(query_for(row))['models'][0]['model']['nested'] == [1, 2]
    with pytest.raises(ValueError):
        CausalTypedReader(raw, versions_of(record(44)), 44)


@pytest.mark.parametrize('fault', ['scope', 'integrity', 'payload', 'revision', 'other_candidate'])
def test_actual_returned_evidence_must_match_accepted_history_and_typed_target(factory, fault):
    row, other = record(43), record(42)
    raw, sdk, _ = factory([row, other])
    reader = CausalTypedReader(raw, versions_of(row, other), 44)
    def edit(response):
        evidence = response['evidence'][0]
        if fault == 'scope':
            response['entity_id'] = 'ent_' + 'b' * 32
        elif fault == 'integrity':
            evidence['integrity_verified'] = False
        elif fault == 'revision':
            evidence['revision'] = 2
        else:
            changed = copy.deepcopy(row if fault == 'payload' else other)
            if fault == 'payload':
                changed['model']['frequency_hz'] = 99
            evidence.update(candidate_id=changed['candidate_id'], claim_key=changed['claim_key'],
                excerpt=changed['claim_key'] + ' declaration ' + json.dumps({'schema': MODEL_SCHEMA, 'model': changed['model']}))
        return response
    sdk.edit = edit
    with pytest.raises((ValueError, CloudMemoryError)):
        reader.recall_models(query_for(row))


@pytest.mark.parametrize('edit', [lambda p: p['semantic'].update(entity_id='other'),
    lambda p: p['semantic'].update(entity_type='unknown'), lambda p: p['semantic'].update(source_id='unverified-source'),
    lambda p: p.update(expected_candidate=None), lambda p: p.update(write=True)])
def test_invalid_query_axes_fail_before_any_HTTP(factory, edit):
    row = record(43)
    raw, _, _ = factory([row])
    reader = CausalTypedReader(raw, versions_of(row), 44)
    plan = json.loads(query_for(row)); edit(plan)
    with pytest.raises(ValueError):
        reader.recall_models(json.dumps(plan))
    assert raw._attempt_counter() == 0


def test_empty_typed_result_never_injects_local_model(factory):
    row = record(43)
    raw, sdk, _ = factory([row]); sdk.empty = True
    reader = CausalTypedReader(raw, versions_of(row), 44)
    assert reader.recall_models(query_for(row))['models'] == []


def test_two_readers_and_grouped_observe_work_with_unmodified_async_cache(factory):
    rows = [record(40, arx=True), record(43)]
    raws = [factory(rows)[0] for _ in range(2)]
    readers = [CausalTypedReader(raw, versions_of(*rows), 44) for raw in raws]
    cache = AsyncCloudMemory(readers, validate_recall=lambda response, version: True,
                            validate_observe=lambda response, ids, version: True, budget=SharedCallBudget(3))
    try:
        cache.start_refresh(planning({'device_id': 'coilA'}, rows, current_index=44), history_version=44)
        assert cache.wait_refresh(2).status == 'published'
        assert {row['candidate_id'] for row in cache.get_snapshot().models} == {'fixture-e40', 'fixture-e43'}
        assert sum(raw._attempt_counter() for raw in raws) == 3
    finally:
        cache.stop()


class HotCache:
    refreshing = True
    def __init__(self):
        self.snapshot = SimpleNamespace(version=1)
        self.calls = 0
    def get_snapshot(self, **_kwargs):
        self.calls += 1
        return self.snapshot


def test_actual_mock_HTTP_pending_through_300_hot_reads_is_instrumented(factory):
    gate = threading.Event(); row = record(43)
    raw, _, entered = factory([row], gate)
    reader = CausalTypedReader(raw, versions_of(row), 44)
    baseline = capture_http_baseline([reader]); cache = HotCache()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(reader.recall_models, query_for(row))
        try:
            assert entered.wait(1)
            result = measure_hot_reads_during_http(cache, cache.snapshot, [reader], context_key='fixture',
                history_version=44, baseline=baseline, timeout_s=.1)
            assert result['http_overlap_proven'] is True and result['hits'] == 300
            assert result['continuous_http_tokens'] == [[raw.client_instance_id, 1]]
            assert result['batch_started_monotonic'] <= result['batch_ended_monotonic']
            assert result['http_after'][raw.client_instance_id]['response_headers'] == 0
        finally:
            gate.set(); future.result(2)
    assert reader.http_probe.capture()['active'] == []


def test_pending_future_after_headers_does_not_claim_network_overlap(factory):
    row = record(43); raw, sdk, _ = factory([row])
    gate = threading.Event(); sdk.after_response_headers = gate
    reader = CausalTypedReader(raw, versions_of(row), 44)
    baseline = capture_http_baseline([reader]); cache = HotCache()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(reader.recall_models, query_for(row))
        try:
            deadline = time.monotonic() + 1
            while reader.http_probe.capture()['response_headers'] < 1 and time.monotonic() < deadline:
                time.sleep(.001)
            assert not future.done()
            result = measure_hot_reads_during_http(cache, cache.snapshot, [reader], context_key='fixture',
                history_version=44, baseline=baseline, timeout_s=.01)
            assert not result['http_overlap_proven'] and result['overlap_kind'] == 'pending_only'
        finally:
            gate.set(); future.result(2)


def test_reader_recreation_does_not_duplicate_HTTP_hooks(factory):
    row = record(43); raw, _, _ = factory([row])
    first = CausalTypedReader(raw, versions_of(row), 44)
    second = CausalTypedReader(raw, versions_of(row), 45)
    assert first.http_probe is second.http_probe
    second.recall_models(query_for(row))
    assert first.http_probe.capture()['started'] == 1
    assert first.http_probe.capture()['response_headers'] == 1


def test_foreign_accepted_receipt_scope_fails_before_HTTP(factory):
    row = record(43); raw, _, _ = factory([row])
    row['receipt']['entity_id'] = 'ent_' + 'b' * 32
    with pytest.raises(ValueError, match='scope_mismatch'):
        CausalTypedReader(raw, versions_of(row), 44)
    assert raw._attempt_counter() == 0


def test_duplicate_JSON_fields_fail_before_HTTP(factory):
    row = record(43); raw, _, _ = factory([row])
    reader = CausalTypedReader(raw, versions_of(row), 44)
    duplicate = '{"query":"first","query":"second","semantic":{"predicate":"declaration"}}'
    with pytest.raises(ValueError):
        reader.recall_models(duplicate)
    assert raw._attempt_counter() == 0


def test_missing_HTTP_instrumentation_is_honestly_pending_only(factory):
    row = record(43); raw, _, _ = factory([row])
    owned = raw._owned_http_client
    raw._owned_http_client = None
    try:
        reader = CausalTypedReader(raw, versions_of(row), 44)
        baseline = capture_http_baseline([reader]); cache = HotCache()
        result = measure_hot_reads_during_http(cache, cache.snapshot, [reader], context_key='fixture',
            history_version=44, baseline=baseline, timeout_s=0)
        assert not result['http_overlap_proven'] and result['overlap_kind'] == 'pending_only'
        assert result['hits'] == 300 and raw._attempt_counter() == 0
    finally:
        raw._owned_http_client = owned
