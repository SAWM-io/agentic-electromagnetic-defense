"""Public SDK contract tests. No network and no private engine imports."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('magnetic_cloud_memory_test', ROOT / 'cloud_memory.py')
cloud = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cloud
SPEC.loader.exec_module(cloud)

SPACE = 'mem_test'
ENTITY = 'ent_' + 'a' * 32
SECRET = 'a-private-key-that-must-never-appear'
MODEL = {'schema': 'episode/v1', 'frequency_hz': 3.2, 'episode_id': 'ep-1',
         'observation_sha256': 'b' * 64, 'scientific_status': cloud.SCIENTIFIC_STATUS}


class FakePublicSDK:
    def __init__(self):
        self.calls = []
        self.edit = lambda operation, response: response
        self.fail = None

    def _result(self, operation, request, response):
        self.calls.append((operation, copy.deepcopy(request)))
        if self.fail:
            raise self.fail
        return self.edit(operation, response)

    def write_assertion(self, **request):
        body = request['assertion']
        result = {'schema': 'basira-assertion/v1', 'memory_space_id': SPACE,
                  'entity_id': ENTITY, 'status': 'completed', 'claim_key': body['claim_key'],
                  'candidate_id': body['candidate_id'], 'revision': 2 if body['declared_revision'] else 1,
                  'epistemic_status': 'observed', 'confidence': body['confidence'],
                  'known_at_ms': 1001, 'valid_time_ms': body.get('valid_time_ms'),
                  'decision_time_ms': None, 'belief_time_ms': 1001, 'odvl_applied': True}
        return self._result('write_assertion', request, result)

    def recall_tsst(self, **request):
        result = {'schema': 'basira-tsst-evidence/v1', 'memory_space_id': SPACE,
                  'entity_id': ENTITY, 'count': 1, 'flashback': bool(request['temporal']),
                  'evidence': [{'evidence_id': 'ev_' + 'c' * 32, 'candidate_id': 'ep-1',
                                'claim_key': 'episode', 'revision': 1, 'integrity_verified': True,
                                'score': .8, 'status': 'observed',
                                'excerpt': 'episode declaration ' + json.dumps({'schema': cloud.MODEL_SCHEMA, 'model': MODEL})}]}
        return self._result('recall_tsst', request, result)

    def _observation(self, operation, request):
        identifiers = request.get('candidate_ids', [row['candidate_id'] for row in request.get('stimuli', [])])
        stimulation = operation == 'odvl_stimulate'
        result = {'schema': 'basira-odvl-stimulation/v1' if stimulation else 'basira-odvl-observation/v1',
                  'memory_space_id': SPACE, 'entity_id': ENTITY, 'mode': 'on', 'available': True,
                  'clock': 'game' if stimulation else 'frozen', 'count': len(identifiers),
                  'requested_count': len(identifiers),
                  'observations': [{'candidate_id': value, 'present': True, 'resonance': .4,
                                    'normalized': .5, 'normalized_global': .4, 'depth': .1,
                                    'proof': .3, 'carried_by': 'surface', 'cluster_id': None}
                                   for value in identifiers],
                  'culminations': {'global': 1.0, 'clusters': {}}, 'last_step_at': None}
        if stimulation:
            result.update(applied=True, elapsed_units=request['elapsed_units'],
                          steps_applied=round(request['elapsed_units'] / request['dt_units']),
                          settle_steps=request['settle_steps'], touched=len(identifiers), ingested=0)
        return self._result(operation, request, result)

    def odvl_observe(self, **request):
        return self._observation('odvl_observe', request)

    def odvl_stimulate(self, **request):
        return self._observation('odvl_stimulate', request)


@pytest.fixture
def session(tmp_path):
    client = FakePublicSDK()
    memory = cloud.CloudMemory(client, SPACE, ENTITY, tmp_path / 'receipts.jsonl', _secrets=(SECRET,))
    return memory, client


def remember(memory, **changes):
    return memory.remember_model(MODEL, **{'claim_key': 'episode', 'candidate_id': 'ep-1',
                                         'provenance_key': 'simulated-run/ep-1',
                                         'idempotency_key': 'write.001', **changes})


def test_actual_responses_and_simulated_model_round_trip_are_recorded(session):
    memory, client = session
    receipt = remember(memory, valid_time_ms=900)
    assert receipt['known_at_ms'] == 1001
    result = memory.recall_models('simulated episode')
    assert result['models'][0]['model'] == MODEL
    assert result['models'][0]['candidate_id'] == 'ep-1'
    assert result['raw_response']['evidence'] == result['evidence']
    records = [json.loads(line) for line in memory.receipt_path.read_text().splitlines()]
    assert [row['phase'] for row in records] == ['request', 'response', 'request', 'response']
    assert records[1]['response'] == receipt
    assert records[3]['response'] == result['raw_response']
    assert records[1]['elapsed_ms'] >= 0 and records[3]['logical_sdk_call_index'] == 2
    assert records[1]['sdk_max_retries'] == 0 and records[1]['http_attempt_ceiling'] == 1
    assert records[1]['http_attempts'] is None and not records[1]['http_attempts_instrumented']
    assert all(row['scientific_status'] == 'simulated_sensor_data' and not row['physical_measurements'] for row in records)
    assert client.calls[1][1]['semantic'] == {'predicate': 'declaration'}


@pytest.mark.parametrize('field,value', [('entity_id', 'ent_' + 'b' * 32),
                                       ('memory_space_id', 'mem_elsewhere'),
                                       ('schema', 'other'), ('candidate_id', 'not-requested'),
                                       ('revision', 2), ('odvl_applied', False), ('known_at_ms', None)])
def test_assertion_receipts_fail_closed(session, field, value):
    memory, client = session
    client.edit = lambda operation, response: {**response, field: value}
    with pytest.raises(cloud.CloudMemoryError):
        remember(memory)
    assert json.loads(memory.receipt_path.read_text().splitlines()[-1])['accepted'] is False


def test_revision_is_explicit_and_flashback_uses_server_transaction_time(session):
    memory, client = session
    receipt = remember(memory, revision=2, reason='Corrected simulated sensor fit.', candidate_id='ep-1-r2')
    payload = client.calls[0][1]['assertion']
    assert receipt['revision'] == 2 and payload['declared_revision'] is True
    assert payload['evidence'] and payload['reason']
    memory.flashback_models('episode', as_of_transaction_time_ms=1000, as_of_valid_time_ms=900)
    request = client.calls[-1][1]
    assert request['temporal'] == {'as_of_transaction_time_ms': 1000, 'as_of_valid_time_ms': 900}
    assert request['epistemic']['include_belief_history'] is True
    assert 'as_of_belief_time_ms' not in request['temporal']


def test_bare_json_content_is_also_accepted_without_rewriting_raw_response(session):
    memory, client = session
    content = json.dumps({'schema': cloud.MODEL_SCHEMA, 'model': MODEL})
    def edit(operation, response):
        response['evidence'][0]['excerpt'] = content
        return response
    client.edit = edit
    result = memory.recall_models('episode')
    assert result['models'][0]['model'] == MODEL
    assert result['raw_response']['evidence'][0]['excerpt'] == content


@pytest.mark.parametrize('prefix', [
    'other-claim declaration ', 'episode other-predicate ',
    'untrusted commentary episode declaration ', 'episode declaration junk ',
    'episode declaration episode declaration ',
])
def test_hydrated_prefix_must_match_exact_claim_and_predicate(session, prefix):
    memory, client = session
    def edit(operation, response):
        response['evidence'][0]['excerpt'] = prefix + json.dumps({'schema': cloud.MODEL_SCHEMA, 'model': MODEL})
        return response
    client.edit = edit
    with pytest.raises(cloud.CloudMemoryError, match='unparseable_tsst_model'):
        memory.recall_models('episode')
    assert json.loads(memory.receipt_path.read_text().splitlines()[-1])['accepted'] is False


@pytest.mark.parametrize('change', [
    lambda r: r.update(integrity_verified=False),
    lambda r: r.update(candidate_id=None),
    lambda r: r.update(evidence_id='invalid'),
    lambda r: r.update(revision=True),
    lambda r: r.update(excerpt='not JSON'),
    lambda r: r.update(score=float('nan')),
    lambda r: r.update(excerpt=json.dumps({'schema': cloud.MODEL_SCHEMA, 'model': {'scientific_status': 'physical'}})),
])
def test_tsst_evidence_contract_is_checked_before_model_use(session, change):
    memory, client = session
    def edit(operation, response):
        change(response['evidence'][0])
        return response
    client.edit = edit
    with pytest.raises(cloud.CloudMemoryError):
        memory.recall_models('episode')


def test_observe_is_pure_public_sdk_read_and_preserves_explicit_absence(session):
    memory, client = session
    def edit(operation, response):
        response['observations'][0].update(present=False, resonance=0, normalized=0, normalized_global=0, depth=0, proof=0)
        return response
    client.edit = edit
    response = memory.observe(['ep-1'])
    assert response['observations'][0]['present'] is False
    assert [call[0] for call in client.calls] == ['odvl_observe']


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(available=False),
    lambda r: r.update(clock='wall'),
    lambda r: r.update(entity_id='ent_' + 'b' * 32),
    lambda r: r['observations'][0].update(candidate_id='unknown'),
    lambda r: r['observations'][0].update(present=1),
    lambda r: r['observations'][0].update(resonance=-1),
    lambda r: r.update(observations=[], count=0),
])
def test_odvl_scope_availability_and_complete_measurement_are_required(session, mutation):
    memory, client = session
    def edit(operation, response):
        mutation(response)
        return response
    client.edit = edit
    with pytest.raises(cloud.CloudMemoryError):
        memory.observe(['ep-1'])


def test_stimulation_captures_persistent_mutation_receipt_and_exact_retry_key(session):
    memory, client = session
    response = memory.stimulate([{'candidate_id': 'ep-1', 'score': .8}],
                                elapsed_units=.1, dt_units=.02, idempotency_key='outcome.1')
    assert response['applied'] and response['steps_applied'] == 5
    request = client.calls[-1][1]
    assert request['idempotency_key'] == 'outcome.1'
    assert request['qualification_floor'] == 0
    assert request['entity_id'] == ENTITY


def test_secrets_never_appear_in_receipt_or_exception(session):
    memory, client = session
    client.fail = RuntimeError('network failed ' + SECRET)
    with pytest.raises(cloud.CloudMemoryError) as caught:
        memory.observe(['ep-1'])
    assert SECRET not in str(caught.value)
    client.fail = None
    client.edit = lambda operation, response: {**response, 'authorization': SECRET, 'echo': SECRET}
    with pytest.raises(cloud.CloudMemoryError):
        memory.observe(['ep-1'])
    assert SECRET not in memory.receipt_path.read_text()
    assert SECRET not in repr(memory)


def test_invalid_inputs_make_no_network_calls(session):
    memory, client = session
    with pytest.raises(cloud.CloudMemoryError):
        remember(memory, revision=2)
    with pytest.raises(cloud.CloudMemoryError):
        memory.remember_model({'scientific_status': 'real'}, claim_key='a', candidate_id='a', provenance_key='a', idempotency_key='a')
    with pytest.raises(cloud.CloudMemoryError):
        memory.remember_model({**MODEL, 'large': 'x' * 1200}, claim_key='a', candidate_id='a', provenance_key='a', idempotency_key='a')
    with pytest.raises(cloud.CloudMemoryError):
        memory.stimulate([{'candidate_id': 'a', 'score': .3}], idempotency_key='')
    assert client.calls == []


def test_credentials_are_private_and_loading_performs_no_calls(tmp_path, monkeypatch):
    path = tmp_path / 'env'
    path.write_text(f'BASIRA_BASE=https://example.test/api/v1/basira\nBASIRA_SPACE={SPACE}\nBASIRA_ENTITY={ENTITY}\nBASIRA_KEY={SECRET}\n')
    path.chmod(0o644)
    with pytest.raises(cloud.CloudMemoryError, match='0600'):
        cloud._load_credentials(path)
    path.chmod(0o600)
    assert cloud._load_credentials(path)['BASIRA_ENTITY'] == ENTITY
    link = tmp_path / 'linked'
    link.symlink_to(path)
    with pytest.raises(cloud.CloudMemoryError):
        cloud._load_credentials(link)
    with pytest.raises(cloud.CloudMemoryError, match='dedicated_entity_mismatch'):
        cloud.CloudMemory.from_credentials(path, tmp_path / 'log', expected_entity_id='ent_' + 'b' * 32)
    assert not (tmp_path / 'log').exists()


def test_exact_public_sdk_http_attempt_is_instrumented_without_network(tmp_path, monkeypatch):
    """Exercise the actual SDK transport contract against an in-process response."""
    sdk, digest = cloud._load_sdk(cloud.DEFAULT_SDK_PATH)
    import httpx
    real_http_client = httpx.Client
    fake = FakePublicSDK()
    attempts = []

    def transport(request):
        attempts.append(request.method)
        body = json.loads(request.content)
        return httpx.Response(200, json=fake.odvl_observe(**body))

    monkeypatch.setattr(sdk.httpx, 'Client',
                        lambda **kwargs: real_http_client(transport=httpx.MockTransport(transport), **kwargs))
    credentials = tmp_path / 'env'
    credentials.write_text(f'BASIRA_BASE=https://example.test/api/v1/basira\nBASIRA_SPACE={SPACE}\nBASIRA_ENTITY={ENTITY}\nBASIRA_KEY={SECRET}\n')
    credentials.chmod(0o600)
    memory = cloud.CloudMemory.from_credentials(credentials, tmp_path / 'receipts', expected_entity_id=ENTITY)
    assert attempts == []
    result = memory.observe(['ep-1'])
    fresh = memory.reconnect()
    assert attempts == ['POST']
    assert fresh.client_instance_id != memory.client_instance_id
    assert fresh._owned_http_client is not memory._owned_http_client
    assert fresh.space_id == memory.space_id and fresh.entity_id == memory.entity_id
    fresh.observe(['ep-1'])
    fresh.close()
    memory.close()
    assert result['schema'] == 'basira-odvl-observation/v1'
    assert attempts == ['POST', 'POST']
    receipt = json.loads(memory.receipt_path.read_text().splitlines()[-1])
    assert receipt['http_attempts_instrumented'] and receipt['http_attempts'] == 1
    assert receipt['sdk_source_sha256'] == digest
    assert len(digest) == 64 and SECRET not in memory.receipt_path.read_text()
