"""Public Cloud memory for synthetic magnetic-control episodes.

Only the explicitly selected public SDK is loaded. No local Basira, TSST or
ODVL engine is imported. The caller must provision a dedicated laboratory
entity; construction and credential loading make no network calls.

Assertions describe observations of a simulation, never physical measurements.
ODVL values describe its recognition state, never the truth of those assertions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid


DEFAULT_SDK_PATH = Path(__file__).resolve().parent / 'vendor' / 'basira_client.py'
MODEL_SCHEMA = 'magnetic-frequency-model/v1'
SCIENTIFIC_STATUS = 'simulated_sensor_data'
_ID = re.compile(r'^[A-Za-z0-9._~:+/=-]{1,160}$')
_ENTITY = re.compile(r'^ent_[0-9a-f]{32}$')
_EVIDENCE = re.compile(r'^ev_[0-9a-f]{32}$')
_SENSITIVE = re.compile(r'(secret|password|authorization|api.?key|access.?token|refresh.?token|basira_key)', re.I)


class CloudMemoryError(RuntimeError):
    """A bounded, credential-free error; there is no local fallback."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError):
        raise CloudMemoryError('non_json_or_nonfinite_value') from None


def _number(value: Any, code: str, minimum=0.0, maximum=None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise CloudMemoryError(code)
    if maximum is not None and value > maximum:
        raise CloudMemoryError(code)
    return float(value)


def _ids(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise CloudMemoryError('invalid_candidate_ids')
    result = list(values)
    if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in result):
        raise CloudMemoryError('invalid_candidate_ids')
    if not 1 <= len(result) <= 128 or len(set(result)) != len(result):
        raise CloudMemoryError('invalid_candidate_ids')
    return result


def _time(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or not 0 <= value <= 253402300799000):
        raise CloudMemoryError('invalid_time_ms')
    return value


def _load_credentials(path: str | Path) -> dict[str, str]:
    """Read a same-user regular 0600 file without shell expansion or symlinks."""
    try:
        descriptor = os.open(Path(path).expanduser(), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'r', encoding='utf-8') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid() or info.st_size > 65536):
                raise CloudMemoryError('credentials_require_owned_regular_0600_file')
            content = stream.read(65537)
        result = {}
        for line in content.splitlines():
            words = shlex.split(line, comments=True, posix=True)
            if words and words[0] == 'export':
                words = words[1:]
            if not words:
                continue
            if len(words) != 1 or '=' not in words[0]:
                raise CloudMemoryError('invalid_credentials_format')
            name, value = words[0].split('=', 1)
            if name in {'BASIRA_BASE', 'BASIRA_SPACE', 'BASIRA_KEY', 'BASIRA_ENTITY'}:
                if name in result or not value:
                    raise CloudMemoryError('invalid_credentials_fields')
                result[name] = value
        if set(result) != {'BASIRA_BASE', 'BASIRA_SPACE', 'BASIRA_KEY', 'BASIRA_ENTITY'}:
            raise CloudMemoryError('missing_dedicated_cloud_credentials')
        base = urlsplit(result['BASIRA_BASE'])
        if (base.scheme != 'https' or not base.hostname or base.username or base.password
                or base.query or base.fragment):
            raise CloudMemoryError('invalid_cloud_base_url')
        return result
    except CloudMemoryError:
        raise
    except Exception:
        raise CloudMemoryError('credentials_unreadable') from None


def _load_sdk(path: str | Path):
    source_path = Path(path).expanduser().resolve(strict=True)
    source = source_path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    name = '_magnetic_public_sdk_' + digest[:24]
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, source_path)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        try:
            exec(compile(source, str(source_path), 'exec'), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name], digest


class CloudMemory:
    """Typed episodic memory, explicit historical probes and frozen ODVL reads.

    Mutation keys must be stable across retries and unique per intended action.
    Read keys are freshly generated by the public SDK. ``receipt_path`` stores
    durable request intents and actual public responses, with credential redaction.
    """

    def __init__(self, client, space_id: str, entity_id: str,
                 receipt_path: str | Path, *, sdk_source_sha256='injected-test-client',
                 max_retries: int = 0, _attempt_counter=None, _secrets: Sequence[str] = ()):
        if not isinstance(space_id, str) or not re.fullmatch(r'mem_[A-Za-z0-9_-]+', space_id):
            raise CloudMemoryError('invalid_memory_space_id')
        if not isinstance(entity_id, str) or not _ENTITY.fullmatch(entity_id):
            raise CloudMemoryError('invalid_dedicated_entity_id')
        self._client, self.space_id, self.entity_id = client, space_id, entity_id
        self.receipt_path = Path(receipt_path)
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.sdk_source_sha256 = sdk_source_sha256
        self._secrets = tuple(value for value in _secrets if value)
        self._lock = threading.RLock()
        self._max_retries, self._attempt_counter = max_retries, _attempt_counter
        self._logical_calls = 0
        self._owned_http_client = None
        self.client_instance_id = uuid.uuid4().hex
        self._reconnect_factory = None
        self._cloud_base = None

    @classmethod
    def from_credentials(cls, path: str | Path, receipt_path: str | Path, *,
                         expected_entity_id: str | None = None,
                         sdk_path: str | Path = DEFAULT_SDK_PATH,
                         max_retries: int = 0):
        if type(max_retries) is not int or not 0 <= max_retries <= 10:
            raise CloudMemoryError('invalid_sdk_retry_budget')
        values = _load_credentials(path)
        if expected_entity_id is not None and values['BASIRA_ENTITY'] != expected_entity_id:
            raise CloudMemoryError('dedicated_entity_mismatch')
        sdk, digest = _load_sdk(sdk_path)
        attempts = [0]
        def count_attempt(_request):
            attempts[0] += 1  # Count requests without inspecting URLs, headers or bodies.
        http = sdk.httpx.Client(event_hooks={'request': [count_attempt]})
        try:
            client = sdk.BasiraClient(values['BASIRA_SPACE'], values['BASIRA_KEY'],
                                     base_url=values['BASIRA_BASE'], max_retries=max_retries,
                                     http_client=http)
            instance = cls(client, values['BASIRA_SPACE'], values['BASIRA_ENTITY'], receipt_path,
                           sdk_source_sha256=digest, max_retries=max_retries,
                           _attempt_counter=lambda: attempts[0], _secrets=(values['BASIRA_KEY'],))
            instance._owned_http_client = http
            instance._cloud_base = values['BASIRA_BASE'].rstrip('/')
            credential_path = Path(path).expanduser().resolve(strict=True)
            source_path = Path(sdk_path).expanduser().resolve(strict=True)
            expected_entity = instance.entity_id
            instance._reconnect_factory = lambda: cls.from_credentials(
                credential_path, receipt_path, expected_entity_id=expected_entity,
                sdk_path=source_path, max_retries=max_retries)
            return instance
        except BaseException:
            http.close()
            raise

    def __repr__(self):
        return f'CloudMemory(entity_id={self.entity_id!r}, credentials=<redacted>)'

    def close(self):
        self._client.close()
        if self._owned_http_client is not None:
            self._owned_http_client.close()

    def reconnect(self):
        """Return a fresh HTTP client for the same entity; make no service call.

        Credentials are reread privately. The space, entity and exact public SDK
        bytes must remain unchanged. The original client remains independently
        closable. A following recall, not construction, tests server persistence.
        """
        if self._reconnect_factory is None:
            raise CloudMemoryError('fresh_client_factory_unavailable')
        fresh = self._reconnect_factory()
        if (fresh.space_id != self.space_id or fresh.entity_id != self.entity_id
                or fresh.sdk_source_sha256 != self.sdk_source_sha256
                or fresh._cloud_base != self._cloud_base):
            fresh.close()
            raise CloudMemoryError('reconnected_scope_or_sdk_changed')
        return fresh

    def _redact(self, value):
        if isinstance(value, Mapping):
            return {str(key): ('<redacted>' if _SENSITIVE.search(str(key)) else self._redact(item))
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, '<redacted>')
        if isinstance(value, float) and not math.isfinite(value):
            return '<nonfinite>'
        return value

    def _record(self, row):
        document = {'schema': 'magnetic-cloud-receipt/v1',
                    'recorded_at': datetime.now(timezone.utc).isoformat(),
                    'scientific_status': SCIENTIFIC_STATUS,
                    'physical_measurements': False, 'space_id': self.space_id,
                    'entity_id': self.entity_id, 'sdk_source_sha256': self.sdk_source_sha256,
                    'client_instance_id': self.client_instance_id,
                    **row}
        encoded = (_canonical(self._redact(document)) + '\n').encode('utf-8')
        with self._lock:
            descriptor = os.open(self.receipt_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, 'ab') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

    def _call(self, operation, request, validate):
        # Serialize one adapter's calls so per-call HTTP attempt counts are exact.
        with self._lock:
            return self._perform_call(operation, request, validate)

    def _perform_call(self, operation, request, validate):
        request_id = uuid.uuid4().hex
        if self._redact(request) != request:
            raise CloudMemoryError('secret_like_request_data_rejected')
        self._logical_calls += 1
        attempt_start = self._attempt_counter() if self._attempt_counter else None
        call_metadata = {'logical_sdk_call_index': self._logical_calls,
                         'sdk_max_retries': self._max_retries,
                         'http_attempt_ceiling': self._max_retries + 1}
        self._record({'request_id': request_id, 'operation': operation, **call_metadata,
                      'phase': 'request', 'request': request})
        started = time.monotonic()
        def timing():
            return {**call_metadata, 'elapsed_ms': round((time.monotonic() - started) * 1000, 3),
                    'http_attempts': (self._attempt_counter() - attempt_start
                                      if self._attempt_counter else None),
                    'http_attempts_instrumented': self._attempt_counter is not None}
        response = None
        try:
            response = getattr(self._client, operation)(**request)
            if not isinstance(response, dict):
                raise CloudMemoryError('invalid_response_object')
            _canonical(response)
            if self._redact(response) != response:
                raise CloudMemoryError('secret_like_response_data_rejected')
            validate(response)
        except Exception as error:
            code = str(error) if isinstance(error, CloudMemoryError) else getattr(error, 'code', 'cloud_request_failed')
            code = code if isinstance(code, str) and re.fullmatch(r'[a-z0-9_]{1,100}', code) else 'cloud_request_failed'
            self._record({'request_id': request_id, 'operation': operation, 'phase': 'response', **timing(),
                          'accepted': False, 'error_code': code,
                          'response': response if isinstance(response, dict) else None})
            raise CloudMemoryError(code) from None
        self._record({'request_id': request_id, 'operation': operation, 'phase': 'response', **timing(),
                      'accepted': True, 'response': response})
        return copy.deepcopy(response)

    def _scope(self, response, schema):
        if response.get('schema') != schema:
            raise CloudMemoryError('cloud_response_schema_mismatch')
        if response.get('memory_space_id') != self.space_id or response.get('entity_id') != self.entity_id:
            raise CloudMemoryError('cloud_response_scope_mismatch')

    @staticmethod
    def _mutation_key(value):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9._~:+/=-]{8,160}', value):
            raise CloudMemoryError('explicit_idempotency_key_required')
        return value

    def remember_model(self, model: Mapping[str, Any], *, claim_key: str,
                       candidate_id: str, provenance_key: str, revision: int = 1,
                       reason: str = '', valid_time_ms: int | None = None,
                       decision_time_ms: int | None = None, belief_time_ms: int | None = None,
                       confidence: float = 1.0, importance: float = 0.0,
                       anchor: bool = False, cluster_id: str = 'magnetic-simulation',
                       idempotency_key: str):
        """Store a completed synthetic episode, or an explicit corrected claim.

        Successful first/revised writes also ingest ODVL at server-defined
        salience. This effect is not an independent outcome-success reward.
        """
        _ids([candidate_id])
        _ids([cluster_id])
        if (not isinstance(claim_key, str) or not claim_key.strip() or len(claim_key) > 128
                or not isinstance(provenance_key, str) or not provenance_key.strip() or len(provenance_key) > 160):
            raise CloudMemoryError('invalid_assertion_identity')
        if type(revision) is not int or revision < 1 or (revision > 1 and not reason.strip()):
            raise CloudMemoryError('explicit_revision_reason_required')
        if not isinstance(model, Mapping) or model.get('scientific_status') != SCIENTIFIC_STATUS:
            raise CloudMemoryError('simulated_sensor_status_required')
        content = _canonical({'schema': MODEL_SCHEMA, 'model': dict(model)})
        if len(content) > 1200:
            raise CloudMemoryError('model_exceeds_public_assertion_1200_chars')
        payload = {'claim_key': claim_key, 'candidate_id': candidate_id,
                   'content': content, 'epistemic_status': 'observed',
                   'confidence': _number(confidence, 'invalid_confidence', maximum=1),
                   'provenance_key': provenance_key,
                   'importance': _number(importance, 'invalid_importance', maximum=1),
                   'anchor': anchor, 'cluster_id': cluster_id,
                   'declared_revision': revision > 1}
        if type(anchor) is not bool or len(reason) > 1000:
            raise CloudMemoryError('invalid_assertion_metadata')
        for name, value in [('valid_time_ms', valid_time_ms), ('decision_time_ms', decision_time_ms), ('belief_time_ms', belief_time_ms)]:
            if value is not None:
                payload[name] = _time(value)
        if revision > 1:
            payload.update(reason=reason,
                           evidence='Revised simulated observation; content SHA256=' + hashlib.sha256(content.encode()).hexdigest())

        def validate(response):
            self._scope(response, 'basira-assertion/v1')
            if (response.get('status') != 'completed' or response.get('claim_key') != claim_key
                    or response.get('candidate_id') != candidate_id or type(response.get('revision')) is not int
                    or response['revision'] != revision or response.get('epistemic_status') != 'observed'):
                raise CloudMemoryError('assertion_receipt_identity_mismatch')
            if response.get('odvl_applied') is not True:
                raise CloudMemoryError('assertion_odvl_not_applied')
            if response.get('known_at_ms') is None:
                raise CloudMemoryError('assertion_server_time_missing')
            _time(response['known_at_ms'])
            _number(response.get('confidence'), 'invalid_assertion_confidence', maximum=1)
            for field in ('valid_time_ms', 'decision_time_ms', 'belief_time_ms'):
                _time(response.get(field))
        return self._call('write_assertion', {'assertion': payload, 'entity_id': self.entity_id,
                         'idempotency_key': self._mutation_key(idempotency_key)}, validate)

    def recall_models(self, query: str, *, limit: int = 10,
                      as_of_valid_time_ms: int | None = None,
                      as_of_transaction_time_ms: int | None = None):
        if not isinstance(query, str) or not query.strip() or len(query) > 8192 or type(limit) is not int or not 1 <= limit <= 10:
            raise CloudMemoryError('invalid_recall_query')
        temporal = {}
        for name, value in [('as_of_valid_time_ms', as_of_valid_time_ms), ('as_of_transaction_time_ms', as_of_transaction_time_ms)]:
            if value is not None:
                temporal[name] = _time(value)
        models = []

        def validate(response):
            self._scope(response, 'basira-tsst-evidence/v1')
            evidence = response.get('evidence')
            if (not isinstance(evidence, list) or type(response.get('count')) is not int
                    or response['count'] != len(evidence) or len(evidence) > limit
                    or response.get('flashback') is not bool(temporal)):
                raise CloudMemoryError('invalid_tsst_count_or_temporal_contract')
            seen = set()
            identities = set()
            for row in evidence:
                if not isinstance(row, dict) or row.get('integrity_verified') is not True:
                    raise CloudMemoryError('unverified_tsst_evidence')
                identifier = row.get('evidence_id')
                candidate = row.get('candidate_id')
                claim = row.get('claim_key')
                revision = row.get('revision')
                if (not isinstance(identifier, str) or not _EVIDENCE.fullmatch(identifier)
                        or identifier in seen or not isinstance(claim, str) or not claim.strip()
                        or type(revision) is not int or revision < 1):
                    raise CloudMemoryError('invalid_tsst_identity')
                _ids([candidate])
                identity = (claim, revision)
                if identity in identities:
                    raise CloudMemoryError('duplicate_tsst_claim_revision')
                seen.add(identifier)
                identities.add(identity)
                if row.get('status') != 'observed':
                    raise CloudMemoryError('unexpected_tsst_epistemic_status')
                score = _number(row.get('score'), 'invalid_tsst_score', maximum=1)
                try:
                    excerpt = row['excerpt']
                    # The verified TSST hydration lane renders the sealed
                    # subject + predicate + object. Accept exactly this public
                    # declaration prefix, bound to the returned claim identity,
                    # or a bare JSON object from a content-only representation.
                    # Never search an arbitrary excerpt for an opening brace.
                    prefix = claim + ' declaration '
                    if isinstance(excerpt, str) and excerpt.startswith(prefix):
                        excerpt = excerpt[len(prefix):]
                    decoded = json.loads(excerpt)
                except (ValueError, KeyError, TypeError):
                    raise CloudMemoryError('unparseable_tsst_model') from None
                if (not isinstance(decoded, dict) or decoded.get('schema') != MODEL_SCHEMA
                        or not isinstance(decoded.get('model'), dict)
                        or decoded['model'].get('scientific_status') != SCIENTIFIC_STATUS):
                    raise CloudMemoryError('unexpected_tsst_model_schema')
                _canonical(decoded)
                models.append({'candidate_id': candidate, 'claim_key': claim, 'revision': revision,
                               'model': decoded['model'], 'tsst_score': score,
                               'evidence_id': identifier, 'rank': len(models) + 1})
        response = self._call('recall_tsst', {'query': query, 'entity_id': self.entity_id,
                              'limit': limit, 'semantic': {'predicate': 'declaration'},
                              'temporal': temporal,
                              'epistemic': {'allowed_epistemic': ['observed'],
                                            'include_belief_history': bool(temporal)}}, validate)
        return {**response, 'models': models, 'raw_response': response}

    def flashback_models(self, query: str, *, as_of_transaction_time_ms: int,
                         as_of_valid_time_ms: int | None = None, limit: int = 10):
        _time(as_of_transaction_time_ms)
        if as_of_transaction_time_ms is None:
            raise CloudMemoryError('flashback_requires_server_time_cutoff')
        return self.recall_models(query, limit=limit, as_of_valid_time_ms=as_of_valid_time_ms,
                                  as_of_transaction_time_ms=as_of_transaction_time_ms)

    def _observations(self, response, expected, *, schema, clock):
        self._scope(response, schema)
        if response.get('available') is not True or response.get('mode') not in {'on', 'shadow'}:
            raise CloudMemoryError('odvl_measurement_unavailable')
        if response.get('clock') != clock:
            raise CloudMemoryError('odvl_clock_mismatch')
        rows = response.get('observations')
        if (not isinstance(rows, list) or type(response.get('count')) is not int
                or response['count'] != len(rows) or type(response.get('requested_count')) is not int
                or response['requested_count'] != len(expected)):
            raise CloudMemoryError('invalid_odvl_observation_count')
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or row.get('candidate_id') not in expected or row['candidate_id'] in seen:
                raise CloudMemoryError('invalid_odvl_candidate_identity')
            seen.add(row['candidate_id'])
            if type(row.get('present')) is not bool:
                raise CloudMemoryError('invalid_odvl_presence')
            for field in ('resonance', 'normalized', 'normalized_global', 'depth', 'proof'):
                _number(row.get(field), 'invalid_odvl_measurement')
        if seen != set(expected):
            raise CloudMemoryError('missing_odvl_candidate_measurement')

    def observe(self, candidate_ids: Sequence[str], *, cluster_ids: Sequence[str] = ()):
        identifiers = _ids(candidate_ids)
        clusters = _ids(cluster_ids) if cluster_ids else []
        return self._call('odvl_observe', {'candidate_ids': identifiers, 'entity_id': self.entity_id,
                          'cluster_ids': clusters, 'limit': len(identifiers)},
                          lambda response: self._observations(response, identifiers,
                            schema='basira-odvl-observation/v1', clock='frozen'))

    def stimulate(self, stimuli: Sequence[Mapping[str, Any]] = (), *,
                  elapsed_units: float = 0.0, dt_units: float = 1.0,
                  settle_steps: int = 0, qualification_floor: float = 0.0,
                  idempotency_key: str):
        """Apply caller-declared stimulation after measurement, then game time.

        Even score=0 supplies a nonzero server stimulus. Omit failed candidates
        when the frozen protocol intends no positive reinforcement.
        """
        rows = [dict(row) for row in stimuli]
        identifiers = _ids([row.get('candidate_id') for row in rows]) if rows else []
        touches = 0
        for row in rows:
            if set(row) - {'candidate_id', 'score', 'importance', 'anchor', 'cluster_id', 'touch_count'}:
                raise CloudMemoryError('unexpected_stimulus_field')
            _number(row.get('score'), 'invalid_stimulus_score', maximum=1)
            _number(row.get('importance', 0), 'invalid_stimulus_importance', maximum=1)
            if type(row.get('anchor', False)) is not bool:
                raise CloudMemoryError('invalid_stimulus_anchor')
            count = row.get('touch_count', 1)
            if type(count) is not int or not 1 <= count <= 128:
                raise CloudMemoryError('invalid_stimulus_touch_count')
            touches += count
            if row.get('cluster_id') is not None:
                _ids([row['cluster_id']])
        if touches > 128 or type(settle_steps) is not int or not 0 <= settle_steps <= 64:
            raise CloudMemoryError('invalid_stimulation_budget')
        units = _number(elapsed_units, 'invalid_elapsed_units', maximum=512)
        dt = _number(dt_units, 'invalid_dt_units', minimum=1e-4, maximum=1)
        floor = _number(qualification_floor, 'invalid_qualification_floor', maximum=1)
        if floor >= 1 or math.ceil(units / dt - 1e-10) > 512:
            raise CloudMemoryError('invalid_stimulation_budget')

        def validate(response):
            self._observations(response, identifiers, schema='basira-odvl-stimulation/v1', clock='game')
            if (response.get('applied') is not True or response.get('settle_steps') != settle_steps
                    or type(response.get('steps_applied')) is not int
                    or response['steps_applied'] != math.ceil(units / dt - 1e-10)
                    or any(row['present'] is not True for row in response['observations'])
                    or not math.isclose(_number(response.get('elapsed_units'), 'invalid_applied_elapsed'), units,
                                        rel_tol=1e-6, abs_tol=1e-6)):
                raise CloudMemoryError('odvl_stimulation_not_applied')
        return self._call('odvl_stimulate', {'stimuli': rows, 'entity_id': self.entity_id,
                          'qualification_floor': floor, 'elapsed_units': units, 'dt_units': dt,
                          'settle_steps': settle_steps,
                          'idempotency_key': self._mutation_key(idempotency_key)}, validate)
