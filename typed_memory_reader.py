"""Causal public TSST selectors and passive HTTP/cache timing instrumentation.

This module neither writes memory nor supplies local models as retrieval results.
Wrap CausalTypedReader *inside* the caller's PacedClient so pacing surrounds the
complete public read. HTTP probes record counters/times only, never request data.
"""
from __future__ import annotations

import copy
import json
import math
import statistics
import threading
import time
import weakref
from collections.abc import Mapping
from types import MappingProxyType

from cloud_memory import CloudMemory
from candidate_validation import assert_current_candidates
from candidate_validation import _RecallView


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _text(value, maximum=512):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_typed_query_field')
        result[key] = value
    return result


def _accepted_record(record, current_index=None):
    receipt = record.get('receipt', {})
    index = record.get('model', {}).get('completed_episode_index')
    if (type(index) is not int or index < 0 or
            (current_index is not None and index >= current_index) or
            receipt.get('status') != 'completed' or receipt.get('epistemic_status') != 'observed' or
            any(receipt.get(key) != record.get(key) for key in ('candidate_id', 'claim_key', 'revision', 'known_at_ms')) or
            not _text(record.get('candidate_id')) or not _text(record.get('claim_key')) or
            type(record.get('revision')) is not int or record['revision'] < 1):
        raise ValueError('planning_requires_accepted_prior_assertion_metadata')
    return index


def planning(context, bank, *, current_index=None):
    """Choose identifiers from completed accepted metadata, never local predictions.

    A residual_model is present only for an accepted ARX fit in this campaign's
    assertion schema. Optional provenance_key must have been recovered from the
    corresponding accepted assertion request by the caller; it is never guessed.
    """
    if (not isinstance(context, dict) or set(context) != {'device_id'} or
            context['device_id'] not in ('coilA', 'coilB')):
        raise ValueError('explicit_observable_device_required')
    if current_index is not None and (type(current_index) is not int or current_index < 0):
        raise ValueError('invalid_current_episode_index')
    latest = {}
    for record in bank:
        _accepted_record(record, current_index)
        key = record['candidate_id']
        if key not in latest or record['revision'] > latest[key]['revision']:
            latest[key] = record
        elif record['revision'] == latest[key]['revision'] and record != latest[key]:
            raise ValueError('conflicting_accepted_planning_record')
    matching = [row for row in latest.values() if row['model'].get('context_id') == context['device_id']]
    matching.sort(key=lambda row: (row['model']['completed_episode_index'], row['candidate_id']), reverse=True)
    recent = matching[0] if matching else None
    arx = next((row for row in matching if isinstance(row['model'].get('residual_model'), dict)
                and row['model']['residual_model'].get('v') == 1
                and row['model']['residual_model'].get('ep') == row['model']['completed_episode_index']), None)
    base = 'magnetic compensation completed sensor observation device ' + context['device_id']

    def encode(record, suffix):
        plan = {'query': base + suffix, 'semantic': {'predicate': 'declaration'}, 'expected_candidate': None}
        if record is not None:
            plan['query'] = base + ' ' + record['candidate_id']
            plan['expected_candidate'] = record['candidate_id']
            plan['semantic'].update(entity=record['claim_key'], entity_type='entity')
            if 'provenance_key' in record:
                if not _text(record['provenance_key']):
                    raise ValueError('invalid_accepted_provenance_key')
                plan['semantic']['source_id'] = record['provenance_key']
        return json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False)

    second = arx if arx is not None and (recent is None or arx['candidate_id'] != recent['candidate_id']) else None
    return (encode(recent, ''), encode(second, ' observed residual response command'))


class HTTPFlightProbe:
    """Passive HTTPX hooks: attempt dispatched, response headers not yet received.

    This interval includes connection setup and transport; it is not a server
    execution timer. Receiving headers ends it even if parsing/logging continues.
    """
    def __init__(self, adapter):
        self.client_instance_id = adapter.client_instance_id
        self._counter = getattr(adapter, '_attempt_counter', None)
        client = getattr(adapter, '_owned_http_client', None)
        self._lock = threading.Lock()
        self._active = {}
        self._started = self._headers = 0
        self.available = callable(self._counter) and client is not None and isinstance(getattr(client, 'event_hooks', None), dict)
        if self.available:
            client.event_hooks.setdefault('request', []).append(self._request)
            client.event_hooks.setdefault('response', []).append(self._response)

    def _request(self, request):
        now = time.monotonic()
        with self._lock:
            self._started += 1
            self._active[id(request)] = {'token': self._started, 'started_monotonic': now,
                                        'instrumented_attempt': self._counter(), 'thread': threading.get_ident()}

    def _response(self, response):
        with self._lock:
            self._active.pop(id(response.request), None)
            self._headers += 1

    def finish_read(self):
        # An exception can occur without a response hook. The public call is now
        # finished, so never leave that failed attempt looking perpetually active.
        with self._lock:
            self._active = {key: row for key, row in self._active.items() if row['thread'] != threading.get_ident()}

    def capture(self):
        with self._lock:
            return {'available': self.available, 'client_instance_id': self.client_instance_id,
                    'attempts': self._counter() if self.available else None,
                    'started': self._started, 'response_headers': self._headers,
                    'active': [{key: value for key, value in row.items() if key != 'thread'}
                               for row in self._active.values()]}


_PROBES = weakref.WeakKeyDictionary()
_PROBE_LOCK = threading.Lock()


def _probe(adapter):
    with _PROBE_LOCK:
        if adapter not in _PROBES:
            _PROBES[adapter] = HTTPFlightProbe(adapter)
        return _PROBES[adapter]


class CausalTypedReader:
    """Read/observe adapter bound to a deep-frozen accepted-history generation."""
    _FORWARDED = frozenset(('space_id', 'entity_id', 'sdk_source_sha256', 'client_instance_id',
                           '_cloud_base', '_client', '_attempt_counter', '_max_retries',
                           'receipt_path', '_logical_calls'))

    def __init__(self, adapter, versions, current_index):
        if type(current_index) is not int or current_index < 0:
            raise ValueError('invalid_current_episode_index')
        confirmed = copy.deepcopy(versions)
        for (key, revision), record in confirmed.items():
            _accepted_record(record, current_index)
            if (key, revision) != (record['candidate_id'], record['revision']):
                raise ValueError('confirmed_history_identity_mismatch')
            if (record['receipt'].get('memory_space_id'), record['receipt'].get('entity_id')) != (adapter.space_id, adapter.entity_id):
                raise ValueError('confirmed_history_scope_mismatch')
        self._adapter = adapter
        self._confirmed = MappingProxyType({key: _freeze(record) for key, record in confirmed.items()})
        self._current_index = current_index
        self.http_probe = _probe(adapter)

    @property
    def current_index(self):
        return self._current_index

    def __getattr__(self, name):
        if name not in self._FORWARDED:
            raise AttributeError(name)
        return getattr(self._adapter, name)

    def _versions(self):
        return {key: _thaw(record) for key, record in self._confirmed.items()}

    def recall_models(self, query, *, limit=10):
        if not _text(query, 16384):
            raise ValueError('encoded_typed_query_required')
        try:
            plan = json.loads(query, object_pairs_hook=_unique_object)
        except (ValueError, TypeError):
            raise ValueError('invalid_typed_query_json') from None
        if (not isinstance(plan, dict) or set(plan) - {'query', 'semantic', 'expected_candidate'} or
                not _text(plan.get('query'), 8192)):
            raise ValueError('invalid_typed_query_plan')
        axes = plan.get('semantic')
        if (not isinstance(axes, dict) or set(axes) - {'predicate', 'entity', 'entity_type', 'source_id'} or
                axes.get('predicate') != 'declaration' or any(not _text(value) for value in axes.values()) or
                ('entity' in axes and axes.get('entity_type') != 'entity') or
                ('entity_type' in axes and ('entity' not in axes or axes['entity_type'] != 'entity'))):
            raise ValueError('unsupported_typed_semantic_axes')
        expected = plan.get('expected_candidate')
        typed = 'entity' in axes or 'source_id' in axes
        versions = self._versions()
        if typed:
            records = [row for (key, _), row in versions.items() if key == expected]
            if not _text(expected) or not records:
                raise ValueError('typed_target_requires_confirmed_prior_receipt')
            record = max(records, key=lambda row: row['revision'])
            if (('entity' in axes and axes['entity'] != record['claim_key']) or
                    ('source_id' in axes and axes['source_id'] != record.get('provenance_key'))):
                raise ValueError('selector_does_not_match_accepted_target_metadata')
        elif expected is not None:
            raise ValueError('broad_query_cannot_claim_exact_target')
        try:
            response = CloudMemory.recall_models(_RecallView(self._adapter, axes), plan['query'], limit=limit)
            assert_current_candidates(response, versions, self.current_index)
            if typed and any(row['candidate_id'] != expected for row in response['models']):
                raise ValueError('typed_selector_returned_other_candidate')
            return response  # Empty is truthful absence; never inject local data.
        finally:
            self.http_probe.finish_read()

    def observe(self, candidate_ids, cluster_ids=()):
        ids = tuple(candidate_ids)
        if any(key not in {candidate for candidate, _ in self._confirmed} for key in ids):
            raise ValueError('observe_without_confirmed_prior_candidate')
        try:
            return self._adapter.observe(ids, cluster_ids=cluster_ids)
        finally:
            self.http_probe.finish_read()


def capture_http_baseline(readers):
    """Call before start_refresh; no I/O and no inspection of request content."""
    return {reader.client_instance_id: reader.http_probe.capture() for reader in readers}


def _active_since(readers, baseline):
    states = capture_http_baseline(readers)
    active = set()
    for key, state in states.items():
        previous = baseline.get(key, {})
        if not state['available'] or not previous.get('available'):
            continue
        for row in state['active']:
            if (row['token'] > previous['started'] and state['attempts'] > previous['attempts'] and
                    row['instrumented_attempt'] > previous['attempts']):
                active.add((key, row['token']))
    return states, active


def _percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)


def measure_hot_reads_during_http(cache, snapshot, readers, *, context_key, history_version,
                                  baseline, samples=300, timeout_s=8.):
    """Offline preparation wait, then time only local get_snapshot calls.

    The same instrumented request must remain without response headers across
    the entire batch to claim overlap. Otherwise the explicit pending_only label
    applies, even if the refresh Future remains unfinished. No retries/calls.
    """
    if type(samples) is not int or samples < 1 or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 60:
        raise ValueError('invalid_bounded_hot_read_measurement')
    wait_started = time.monotonic()
    before, active_before = _active_since(readers, baseline)
    while not active_before and cache.refreshing and time.monotonic() - wait_started < timeout_s:
        time.sleep(.001)
        before, active_before = _active_since(readers, baseline)
    before_pending = cache.refreshing
    batch_started = time.monotonic()
    elapsed = []; versions = []; hits = 0
    for _ in range(samples):
        started = time.perf_counter_ns()
        current = cache.get_snapshot(context_key=context_key, min_history_version=history_version)
        elapsed.append((time.perf_counter_ns() - started) / 1e9)
        if current is not None:
            hits += 1
            versions.append(current.version)
    batch_ended = time.monotonic()
    after, active_after = _active_since(readers, baseline)
    continuous = sorted(active_before & active_after)
    proven = bool(continuous)
    return {'samples': samples, 'hits': hits, 'misses': samples - hits,
            'captured_snapshot_version': snapshot.version, 'returned_snapshot_versions': sorted(set(versions)),
            'control_uses_captured_snapshot': True,
            'refresh_in_flight_before_hot_reads': before_pending,
            'refresh_in_flight_after_hot_reads': cache.refreshing,
            'p50_s': statistics.median(elapsed), 'p95_s': _percentile(elapsed, .95), 'max_s': max(elapsed),
            'offline_wait_s': batch_started - wait_started,
            'batch_started_monotonic': batch_started, 'batch_ended_monotonic': batch_ended,
            'http_overlap_proven': proven,
            'overlap_kind': 'same_HTTP_attempt_without_response_headers_throughout_batch' if proven else 'pending_only',
            'continuous_http_tokens': [list(row) for row in continuous],
            'http_before': before, 'http_after': after,
            'timing_scope': 'local get_snapshot only; offline wait excluded; HTTP hooks are dispatch-to-response-headers evidence, not server execution duration'}
