#!/usr/bin/env python3
"""Verify the published numerical evidence without Cloud access or private engines.

This checks the publication copy. It cannot authenticate omitted Cloud receipts.
Run from any directory: python scripts/verify_evidence.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'evidence' / 'current-decision'
ARMS = ('online', 'hybrid_v1', 'cached_hybrid', 'utility', 'optimized', 'fresh_current_response')


def read(path):
    return json.loads(path.read_text())


def close(actual, expected, name):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f'Numerical mismatch: {name}')


def predict_from_past(reference, decision, config):
    """Reconstruct a stored forecast using only samples available at its step."""
    k = decision['step']; window = round(config['window'] / config['dt'])
    start = max(0, k - window + 1)
    past = reference[start:k+1]
    times = (np.arange(start, k+1) - k) * config['dt']
    frequency = decision.get('online_frequency_hz') or decision['frequency_hz']
    weight = float(decision.get('memory_weight', 0.))
    frequencies = [frequency]
    if weight > 0:
        frequencies.append(decision['memory_frequency_hz'])
    forecasts = []
    for f in frequencies:
        x = np.column_stack((np.sin(2*np.pi*f*times), np.cos(2*np.pi*f*times), np.ones(len(times))))
        coefficients = np.linalg.solve(x.T@x + np.eye(3)*1e-7, x.T@past)
        future = np.array([np.sin(2*np.pi*f*config['forecast_lead']), np.cos(2*np.pi*f*config['forecast_lead']), 1.])
        forecasts.append(float(future@coefficients))
    value = (1-weight)*forecasts[0] + weight*forecasts[1] if weight > 0 else forecasts[0]
    return float(np.clip(value, -config['command_limit'], config['command_limit']))


def verify(folder=DEFAULT):
    folder = Path(folder)
    manifest = read(folder/'manifest.json')['files']
    for name, digest in manifest.items():
        if Path(name).name != name:
            raise ValueError('Manifest must name only immediate artifact files')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Publication checksum mismatch: {name}')
    summary = read(folder/'summary.json'); protocol = read(folder/'protocol.json')
    rows = read(folder/'episodes.json'); decisions = read(folder/'decisions.json')
    flow = read(folder/'memory-flow.json'); audit = read(folder/'audit-summary.json')
    config = protocol['config']; dt = config['dt']; prefix = round(config['prefix']/dt)
    assert summary['episodes'] == 4 and summary['controller_runs'] == 24
    assert protocol['arms'] == list(ARMS)
    assert len(rows) == len(decisions) == 24
    assert Counter((r['episode_index'], r['arm']) for r in rows) == Counter((i,a) for i in range(44,48) for a in ARMS)
    assert summary['public_protocol_sha256'] == hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()
    assert audit['all_audit_checks_passed'] and len(audit['checks']) == 44 and all(audit['checks'].values())
    assert summary['actual_cloud_calls'] == audit['cloud_counts']['accepted'] == 34
    total_forecasts = 0; aggregate = {a:[] for a in ARMS}
    rms = {a:[] for a in ARMS}; corrected = []
    with np.load(folder/'traces.npz', allow_pickle=False) as traces:
        for row in rows:
            i, arm = row['episode_index'], row['arm']; base = f'e{i:02d}'
            residual = traces[f'{base}_{arm}_residual']; command = traces[f'{base}_{arm}_command']
            measured = traces[f'{base}_{arm}_measured']; ff = traces[f'{base}_{arm}_feedforward']
            reference = traces[f'{base}_reference']; disturbance = traces[f'{base}_disturbance']
            assert np.isfinite(residual).all() and np.isfinite(command).all() and np.isfinite(measured).all()
            for name in ('residual','command','measured'):
                assert np.array_equal(traces[f'{base}_{arm}_{name}'][:prefix+1], traces[f'{base}_online_{name}'][:prefix+1])
            assert np.all(ff[:prefix+1] == 0)
            assert np.max(np.abs(command)) <= config['command_limit'] + 1e-10
            assert np.max(np.abs(np.diff(np.r_[0.,command]))) <= config['slew_per_second']*dt + 1e-10
            peak = float(np.max(np.abs(residual)))
            computed = {
                'rms_uT':np.sqrt(np.mean(residual**2)),
                'late_rms_uT':np.sqrt(np.mean(residual[prefix:]**2)),
                'peak_uT':peak,
                'peak_ratio_to_uncontrolled':peak/max(float(np.max(np.abs(disturbance))),1e-12),
                'effort_uT2_s':np.sum(command**2)*dt,
                'saturation_fraction':np.mean(np.abs(command)>=config['command_limit']-1e-8),
                'first_event_residual_uT':residual[round(2./dt)],
            }
            for key, value in computed.items(): close(value,row['metrics'][key], f'{i}/{arm}/{key}')
            rms[arm].append(float(computed['rms_uT']))
            normalized = np.sqrt(np.mean(measured[prefix:]**2))/max(np.sqrt(np.mean(reference[prefix:]**2)),.06)
            close(np.sqrt(np.mean(measured[prefix:]**2)),row['measured_outcome']['measured_rms_uT'],'measured RMS')
            close(normalized,row['measured_outcome']['normalized_measured_rms'],'normalized RMS')
            for refarm in ('online','utility'):
                delta = np.max(np.abs(command-traces[f'{base}_{refarm}_command']))
                close(delta,row[f'command_max_abs_delta_vs_{refarm}'],'command difference')
                close(np.max(np.abs(ff-traces[f'{base}_{refarm}_feedforward'])),row[f'feedforward_max_abs_delta_vs_{refarm}'],'FF difference')
            assert row['actual_command_changed_vs_online'] == (row['command_max_abs_delta_vs_online']>1e-10)
            chosen = next(d for d in decisions if d['episode_index']==i and d['arm']==arm)
            if arm == 'optimized':
                assert chosen['residual_decision']['applied'] == row['actual_residual_FF_effect_vs_nominal']
                corrected.append(chosen['residual_decision']['applied'])
            for decision in chosen['decisions']:
                assert decision.get('max_observed_step', decision['step']) <= decision['step']
            for point in row['forecast_32ms']['points']:
                decision = next(d for d in chosen['decisions'] if d['step']==point['decision_step'])
                assert point['last_observed_step_used'] == point['decision_step'] < point['target_step']
                assert point['target_step']-point['decision_step'] == round(config['forecast_lead']/dt)
                prediction = predict_from_past(reference,decision,config)
                close(prediction,point['predicted_reference_uT'],'causal forecast')
                close(reference[point['target_step']],point['observed_future_reference_uT'],'future evaluation target')
                error = prediction-reference[point['target_step']]
                close(error,point['error_uT'],'forecast error')
                aggregate[arm].append(error); total_forecasts += 1
            for name in ('residual','command','measured','feedforward'):
                assert np.array_equal(traces[f'{base}_hybrid_v1_{name}'],traces[f'{base}_cached_hybrid_{name}'])
    for arm in ARMS:
        close(np.mean(rms[arm]),summary['arms'][arm]['mean_rms_uT'],'summary RMS')
        close(np.sqrt(np.mean(np.square(aggregate[arm]))),summary['arms'][arm]['forecast_32ms_RMSE_uT'],'summary forecast RMS')
        close(np.mean(np.abs(aggregate[arm])),summary['arms'][arm]['forecast_32ms_MAE_uT'],'summary forecast MAE')
    assert corrected == [True,True,False,False]
    assert len(flow)==4 and all(r['latest_target_returned'] and r['latest_target'] in r['returned_candidates'] for r in flow)
    assert all(r['maximum_command_delta_vs_utility']>0 for r in flow[:2])
    assert all(r['maximum_command_delta_vs_utility']==0 for r in flow[2:])
    hot = read(folder/'cache-timings.json')
    assert sum(r['hits'] for r in hot)==sum(r['samples'] for r in hot)==1200
    return {'publication_hashes_verified':len(manifest),'numerical_runs_verified':len(rows),
            'causal_forecasts_reconstructed':total_forecasts,'latest_memory_returned_reported':4,
            'effective_calibration_corrections':sum(corrected),
            'mean_RMS_uT':{a:float(np.mean(rms[a])) for a in ARMS},
            'scope':'Published numerical copies only. No live Cloud authentication, hardware test or private-engine replay is performed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence),indent=2))

if __name__=='__main__': main()
