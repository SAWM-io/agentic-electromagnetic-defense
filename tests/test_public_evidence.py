from pathlib import Path
import importlib.util
import json
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_verifier',ROOT/'scripts/verify_evidence.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)

def test_public_numerical_evidence_reconciles():
    result=verifier.verify()
    assert result['numerical_runs_verified']==24
    assert result['causal_forecasts_reconstructed']==168
    assert result['effective_calibration_corrections']==2

def test_tampered_summary_is_rejected_before_comparison(tmp_path):
    folder=tmp_path/'evidence';shutil.copytree(verifier.DEFAULT,folder)
    path=folder/'summary.json';data=json.loads(path.read_text())
    data['arms']['optimized']['mean_rms_uT']=0.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum mismatch'):
        verifier.verify(folder)

def test_future_samples_do_not_enter_public_forecast_reconstruction():
    import numpy as np
    folder=verifier.DEFAULT;config=verifier.read(folder/'protocol.json')['config']
    decisions=verifier.read(folder/'decisions.json')[0]['decisions']
    with np.load(folder/'traces.npz',allow_pickle=False) as traces:
        original=traces['e44_reference'].copy()
    decision=decisions[0];changed=original.copy();changed[decision['step']+1:]+=10.0
    assert verifier.predict_from_past(original,decision,config)==verifier.predict_from_past(changed,decision,config)
