"""Analytic observable fixtures only; no future campaign or Cloud execution."""
import copy
import json

import numpy as np
import pytest

from residual_experience import (export_compact_experience, fit_residual_experience,
                                 propose_residual_correction)


CONTEXT = {'device_id': 'analytic-fixture-A', 'control_mode': 'pi-baseline'}
PROVENANCE = {'scientific_status': 'simulated_sensor_data', 'completed': True,
              'baseline_policy': 'pi_only', 'source': 'observed_sensor_trace'}


def fixture(*, gain=.65, a=.8, delay=3, noise=.001, count=500):
    """Known ARX is used solely to generate measured unit-test channels."""
    rng = np.random.default_rng(519)
    command = np.repeat(rng.uniform(-1,1,(count+4)//5),5)[:count]
    contrast = np.zeros(count)
    for k in range(1,count):
        contrast[k] = a*contrast[k-1]+gain*(1-a)*(command[k-delay] if k>=delay else 0)+.002
    reference = np.sin(np.arange(count)*.043)
    measured = reference+contrast+rng.normal(0,noise,count)
    return reference,measured,command


def fit(channels=None, **kwargs):
    ref,residual,command = channels or fixture()
    return fit_residual_experience(ref,residual,command,dt_s=.002,context=CONTEXT,
        episode_index=1,episode_id='analytic-completed-1',baseline_slice=(0,len(ref)),
        command_limit=8,provenance=PROVENANCE,**kwargs)


def propose(experience, channels=None, **kwargs):
    ref,residual,command = channels or fixture()
    args = {'current_reference':ref,'current_measured_residual':residual,
        'current_applied_command':command,'current_context':CONTEXT,'current_episode_index':2,
        'nominal_feedforward':np.r_[np.zeros(len(ref)),np.ones(100)],
        'prefix_samples':len(ref),'dt_s':.002,'command_limit':8}
    args.update(kwargs)
    return propose_residual_correction(experience,**args)


def test_identifies_observed_response_and_exports_small_real_evidence():
    result = fit()
    assert result['accepted'], result['reasons']
    model = export_compact_experience(result)
    assert model['d']==3
    assert abs(model['a']-.8)<.005
    assert abs(model['g']-.65)<.01
    assert model['lo']<.65<model['hi']
    assert len(json.dumps(model,separators=(',',':'))) <= 350
    diagnostics = result['diagnostics']
    assert diagnostics['training_end_exclusive']==diagnostics['validation_start']
    assert diagnostics['validation_nmse']<.01
    assert diagnostics['correction_effectiveness_established'] is False


def test_fit_ignores_every_sample_after_explicit_baseline():
    channels=fixture();changed=[np.r_[v,np.full(50,np.nan)] for v in channels]
    before=fit_residual_experience(*channels,dt_s=.002,context=CONTEXT,
        episode_index=1,episode_id='fixture',baseline_slice=(0,500),command_limit=8,provenance=PROVENANCE)
    after=fit_residual_experience(*changed,dt_s=.002,context=CONTEXT,
        episode_index=1,episode_id='fixture',baseline_slice=(0,500),command_limit=8,provenance=PROVENANCE)
    assert before==after


def test_heldout_inconsistency_rejects_instead_of_selecting_on_validation():
    ref,measured,command=fixture()
    measured[325:]+=np.random.default_rng(877).normal(0,3,len(measured)-325)
    result=fit((ref,measured,command))
    assert not result['accepted']
    assert 'heldout_response_prediction_failed' in result['reasons']
    assert result['model'] is None


def test_constant_command_cannot_identify_actuator():
    ref,measured,command=fixture()
    result=fit((ref,measured,np.zeros_like(command)))
    assert not result['accepted']
    assert 'insufficient_command_excitation' in result['reasons']
    with pytest.raises(ValueError,match='only_accepted'):
        export_compact_experience(result)


def test_saturation_invalidates_learning_and_current_application():
    ref,measured,command=fixture();command[:50]=8
    bad=fit((ref,measured,command))
    assert not bad['accepted'] and 'saturated_baseline' in bad['reasons']
    _,decision=propose(fit(),(ref,measured,command))
    assert not decision['applied'] and decision['reason']=='current_saturated_baseline'


def test_bounded_correction_changes_only_future_feedforward():
    ff,decision=propose(fit())
    assert decision['applied'] and decision['scale']==1.2
    np.testing.assert_array_equal(ff[:500],np.zeros(500))
    np.testing.assert_array_equal(ff[500:],np.ones(100)*1.2)
    assert decision['max_observed_step']==499
    assert decision['correction_start_step']==500
    assert decision['isolated_physical_benefit_claimed'] is False
    ff,decision=propose(fit(),nominal_feedforward=np.r_[np.zeros(500),np.ones(100)*7.9])
    assert np.max(ff)<=8


def test_future_observations_and_nominal_actions_cannot_change_choice():
    experience=fit();channels=fixture();_,before=propose(experience)
    future=[np.r_[v,np.full(100,np.nan)] for v in channels]
    _,after=propose(experience,future,prefix_samples=500,
        nominal_feedforward=np.r_[np.zeros(500),np.linspace(-7,7,100)])
    assert before==after


def test_stale_same_device_response_is_rejected_by_current_PI_prefix():
    learned=fit();_,decision=propose(learned,fixture(gain=1.))
    assert not decision['applied']
    assert decision['reason']=='old_response_inconsistent_with_current_PI_prefix'
    assert decision['current_response_gain_ratio']>1.15


def test_context_sample_rate_and_future_identity_are_fail_closed():
    learned=fit()
    for kwargs,reason in [({'current_context':{'device_id':'other-device'}},'observable_context_mismatch'),
            ({'dt_s':.004},'sample_interval_mismatch'),
            ({'current_episode_index':1},'experience_not_completed_before_current_episode')]:
        ff,decision=propose(learned,**kwargs)
        assert not decision['applied'] and decision['reason']==reason
        np.testing.assert_array_equal(ff,np.r_[np.zeros(500),np.ones(100)])


def test_unobserved_or_uncompleted_provenance_and_hidden_context_rejected():
    for override in [{'completed':False},{'completed':1},{'source':'generator_truth'},
                     {'scientific_status':'hardware'},{'baseline_policy':'hybrid'}]:
        with pytest.raises(ValueError,match='provenance'):
            fit_residual_experience(*fixture(),dt_s=.002,context=CONTEXT,
                episode_index=1,episode_id='bad',baseline_slice=(0,500),command_limit=8,
                provenance={**PROVENANCE,**override})
    with pytest.raises(ValueError,match='observable_context'):
        fit_residual_experience(*fixture(),dt_s=.002,context={'device_id':'A','hidden_gain':.65},
            episode_index=1,episode_id='bad',baseline_slice=(0,500),command_limit=8,provenance=PROVENANCE)


def test_uncertain_gain_and_inconsistent_compact_parameters_cannot_actuate():
    model=export_compact_experience(fit())
    for update in [{'lo':.3,'hi':1.2},{'g':1.3},{'a':float('nan')},{'ep':-1},{'v':True}]:
        _,decision=propose({**model,**update})
        assert not decision['applied']


def test_unity_gain_uncertainty_does_not_manufacture_correction():
    learned=fit(fixture(gain=1.))
    assert learned['accepted']
    _,decision=propose(learned,fixture(gain=1.))
    assert not decision['applied']
    assert decision['reason']=='gain_correction_direction_uncertain'


def test_high_gain_reduces_feedforward_and_requires_PI_prefix():
    learned=fit(fixture(gain=1.5))
    _,decision=propose(learned,fixture(gain=1.5))
    assert decision['applied'] and decision['scale']==.8
    _,decision=propose(learned,fixture(gain=1.5),nominal_feedforward=np.ones(600))
    assert not decision['applied'] and decision['reason']=='current_prefix_not_PI_only'


def test_short_or_nonfinite_observations_do_not_create_experience():
    result=fit(tuple(v[:60] for v in fixture()))
    assert not result['accepted'] and 'insufficient_baseline_samples' in result['reasons']
    ref,residual,command=fixture();ref[3]=np.nan
    with pytest.raises(ValueError,match='finite_observed'):
        fit((ref,residual,command))
