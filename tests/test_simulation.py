import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from simulation import Config, disturbance, sensor, plant, forecast, make_bank

def test_actuator_cannot_cancel_unseen_leading_edge():
    c=Config(); _,d,n=disturbance('surprise_pulse',10012,c)
    tr=plant(d,np.zeros_like(d),np.zeros_like(d),c)
    index=round(2/c.dt)
    assert tr['residual'][index]==d[index]==2.0

def test_actuator_limit_and_slew():
    c=Config(); _,d,n=disturbance('saturation',10001,c)
    tr=plant(d,n[1],-np.ones_like(d)*200,c)
    assert np.max(np.abs(tr['command']))<=c.command_limit+1e-12
    assert np.max(np.abs(np.diff(tr['command'])))<=c.slew_per_second*c.dt+1e-12

def test_forecast_cannot_read_changed_future():
    c=Config(); _,d,n=disturbance('recurrent',10000,c)
    ref=sensor(d,n[0],c); altered=ref.copy()
    split=round(2/c.dt); altered[split:]+=50
    bank=make_bank(c)
    for mode in ['online','memory']:
        a,da=forecast(ref,c,mode,bank)
        b,db=forecast(altered,c,mode,bank)
        np.testing.assert_array_equal(a[:split],b[:split])
        assert [x for x in da if x['step']<split]==[x for x in db if x['step']<split]

def test_memory_has_only_completed_calibration():
    bank=make_bank(Config())
    assert len({e['node_id'] for e in bank})==len(bank)==36
    assert all(e['seed']<2000 and e['available_before_test'] for e in bank)

def test_uncontrolled_output_is_disturbance():
    c=Config(); _,d,n=disturbance('chirp',10000,c)
    tr=plant(d,n[1],np.zeros_like(d),c,active=False)
    np.testing.assert_array_equal(tr['residual'],d)

def test_reference_sensor_is_causal():
    c=Config(); x=np.zeros(200); y=x.copy(); y[100:]=10.
    noise=np.zeros(200)
    np.testing.assert_array_equal(sensor(x,noise,c)[:100],sensor(y,noise,c)[:100])

def test_quiet_prefix_abstains():
    c=Config(); ff,decisions=forecast(np.zeros(2000),c,'memory',make_bank(c))
    assert not np.any(ff)
    assert all(not d['accepted'] for d in decisions)

def test_candidate_frequencies_are_distinct():
    c=Config(); _,d,n=disturbance('recurrent',12345,c)
    bank=make_bank(c); frequencies={r['node_id']:r['frequency_hz'] for r in bank}
    _,decisions=forecast(sensor(d,n[0],c),c,'memory',bank)
    for decision in decisions:
        if 'eligible_ids' in decision:
            ff=[frequencies[key] for key in decision['eligible_ids']]
            assert len(ff)==len(set(ff))

def test_ranker_only_receives_physically_qualified_candidates():
    class RecordingRanker:
        calls=0
        def rank(self, candidates):
            self.calls+=1
            assert all(r['score'] >= np.exp(-(1-Config().minimum_explained))-1e-12 for r in candidates)
            return list(reversed(candidates))
    c=Config(); _,d,n=disturbance('recurrent',1300,c)
    ranker=RecordingRanker()
    forecast(sensor(d,n[0],c),c,'odvl',make_bank(c),ranker)
    assert ranker.calls>0
