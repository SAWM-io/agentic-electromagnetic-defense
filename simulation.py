"""Public numerical demonstration helpers. No private memory engines are included."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, math
from pathlib import Path
import numpy as np
CONDITIONS=("recurrent","chirp","regime_change","surprise_pulse","ambiguous_continuation","plant_drift","saturation")
FREQUENCY_GRID=np.arange(.4,9.01,.05)

@dataclass(frozen=True)
class Config:
    dt: float = .002
    duration: float = 4.0
    coil_tau: float = .018
    sensor_tau: float = .004
    sensor_delay: float = .006
    command_delay: float = .004
    command_limit: float = 8.0
    slew_per_second: float = 200.0
    noise_std: float = .02
    prefix: float = .75
    update_period: float = .5
    window: float = .75
    forecast_lead: float = .032
    minimum_std: float = .06
    minimum_explained: float = .55
    memory_candidates: int = 5



def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def disturbance(condition, seed, c):
    """Generator truth is used only by the plant and evaluation, never recall."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(c.duration / c.dt)) * c.dt
    f = rng.choice([.7, 1.3, 2.2, 3.5, 5.2, 7.4]) + rng.uniform(-.025, .025)
    amp, phase = rng.uniform(.5, 1.8), rng.uniform(-np.pi, np.pi)
    d = amp * np.sin(2 * np.pi * f * t + phase)
    if condition == 'chirp':
        d = amp * np.sin(2 * np.pi * (1.1*t + .8*t*t) + phase)
    elif condition == 'regime_change':
        d = np.where(t < 2, d, .7*amp*np.sin(2*np.pi*6.3*t+phase)+.5)
    elif condition == 'surprise_pulse':
        d = np.where(t < 2, 0.0, 2.0*np.exp(-((t-2)/.004)**2))
    elif condition == 'ambiguous_continuation':
        d = np.where(t < 2, d, -d)
    elif condition == 'saturation':
        d = np.where(t < 2, 0., 12.)
    elif condition not in ('recurrent', 'plant_drift'):
        raise ValueError(condition)
    return t, d, rng.normal(0, c.noise_std, (2, len(t)))



def sensor(values, noise, c):
    """Causal reference magnetometer, ideal spatial match to protected point."""
    delay = round(c.sensor_delay/c.dt)
    alpha = math.exp(-c.dt/c.sensor_tau)
    filt = np.empty_like(values); state = 0.
    for k, value in enumerate(values):
        state = alpha*state+(1-alpha)*float(value)
        filt[k] = state
    result = np.zeros_like(values)
    if delay: result[delay:] = filt[:-delay]
    else: result[:] = filt
    return result + noise



def harmonic_fit(t, observed, frequencies):
    """Fit only supplied observed samples; frequency is never read from truth."""
    frequencies = np.asarray(frequencies)
    angles = 2*np.pi*frequencies[:, None]*t[None, :]
    x = np.stack([np.sin(angles), np.cos(angles), np.ones_like(angles)], axis=2)
    gram = np.einsum('fnp,fnq->fpq', x, x)
    rhs = np.einsum('fnp,n->fp', x, observed)
    coef = np.linalg.solve(gram+np.eye(3)[None, :, :]*1e-7, rhs[..., None])[..., 0]
    errors = np.mean((np.einsum('fnp,fp->fn', x, coef)-observed)**2, axis=1)
    return coef, errors



def make_bank(c):
    """36 completed synthetic observations, separate seeds from all tests."""
    bank = []
    for j in range(36):
        t, d, noise = disturbance('recurrent', 1000+j, c)
        ref = sensor(d, noise[0], c)
        coef, errors = harmonic_fit(t, ref, FREQUENCY_GRID)
        ix = int(np.argmin(errors))
        bank.append({'node_id': f'cal-{j:03d}', 'seed':1000+j,
                     'frequency_hz':float(FREQUENCY_GRID[ix]),
                     'observed_fit_mse':float(errors[ix]),
                     'observation_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
                     'status':'synthetic_completed_calibration',
                     'available_before_test':True})
    return bank



def forecast(ref, c, mode, bank=(), ranker=None, blend=.7):
    """Schedule forecasts using ONLY a trailing prefix available at each step.

    The simulator's full reference array is never passed to ranker/fitting:
    each fit receives ref[start:k+1]. Future output is extrapolated from that fit.
    This is a simulated local controller clock, not measured ODVL runtime.
    """
    n = len(ref); t = np.arange(n)*c.dt
    out = np.zeros(n); decisions = []
    if mode in ('none', 'pi'): return out, decisions
    window = round(c.window/c.dt); stride = round(c.update_period/c.dt)
    for k in range(round(c.prefix/c.dt), n, stride):
        start = max(0, k-window+1)
        past = ref[start:k+1]
        now_t = t[start:k+1]-t[k]
        variance = float(np.var(past))
        stop = min(n, k+stride)
        decision = {'step':k, 'max_observed_step':k, 'accepted':False}
        if variance < c.minimum_std**2:
            decision['reason'] = 'insufficient_signal'
            decisions.append(decision); continue
        frequencies = FREQUENCY_GRID if mode == 'online' else [e['frequency_hz'] for e in bank]
        coef, errors = harmonic_fit(now_t, past, frequencies)
        if mode == 'online':
            selected = int(np.argmin(errors))
        else:
            # One representative per learned physical frequency: duplicate
            # calibration observations must not crowd out alternative models.
            eligible=[]; seen=set()
            for i in np.argsort(errors, kind='stable'):
                # ODVL orders already-qualified evidence; it is NOT the
                # physical compatibility validator. Apply the SAME fit gate
                # before either ordinary recall or ODVL sees a candidate.
                if 1-float(errors[int(i)])/variance < c.minimum_explained: continue
                f=float(frequencies[int(i)])
                if f in seen: continue
                seen.add(f); eligible.append(int(i))
                if len(eligible)==c.memory_candidates: break
            if not eligible:
                decision['reason']='no_qualified_memory'
                decisions.append(decision); continue
            candidates = [{'node_id':bank[int(i)]['node_id'],
                           'score':float(np.exp(-errors[int(i)]/max(variance, 1e-9)))} for i in eligible]
            ordered = ranker.rank(candidates) if mode == 'odvl' else candidates
            selected_id = ordered[0]['node_id']
            selected = next(i for i,e in enumerate(bank) if e['node_id']==selected_id)
            decision['eligible_ids'] = [e['node_id'] for e in candidates]
            decision['selected_episode'] = selected_id
            decision['ranked_ids'] = [e['node_id'] for e in ordered]
        explained = float(1-errors[selected]/variance)
        decision['explained_variance'] = explained
        if explained >= c.minimum_explained:
            f = float(frequencies[selected]); a,b,dc = coef[selected]
            future = t[k:stop]-t[k]+c.forecast_lead
            prediction = a*np.sin(2*np.pi*f*future)+b*np.cos(2*np.pi*f*future)+dc
            prediction = np.clip(prediction, -c.command_limit, c.command_limit)
            ramp = np.minimum((t[k:stop]-t[k])/.04, 1.)
            # Blend from previous FF value; changes remain slew limited in plant.
            last = out[k-1] if k else 0.
            out[k:stop] = last*(1-ramp)-blend*prediction*ramp
            decision.update(accepted=True, frequency_hz=f, reason='causal_harmonic_fit')
        else:
            decision['reason']='poor_fit_abstain'
        decisions.append(decision)
    return out, decisions



def plant(d, residual_noise, ff, c, gains=(.6, 30.), drift=False, active=True):
    """One-axis low-pass coil + delayed sensor + bounded PI/FF actuation.

    Output at k is recorded BEFORE action k changes coil at k+1. This prevents
    instantaneous cancellation or access to future disturbances. Command is in
    equivalent nominal microtesla, not an electrical hardware specification.
    """
    n=len(d); field=np.zeros(n); measured=np.zeros(n); command=np.zeros(n)
    sensor_state=0.; coil=0.; previous_y=0.; previous_ff=0.; desired=0.
    filtered=np.zeros(n); delay=round(c.sensor_delay/c.dt)
    command_delay=round(c.command_delay/c.dt)
    alpha_s=math.exp(-c.dt/c.sensor_tau)
    kp, ki=gains
    for k in range(n):
        field[k]=d[k]+coil
        sensor_state=alpha_s*sensor_state+(1-alpha_s)*field[k]
        filtered[k]=sensor_state
        y=(filtered[k-delay] if k>=delay else 0.)+residual_noise[k]
        measured[k]=y
        if active:
            desired += -kp*(y-previous_y)-ki*y*c.dt+(ff[k]-previous_ff)
            # Clamped incremental PI: no hidden integral can keep winding up.
            desired=float(np.clip(desired, -c.command_limit, c.command_limit))
            last=command[k-1] if k else 0.
            command[k]=float(np.clip(desired, last-c.slew_per_second*c.dt,
                                                last+c.slew_per_second*c.dt))
            desired=command[k]
        previous_y=y; previous_ff=ff[k]
        tau=c.coil_tau; gain=1.
        if drift and k*c.dt>=2:
            tau*=2.5; gain=.65
        applied=command[k-command_delay] if k>=command_delay else 0.
        alpha=math.exp(-c.dt/tau)
        coil=alpha*coil+(1-alpha)*gain*applied
    return {'residual':field, 'command':command, 'measured':measured, 'feedforward':ff}



def metrics(trace, d, c):
    r=trace['residual']; u=trace['command']; peak=float(np.max(np.abs(r)))
    return {'rms_uT':float(np.sqrt(np.mean(r*r))),
            'late_rms_uT':float(np.sqrt(np.mean(r[round(c.prefix/c.dt):]**2))),
            'peak_uT':peak,
            'peak_ratio_to_uncontrolled':peak/max(float(np.max(np.abs(d))),1e-12),
            'effort_uT2_s':float(np.sum(u*u)*c.dt),
            'saturation_fraction':float(np.mean(np.abs(u)>=c.command_limit-1e-8)),
            'first_event_residual_uT':float(r[round(2./c.dt)])}
