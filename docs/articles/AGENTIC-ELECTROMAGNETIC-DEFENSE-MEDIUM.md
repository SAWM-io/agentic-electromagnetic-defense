# Agentic Electromagnetic Defense: When a Controller Can Learn From Its Own History

### SAWM × Basira connects recent memory to present decisions. A Cloud experiment shows the connection working—and gives us a concrete next step toward more resilient machines.

*Research update · 29 October 2026*

A robot enters a familiar workspace. A motor starts nearby, a sensor becomes less reliable, and the robot has to decide whether yesterday's calibration still applies. An older measurement could help. It could also be precisely the wrong measurement to trust.

That is the question behind **Agentic Electromagnetic Defense**: can a system retrieve a relevant experience, test it against the present, use it to adjust its next action, and retain the outcome?

We have published a numerical demonstrator of that loop. Its sensors and magnetic compensation equipment are simulated. Its **Basira, TSST and ODVL Cloud calls are real**. The release includes the client program, comparison results and a path for developers to run their own isolated experiment.

The first result is modest but useful: the remembered response of an actuator actually changes the command when current observations support it. When the equipment changes, the controller rejects the old calibration.

## What the experiment actually does

The simulated equipment has one magnetic axis, one observation point, noise, delay and bounded commands. It is a reduced control model, not a full electromagnetic field solver.

The loop is explicit:

1. **Observe** the initial sensor measurements.
2. **Recall** the latest experience of that equipment and an earlier accepted calibration when needed.
3. **Check** their provenance, age, uncertainty and agreement with current measurements.
4. **Act** with a bounded correction—or keep the baseline when the evidence is insufficient.
5. **Compare and record** the resulting observations for a later decision.

Basira stores the completed experiences. TSST retrieves them through explicit identity filters. ODVL supplies memory state used in ranking frequency candidates. A local numerical controller calculates the command and enforces its limits. A language model is not issuing commands in this experiment.

We also found a practical integration problem: including an experience identifier in a text query did not ensure that the experience appeared in the returned top ten results. The stored data was present. Using the appropriate typed identity filters made the recent experiences accessible to the controller. Memory only helps if the relevant memory reaches the decision.

## The measured result

The latest study used four new sequential situations and six controller variants: **24 compared executions, 34 successful Cloud calls and 44 passing independent audit checks**. The original laboratory software also passed 255 local tests; that is a separate count from tests of this public release.

| Observation | Result |
| --- | --- |
| Latest equipment experience retrieved | 4 of 4 situations |
| Remembered calibration actually changes the command | 2 of 4 situations |
| Old calibration rejected after equipment drift | Both drift situations |
| Mean residual error versus recent measurements alone | 1.22% lower |
| Mean residual error versus current-only recalibration | 0.60% lower |

Mean residual RMS was **0.128530 simulated µT**, compared with **0.130112** using recent measurements alone and **0.129312** with current-only recalibration. RMS measures the size of the remaining error; these percentages are not success rates.

In the first situation, fresh recalibration did just as well. In the second, a previously accepted calibration remained compatible with present observations, while fitting a new calibration from those observations was too uncertain. This is the clearest example of the memory being useful.

Then we changed the equipment. The controller recalled the recent experiences but rejected the incompatible old response. Its new calibration remained uncertain, so complete relearning after that change is **not yet demonstrated**.

These are four dependent situations, not a large independent benchmark. They demonstrate a working connection and a small measured benefit. They do not establish general superiority or isolate ODVL's individual contribution. The [published results](https://github.com/SAWM-io/agentic-electromagnetic-defense/blob/main/docs/results.md) retain the comparisons and the cases without improvement.

## Anticipation does not require knowing the future

The controller estimates what comes next from measurements already received. The experiment evaluates forecasts 32 milliseconds into simulated time. Future observations are used afterward to score those forecasts, never to select the command.

The network is separated from the local control calculation: memory is retrieved and validated before use, then refreshed asynchronously. In the latest study, all 1,200 measured cache reads succeeded while actual HTTP requests were in flight. This measures access to a prepared local snapshot. It does **not** mean a Cloud request completes in microseconds or prove a hardware deadline.

For a future deployed machine, the fast protection loop would need to remain local, including when connectivity is lost. Cloud memory could support preparation, supervised learning between episodes and traceability.

## Where this research could lead

The following are development directions, not capabilities validated by this simulator.

**Robots and industrial equipment.** A robot could associate disturbances with a machine configuration, detect that a known calibration no longer fits, and request recalibration or enter a permitted degraded mode. The value would be fewer repeated investigations and a clearer record of why a control setting changed. Electromagnetic compatibility is already a real robotics engineering requirement: NASA/JPL describes chamber testing of its CADRE rover to examine subsystem interference and expected disturbances. That work is independent of SAWM. [NASA/JPL: CADRE testing](https://www.jpl.nasa.gov/images/pia26166-cadre-rover-undergoes-electromagnetic-testing/).

**Protective wear and armour electronics.** A future research system could track the reliability of sensors, power electronics and communications carried by a person, relate anomalies to earlier operating conditions, and recommend an approved fallback. Body proximity, motion, power limits and human safety would require dedicated validation. No protective suit, armour system or personal electromagnetic shield has been tested here.

**Satellites and space instruments.** Memory could help engineers distinguish recurring onboard interference from a new equipment condition and retain the justification for calibration changes. Spacecraft EMC is a substantial engineering discipline: ESA describes the need to verify both operation in an interference environment and compatibility between units within the same spacecraft. Our proposed contribution is a memory and decision layer around qualified instrumentation. [ESA: electromagnetic compatibility](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Test_centre/Electromagnetic_compatibility).

These possibilities require real sensors, qualified electronics and measured operating limits. Shielding, filtering and grounding remain engineering foundations; NASA Glenn's EMI laboratory explicitly evaluates them. This software experiment does not demonstrate cancellation of a destructive electromagnetic pulse, protection against component damage or radio anti-jamming. [NASA Glenn: EMI laboratory](https://www.nasa.gov/centers-and-facilities/glenn/electromagnetic-interference-laboratory/).

## Try the program with your own Cloud environment

The [public repository](https://github.com/SAWM-io/agentic-electromagnetic-defense) provides the demonstrator and HTTP integration. It contains no private implementation of SAWM, Basira, TSST or ODVL. Their services remain behind the authenticated Cloud API.

Install the repository's requirements, then run the baseline:

```bash
python run_demo.py --mode offline --output runs/offline-demo
```

Offline mode makes no Cloud calls and does not exercise Basira or ODVL. For an actual connected run, create your own isolated laboratory entity and restricted credentials through [Basira Cloud](https://basira.sawm.io/espace), then follow the repository's setup instructions:

```bash
python run_demo.py --mode cloud \
  --credentials /private/path/lab.env \
  --confirm-dedicated-lab \
  --output runs/cloud-demo
```

The credentials file stays outside Git. A fresh run creates its own history; it does not inherit the earlier laboratory's 44 experiences or promise identical results.

Our next milestones are larger independent comparisons, more reliable recalibration after drift, and a controlled hardware pilot. The ambition is practical: **a machine that can use experience without confusing an old success with evidence that the same action is right today.**

Explore the [code and results](https://github.com/SAWM-io/agentic-electromagnetic-defense) and the [development roadmap](https://github.com/SAWM-io/agentic-electromagnetic-defense/blob/main/docs/roadmap.md).
