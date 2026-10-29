# Development roadmap

This roadmap describes proposed work. It is not a list of capabilities already delivered or a commitment to deployment dates.

## Starting point

The historical connected study retrieved the latest equipment experience in four of four sequential situations. A remembered actuator-response calibration changed the command in two situations; after drift, the old calibration was rejected and the new calibration remained too uncertain to apply. Mean residual error was 1.22% lower than the recent-measurements baseline on this small study. See [results and qualifications](results.md).

The physics is simulated, the Cloud calls are real, and the control policy is numerical. This release does not demonstrate a physical shield, EMP cancellation, radio anti-jamming, or autonomous language-model control of hardware.

## Work packages and evidence required

| Work package | What to improve | Evidence required before claiming success |
| --- | --- | --- |
| Reliable learning after drift | Detect a stale response, collect sufficient new observations and accept a replacement calibration only when its uncertainty is controlled. | Frozen unseen drift scenarios; calibration error, rejection rate, recovery time and cases where adaptation worsens control. |
| Wider generalisation | Extend beyond four dependent episodes, two device contexts and the existing operating ranges. | Multiple independent histories, noise conditions and equipment changes; equal-information comparators; uncertainty intervals and retained negative results. |
| Component attribution | Separate the benefit of memory availability, typed retrieval, recency, ODVL ranking and current-only identification. | Ablations with identical eligible candidates and observations; report prediction, command changes and resulting error separately. |
| Retrieval at scale | Make the latest relevant experience available without treating textual similarity as an identity guarantee. | Large synthetic histories; exact-context recall, version correctness, access isolation and query latency distributions. |
| Edge continuity | Preserve a valid local fallback through Cloud latency, quota refusal, disconnection and process restart. | Bounded stale-state behaviour, measured deadlines, durable receipts, no duplicate mutation effects and explicit recovery outcomes. |
| Richer physics | Add multiple sensing points, axes, coupling and spatial mismatch using an appropriate electromagnetic model. | Resolution and model-validation studies; comparison with measurements; clearly stated domains where the model is inadequate. |
| Hardware pilot | Connect qualified sensing and bounded actuation in a controlled laboratory. | Independent metrology, measured actuator limits, repeatable paired comparisons and a documented stop/fallback procedure. |
| Product integration | Add authenticated operator roles, maintenance notices, configuration history and application-specific observability. | Traceable authority changes and decisions, reproducible incident review, and domain-specific acceptance tests. |

## Candidate applications

### Robotics and industrial measurement

A prospective system could retain the relationship between equipment configuration, sensor disturbance and an accepted calibration. The desired customer outcome is reduced repeated diagnosis and documented recovery after change. Hardware susceptibility and interaction with the robot's existing safety controller must be assessed separately. NASA/JPL's [CADRE rover test description](https://www.jpl.nasa.gov/images/pia26166-cadre-rover-undergoes-electromagnetic-testing/) illustrates the established role of EMI/EMC verification in robotics; it is not a validation or endorsement of this project.

### Protective wear and armour electronics

An exploratory integration could monitor the reliability of carried electronics, keep a history of anomalies and select only approved degraded operating modes. Before any human use, body proximity, motion, ergonomics, thermal behaviour, electromagnetic exposure and failure consequences would need independent assessment. No wearable protection or armour capability has been measured in this release.

### Satellites and space instrumentation

A prospective memory layer could preserve calibration lineage and help distinguish recurring onboard disturbances from new conditions. Its interface would need to respect flight-software authority, resource limits and disconnected operation. ESA documents [spacecraft electromagnetic compatibility verification](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Test_centre/Electromagnetic_compatibility); these established engineering obligations remain outside the present simulator's validation.

## Engineering boundary

The intended architecture is layered: physical protection and qualified electronics, a local bounded controller, then experience retrieval and supervisory analysis. The present API-connected prototype must not be represented as replacing shielding, filtering, grounding or hardware qualification. NASA Glenn's [EMI laboratory](https://www.nasa.gov/centers-and-facilities/glenn/electromagnetic-interference-laboratory/) describes those evaluation activities.

The next useful result is a reproducible improvement over strong baselines under an explicit operating envelope, including correct abstention and recovery. A broader product claim should follow that evidence.
