# Publication validation — 29 October 2026

The public package was validated separately from the original connected experiment.

| Check | Result | What it establishes |
| --- | --- | --- |
| Public repository test suite | 166 passed | Numerical behavior, causal inputs, client validation, failure handling and mocked Cloud integration |
| Public offline CLI | 6 episodes, 12 calculations, zero HTTP attempts | The standalone numerical example runs without credentials or private engines |
| Evidence verifier | 9 publication file hashes, 24 numerical runs, 168 reconstructed forecasts | Published measurements and their causal forecast calculations are internally consistent |
| Original connected experiment | 34 real Cloud calls; 44 independent checks passed | The separately recorded experiment used real Basira/TSST/ODVL APIs |
| Gitleaks 8.30.1 and publication-boundary checks | No detected secrets or forbidden private-engine imports | Checks of the prepared public snapshot, not a guarantee against every possible disclosure |

The new six-episode CLI was tested offline and with a mocked Cloud transport. It was not presented as a new real-Cloud campaign. The real-Cloud results in this repository come from the recorded four-episode experiment, with its preceding history. Do not equate these two protocols or promise identical results on a new account.

The original 255-test laboratory suite is a different test set from this public package's 166 tests. Historical results are publication copies; original private Cloud receipt journals are not shipped.

## Recheck locally

```sh
python -m pytest -q
python scripts/verify_evidence.py
python run_demo.py --mode offline --output runs/another-offline-run
python scripts/check_publication.py
```

GitHub Actions runs these commands on Linux with Python 3.12. A green workflow verifies this public package; it does not execute billable Cloud requests or qualify physical protection hardware.
