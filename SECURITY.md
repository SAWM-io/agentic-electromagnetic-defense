# Security and responsible integration

This repository is a numerical research demonstrator and Cloud API client. It does not install a device driver, control physical emitters, run a jammer, or claim EMP protection.

## Keep access outside Git

- Use a dedicated laboratory space and entity, with grants limited to that entity and the operations you need.
- Store populated credential files outside the checkout, readable only by their owner (`chmod 600`). Never paste keys into an issue, command argument, screenshot, or article.
- Runtime journals and generated runs can contain Cloud identifiers. `runs/` is ignored. Review and redact them before sharing.
- Rotate an exposed key through your Cloud account. Removing a file from the latest commit does not remove it from Git history.

## Preserve the separation of responsibilities

The hosted services supply memory and retrieval results. A local controller validates observations and enforces bounded simulated commands. A recalled experience is not automatically correct or applicable. Missing, unauthorized or incompatible Cloud data must not be silently replaced with a claimed Cloud result.

Physical implementations would require independent hardware protection, a local real-time control loop, authenticated configuration, sensor validation and a tested safe state. Cloud availability is not a substitute for those controls. Neither local cache timing nor numerical convergence establishes a hardware deadline.

Report defects without secrets or customer data through the repository issues. For a sensitive issue, contact SAWM through the official website rather than posting exploitation details or access credentials publicly.
