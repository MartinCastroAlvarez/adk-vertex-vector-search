---
title: Canary deploys in Ferry 1.5
product: ferry
version: 1.5
doc_type: guide
released: 2025-05-15
---

## Starting a canary

`ferry deploy --canary 10` deploys the new release to 10 percent of the hosts (at least
one) and stops. The rest of the fleet keeps serving the previous release.

## Promoting or aborting

Watch the canary hosts, then run `ferry promote` to deploy the release to the rest of the
fleet, or `ferry abort` to roll the canary hosts back. Ferry 1.5 never promotes or aborts
on its own.
