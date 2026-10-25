---
title: Rolling back with Ferry 1.0
product: ferry
version: 1.0
doc_type: guide
released: 2024-06-01
---

## Rolling back a release

`ferry rollback` points the `current` symlink at the previous release on every host and
restarts the service. `ferry rollback --to <release-id>` picks a specific release; it must
still be among the releases kept on the hosts.

## Listing releases

`ferry releases` lists the releases on each host with their id, git commit and the time
they were deployed.
