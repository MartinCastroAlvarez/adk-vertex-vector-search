---
title: Ferry 2.0 release notes
product: ferry
version: 2.0
doc_type: release-notes
released: 2026-06-30
---

## Highlights

Ferry 2.0 replaces canary deploys with progressive rollouts, adds external secret
providers and an `on_failure` hook, and moves the configuration to `ferry.yaml`.

## Breaking changes

`ferry.yaml` replaces `ferry.toml`. Ferry 2.0 still reads `ferry.toml` with a deprecation
warning; run `ferry config convert` to write the YAML file. `ferry deploy --canary` and
`ferry promote` are removed in favour of progressive rollouts. The default health check
timeout grows from 30 to 60 seconds.

## Upgrading from 1.5

Convert the configuration, replace canary steps in your CI scripts with
`ferry deploy --progressive`, and keep using `ferry.secrets` or move to an external
provider.
