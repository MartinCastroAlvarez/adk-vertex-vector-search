---
title: Lumen 2.0 release notes
product: lumen
version: 2.0
doc_type: release-notes
released: 2025-01-20
---

## Highlights

Lumen 2.0 replaces the shared password with API tokens, adds automatic compaction,
key expiry (TTL) and Mirror, an asynchronous replication feature. The maximum value size
grows from 1 MiB to 4 MiB.

## Breaking changes

The default server port changes from 7400 to 7410, so that 1.x and 2.x servers can run
side by side during an upgrade. The `auth_password` setting is deprecated: it still works
in 2.0 but logs a warning on every start. Create tokens with `lumen token create` instead.

## Upgrading from 1.4

Run `lumen migrate --data ./data` once on a stopped server. The migration rewrites the
segment index to the 2.0 format and cannot be undone, so take a backup first.
