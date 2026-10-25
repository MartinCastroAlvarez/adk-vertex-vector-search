---
title: Lumen 2.0 configuration reference
product: lumen
version: 2.0
doc_type: reference
released: 2025-01-20
---

## Network settings

`port` defaults to 7410 in Lumen 2.0 (it was 7400 in 1.x). `bind` defaults to
`127.0.0.1`, so a new server only accepts local connections until you change it.
`max_connections` defaults to 1024.

## Authentication

Clients authenticate with API tokens. Create one with `lumen token create --name ci` and
pass it with `--token` or the `LUMEN_TOKEN` environment variable. Tokens can be limited to
read-only access with `--read-only` and revoked with `lumen token revoke`.

## Storage limits

`max_value_size` defaults to 4 MiB and can be lowered but not raised. Writes above the
limit fail with error LUM-210.

## Compaction settings

Automatic compaction starts when the share of dead bytes in a segment passes
`compaction.trigger_ratio`, which defaults to 0.5. `compaction.max_concurrent` limits how
many segments are rewritten at once and defaults to 2.
