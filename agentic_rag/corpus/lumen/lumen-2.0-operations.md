---
title: Operating Lumen 2.0
product: lumen
version: 2.0
doc_type: guide
released: 2025-01-20
---

## Key expiry (TTL)

Lumen 2.0 can expire keys. Write with `lumen put session:42 data --ttl 3600` to delete the
key after an hour. Expired keys disappear from reads at once and their space is reclaimed
by the next compaction.

## Backups

`lumen backup --out backup.lum` still takes a full copy. On servers with Mirror replicas,
run backups on a replica to keep the load off the primary.

## Monitoring

Metrics moved to `/metrics` on the main port (7410). New metrics include
`lumen_compactions_total`, `lumen_expired_keys_total` and `lumen_mirror_lag_ms`.
