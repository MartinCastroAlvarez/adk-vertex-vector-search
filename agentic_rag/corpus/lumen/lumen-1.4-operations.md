---
title: Operating Lumen 1.4
product: lumen
version: 1.4
doc_type: guide
released: 2024-03-12
---

## Backups

Take a backup with `lumen backup --out backup.lum`. The command copies every segment and
the write-ahead log into one file while the server keeps running. Restore it with
`lumen restore backup.lum --data ./data` on a stopped server. Backups are always full
copies in Lumen 1.4.

## Monitoring

The server exposes metrics in plain text at `/metrics` on port 7401. The most useful
metrics are `lumen_segments_total`, `lumen_wal_bytes` and `lumen_requests_total`.

## Upgrading from 1.3

Stop the server, replace the binary and start it again. The 1.4 segment format is
backward compatible with 1.3, so no migration step is needed.
