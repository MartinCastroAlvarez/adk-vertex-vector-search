---
title: Lumen 1.4 configuration reference
product: lumen
version: 1.4
doc_type: reference
released: 2024-03-12
---

Lumen 1.4 reads its settings from `lumen.conf` in the data directory. Command-line flags
override the file.

## Network settings

`port` sets the TCP port of the server and defaults to 7400. `bind` sets the listen
address and defaults to `0.0.0.0`. `max_connections` limits concurrent clients and
defaults to 256.

## Authentication

Lumen 1.4 protects the server with a single shared password. Set `auth_password` in
`lumen.conf`; clients send it with `--password` or the `LUMEN_PASSWORD` environment
variable. There are no per-user accounts and no tokens in this release.

## Storage limits

`max_value_size` is fixed at 1 MiB in Lumen 1.4 and cannot be raised. Writes larger than
the limit fail with error LUM-210. `segment_size` controls the size of segment files and
defaults to 64 MiB.

## Compaction

Lumen 1.4 has no automatic compaction. Deleted and overwritten values stay on disk until
you run `lumen compact` by hand, which rewrites every segment and pauses writes while it
runs.
