---
title: Lumen 3.0 configuration reference
product: lumen
version: 3.0
doc_type: reference
released: 2026-02-10
---

## Network settings

`port` defaults to 7410 and `bind` to `127.0.0.1`. `max_connections` defaults to 4096.

## Authentication and TLS

Lumen 3.0 accepts API tokens and mutual TLS. Enable mutual TLS with `tls.mode = "mutual"`,
`tls.cert`, `tls.key` and `tls.client_ca`. Passwords are no longer supported.

## Storage limits

`max_value_size` defaults to 16 MiB. Keys are still limited to 512 bytes.

## Tiered compaction

Lumen 3.0 groups segments into tiers by size and merges segments of the same tier.
`compaction.tiers` sets the number of tiers and defaults to 4; `compaction.trigger_ratio`
still applies inside each tier and defaults to 0.5.
