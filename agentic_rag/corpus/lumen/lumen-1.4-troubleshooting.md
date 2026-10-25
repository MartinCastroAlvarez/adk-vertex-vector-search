---
title: Troubleshooting Lumen 1.4
product: lumen
version: 1.4
doc_type: troubleshooting
released: 2024-03-12
---

## LUM-104: lock file held

LUM-104 means another process holds `LOCK` in the data directory. Only one server can
open a data directory at a time. Stop the other process; if it crashed, delete the
stale `LOCK` file after making sure no `lumen` process is running.

## LUM-210: value too large

LUM-210 means a write exceeded `max_value_size`, which is 1 MiB in Lumen 1.4. Split the
value into several keys or store large blobs elsewhere and keep a reference in Lumen.

## Slow writes after many deletes

Lumen 1.4 never compacts on its own, so heavy delete workloads leave dead values on disk
and slow down writes. Schedule `lumen compact` during a quiet period.
