---
title: Mirror replication in Lumen 2.0
product: lumen
version: 2.0
doc_type: guide
released: 2025-01-20
---

## What Mirror does

Mirror streams the write-ahead log of a primary server to one or more read-only replicas.
Replication is asynchronous: the primary acknowledges a write before replicas apply it,
so a replica can lag behind and a failed primary can lose the most recent writes.

## Adding a replica

Run `lumen mirror add --primary db1:7410` on the replica. The replica copies a full
snapshot first and then follows the log. `lumen mirror status` shows the lag of every
replica in milliseconds.

## Lag alerts

The primary logs a warning when a replica falls behind by more than
`mirror.lag_alert_ms`, which defaults to 5000 milliseconds. Failover is manual: promote a
replica with `lumen mirror promote`.
